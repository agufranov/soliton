// Where does a GPU step actually go?
//
//   node serve.mjs &
//   node tools/gpu-profile.mjs [n] [rounds]
//
// Times each kernel in isolation by recording M repetitions into one submit and
// waiting for the queue to drain, then converts to effective memory bandwidth.
// A pseudospectral step is entirely memory bound, so bandwidth is the number
// that says whether a kernel is good or bad - "ms" alone tells you nothing.
//
// One sample per kernel is not a measurement on this machine. The iGPU shares
// DDR with the CPU and ramps its clocks over about a second, so a cold or
// unlucky kernel reads at half its real speed - an unchanged rowFFT was seen
// anywhere from 10 to 28 GB/s. So: warm up first, sweep the kernels
// round-robin so they all see the same drift, report the median and the
// spread. Treat any difference smaller than the spread as nothing.
//
// It also measures the fixed cost of onSubmittedWorkDone, because the frame
// loop paces itself with that and a large fixed latency there would make the
// app's own ms/step meaningless.

import { chromium } from 'playwright-core';

const N = Number(process.argv[2] || 512);
const ROUNDS = Number(process.argv[3] || 5);

const browser = await chromium.launch({
  channel: 'msedge', headless: true,
});
const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text()); });
await page.goto('http://localhost:8123/', { waitUntil: 'domcontentloaded' });

const r = await page.evaluate(async ({ n, ROUNDS }) => {
  const { Grid } = await import('/src/core/grid.js');
  const S = await import('/src/core/webgpuShaders.js');
  const { requestGpuContext, WebGpuBackend } = await import('/src/core/webgpuBackend.js');
  const { modelById } = await import('/src/models/index.js');

  const ctx = await requestGpuContext();
  if (!ctx) return { error: 'no adapter' };
  const grid = new Grid(n, n, 45, 45);
  const b = new WebGpuBackend(ctx, grid);
  const dev = b.device;
  const bytes = grid.n * 8;                       // one complex f32 field

  // Five distinct fields: etdStage reads four and writes one, and handing it
  // the same buffer twice would let the cache serve a read the accounting has
  // already charged to DRAM - which is how it used to report 59 GB/s on a
  // machine whose ceiling is 28.
  const a1 = b.alloc(), a2 = b.alloc(), a3 = b.alloc(), a4 = b.alloc(), a5 = b.alloc();

  // Fixed latency of the drain signal itself: submit nothing, await.
  let t0 = performance.now();
  for (let i = 0; i < 20; i++) await dev.queue.onSubmittedWorkDone();
  const drainLatency = (performance.now() - t0) / 20;

  const time = async (reps, record) => {
    b.flush();
    await dev.queue.onSubmittedWorkDone();
    const t = performance.now();
    for (let i = 0; i < reps; i++) record();
    b.flush();
    await dev.queue.onSubmittedWorkDone();
    return (performance.now() - t) / reps;
  };

  // The kernels, as jobs rather than as a sequence of measurements. Timing
  // them one after another was wrong twice over: this iGPU's clocks ramp for
  // about a second, and a busy CPU steals DDR from it, so whichever kernel
  // happened to run first or during a hiccup got a number that had nothing to
  // do with the kernel. Measured spread on one unchanged kernel across runs
  // was 10-28 GB/s. Hence: warm up, then sweep round-robin, then report the
  // median of several rounds and the spread, so a difference smaller than the
  // spread cannot be mistaken for a result.
  const rowsF = S.fftRowSrc(n, false);
  const tr = S.transposeSrc(n, n);
  const redOut = b._buffer(S.RED_GROUPS * 16, GPUBufferUsage.STORAGE);

  const model = modelById('kpI');
  const params = {};
  for (const p of model.params) params[p.key] = p.value;
  const solver = model.create(b, grid, model.dt * (128 / n), params);
  solver.applyPreset(model.presets[0]);

  const jobs = [
    // Elementwise: read one field, write one.
    { name: 'copy', reps: 200, traffic: 2 * bytes, run: () => b.copy(a1, a2) },
    { name: 'scale', reps: 200, traffic: 2 * bytes, run: () => b.scale(a1, 0.5) },
    // The heaviest elementwise kernel ETDRK4 uses: four fields in, one out.
    { name: 'etdStage', reps: 200, traffic: 5 * bytes, run: () => b.etdStage(a5, a1, a2, a3, a4) },
    { name: 'rowFFT', reps: 200, traffic: 2 * bytes, run: () => b._run(rowsF, [a1.buf], [n]) },
    { name: 'transpose', reps: 200, traffic: 2 * bytes, run: () => b._run(tr, [a1.buf, a2.buf], [n / 16, n / 16]) },
    // Three passes each (rows, transpose, rows), two field-touches per pass.
    { name: 'forward (fft2)', reps: 100, traffic: 6 * bytes, run: () => b.forward(a1) },
    { name: 'inverse (fft2)', reps: 100, traffic: 6 * bytes, run: () => b.inverse(a1) },
    { name: 'reduce4', reps: 100, traffic: bytes, run: () => b._run(S.reduceSrc(grid.n), [a1.buf, redOut], [S.RED_GROUPS]) },
    // ETDRK4 does 4 nonlinear evaluations, each an inverse (6) + a forward (6)
    // + the fused derivative/mask (3), plus 28 field-touches of combine kernels.
    { name: 'шаг KP-I', reps: 50, traffic: (4 * (6 + 6 + 3) + 28) * bytes, run: () => solver.step() },
  ];

  // Warm up: compile every pipeline, then hold the device busy long enough for
  // the clocks to come up. Without this the first job measured reads low.
  for (const j of jobs) j.run();
  b.flush();
  await dev.queue.onSubmittedWorkDone();
  const warmUntil = performance.now() + 1500;
  while (performance.now() < warmUntil) await time(50, () => b.copy(a1, a2));

  const samples = new Map(jobs.map((j) => [j.name, []]));
  for (let round = 0; round < ROUNDS; round++) {
    for (const j of jobs) samples.get(j.name).push(await time(j.reps, j.run));
  }

  const out = { n, adapter: ctx.label, drainLatency, bytes, rounds: ROUNDS, ops: [] };
  for (const j of jobs) {
    const s = samples.get(j.name).sort((x, y) => x - y);
    const gbs = (ms) => (j.traffic / 1e9) / (ms / 1000);
    out.ops.push({
      name: j.name, ms: s[s.length >> 1], traffic: j.traffic,
      gbs: gbs(s[s.length >> 1]), best: gbs(s[0]), worst: gbs(s[s.length - 1]),
    });
  }
  out.dispatchesPerStep = (() => {
    const before = b.dispatches; solver.step(); return b.dispatches - before;
  })();
  return out;
}, { n: N, ROUNDS });

if (r.error) { console.log('FAIL:', r.error); await browser.close(); process.exit(1); }

console.log(`адаптер ${r.adapter}, сетка ${r.n}², поле ${(r.bytes / 1e6).toFixed(1)} МБ`);
console.log(`задержка onSubmittedWorkDone на пустой очереди: ${r.drainLatency.toFixed(2)} мс`);
console.log(`${r.rounds} раундов кругового перебора после прогрева\n`);
console.log('  ядро                        мс     ГБ/с медиана   лучш   худш');
for (const o of r.ops) {
  console.log(`  ${o.name.padEnd(24)} ${o.ms.toFixed(4).padStart(8)}   ${o.gbs.toFixed(1).padStart(11)}`
    + `${o.best.toFixed(1).padStart(7)}${o.worst.toFixed(1).padStart(7)}`);
}
const step = r.ops.find((o) => o.name === 'шаг KP-I');
console.log(`\n  шаг: ${r.dispatchesPerStep} диспатчей, трафик ≈ ${(step.traffic / 1e6).toFixed(0)} МБ`);
console.log('  Сравнивай каждое ядро с copy: заметно ниже — виновато оно, все вровень —');
console.log('  упёрлись в память. Разница меньше промежутка лучш/худш ничего не значит.');

await browser.close();
