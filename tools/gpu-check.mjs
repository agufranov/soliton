// Does the WebGPU backend compute the same thing as the CPU one?
//
//   node serve.mjs &
//   node tools/gpu-check.mjs
//
// Runs both backends side by side in a real browser and reports the relative
// difference for the FFT, for each model's evolution, and for the diagnostic
// reductions. Everything on the GPU is f32 against the CPU's f64, so the
// expected answer is "small and O(eps_f32 * sqrt(work))", not "zero".
//
// Headless, with no launch flags. Headless Edge does hand back the real
// hardware adapter; an earlier version of this file claimed otherwise and ran
// headed because of it. The probe that "proved" it was broken two ways: it
// checked navigator.gpu on about:blank, and the one headless configuration it
// did test on a real page passed --use-angle=swiftshader, which is exactly the
// flag that makes requestAdapter return null. Do not add launch flags here.

import { chromium } from 'playwright-core';

const browser = await chromium.launch({
  channel: 'msedge', headless: true,
});
const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text()); });
await page.goto('http://localhost:8123/', { waitUntil: 'domcontentloaded' });

const report = await page.evaluate(async () => {
  const { Grid } = await import('/src/core/grid.js');
  const { CpuBackend } = await import('/src/core/backend.js');
  const { requestGpuContext, WebGpuBackend } = await import('/src/core/webgpuBackend.js');
  const { modelById } = await import('/src/models/index.js');

  const ctx = await requestGpuContext();
  if (!ctx) return { error: 'no WebGPU adapter' };

  // A GPU read is asynchronous by design; pump the event loop until the
  // pending map resolves.
  const settle = async (b, rec) => {
    b.flush();
    await b.device.queue.onSubmittedWorkDone();
    for (let i = 0; i < 200 && rec.busy; i++) await new Promise((r) => setTimeout(r, 5));
    return rec;
  };
  const readField = (b, f) => settle(b, b.snapshot(f));
  const readNums = async (b, f) => {
    b.sumRe(f);
    const rec = b.reductions.get(f);
    await settle(b, rec);
    return { sum: rec.v[0], sumSq: rec.v[1], abs2: rec.v[2], max: rec.v[3] };
  };

  const relL2 = (a, ref) => {
    let s = 0, n = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - ref[i]; s += d * d; n += ref[i] * ref[i]; }
    return Math.sqrt(s / Math.max(n, 1e-300));
  };
  const maxAbs = (a) => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m; };

  const out = { adapter: ctx.label, fft: {}, models: [], reductions: {} };

  // --- FFT ---------------------------------------------------------------
  //
  // The GPU leaves the spectrum TRANSPOSED (it skips the fourth pass; see
  // core/webgpuBackend.js), so the forward comparison has to transpose one
  // side. Getting this wrong is not academic: an earlier version of this test
  // loaded the input through `adopt`, which transposes because it is meant for
  // spectral coefficients, and the two transposes cancelled - the forward test
  // passed while comparing the wrong things, and only the round trip caught it.
  const transposed = (a, n) => {
    const t = new Float64Array(a.length);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) t[i * n + j] = a[j * n + i];
    return t;
  };

  for (const n of [128, 256]) {
    const grid = new Grid(n, n, 40, 40);
    const cpu = new CpuBackend(grid);
    const gpu = new WebGpuBackend(ctx, grid);
    const re = new Float64Array(grid.n), im = new Float64Array(grid.n);
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
    for (let i = 0; i < grid.n; i++) { re[i] = rnd(); im[i] = rnd(); }

    const c = cpu.adopt({ re: Float64Array.from(re), im: Float64Array.from(im) });
    const g = gpu.alloc();
    gpu.zero(g);
    gpu.addHostComplex(g, re, im);        // physical-space upload: no transpose

    cpu.forward(c); gpu.forward(g);
    const gf = await readField(gpu, g);
    const fwd = Math.max(relL2(gf.re, transposed(c.re, n)), relL2(gf.im, transposed(c.im, n)));

    cpu.inverse(c); gpu.inverse(g);
    const gi = await readField(gpu, g);
    const round = Math.max(relL2(gi.re, re), relL2(gi.im, im));

    out.fft[n] = { forward: fwd, roundTrip: round };
    gpu.dispose();
  }

  // --- models ------------------------------------------------------------
  // Same grid, same dt, same preset, same number of steps on both backends.
  const cases = [
    { id: 'kpI', preset: 0, n: 128, dt: 0.012, steps: 400 },
    { id: 'kpII', preset: 0, n: 128, dt: 0.012, steps: 400 },
    { id: 'zk', preset: 0, n: 128, dt: 0.01, steps: 400 },
    { id: 'nlsCQ', preset: 0, n: 128, dt: 0.015, steps: 400 },
    { id: 'nlsSat', preset: 0, n: 128, dt: 0.015, steps: 400 },
    { id: 'nlsCubic', preset: 2, n: 128, dt: 0.015, steps: 400 },
    { id: 'sg', preset: 0, n: 128, dt: 0.03, steps: 400 },
  ];

  for (const cs of cases) {
    const model = modelById(cs.id);
    const grid = new Grid(cs.n, cs.n, model.grid.L, model.grid.L);
    const params = {};
    for (const p of model.params) params[p.key] = p.value;

    const cpu = new CpuBackend(grid);
    const gpu = new WebGpuBackend(ctx, grid);
    const sc = model.create(cpu, grid, cs.dt, params);
    const sg = model.create(gpu, grid, cs.dt, params);
    const preset = model.presets[cs.preset];
    sc.applyPreset(preset); sg.applyPreset(preset);

    const g0 = await readField(gpu, sg.target());
    const start = { cpu: maxAbs(sc.field()), gpu: maxAbs(g0.re), err: relL2(g0.re, sc.field()) };

    for (let s = 0; s < cs.steps; s++) { sc.step(); sg.step(); }

    const g1 = await readField(gpu, sg.target());
    const cpuField = sc.field();
    const dc = sc.diagnostics();

    // Reductions are asynchronous: the first call only schedules them, so let
    // them land and then bust the per-version cache to read the real numbers.
    // (Live, this is why the invariants panel shows zeros for the first frame
    // or two after a preset and then settles.)
    sg.diagnostics();
    gpu.flush();
    await gpu.device.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 100));
    sg.solver.version++;
    const dg = sg.diagnostics();

    out.models.push({
      id: cs.id, preset: preset.label, t: sc.time.toFixed(2), steps: cs.steps,
      spawnErr: start.err, peakCpu: start.cpu, peakGpu: start.gpu,
      err: relL2(g1.re, cpuField),
      finalPeakCpu: maxAbs(cpuField), finalPeakGpu: maxAbs(g1.re),
      diag: Object.keys(dc).map((k) => ({ k, cpu: dc[k], gpu: dg[k] })),
    });
    gpu.dispose();
  }

  // Long-run f32 stability is NOT measured here - see tools/gpu-stability.mjs.
  // Thousands of steps inside a single page.evaluate wedges the tab: the work
  // never yields to the browser, and a headed window that is not in the
  // foreground gets its GPU submissions throttled to a crawl. Driving the app's
  // own frame loop instead keeps the page alive and measures the same thing.

  // --- reductions --------------------------------------------------------
  {
    const grid = new Grid(1024, 1024, 45, 45);
    const cpu = new CpuBackend(grid);
    const gpu = new WebGpuBackend(ctx, grid);
    const re = new Float64Array(grid.n);
    for (let j = 0; j < grid.ny; j++) for (let i = 0; i < grid.nx; i++) {
      const dx = grid.wrapX(grid.x(i) - 20), dy = grid.wrapY(grid.y(j) - 22);
      re[j * grid.nx + i] = Math.exp(-(dx * dx + dy * dy) / 9) - 0.001;
    }
    const c = cpu.adopt({ re, im: new Float64Array(grid.n) });
    const g = gpu.adopt({ re, im: new Float64Array(grid.n) });
    const gv = await readNums(gpu, g);
    out.reductions = {
      n: '1024^2',
      sumRe: { cpu: cpu.sumRe(c), gpu: gv.sum },
      sumSqRe: { cpu: cpu.sumSqRe(c), gpu: gv.sumSq },
      maxAbsRe: { cpu: cpu.maxAbsRe(c), gpu: gv.max },
    };
    gpu.dispose();
  }

  return out;
});

if (report.error) {
  console.log('FAIL:', report.error);
  await browser.close();
  process.exit(1);
}

const e = (v) => Number(v).toExponential(2);
console.log(`адаптер: ${report.adapter}\n`);

console.log('--- FFT (GPU f32 против CPU f64) ---');
for (const [n, r] of Object.entries(report.fft)) {
  console.log(`  ${n}²  forward отн.L2 ${e(r.forward)}   round-trip ${e(r.roundTrip)}`);
}

console.log('\n--- эволюция моделей ---');
let worst = 0;
for (const m of report.models) {
  console.log(`\n  ${m.id} «${m.preset}», ${m.steps} шагов, t=${m.t}`);
  console.log(`    спавн:  отн.L2 ${e(m.spawnErr)}   пик ${m.peakCpu.toFixed(5)} / ${m.peakGpu.toFixed(5)}`);
  console.log(`    после:  отн.L2 ${e(m.err)}   пик ${m.finalPeakCpu.toFixed(5)} / ${m.finalPeakGpu.toFixed(5)}`);
  for (const d of m.diag) {
    console.log(`      ${d.k.padEnd(16)} ${Number(d.cpu).toFixed(5).padStart(12)} / ${Number(d.gpu).toFixed(5).padStart(12)}`);
  }
  worst = Math.max(worst, m.err);
}

console.log('\n--- редукции на 1024² ---');
for (const [k, v] of Object.entries(report.reductions)) {
  if (k === 'n') continue;
  const rel = Math.abs(v.gpu - v.cpu) / Math.max(Math.abs(v.cpu), 1e-300);
  console.log(`  ${k.padEnd(10)} ${v.cpu.toFixed(6)} / ${v.gpu.toFixed(6)}   отн. ${e(rel)}`);
}

console.log(`\nхудшее расхождение по полю: ${e(worst)}`);
await browser.close();
process.exit(worst < 0.05 ? 0 : 1);
