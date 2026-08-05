// Does single precision cost us time step?
//
//   node serve.mjs &
//   node tools/gpu-stability.mjs [modelId] [targetTime]
//
// The old GPU contract warned that f32 is stiff enough for KP/ZK to shorten the
// usable dt. This checks it at every grid size the app offers, with the app's
// own dt policy (0.012 * 128/n for KP), and watches the L2 norm - an exact KP
// invariant, and the first thing to move when a step is too large.
//
// Two things this has to get right or it measures nothing:
//
//  * equal *physical* time, not equal wall time. A fixed number of seconds
//    lets 128^2 reach t=300 while 1024^2 reaches t=0.4, and then the coarse
//    grid looks unstable purely because it ran 700x longer.
//  * a CPU reference at the same grid and the same t. Some L2 is genuinely
//    lost to the 2/3 dealiasing mask, in f64 just as much as in f32; only the
//    difference between the two columns is about precision.
//
// Driven through the UI on purpose: thousands of steps inside one page.evaluate
// never yield to the browser and wedge the tab. Headless, no launch flags.

import { chromium } from 'playwright-core';

const MODEL = process.argv[2] || 'kpI';
const TARGET = Number(process.argv[3] || 2);

const browser = await chromium.launch({
  channel: 'msedge', headless: true,
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

// Clean slate: the app restores model/grid/backend/tempo from localStorage,
// and a tool that inherits them measures the last session, not the default.
await page.addInitScript(() => { try { localStorage.clear(); } catch {} });
await page.goto('http://localhost:8123', { waitUntil: 'networkidle' });
await page.selectOption('#model', MODEL);
await page.selectOption('#tempo', '30');
await page.waitForTimeout(800);

const KEY = MODEL.startsWith('nls') ? 'гамильтониан' : '‖u‖²';

const read = () => page.evaluate(() => {
  const out = {};
  for (const d of document.querySelectorAll('#stats div')) {
    out[d.querySelector('span').textContent] = Number(d.querySelector('b').textContent);
  }
  out._grid = document.getElementById('gridInfo').textContent;
  return out;
});

// Wait until the run has actually started: t must advance and the asynchronous
// GPU readback must have delivered a real invariant (it reads 0 for the first
// frame or two, and a 1024^2 rebuild alone takes seconds).
const started = (key) => page.waitForFunction((k) => {
  const rows = [...document.querySelectorAll('#stats div')];
  const get = (n) => Number(rows.find((d) => d.querySelector('span').textContent === n)
    ?.querySelector('b').textContent);
  return get('t') > 0 && Math.abs(get(k)) > 1e-9;
}, key, { timeout: 120000 });

const reachT = (t) => page.waitForFunction(
  (target) => Number(document.querySelector('#stats b').textContent) >= target,
  t, { timeout: 900000 },
);

// Pause, let the pending GPU readbacks land, then read. The panel keeps
// refreshing while paused, so a short wait is enough.
const settled = async () => {
  await page.locator('#playPause').click();
  await page.waitForTimeout(500);
  return read();
};

console.log(`модель ${MODEL}, инвариант «${KEY}», до t=${TARGET}\n`);
console.log('  сетка   бэкенд        dt      t             инвариант        дрейф        пик');

for (const n of [128, 256, 512, 1024]) {
  for (const backend of ['cpu', 'gpu']) {
    // The CPU is ~1.2 s per step at 1024^2; reaching t=2 there would take an
    // hour and tells us nothing we cannot read off the smaller grids.
    if (backend === 'cpu' && n >= 512) continue;
    await page.selectOption('#backend', backend);
    if (await page.$eval('#backend', (s) => s.value) !== backend) continue;
    await page.selectOption('#resolution', String(n));
    await page.locator('#presets button').first().click();
    await started(KEY);
    // Sample with the clock stopped. The GPU runs 40 steps a frame at 128^2,
    // which is faster than the asynchronous readback refreshes the panel; read
    // it live and both samples come back identical and stale.
    const a = await settled();
    await page.locator('#playPause').click();      // resume
    await reachT(TARGET);
    const b = await settled();
    const drift = (b[KEY] / a[KEY] - 1) * 100;
    const peakKey = 'макс |u|' in a ? 'макс |u|' : 'макс |ψ|²';
    console.log(`  ${String(n).padStart(4)}²   ${backend}   ${a._grid.split('dt=')[1]}`
      + `   ${a.t.toFixed(2)} → ${b.t.toFixed(2)}`
      + `   ${a[KEY].toFixed(4)} → ${b[KEY].toFixed(4)}`
      + `   ${drift.toFixed(4)}%   ${a[peakKey].toFixed(4)} → ${b[peakKey].toFixed(4)}`);
  }
}

console.log('\n--- ошибки консоли ---');
console.log(errors.length ? errors.join('\n') : '  нет');
await browser.close();
process.exit(errors.length ? 1 : 0);
