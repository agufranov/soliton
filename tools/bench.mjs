// Backend x resolution timing, driven through the real UI.
//
//   node serve.mjs &
//   node tools/bench.mjs [outDir] [modelId]
//
// Switches the app between CPU and GPU at every offered grid size, lets the
// steps-per-frame controller settle, and reads the ms/step the app itself
// reports. Also screenshots the canvas at each setting, which is the only way
// to catch a GPU render path that is fast because it is drawing nothing.
//
// Headless, like every tool here: headless Edge hands back the real hardware
// adapter. See the note in tools/gpu-check.mjs.

import { chromium } from 'playwright-core';
import { mkdir } from 'node:fs/promises';

const OUT = process.argv[2] || 'shots';
const MODEL = process.argv[3] || 'kpI';
await mkdir(OUT, { recursive: true });

const browser = await chromium.launch({
  channel: 'msedge', headless: true,
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
page.on('crash', () => console.log('\n!! вкладка упала (нехватка памяти или потеря устройства)'));
page.on('close', () => console.log('\n!! страница закрыта'));

// Clean slate: the app restores model/grid/backend/tempo from localStorage,
// and a tool that inherits them measures the last session, not the default.
await page.addInitScript(() => { try { localStorage.clear(); } catch {} });
await page.goto('http://localhost:8123', { waitUntil: 'networkidle' });
await page.selectOption('#model', MODEL);
await page.selectOption('#tempo', '14');
await page.waitForTimeout(800);

const stats = () => page.evaluate(() => {
  const out = {};
  for (const d of document.querySelectorAll('#stats div')) {
    out[d.querySelector('span').textContent] = d.querySelector('b').textContent;
  }
  out._grid = document.getElementById('gridInfo').textContent;
  out._backend = document.getElementById('backendInfo').textContent;
  return out;
});

const rows = [];
for (const backend of ['cpu', 'gpu']) {
  await page.selectOption('#backend', backend);
  await page.waitForTimeout(700);
  const enabled = await page.$eval('#backend', (s) => s.value);
  if (enabled !== backend) { console.log(`${backend}: недоступен, пропуск`); continue; }

  for (const n of [128, 256, 512, 1024]) {
    process.stdout.write(`${backend} ${n}: `);
    await page.selectOption('#resolution', String(n));
    await page.locator('#presets button').first().click();
    // The ms/step estimate is an EMA with a 0.2 coefficient, so it needs ~15
    // frames; on the CPU at 1024^2 a frame is most of a second.
    const settle = backend === 'cpu' && n >= 512 ? 30000 : 6000;
    await page.waitForTimeout(settle);
    const s = await stats();
    await page.locator('#stage').screenshot({ path: `${OUT}/bench-${MODEL}-${backend}-${n}.png` });
    rows.push({ backend, n, ms: s['мс/шаг'], spf: s['шагов/кадр'], fps: s.fps, t: s.t, s });
    console.log(`${backend.padEnd(4)} ${String(n).padStart(4)}²  мс/шаг ${String(s['мс/шаг']).padStart(8)}`
      + `  шагов/кадр ${String(s['шагов/кадр']).padStart(3)}  fps ${String(s.fps).padStart(3)}  t=${s.t}`);
  }
}

console.log('\n--- сводка (мс на шаг) ---');
const by = (b) => rows.filter((r) => r.backend === b);
for (const n of [128, 256, 512, 1024]) {
  const c = by('cpu').find((r) => r.n === n);
  const g = by('gpu').find((r) => r.n === n);
  const speedup = c && g ? (Number(c.ms) / Number(g.ms)).toFixed(1) + '×' : '—';
  console.log(`  ${String(n).padStart(4)}²   CPU ${String(c ? c.ms : '—').padStart(8)}`
    + `   GPU ${String(g ? g.ms : '—').padStart(7)}   ускорение ${speedup}`);
}

console.log('\n--- ошибки консоли ---');
console.log(errors.length ? errors.join('\n') : '  нет');

await browser.close();
process.exit(errors.length ? 1 : 0);
