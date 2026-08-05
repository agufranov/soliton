// Focused screenshot helper: pick a model, preset, resolution and a list of
// times, and get one canvas-only frame per time.
//   node tools/shoot.mjs <outDir> <modelId> <presetIndex> <resolution> <t1,t2,...> [cpu|gpu]
//
// Headless throughout; headless Edge gets the real hardware WebGPU adapter.

import { chromium } from 'playwright-core';
import { mkdir } from 'node:fs/promises';

const [outDir, modelId, presetIdx = '0', res = '128', times = '4', backend = 'cpu'] =
  process.argv.slice(2);
const gpu = backend === 'gpu';
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch({
  channel: 'msedge', headless: true,
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
page.on('pageerror', (e) => console.log('pageerror:', e.message));
// Clean slate: the app restores model/grid/backend/tempo from localStorage,
// and a tool that inherits them measures the last session, not the default.
await page.addInitScript(() => { try { localStorage.clear(); } catch {} });
await page.goto('http://localhost:8123', { waitUntil: 'networkidle' });

if (gpu) {
  await page.selectOption('#backend', 'gpu');
  await page.waitForTimeout(600);
  if (await page.$eval('#backend', (s) => s.value) !== 'gpu') {
    console.log('GPU недоступен'); await browser.close(); process.exit(1);
  }
}
await page.selectOption('#model', modelId);
if (res !== '128') await page.selectOption('#resolution', res);
await page.waitForTimeout(400);
await page.locator('#presets button').nth(Number(presetIdx)).click();
await page.selectOption('#tempo', '30');

let prev = 0;
for (const t of times.split(',').map(Number)) {
  await page.waitForFunction(
    (target) => Number(document.querySelector('#stats b').textContent) >= target,
    t, { timeout: 180000 },
  );
  const name = `${modelId}-p${presetIdx}-n${res}-${backend}-t${t}`;
  await page.locator('#stage').screenshot({ path: `${outDir}/${name}.png` });
  const stats = await page.evaluate(() => [...document.querySelectorAll('#stats div')]
    .map((d) => `${d.querySelector('span').textContent}=${d.querySelector('b').textContent}`).join('  '));
  console.log(name.padEnd(28), stats);
  prev = t;
}

await browser.close();
