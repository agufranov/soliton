// Drive the real app in a real browser: load it, let KP-I run, take
// screenshots, click to spawn a lump, switch models.
//
// Uses the installed Edge/Chrome via playwright-core, so no browser download.
//   node serve.mjs &
//   node tools/smoke-browser.mjs [outDir] [cpu|gpu] [resolution]
//
// Headless for both backends: headless Edge hands back the real hardware
// WebGPU adapter, with no launch flags. See tools/gpu-check.mjs.

import { chromium } from 'playwright-core';
import { mkdir } from 'node:fs/promises';

const OUT = process.argv[2] || 'shots';
const BACKEND = process.argv[3] || 'cpu';
const RES = process.argv[4] || '128';
const URL_ = 'http://localhost:8123';
const GPU = BACKEND === 'gpu';

await mkdir(OUT, { recursive: true });

const browser = await chromium.launch({
  channel: 'msedge', headless: true,
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

// Boot from a clean slate: the app restores model/grid/backend/tempo from
// localStorage, and a tool that inherits them measures the last session
// instead of the default. Has to run before the page's own scripts, so it is
// an init script and not an evaluate.
await page.addInitScript(() => { try { localStorage.clear(); } catch {} });
await page.goto(URL_, { waitUntil: 'networkidle' });

// The equation picker is a hand-built listbox (it needs hover events, which a
// native <option> does not give), and <select id="model"> is only its value
// store - hidden, so selectOption cannot reach it. Setting the value and
// dispatching `change` is the same path the UI takes.
const pickModel = (id) => page.evaluate((v) => {
  const s = document.getElementById('model');
  s.value = v;
  s.dispatchEvent(new Event('change'));
}, id);
// Pin the backend explicitly, both ways. The app now defaults to the GPU, so a
// run that only selects 'gpu' when asked would quietly test the GPU for a
// 'cpu' run too - which is exactly what happened when the default changed.
await page.selectOption('#backend', BACKEND);
await page.waitForTimeout(600);
const gotBackend = await page.$eval('#backend', (s) => s.value);
if (gotBackend !== BACKEND) {
  console.log(`бэкенд ${BACKEND} недоступен — прогон отменён`);
  await browser.close(); process.exit(1);
}
if (RES !== '128') await page.selectOption('#resolution', RES);
await page.waitForTimeout(1500);

// Is anything actually painted? A WebGPU canvas cannot be read back from
// inside the page - its texture is gone by the time script runs again, and
// createImageBitmap on it hands back pure black - so the check goes through a
// real screenshot, decoded back inside the page. That also makes the CPU and
// GPU numbers comparable, because both now measure what the compositor shows.
const readStats = async () => {
  const shot = (await page.locator('#stage').screenshot()).toString('base64');
  return page.evaluate(async (b64) => {
    const out = {};
    for (const d of document.querySelectorAll('#stats div')) {
      out[d.querySelector('span').textContent] = d.querySelector('b').textContent;
    }
    out._model = document.getElementById('model').value;
    out._grid = document.getElementById('gridInfo').textContent;
    out._backend = document.getElementById('backendInfo').textContent;

    const img = new Image();
    img.src = 'data:image/png;base64,' + b64;
    await img.decode();
    const cv = document.createElement('canvas');
    cv.width = img.width; cv.height = img.height;
    const g = cv.getContext('2d');
    g.drawImage(img, 0, 0);
    const px = g.getImageData(0, 0, cv.width, cv.height).data;
    let min = 765, max = 0, sum = 0, n = 0;
    for (let i = 0; i < px.length; i += 4 * 97) {
      const v = px[i] + px[i + 1] + px[i + 2];
      if (v < min) min = v; if (v > max) max = v; sum += v; n++;
    }
    out._pixels = `min=${min} max=${max} mean=${(sum / n).toFixed(0)}`;
    return out;
  }, shot);
};

const tag = GPU ? `gpu${RES}-` : '';
const step = async (name, note) => {
  await page.screenshot({ path: `${OUT}/${tag}${name}.png` });
  const s = await readStats();
  console.log(`\n== ${name} ${note ? '- ' + note : ''}`);
  console.log('   model:', s._model, '|', s._grid, '|', s._backend);
  console.log('   canvas:', s._pixels);
  console.log('   ', Object.entries(s).filter(([k]) => !k.startsWith('_'))
    .map(([k, v]) => `${k}=${v}`).join('  '));
  return s;
};

const a = await step('01-kpI-start', 'KP-I, стартовый пресет «два лампа под углом»');
await page.waitForTimeout(4000);
const b = await step('02-kpI-running', 'через 4 с');

// spawn a lump by dragging on the canvas
const box = await page.locator('#stage').boundingBox();
await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.75);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.28, box.y + box.height * 0.6, { steps: 12 });
await page.screenshot({ path: `${OUT}/${tag}03-aiming.png` });
await page.mouse.up();
await page.waitForTimeout(1200);
await step('04-after-spawn', 'после спавна лампа мышью');

for (const [id, label, wait] of [
  ['nlsCQ', 'НУШ кубично-квинтичное', 4000],
  ['zk', 'Захаров–Кузнецов', 4000],
  ['kpII', 'KP-II', 4000],
  ['nlsCubic', 'НУШ кубическое (коллапс)', 5000],
  ['sg', 'sine-Gordon', 4000],
]) {
  await pickModel(id);
  await page.waitForTimeout(wait);
  await step(`05-${id}`, label);
}

console.log('\n--- console errors ---');
console.log(errors.length ? errors.join('\n') : '  нет');

const advanced = Number(b.t) > Number(a.t);
console.log(`\nвремя идёт: ${a.t} -> ${b.t}  ${advanced ? 'OK' : 'FAIL'}`);

await browser.close();
process.exit(errors.length || !advanced ? 1 : 0);
