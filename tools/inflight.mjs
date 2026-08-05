// Сколько пачек держать в полёте?
//
//   node serve.mjs &
//   node tools/inflight.mjs [модель]
//
// Кадр ставит не больше одной пачки, поэтому пропускная способность упирается
// в `шагов/кадр × fps` задолго до того, как загрузится само устройство. Глубина
// очереди помогает ровно там, где пачка длиннее кадра.
//
// Мерим пройденным ФИЗИЧЕСКИМ временем за настенное окно. Считать «занятость»
// как шагов/с × мс/шаг нельзя: проход рендера идёт в ту же очередь, но в
// мс/шаг не входит, и недосчёт выглядит как простой, которого нет.
//
// Перебор круговой: состояние машины дрейфует (частоты iGPU, CPU ворует DDR),
// и каждая глубина должна увидеть один и тот же дрейф, а не свой отрезок.

import { chromium } from 'playwright-core';

const MODEL = process.argv[2] || 'kpI';
const CAPS = [2, 3, 4, 6];
const GRIDS = [128, 256, 512, 1024];
const WINDOW = 8000;
const ROUNDS = 2;

const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
// Clean slate: the app restores model/grid/backend/tempo from localStorage,
// and a tool that inherits them measures the last session, not the default.
await page.addInitScript(() => { try { localStorage.clear(); } catch {} });
await page.goto('http://localhost:8123', { waitUntil: 'networkidle' });

const app = (fn) => page.evaluate(fn);
const readT = () => app(() => Number(document.querySelector('#stats b').textContent));

await page.selectOption('#model', MODEL);
await page.selectOption('#backend', 'gpu');
await page.waitForTimeout(700);
if (await page.$eval('#backend', (s) => s.value) !== 'gpu') {
  console.log('GPU недоступен — мерить нечего'); await browser.close(); process.exit(1);
}
const dt0 = await app(() => window.__soliton.model.dt);

console.log(`модель ${MODEL}, темп «обычно»\n`);
for (const n of GRIDS) {
  await page.selectOption('#resolution', String(n));
  await page.locator('#presets button').first().click();
  await page.waitForTimeout(5000);

  const dt = dt0 * (128 / n);
  const { spf, fps } = await app(() => ({
    spf: window.__soliton.stepsPerFrame,
    fps: Number(document.querySelector('#stats div:last-child b').textContent),
  }));
  const acc = new Map(CAPS.map((c) => [c, []]));

  for (let r = 0; r < ROUNDS; r++) {
    for (const cap of CAPS) {
      await page.evaluate((c) => { window.__soliton.inFlightCap = c; }, cap);
      await page.waitForTimeout(2500);            // дать конвейеру перестроиться
      const t0 = await readT(); const w0 = Date.now();
      await page.waitForTimeout(WINDOW);
      const t1 = await readT();
      acc.get(cap).push((t1 - t0) / dt / ((Date.now() - w0) / 1000));
    }
  }

  const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
  const cells = CAPS.map((c) => med(acc.get(c)).toFixed(1).padStart(8)).join('');
  if (n === GRIDS[0]) console.log('  сетка  шаг/кадр  потолок rAF' + CAPS.map((c) => `  гл.${c}`.padStart(8)).join(''));
  console.log(`  ${String(n).padStart(4)}²  ${String(spf).padStart(8)}  ${(spf * fps).toFixed(0).padStart(11)}${cells}`);
}

await app(() => { window.__soliton.inFlightCap = 3; });
if (errors.length) { console.log('\n--- ошибки ---\n  ' + errors.join('\n  ')); }
console.log('\n  Потолок rAF = шагов/кадр × fps. Где приложение уже на нём, глубина очереди');
console.log('  не поможет — помогает темп. Глубже 3 берём только ценой задержки ввода:');
console.log('  в худшем случае она равна глубина × длительность пачки.');
await browser.close();
