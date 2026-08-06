// Layout check for phones: drive the real app at phone viewports with touch
// emulation, measure the boxes, work the bottom sheet with taps and swipes,
// and shoot every state.
//
//   node serve.mjs &
//   node tools/mobile-shots.mjs [outDir] [cpu|gpu]
//
// Why a tool and not an eyeball: the mobile layout is three overlapping
// constraints - a square stage, a sheet pinned to the bottom, and a viewport
// that is shorter than either of them wants - and every one of them fails
// silently. A stage taller than the space left over just slides under the
// sheet; a page one pixel too wide scrolls sideways forever. Both look fine in
// a screenshot of the top half.
//
// CPU by default: this measures boxes, not throughput, and headless WebGPU on
// an emulated phone is a different adapter path with nothing to do with layout.

import { chromium } from 'playwright-core';
import { mkdir } from 'node:fs/promises';

const OUT = process.argv[2] || 'shots';
const BACKEND = process.argv[3] || 'cpu';
const TAG = BACKEND === 'gpu' ? 'gpu-' : '';   // both backends share one out dir
const URL_ = 'http://localhost:8123';

const DEVICES = [
  { id: 'iphone-se',  w: 375, h: 667, dpr: 2 },
  { id: 'android',    w: 360, h: 740, dpr: 3 },
  { id: 'iphone-pro', w: 393, h: 852, dpr: 3 },
  { id: 'tablet',     w: 768, h: 1024, dpr: 2 },
  { id: 'landscape',  w: 852, h: 393, dpr: 3 },
];

await mkdir(OUT, { recursive: true });

const browser = await chromium.launch({ channel: 'msedge', headless: true });
const fails = [];
const errors = [];

// Everything the layout has to satisfy, measured in the page. Each check is a
// number and a verdict, so a regression shows up as a changed number rather
// than as "looks off".
const measure = () => ({
  vw: innerWidth,
  vh: innerHeight,
  docW: document.documentElement.scrollWidth,
  stage: (({ x, y, width, height }) => ({ x, y, w: width, h: height }))(
    document.getElementById('stage').getBoundingClientRect()),
  bar: (({ y, height }) => ({ y, h: height }))(
    document.getElementById('sheetBar').getBoundingClientRect()),
  sheet: (({ y, height }) => ({ y, h: height }))(
    document.querySelector('aside').getBoundingClientRect()),
  open: document.querySelector('aside').classList.contains('open'),
  // The physical box, straight out of the running solver.
  box: { Lx: globalThis.__soliton.grid.Lx, Ly: globalThis.__soliton.grid.Ly },
  sheetVisible: getComputedStyle(document.getElementById('sheetBar')).display !== 'none',
  // iOS zooms the whole page when a control smaller than 16px takes focus.
  // The equation is picked by #modelBtn now, not by a <select> - #model is
  // hidden and its computed size says nothing about what a finger touches.
  selectFont: parseFloat(getComputedStyle(document.getElementById('resolution')).fontSize),
  // The picker moved above the field on a phone: in the sheet it was the first
  // line of content, i.e. unreachable until the sheet was opened.
  picker: (({ y, height, width }) => ({ y, h: height, w: width }))(
    document.getElementById('modelBtn').getBoundingClientRect()),
  pickerInHead: !!document.getElementById('headSlot')
    .contains(document.getElementById('modelBox')),
  // smallest tap target among the generated controls
  minTap: Math.min(...[...document.querySelectorAll('aside button')]
    .map((b) => b.getBoundingClientRect().height).filter((h) => h > 0)),
  t: Number(document.querySelector('#stats b')?.textContent || 0),
});

const check = (dev, name, cond, detail) => {
  if (!cond) fails.push(`${dev}: ${name} — ${detail}`);
  console.log(`   ${cond ? 'OK  ' : 'FAIL'} ${name}  ${detail}`);
};

for (const d of DEVICES) {
  const portrait = d.h > d.w;
  const ctx = await browser.newContext({
    viewport: { width: d.w, height: d.h },
    deviceScaleFactor: d.dpr,
    isMobile: true,
    hasTouch: true,
  });
  const page = await ctx.newPage();
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`${d.id}: ${m.text()}`); });
  page.on('pageerror', (e) => errors.push(`${d.id}: pageerror: ${e.message}`));

  await page.addInitScript(() => { try { localStorage.clear(); } catch {} });
  await page.goto(URL_, { waitUntil: 'networkidle' });
  await page.selectOption('#backend', BACKEND);
  await page.waitForTimeout(1200);

  console.log(`\n== ${d.id}  ${d.w}x${d.h} @${d.dpr}x  ${portrait ? 'портрет' : 'ландшафт'}`);
  const m = await page.evaluate(measure);

  check(d.id, 'нет горизонтального скролла', m.docW <= m.vw + 1, `doc=${m.docW} vp=${m.vw}`);

  // The one check that matters most: the box the solver integrates and the box
  // the picture is stretched into have to be the same shape. Any drift here
  // draws round lumps as ellipses and sends them off the aim arrow, and it
  // does it without a single error anywhere.
  const stageA = m.stage.h / m.stage.w, boxA = m.box.Ly / m.box.Lx;
  check(d.id, 'ящик по форме сцены', Math.abs(boxA / stageA - 1) < 0.03,
    `ящик ${m.box.Lx}×${m.box.Ly.toFixed(1)} (${boxA.toFixed(3)}) против сцены ${stageA.toFixed(3)}`);
  check(d.id, portrait ? 'сцена заполняет ширину' : 'сцена квадратная',
    portrait ? m.stage.w >= m.vw - 24 : Math.abs(m.stage.w - m.stage.h) <= 1,
    `${m.stage.w.toFixed(0)}x${m.stage.h.toFixed(0)}`);
  check(d.id, 'сцена целиком на экране',
    m.stage.y >= -1 && m.stage.y + m.stage.h <= m.vh + 1 && m.stage.x >= -1,
    `верх=${m.stage.y.toFixed(0)} низ=${(m.stage.y + m.stage.h).toFixed(0)} из ${m.vh}`);
  check(d.id, 'сцена не прячется под шторкой',
    !m.sheetVisible || m.stage.y + m.stage.h <= m.bar.y + 1,
    `низ сцены=${(m.stage.y + m.stage.h).toFixed(0)} верх шторки=${m.bar.y.toFixed(0)}`);
  check(d.id, 'шторка там, где ждём', m.sheetVisible === portrait,
    `видна=${m.sheetVisible}`);
  check(d.id, 'селект не зумит iOS', m.selectFont >= 16, `${m.selectFont}px`);
  check(d.id, 'палец попадает по кнопкам', m.minTap >= 38, `мин ${m.minTap.toFixed(0)}px`);
  const used = (m.stage.w * m.stage.h) / (m.vw * m.vh);
  check(d.id, 'поле занимает экран', !portrait || used > 0.6, `${(used * 100).toFixed(0)}%`);

  await page.screenshot({ path: `${OUT}/m-${TAG}${d.id}-01-closed.png` });

  if (portrait) {
    // Tap the handle: the sheet must come all the way up and the controls at
    // its top must land inside the viewport.
    await page.locator('#sheetBar').tap();
    await page.waitForTimeout(450);
    const o = await page.evaluate(measure);
    check(d.id, 'тап по ручке открывает', o.open && o.sheet.y < m.sheet.y - 40,
      `верх ${m.sheet.y.toFixed(0)} -> ${o.sheet.y.toFixed(0)}`);
    check(d.id, 'открытая шторка помещается', o.sheet.y >= -1 && o.sheet.h <= m.vh,
      `y=${o.sheet.y.toFixed(0)} h=${o.sheet.h.toFixed(0)} из ${m.vh}`);
    const presetBox = await page.locator('.preset').first().boundingBox();
    check(d.id, 'сценарии видны без скролла',
      !!presetBox && presetBox.y + presetBox.height <= m.vh,
      presetBox ? `низ ${(presetBox.y + presetBox.height).toFixed(0)} из ${m.vh}` : 'нет кнопки');
    await page.screenshot({ path: `${OUT}/m-${TAG}${d.id}-02-open.png` });

    // The panel has to scroll to its end inside the sheet, not drag the page.
    await page.evaluate(() => { document.querySelector('aside').scrollTop = 1e5; });
    await page.waitForTimeout(250);
    const bottom = await page.evaluate(() => {
      const a = document.querySelector('aside');
      return { scrolled: a.scrollTop, max: a.scrollHeight - a.clientHeight, bodyTop: scrollY };
    });
    check(d.id, 'панель прокручивается внутри шторки',
      bottom.max > 0 && bottom.scrolled >= bottom.max - 2 && bottom.bodyTop === 0,
      `scrollTop=${bottom.scrolled.toFixed(0)}/${bottom.max.toFixed(0)} страница=${bottom.bodyTop}`);
    await page.screenshot({ path: `${OUT}/m-${TAG}${d.id}-03-scrolled.png` });

    // Real touch drags, through CDP: playwright's touchscreen can only tap,
    // and a mouse drag would not exercise the one thing that actually breaks
    // here - a missing `touch-action: none`, which lets the browser swallow
    // the gesture as a scroll and never send the pointermove.
    const cdp = await ctx.newCDPSession(page);
    const touch = (type, x, y) => cdp.send('Input.dispatchTouchEvent', {
      type, touchPoints: type === 'touchEnd' ? [] : [{ x, y }],
    });
    const swipe = async (x, y, dy, steps = 10) => {
      await touch('touchStart', x, y);
      for (let i = 1; i <= steps; i++) await touch('touchMove', x, y + (dy * i) / steps);
      await touch('touchEnd', x, y + dy);
      await page.waitForTimeout(420);
    };
    const isOpen = () => page.evaluate(() => document.querySelector('aside').classList.contains('open'));

    await page.evaluate(() => { document.querySelector('aside').scrollTop = 0; });
    let bar = await page.locator('#sheetBar').boundingBox();
    await swipe(bar.x + bar.width / 2, bar.y + bar.height / 2, 220);   // вниз
    check(d.id, 'протяжка вниз закрывает', !(await isOpen()), '');

    bar = await page.locator('#sheetBar').boundingBox();
    await swipe(bar.x + bar.width / 2, bar.y + bar.height / 2, -220);  // вверх
    check(d.id, 'протяжка вверх открывает', await isOpen(), '');

    bar = await page.locator('#sheetBar').boundingBox();
    await page.touchscreen.tap(bar.x + bar.width / 2, bar.y + bar.height / 2);
    await page.waitForTimeout(450);
    check(d.id, 'повторный тап закрывает', !(await isOpen()), '');

    // A tap on the field spawns; a drag aims and throws. The whole point of
    // the app has to survive being done with a finger.
    const st = await page.locator('#stage').boundingBox();
    await page.touchscreen.tap(st.x + st.width * 0.3, st.y + st.height * 0.35);
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${OUT}/m-${TAG}${d.id}-04-tap-spawn.png` });
    const after = await page.evaluate(measure);
    check(d.id, 'время идёт после тапа', after.t > 0, `t=${after.t}`);

    const ax = st.x + st.width * 0.3, ay = st.y + st.height * 0.7;
    await touch('touchStart', ax, ay);
    for (let i = 1; i <= 8; i++) await touch('touchMove', ax + 7 * i, ay - 5 * i);
    await page.waitForTimeout(120);
    const aimed = await page.evaluate(() => !!globalThis.__soliton.drag);
    check(d.id, 'протяжка по полю целится', aimed, aimed ? 'прицел живой' : 'жест съеден');
    await page.screenshot({ path: `${OUT}/m-${TAG}${d.id}-05-aiming.png` });
    await touch('touchEnd', ax + 56, ay - 40);
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${OUT}/m-${TAG}${d.id}-06-thrown.png` });
  } else {
    const presetBox = await page.locator('.preset').first().boundingBox();
    check(d.id, 'сценарии доступны в ландшафте', !!presetBox, presetBox ? 'да' : 'нет');
  }

  // Выбор уравнения. В портрете он живёт над полем, в ландшафте — в боковой
  // панели, но работает одинаково: тап по пункту показывает формулу, а
  // применяет её кнопка. Тап, применяющий модель сразу, означал бы, что
  // формулу видно только у той модели, с которой уже ушёл.
  check(d.id, 'выбор уравнения над полем', m.pickerInHead === portrait,
    `в шапке=${m.pickerInHead}`);
  check(d.id, 'по кнопке уравнения можно попасть пальцем', m.picker.h >= 38,
    `${m.picker.h.toFixed(0)}px`);

  // Превью в портрете ложится над закрытой шторкой, так что шторку надо
  // закрыть — но только если она открыта, иначе тап её как раз откроет.
  const sheetOpen = () => page.evaluate(() =>
    document.querySelector('aside').classList.contains('open'));
  if (portrait && await sheetOpen()) await page.locator('#sheetBar').tap();
  await page.waitForTimeout(350);
  await page.locator('#modelBtn').tap();
  await page.waitForTimeout(250);
  await page.locator('.mitem').nth(2).tap();             // nlsSat, не текущая модель
  await page.waitForTimeout(250);
  const prev = await page.evaluate(() => {
    const p = document.getElementById('eqPrev');
    const r = p.getBoundingClientRect();
    return {
      on: p.classList.contains('on'), compact: p.classList.contains('compact'),
      terms: p.querySelectorAll('.peq .tm').length,
      pick: p.querySelector('.pick')?.getBoundingClientRect().height || 0,
      y: r.y, bottom: r.y + r.height,
      model: document.getElementById('model').value,
    };
  });
  check(d.id, 'тап по уравнению показывает формулу',
    prev.on && prev.compact && prev.terms >= 2,
    `видно=${prev.on} подсвечено членов=${prev.terms}`);
  check(d.id, 'превью помещается на экран', prev.y >= -1 && prev.bottom <= m.vh + 1,
    `${prev.y.toFixed(0)}..${prev.bottom.toFixed(0)} из ${m.vh}`);
  check(d.id, 'в превью есть кнопка «выбрать»', prev.pick >= 38, `${prev.pick.toFixed(0)}px`);
  await page.screenshot({ path: `${OUT}/m-${TAG}${d.id}-07-formula.png` });

  await page.locator('#eqPrev .pick').tap();
  await page.waitForTimeout(900);
  const picked = await page.evaluate(() => document.getElementById('model').value);
  check(d.id, 'кнопка «выбрать» применяет уравнение', picked === 'nlsSat',
    `${prev.model} -> ${picked}`);
  await page.screenshot({ path: `${OUT}/m-${TAG}${d.id}-08-picked.png` });

  await ctx.close();
}

console.log('\n--- ошибки консоли ---');
console.log(errors.length ? errors.join('\n') : '  нет');
console.log('\n--- итог ---');
console.log(fails.length ? fails.map((f) => '  FAIL ' + f).join('\n') : '  все проверки прошли');

await browser.close();
process.exit(fails.length || errors.length ? 1 : 0);
