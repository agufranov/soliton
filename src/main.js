// App wiring: backend and model selection, controls, mouse spawning, render loop.
//
// Everything numerical lives behind `solver` (models/) and `backend` (core/);
// this file only decides *when* to step and *what* to draw.
//
// The backend switch is the reason several things here are duplicated. A CPU
// step is synchronous, so it can be timed with a stopwatch around the call and
// the field can be read back for free. A GPU step only *records* work, so it is
// timed by waiting for the queue to drain, and the field is drawn straight out
// of device memory. Both paths feed the same adaptive steps-per-frame control.

import { Grid } from './core/grid.js';
import { CpuBackend } from './core/backend.js';
import { requestGpuContext, WebGpuBackend } from './core/webgpuBackend.js';
import { MODELS, modelById } from './models/index.js';
import { formulaFor } from './models/formula.js';
import { Renderer, AimOverlay } from './render/renderer.js';
import { GpuRenderer } from './render/gpuRenderer.js';
import { presetIcon, kindIcon } from './render/icons.js';

const $ = (id) => document.getElementById(id);

const state = {
  model: null,
  grid: null,
  backend: null,
  solver: null,
  params: {},
  resolution: 128,
  backendKind: 'cpu',
  running: true,
  spawnKind: null,
  budgetMs: 14,
  msPerStep: 8,
  stepsPerFrame: 1,
  gpuInFlight: 0,
  // How many submitted batches may be outstanding before the frame loop stops
  // queueing more. Named rather than inlined so tools/inflight.mjs can sweep
  // it: this is a throughput/latency trade and the value is a measurement.
  //
  // 3, not 2. A frame submits at most one batch, so the ceiling is
  // stepsPerFrame*fps, and a batch longer than a frame cannot be topped up
  // fast enough at depth 2. Measured steps/s: 512^2 115.2 -> 119.5 (its rAF
  // ceiling is 120), 1024^2 26.7 -> 27.4, and on to 29.1 at depth 6. Neutral
  // at 128^2/256^2, which sit at the rAF ceiling already. Deeper than 3 buys
  // a few percent at 1024^2 only, and costs latency: worst-case input lag is
  // depth * batch, and a 1024^2 batch is ~30 ms.
  inFlightCap: 3,
  drag: null,
  lastPreset: null,
  note: '',
  noteUntil: 0,
};

// Debug hook for tools/. Everything under tools/ drives the real app rather
// than a copy of it, so a pacing knob that cannot be reached from outside
// cannot be measured either.
globalThis.__soliton = state;

// --- settings that survive a reload -----------------------------------------
// Model, grid, backend, tempo and relief. Solver parameters (q, s) are
// deliberately NOT saved: q can be restored to a value for which no soliton
// exists (cubic-quintic diverges from q ~ 0.35 at mu = 0.7), and opening on an
// empty box is precisely the silent-zero trap documented in CLAUDE.md.
// Storage may throw outright under file:// or a locked-down profile, so every
// access is guarded - losing preferences is not a reason to fail to start.
const PREFS_KEY = 'soliton.prefs.v1';

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; } catch { return {}; }
}

function savePrefs() {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify({
      model: state.model && state.model.id,
      resolution: state.resolution,
      backend: state.backendKind,
      tempo: state.budgetMs,
      relief: Number($('relief').value),
    }));
  } catch { /* приватный режим или переполненное хранилище — не повод падать */ }
}

const renderer = new Renderer($('view'));
const aim = new AimOverlay($('overlay'));
let gpuCtx = null;         // { device, label, ... } once WebGPU has been asked
let gpuRenderer = null;

// --- dt policy -------------------------------------------------------------
// The KdV-type models are limited by a nonlinear CFL that scales with dx, so
// their step has to shrink as the grid is refined. The NLS split-step and the
// sine-Gordon wave solver propagate their linear parts exactly and are
// essentially dt-insensitive over the range we allow, so they keep their dt.
function dtFor(model, n) {
  const scale = model.family === 'kp' || model.family === 'zk' ? 128 / n : 1;
  return model.dt * scale;
}

// --- box shape -------------------------------------------------------------
// The simulation box follows the shape of the stage, not the other way round.
// A square field on a 20:9 phone leaves a third of the screen black, and the
// grid does not have to become rectangular to fix that: `Grid` keeps Lx and Ly
// apart, so a portrait box is nx = ny points over Lx x (Lx * aspect). The cell
// count is unchanged, so a step costs exactly what it cost before; what this
// buys in screen area it pays for in vertical resolution, dy = aspect * dx.
// That is the cheap direction to give up in the KdV-type models - y enters
// through a second derivative where x enters through a third.
//
// Rectangular *grids* are the other way to do it and are not available: the
// GPU FFT transposes in place and refuses nx != ny (webgpuBackend.js).
//
// The two aspects must agree. The picture maps [0,Lx] x [0,Ly] onto the stage
// box, so a mismatch turns every lump into an ellipse and makes the aim arrow
// point somewhere the soliton will not go.
function stageAspect() {
  const r = $('stage').getBoundingClientRect();
  if (!(r.width > 0 && r.height > 0)) return 1;
  return r.height / r.width;
}

// Coarse enough that an address bar sliding in and out cannot trigger a
// rebuild, fine enough that a rotation always does.
const aspectStep = (a) => Math.round(a * 20) / 20;

// --- backend ---------------------------------------------------------------
function makeBackend(grid) {
  if (state.backendKind === 'gpu' && gpuCtx) return new WebGpuBackend(gpuCtx, grid);
  return new CpuBackend(grid);
}

function usingGpu() { return !!(state.backend && state.backend.gpu); }

// Build grid + backend + solver for the current model, resolution and backend.
// Everything below the model interface is thrown away and rebuilt: the ETDRK4
// coefficients, the FFT plans and every device buffer depend on all three.
function buildSolver() {
  const n = state.resolution;
  const model = state.model;
  // Let the old buffers go BEFORE allocating the new ones. A KP solver at
  // 1024^2 is ~300 MB of Float64Array (13 ETDRK4 fields plus the grid tables),
  // and holding the outgoing one alive across `create` is what killed the tab
  // when stepping up through the resolutions on the CPU.
  if (state.backend) state.backend.dispose();
  state.solver = null;
  state.backend = null;
  state.grid = null;

  state.aspect = stageAspect();
  state.grid = new Grid(n, n, model.grid.L, model.grid.L * state.aspect);
  state.backend = makeBackend(state.grid);
  state.solver = model.create(state.backend, state.grid, dtFor(model, n), state.params);

  // Only size the backing store that is actually going to be painted: the CPU
  // renderer's ImageData is another 4 MB at 1024^2, and in GPU mode nothing
  // ever writes to it.
  if (usingGpu()) {
    gpuRenderer.resizeFor(state.grid);
  } else {
    renderer.resizeFor(state.grid);
  }
  // Bind groups point at buffers that have just been destroyed, whichever way
  // the switch went.
  if (gpuRenderer) gpuRenderer.invalidate();
  aim.resize();
  $('view').classList.toggle('off', usingGpu());
  $('viewGpu').classList.toggle('off', !usingGpu());

  state.msPerStep = usingGpu() ? 0.5 : 8;
  state.gpuInFlight = 0;
  const g = state.grid;
  const box = g.Ly === g.Lx
    ? `L=${g.Lx}, dx=${g.dx.toFixed(3)}`
    : `L=${g.Lx}×${g.Ly.toFixed(1)}, dx=${g.dx.toFixed(3)}, dy=${g.dy.toFixed(3)}`;
  $('gridInfo').textContent = `${n}×${n}, ${box}, dt=${dtFor(model, n).toFixed(4)}`;
  $('backendInfo').textContent = state.backend.name;
}

function buildModel(id, keepParams = false) {
  const model = modelById(id);
  const prev = state.params;
  state.model = model;
  state.params = {};
  for (const p of model.params) {
    state.params[p.key] = keepParams && p.key in prev ? prev[p.key] : p.value;
  }
  state.spawnKind = model.spawnKinds[0].id;
  state.lastPreset = null;

  buildSolver();

  $('blurb').textContent = model.blurb;
  syncModelBtn();
  buildParamControls();
  buildPresetButtons();
  buildSpawnKinds();
  if (model.presets.length) applyPreset(model.presets[0]);
}

// Rebuild in place and put the same scene back, so that switching backend or
// resolution is a like-for-like comparison rather than a blank canvas.
function rebuildSolver() {
  buildSolver();
  if (state.lastPreset) state.solver.applyPreset(state.lastPreset);
  renderer.scaleSmoothed = 1;
  if (gpuRenderer) gpuRenderer.scaleSmoothed = 1;
}

function buildParamControls() {
  const box = $('params');
  box.innerHTML = '';
  for (const p of state.model.params) {
    const row = document.createElement('div');
    row.className = 'ctl';
    row.innerHTML = `
      <label>${p.label}<span class="val" id="v_${p.key}">${state.params[p.key]}</span></label>
      <input type="range" id="p_${p.key}" min="${p.min}" max="${p.max}" step="${p.step}"
             value="${state.params[p.key]}">
      ${p.hint ? `<div class="hint">${p.hint}</div>` : ''}`;
    box.appendChild(row);
    const input = row.querySelector('input');
    input.addEventListener('input', () => {
      state.params[p.key] = Number(input.value);
      $(`v_${p.key}`).textContent = input.value;
      // `q` (and `s`) enter the nonlinearity itself, which the NLS solver
      // captured when it was built - and on the GPU it is baked into the
      // shader source - so it needs a fresh solver.
      if (p.key === 'q') { rebuildSolver(); note('солвер пересобран под новое q'); }
    });
  }
}

// The icon is drawn from the preset's own items (render/icons.js), so a new
// preset gets a picture of itself for free - and one that cannot disagree with
// what the button does. `innerHTML` is safe here: every label in models/ is
// ours, and the icon is a string of SVG we just generated.
function buildPresetButtons() {
  const box = $('presets');
  box.innerHTML = '';
  for (const p of state.model.presets) {
    const b = document.createElement('button');
    b.className = 'preset';
    b.innerHTML = presetIcon(state.model, p) + `<span>${p.label}</span>`;
    b.addEventListener('click', () => applyPreset(p));
    box.appendChild(b);
  }
}

function buildSpawnKinds() {
  const box = $('kinds');
  const kinds = state.model.spawnKinds;
  box.innerHTML = '';
  if (kinds.length < 2) { box.style.display = 'none'; return; }
  box.style.display = '';
  for (const k of kinds) {
    const b = document.createElement('button');
    b.className = 'kind' + (k.id === state.spawnKind ? ' on' : '');
    b.innerHTML = kindIcon(k.id) + `<span>${k.label}</span>`;
    b.addEventListener('click', () => {
      state.spawnKind = k.id;
      [...box.children].forEach((c) => c.classList.toggle('on', c === b));
    });
    box.appendChild(b);
  }
}

// Run state lives in two places on a phone: the panel button and the mirror on
// the sheet handle, which is the only one visible while the sheet is closed.
// Everything that pauses or resumes goes through here so the two cannot drift.
function setRunning(on) {
  state.running = on;
  $('playPause').textContent = on ? '⏸ пауза' : '▶ пуск';
  $('sheetPlay').textContent = on ? '⏸' : '▶';
}

function applyPreset(p) {
  // A preset can fail the same way a click can: no soliton exists at these
  // parameters. Without this it just left an empty box and said "сценарий: ...".
  const ok = state.solver.applyPreset(p);
  state.lastPreset = p;
  renderer.scaleSmoothed = 1;
  if (gpuRenderer) gpuRenderer.scaleSmoothed = 1;
  setRunning(true);
  note(ok === false
    ? 'профиль не сходится: солитона при таких параметрах нет'
    : `сценарий: ${p.label}`);
}

function note(msg) {
  state.note = msg;
  state.noteUntil = performance.now() + 2600;
}

// --- mobile sheet ----------------------------------------------------------
// On a phone the panel is a bottom sheet: the same <aside>, translated down
// until only its handle (#sheetBar) shows. Open/closed is one class; the CSS
// media query is what decides whether any of this is visible at all, so on a
// desktop these handlers simply never fire - #sheetBar is display:none there.
const panel = document.querySelector('aside');
const sheetBar = $('sheetBar');
// The single query that decides whether the panel is a sheet at all - it has
// to match the media query in index.html exactly.
const sheetLayout = matchMedia('(max-width: 860px) and (orientation: portrait)');

function sheetIsOpen() { return panel.classList.contains('open'); }

function setSheet(open) {
  panel.classList.toggle('open', open);
  sheetBar.setAttribute('aria-expanded', String(open));
  // Closing with the content scrolled would leave the sticky handle sitting on
  // top of some random middle of the panel next time it opens.
  if (!open) panel.scrollTop = 0;
}

// How far down the sheet sits when collapsed, in pixels. Measured rather than
// computed: the resting offset is `--peek` plus the safe-area inset, and
// env() read back through getComputedStyle is not something to rely on. The
// sheet always starts closed, so the first drag measures the real number and
// every later one reuses it.
let collapsedSpan = null;

function sheetSpan() {
  const openTop = window.innerHeight - panel.offsetHeight;   // where translateY(0) puts it
  const now = panel.getBoundingClientRect().top - openTop;
  if (!sheetIsOpen() && now > 20) collapsedSpan = now;
  if (collapsedSpan != null) return collapsedSpan;
  const peek = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--peek'));
  return panel.offsetHeight - (Number.isFinite(peek) ? peek : 62);
}

// Drag-to-open. The sheet follows the finger (transition off, transform set
// inline) and snaps on release; a move shorter than a few pixels is a tap and
// just toggles. Without this the handle is a button that happens to look like
// a sheet, which reads as broken on a touch screen.
let sheetDrag = null;

sheetBar.addEventListener('pointerdown', (ev) => {
  if (ev.target.closest('button')) return;      // the play mirror is not a grip
  sheetBar.setPointerCapture(ev.pointerId);
  sheetDrag = { y0: ev.clientY, dy: 0, open: sheetIsOpen(), span: sheetSpan() };
  panel.classList.add('dragging');
});

sheetBar.addEventListener('pointermove', (ev) => {
  if (!sheetDrag) return;
  sheetDrag.dy = ev.clientY - sheetDrag.y0;
  const base = sheetDrag.open ? 0 : sheetDrag.span;
  const y = Math.max(0, Math.min(sheetDrag.span, base + sheetDrag.dy));
  panel.style.transform = `translateY(${y}px)`;
});

function endSheetDrag() {
  if (!sheetDrag) return;
  const d = sheetDrag;
  sheetDrag = null;
  panel.style.transform = '';
  panel.classList.remove('dragging');
  setSheet(Math.abs(d.dy) < 8 ? !d.open : d.dy < 0);
}
sheetBar.addEventListener('pointerup', endSheetDrag);
sheetBar.addEventListener('pointercancel', endSheetDrag);
sheetBar.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); setSheet(!sheetIsOpen()); }
});

$('sheetPlay').addEventListener('click', (ev) => {
  ev.stopPropagation();               // …and not a toggle of the sheet
  setRunning(!state.running);
});

// --- mouse and touch --------------------------------------------------------
const stage = $('stage');

function canvasToWorld(ev) {
  const r = stage.getBoundingClientRect();
  const fx = (ev.clientX - r.left) / r.width;
  const fy = (ev.clientY - r.top) / r.height;
  return {
    x: fx * state.grid.Lx, y: fy * state.grid.Ly,
    px: fx * $('overlay').width, py: fy * $('overlay').height,
  };
}

stage.addEventListener('pointerdown', (ev) => {
  stage.setPointerCapture(ev.pointerId);
  // Touching the field means you are done with the panel: on a phone the open
  // sheet covers the lower two thirds of what you are aiming at.
  if (sheetIsOpen()) setSheet(false);
  const w = canvasToWorld(ev);
  state.drag = { start: w, current: w };
});
stage.addEventListener('pointermove', (ev) => {
  if (state.drag) state.drag.current = canvasToWorld(ev);
});
stage.addEventListener('pointerup', (ev) => {
  if (!state.drag) return;
  const { start } = state.drag;
  const end = canvasToWorld(ev);
  // Normalise the drag to [-1,1]: a quarter of the box is "full throttle".
  const q = state.grid.Lx / 4;
  let dx = (end.x - start.x) / q, dy = (end.y - start.y) / q;
  const len = Math.hypot(dx, dy);
  if (len > 1) { dx /= len; dy /= len; }
  state.drag = null;
  aim.clear();
  const ok = state.solver.spawn(start.x, start.y, { x: dx, y: dy }, state.spawnKind);
  if (ok === false) note('профиль не сходится: солитона при таких параметрах нет');
});
stage.addEventListener('pointercancel', () => { state.drag = null; aim.clear(); });

// --- picking the equation ---------------------------------------------------
// A hand-built listbox instead of a <select>, for one reason: a native <option>
// raises no hover events, and the formula has to appear while you are still
// choosing, not after. The <select id="model"> stays in the DOM as the value
// store - tools/ set it and dispatch `change`, which is the same path the list
// below goes through, so nothing can be selected two different ways.
//
// Desktop: hover an item, the card shows up beside the list.
// Narrow or touch: the first tap shows the formula, the second (or the button
// inside the card) applies it. Applying on the first tap would mean the formula
// is only ever seen for the model you already left.
const modelSel = $('model');
const modelBtn = $('modelBtn');
const modelList = $('modelList');
const eqPrev = $('eqPrev');

// Not `sheetLayout`: this is about where the card fits and whether hovering
// exists at all, which is also true of a landscape phone and of a narrow
// desktop window. The comma is "or".
const compact = matchMedia('(hover: none), (max-width: 900px)');

let hiIdx = -1;                      // пункт под курсором/пальцем, не выбранный

modelSel.innerHTML = MODELS.map((m) => `<option value="${m.id}">${m.name}</option>`).join('');
modelList.innerHTML = MODELS.map((m, i) =>
  `<div class="mitem" role="option" data-i="${i}">${m.name}</div>`).join('');

modelSel.addEventListener('change', (e) => { buildModel(e.target.value); savePrefs(); });

function syncModelBtn() {
  modelBtn.querySelector('.nm').textContent = state.model ? state.model.name : '';
  modelSel.value = state.model ? state.model.id : MODELS[0].id;
  markList();
}

const listOpen = () => modelList.classList.contains('on');

function previewHtml(model, i) {
  const f = formulaFor(model.id);
  if (!f) return `<div class="ptit">${model.name}</div>`;
  return `<div class="ptit">${model.name}</div>`
    + `<div class="peq">${f.eq}</div>`
    + `<ul class="pleg">${f.parts.map((p) =>
      `<li><span class="dot k-${p.cls}"></span><span><b>${p.html}</b> — ${p.text}</span></li>`)
      .join('')}</ul>`
    + `<button class="pick" type="button" data-i="${i}">выбрать</button>`;
}

function showPrev(i) {
  const el = modelList.children[i];
  if (!el) return;
  eqPrev.innerHTML = previewHtml(MODELS[i], i);
  eqPrev.classList.toggle('compact', compact.matches);
  eqPrev.classList.add('on');
  if (compact.matches) {
    // Место и размер карточки задаёт CSS (она внизу, во всю ширину), а вот
    // список приходится подрезать здесь: на 375x667 карточка с разбором
    // формулы съедает низ экрана, и половина списка ушла бы под неё. Короткий
    // прокручиваемый список лучше длинного наполовину закрытого.
    eqPrev.style.left = ''; eqPrev.style.top = '';
    const top = modelList.getBoundingClientRect().top;
    const cardTop = eqPrev.getBoundingClientRect().top;
    modelList.style.maxHeight = `${Math.max(140, cardTop - top - 10)}px`;
    return;
  }
  // Beside the list, level with the item. To the right if it fits, otherwise
  // to the left - on a narrow desktop window the panel can sit close enough to
  // the edge that the card would hang off screen.
  const lr = modelList.getBoundingClientRect(), ir = el.getBoundingClientRect();
  const w = eqPrev.offsetWidth, h = eqPrev.offsetHeight;
  const right = lr.right + 10;
  eqPrev.style.left = `${right + w + 8 <= innerWidth ? right : Math.max(8, lr.left - w - 10)}px`;
  eqPrev.style.top = `${Math.max(8, Math.min(ir.top - 12, innerHeight - h - 8))}px`;
}

function hidePrev() {
  eqPrev.classList.remove('on');
  modelList.style.maxHeight = '';        // подрезка из showPrev, см. там
}

function markList() {
  const cur = state.model ? MODELS.findIndex((m) => m.id === state.model.id) : -1;
  [...modelList.children].forEach((el, i) => {
    el.classList.toggle('hi', i === hiIdx);
    el.classList.toggle('cur', i === cur);
    el.setAttribute('aria-selected', String(i === cur));
  });
  if (hiIdx >= 0 && listOpen()) showPrev(hiIdx); else hidePrev();
}

function openList(on) {
  modelList.classList.toggle('on', on);
  modelBtn.classList.toggle('open', on);
  modelBtn.setAttribute('aria-expanded', String(on));
  // Opening highlights the current model - on a touch screen that also means
  // its formula is already on screen without a tap.
  hiIdx = on ? MODELS.findIndex((m) => m.id === state.model.id) : -1;
  markList();
  if (on && hiIdx >= 0) modelList.children[hiIdx].scrollIntoView({ block: 'nearest' });
}

function chooseModel(i) {
  openList(false);
  modelBtn.blur();          // иначе пробел снова откроет список, а не пустит счёт
  modelSel.value = MODELS[i].id;
  modelSel.dispatchEvent(new Event('change'));
}

modelBtn.addEventListener('click', () => openList(!listOpen()));
modelList.addEventListener('pointerover', (e) => {
  if (compact.matches) return;
  const it = e.target.closest('.mitem');
  if (it) { hiIdx = Number(it.dataset.i); markList(); }
});
// On a touch screen the finger leaves the item the moment it lands, so the
// card must not follow it out.
modelList.addEventListener('pointerleave', () => { if (!compact.matches) hidePrev(); });
modelList.addEventListener('click', (e) => {
  const it = e.target.closest('.mitem');
  if (!it) return;
  const i = Number(it.dataset.i);
  if (compact.matches && hiIdx !== i) { hiIdx = i; markList(); return; }
  chooseModel(i);
});
eqPrev.addEventListener('click', (e) => {
  const b = e.target.closest('.pick');
  if (b) chooseModel(Number(b.dataset.i));
});
// The card lives outside #modelBox, so a tap on it must not count as a tap
// outside the list - the button would vanish from under the finger before the
// click ever fired.
document.addEventListener('pointerdown', (e) => {
  if (listOpen() && !$('modelBox').contains(e.target) && !eqPrev.contains(e.target)) openList(false);
});
modelBtn.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { openList(false); return; }
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (!listOpen()) { openList(true); return; }
    const step = e.key === 'ArrowDown' ? 1 : -1;
    hiIdx = Math.max(0, Math.min(MODELS.length - 1, (hiIdx < 0 ? 0 : hiIdx) + step));
    markList();
    modelList.children[hiIdx].scrollIntoView({ block: 'nearest' });
    return;
  }
  if (listOpen() && (e.key === 'Enter' || e.code === 'Space')) {
    e.preventDefault();
    if (hiIdx >= 0) chooseModel(hiIdx);
  }
});

// On a phone the picker moves out of the sheet and up above the field: inside
// the sheet it was the first line of content, i.e. invisible until the sheet
// was opened. It is the same element in both places - main.js finds controls
// by id, and a second copy would be a second source of truth.
function placeModelBox() {
  const box = $('modelBox');
  const host = sheetLayout.matches ? $('headSlot') : $('modelSlot');
  if (box.parentElement !== host) host.appendChild(box);
  openList(false);
}
sheetLayout.addEventListener('change', placeModelBox);
compact.addEventListener('change', () => openList(false));

// --- controls --------------------------------------------------------------

$('resolution').addEventListener('change', (e) => {
  state.resolution = Number(e.target.value);
  if (!usingGpu() && state.resolution >= 512) {
    note(`${state.resolution}² на CPU — секунды на кадр, это и есть демонстрация`);
  }
  rebuildSolver();
  savePrefs();
  if (usingGpu() || state.resolution < 512) note(`сетка ${state.resolution}×${state.resolution}`);
});

// WebGPU is asked for once, on demand, and the answer is cached. A machine
// without it keeps the option disabled instead of failing at switch time.
async function ensureGpu() {
  if (gpuCtx) return gpuCtx;
  gpuCtx = await requestGpuContext();
  if (gpuCtx && !gpuRenderer) gpuRenderer = new GpuRenderer($('viewGpu'), gpuCtx);
  return gpuCtx;
}

$('backend').addEventListener('change', async (e) => {
  const want = e.target.value;
  if (want === 'gpu' && !(await ensureGpu())) {
    e.target.value = 'cpu';
    note('WebGPU в этом браузере недоступен');
    return;
  }
  state.backendKind = want;
  rebuildSolver();
  savePrefs();
  note(want === 'gpu'
    ? `считает GPU (${gpuCtx.label}), одинарная точность`
    : 'считает CPU, двойная точность');
});

$('tempo').addEventListener('change', (e) => {
  state.budgetMs = Number(e.target.value);
  savePrefs();
});

$('playPause').addEventListener('click', () => setRunning(!state.running));
$('clear').addEventListener('click', () => {
  state.solver.clear();
  state.lastPreset = null;
  renderer.scaleSmoothed = 1;
  if (gpuRenderer) gpuRenderer.scaleSmoothed = 1;
  note('поле очищено — кликай, чтобы породить горбы');
});
$('relief').addEventListener('input', (e) => {
  renderer.relief = Number(e.target.value);
  if (gpuRenderer) gpuRenderer.relief = Number(e.target.value);
  savePrefs();
});

// --- the stage box can change without a rebuild ------------------------------
// Both the aim overlay and the WebGPU canvas size their backing store from the
// element's CSS box, and until now that was only ever read inside buildSolver.
// Rotating a phone changes the box without touching the solver: the arrow then
// gets drawn into a stale buffer and the GPU picture is resampled from the old
// size. Debounced, because an orientation change fires a burst of these and
// reallocating a 1024^2-sized canvas per event is not free.
let resizeTimer = 0;
function onViewportResize() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    // A rotation changes the shape of the box, not just its size on screen, so
    // the grid tables and every solver coefficient have to be rebuilt. Only
    // when the shape really moved: rebuilding drops everything but the last
    // preset, and this fires on every address-bar twitch.
    if (aspectStep(stageAspect()) !== aspectStep(state.aspect)) {
      rebuildSolver();
      note('ящик пересобран под новую форму экрана');
      return;
    }
    aim.resize();
    if (usingGpu() && gpuRenderer && state.grid) gpuRenderer.resizeFor(state.grid);
    else if (state.grid) renderer.resizeFor(state.grid);
  }, 150);
}
window.addEventListener('resize', onViewportResize);
window.addEventListener('orientationchange', onViewportResize);

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  // On the equation button the keyboard belongs to the list: space opens it,
  // arrows walk it. Pausing on space there would fire on the same keypress.
  if (e.target === modelBtn) return;
  if (e.code === 'Space') { e.preventDefault(); $('playPause').click(); }
  if (e.key === 'c') $('clear').click();
});

// --- loop ------------------------------------------------------------------
let lastFrame = performance.now();
let fps = 60;

function frame() {
  const now = performance.now();
  fps += (1000 / Math.max(now - lastFrame, 1) - fps) * 0.1;
  lastFrame = now;

  const gpu = usingGpu();
  // 128^2 on the GPU is ~0.06 ms a step, so the cap has to be generous or the
  // grid the app opens on is the one place the GPU looks pointless.
  const cap = gpu ? 256 : 48;
  state.stepsPerFrame = Math.max(1, Math.min(cap, Math.round(state.budgetMs / state.msPerStep)));

  // GPU pacing. `onSubmittedWorkDone` costs ~18 ms of fixed round trip on this
  // machine whatever the queue holds, so gating on a single outstanding batch
  // idles the device for that long between batches - at 1024^2 that turned a
  // 50 ms step into a 142 ms one. Keeping two batches in flight means the
  // device always has the next one queued while the current one drains, and
  // two is still few enough that the picture cannot lag behind the mouse.
  let stepped = 0;
  if (state.running && (!gpu || state.gpuInFlight < state.inFlightCap)) {
    const t0 = performance.now();
    for (let s = 0; s < state.stepsPerFrame; s++) state.solver.step();
    stepped = state.stepsPerFrame;
    if (!gpu) {
      const per = (performance.now() - t0) / stepped;
      state.msPerStep += (per - state.msPerStep) * 0.2;
    }
  }

  const d = state.solver.diagnostics();
  const peak = state.solver.peak();

  // Blow-up guard. For the cubic NLS this is the physics (collapse); anywhere
  // else it means the step is too large, so say which one it is.
  const guard = state.model.collapseGuard;
  const bad = !Number.isFinite(peak) || (guard && peak > guard);
  if (bad && state.running) {
    setRunning(false);
    note(guard
      ? 'коллапс: амплитуда ушла в бесконечность за конечное время'
      : 'решение разошлось — уменьши шаг или амплитуду');
  }

  if (gpu) {
    const target = state.solver.target();
    // The GPU reports the pass duration itself. Timing it from here cannot
    // work: with two batches overlapping, submit-to-drain measures queueing,
    // and with one batch it measures the ~18 ms completion round trip. Both
    // feed back into steps-per-frame and both collapse it to 1.
    state.backend.flush(stepped > 0 ? (ms) => {
      state.msPerStep += (ms / stepped - state.msPerStep) * 0.3;
    } : null);
    gpuRenderer.draw(target, state.model.field.kind, peak);
    state.gpuInFlight++;
    state.backend.device.queue.onSubmittedWorkDone().then(() => { state.gpuInFlight--; });
  } else {
    renderer.draw(state.solver.field(), state.model.field.kind);
  }

  if (state.drag) {
    aim.draw(
      { x: state.drag.start.px, y: state.drag.start.py },
      { x: state.drag.current.px, y: state.drag.current.py },
    );
  }

  const num = (v) => (Number.isFinite(v) ? v.toFixed(4) : '—');
  $('stats').innerHTML =
    `<div><span>t</span><b>${state.solver.time.toFixed(2)}</b></div>`
    + Object.entries(d).map(([k, v]) => `<div><span>${k}</span><b>${num(v)}</b></div>`).join('')
    + `<div><span>мс/шаг</span><b>${state.msPerStep.toFixed(2)}</b></div>`
    + `<div><span>шагов/кадр</span><b>${state.stepsPerFrame}</b></div>`
    + `<div><span>fps</span><b>${fps.toFixed(0)}</b></div>`;

  // The chips under the field on a phone. `matches` on a stored MediaQueryList
  // is a plain flag read - no layout, unlike asking the element whether it is
  // displayed - so this stays a cheap per-frame check.
  if (sheetLayout.matches) {
    $('miniStats').innerHTML =
      `<span class="chip">t <b>${state.solver.time.toFixed(1)}</b></span>`
      + Object.entries(d).map(([k, v]) => `<span class="chip">${k} <b>${num(v)}</b></span>`).join('')
      + `<span class="chip">fps <b>${fps.toFixed(0)}</b></span>`;
  }

  $('note').textContent = performance.now() < state.noteUntil ? state.note : '';
  requestAnimationFrame(frame);
}

// Boot. The GPU is now the default, so the device really is created up front
// rather than on first use - and the whole decision has to finish before the
// first solver is built, or the app would allocate a CPU solver and throw it
// away one frame later (which at 1024^2 is 600 MB of Float64Array churn, the
// exact thing that used to kill the tab; see "Ловушки" in CLAUDE.md).
(async () => {
  const prefs = loadPrefs();
  const ok = typeof navigator !== 'undefined' && !!navigator.gpu;
  $('backend').querySelector('option[value=gpu]').disabled = !ok;

  const wantGpu = (prefs.backend ?? 'gpu') === 'gpu';
  state.backendKind = wantGpu && ok && await ensureGpu() ? 'gpu' : 'cpu';
  $('backend').value = state.backendKind;
  $('backendHint').textContent = state.backendKind === 'gpu'
    ? 'GPU считает в float32: инварианты «дышат» в 5–6 знаке, это не ошибка шага.'
    : ok
      ? 'GPU не отдал устройство — считает CPU в двойной точности.'
      : 'WebGPU в этом браузере недоступен — доступен только CPU.';

  // A grid saved while on the GPU can be ruinous on the CPU: 1024^2 there is
  // over a second per step. Restore it only if we actually got the GPU - and
  // on a phone go one notch lower still on both paths: the CPU there is a
  // fraction of a laptop's, and a 1024^2 GPU solver is ~100 MB of device
  // buffers on a chip that shares its memory with everything else.
  // This caps only what is *restored*; every grid stays selectable by hand.
  const phone = matchMedia('(pointer: coarse) and (max-width: 900px)').matches;
  const cap = state.backendKind === 'gpu' ? (phone ? 512 : 1024) : (phone ? 128 : 256);
  const savedN = Number(prefs.resolution);
  if ([128, 256, 512, 1024].includes(savedN)) {
    state.resolution = Math.min(savedN, cap);
    if (state.resolution !== savedN) {
      note(phone && state.backendKind === 'gpu'
        ? `сетка ${savedN}² сохранена, но это телефон — открываю ${state.resolution}²`
        : `сетка ${savedN}² сохранена, но без GPU это секунды на кадр — открываю ${state.resolution}²`);
    }
  }
  $('resolution').value = String(state.resolution);

  if ([6, 14, 30].includes(Number(prefs.tempo))) {
    state.budgetMs = Number(prefs.tempo);
    $('tempo').value = String(state.budgetMs);
  }
  if (Number.isFinite(prefs.relief)) {
    $('relief').value = String(prefs.relief);
    renderer.relief = prefs.relief;
    if (gpuRenderer) gpuRenderer.relief = prefs.relief;
  }

  const model = MODELS.some((m) => m.id === prefs.model) ? prefs.model : MODELS[0].id;
  modelSel.value = model;
  placeModelBox();          // до первого показа: на телефоне список живёт наверху
  buildModel(model);
  // On a phone the legend explaining what a tap does is inside the collapsed
  // sheet, i.e. invisible until you already know to open it. This replaces the
  // "сценарий: ..." note that buildModel just raised, which matters less.
  if (phone && !prefs.model) note('тап — солитон, протяжка — бросок');
  savePrefs();
  requestAnimationFrame(frame);

  // The box was built from the stage box as it stood before anything had been
  // painted into it. Web fonts, the chips filling in, a scrollbar appearing -
  // any of it moves the stage, and the box has to follow or the picture is
  // drawn stretched. Settle once, two frames in, instead of trusting the
  // boot-time measurement.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    if (aspectStep(stageAspect()) !== aspectStep(state.aspect)) rebuildSolver();
  }));
})();
