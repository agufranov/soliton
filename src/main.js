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
import { Renderer, AimOverlay } from './render/renderer.js';
import { GpuRenderer } from './render/gpuRenderer.js';

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

  state.grid = new Grid(n, n, model.grid.L, model.grid.L);
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
  $('gridInfo').textContent =
    `${n}×${n}, L=${model.grid.L}, dx=${state.grid.dx.toFixed(3)}, dt=${dtFor(model, n).toFixed(4)}`;
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

function buildPresetButtons() {
  const box = $('presets');
  box.innerHTML = '';
  for (const p of state.model.presets) {
    const b = document.createElement('button');
    b.className = 'preset';
    b.textContent = p.label;
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
    b.textContent = k.label;
    b.addEventListener('click', () => {
      state.spawnKind = k.id;
      [...box.children].forEach((c) => c.classList.toggle('on', c === b));
    });
    box.appendChild(b);
  }
}

function applyPreset(p) {
  // A preset can fail the same way a click can: no soliton exists at these
  // parameters. Without this it just left an empty box and said "сценарий: ...".
  const ok = state.solver.applyPreset(p);
  state.lastPreset = p;
  renderer.scaleSmoothed = 1;
  if (gpuRenderer) gpuRenderer.scaleSmoothed = 1;
  state.running = true;
  $('playPause').textContent = '⏸ пауза';
  note(ok === false
    ? 'профиль не сходится: солитона при таких параметрах нет'
    : `сценарий: ${p.label}`);
}

function note(msg) {
  state.note = msg;
  state.noteUntil = performance.now() + 2600;
}

// --- mouse -----------------------------------------------------------------
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

// --- controls --------------------------------------------------------------
$('model').innerHTML = MODELS.map((m) => `<option value="${m.id}">${m.name}</option>`).join('');
$('model').addEventListener('change', (e) => { buildModel(e.target.value); savePrefs(); });

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

$('playPause').addEventListener('click', () => {
  state.running = !state.running;
  $('playPause').textContent = state.running ? '⏸ пауза' : '▶ пуск';
});
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

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
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
    state.running = false;
    $('playPause').textContent = '▶ пуск';
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

  $('stats').innerHTML =
    `<div><span>t</span><b>${state.solver.time.toFixed(2)}</b></div>`
    + Object.entries(d).map(([k, v]) =>
      `<div><span>${k}</span><b>${Number.isFinite(v) ? v.toFixed(4) : '—'}</b></div>`).join('')
    + `<div><span>мс/шаг</span><b>${state.msPerStep.toFixed(2)}</b></div>`
    + `<div><span>шагов/кадр</span><b>${state.stepsPerFrame}</b></div>`
    + `<div><span>fps</span><b>${fps.toFixed(0)}</b></div>`;

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
  // over a second per step. Restore it only if we actually got the GPU.
  const savedN = Number(prefs.resolution);
  if ([128, 256, 512, 1024].includes(savedN)) {
    state.resolution = state.backendKind === 'gpu' ? savedN : Math.min(savedN, 256);
    if (state.resolution !== savedN) note(`сетка ${savedN}² сохранена, но без GPU это секунды на кадр — открываю ${state.resolution}²`);
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
  $('model').value = model;
  buildModel(model);
  savePrefs();
  requestAnimationFrame(frame);
})();
