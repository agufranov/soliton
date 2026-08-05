// 2D nonlinear Schrodinger family
//
//     i psi_t + (1/2) laplacian(psi) + f(|psi|^2) psi = 0
//
// Three nonlinearities, and the difference between them is the whole story of
// why 2D solitons are hard:
//
//   f(I) = I                critical. The Townes profile exists but is
//                           unstable: nudge it up and it collapses to a
//                           singularity in finite time, nudge it down and it
//                           disperses. There is no stable 2D soliton here.
//   f(I) = I - q I^2        cubic-quintic. The quintic term arrests collapse
//                           and 2D "light bullets" become genuinely stable.
//   f(I) = I / (1 + s I)    saturable (photorefractive media). Same cure.
//
// Not integrable, so collisions are only near-elastic: at small crossing
// angles and with the wrong relative phase two bullets can fuse or repel
// instead of passing through. That is physics, not a numerical artefact.
//
// Unlike KP and ZK, the NLS family is Galilean invariant, so a soliton can be
// boosted to *any* velocity by multiplying it with exp(i (vx x + vy y)).
// This is the model to use for oblique, mouse-aimed collisions.

import { Grid } from '../core/grid.js';
import { NlsSolver } from '../core/nlsSolver.js';
import { solveProfile, upsamplePeriodic, cachePut, PROFILE_N } from '../core/petviashvili.js';
import { addCenteredBoosted } from '../core/util.js';

const cache = new Map();

// WGSL float literal: bare integers are i32 there, so 2 would not type-check
// against an f32 expression.
const L = (v) => (Number.isInteger(v) ? v.toFixed(1) : String(v));

// Each nonlinearity is given twice, as JS and as a WGSL expression in `x`.
// The backend picks whichever it can run; the pair is why the same solver code
// drives both. Parameters are baked into the WGSL, so changing q rebuilds the
// solver - which it already did, because the CPU closures captured it too.
const nonlinearity = {
  cubic: {
    gain: () => ({ js: (I) => I, wgsl: 'x' }),
    pot: () => ({ js: (I) => (I * I) / 2, wgsl: '0.5 * x * x' }),
    profileN: () => (phi, out) => { for (let p = 0; p < phi.length; p++) out[p] = phi[p] ** 3; },
  },
  cq: {
    gain: (p) => ({ js: (I) => I - p.q * I * I, wgsl: `x - ${L(p.q)} * x * x` }),
    pot: (p) => ({
      js: (I) => (I * I) / 2 - (p.q * I ** 3) / 3,
      wgsl: `0.5 * x * x - ${L(p.q)} * x * x * x / 3.0`,
    }),
    profileN: (p) => (phi, out) => {
      for (let i = 0; i < phi.length; i++) {
        const a = phi[i], a2 = a * a;
        out[i] = a2 * a - p.q * a2 * a2 * a;
      }
    },
  },
  sat: {
    gain: (p) => ({ js: (I) => I / (1 + p.q * I), wgsl: `x / (1.0 + ${L(p.q)} * x)` }),
    pot: (p) => ({
      js: (I) => (I - Math.log(1 + p.q * I) / p.q) / p.q,
      wgsl: `(x - log(1.0 + ${L(p.q)} * x) / ${L(p.q)}) / ${L(p.q)}`,
    }),
    profileN: (p) => (phi, out) => {
      for (let i = 0; i < phi.length; i++) {
        const a = phi[i], a2 = a * a;
        out[i] = (a2 * a) / (1 + p.q * a2);
      }
    },
  },
};

// Solved on at most PROFILE_N points and interpolated up; see petviashvili.js.
function nlsProfile(grid, variant, mu, q) {
  const key = `${variant}:${grid.nx}:${grid.Lx}:${mu.toFixed(4)}:${q.toFixed(4)}`;
  if (cache.has(key)) return cache.get(key);

  const n = Math.min(grid.nx, PROFILE_N);
  const g = n === grid.nx ? grid : new Grid(n, n, grid.Lx, grid.Ly);

  const Lop = new Float64Array(g.n);
  for (let p = 0; p < g.n; p++) Lop[p] = g.k2[p] / 2 + mu;

  // Townes-like seed: peak ~1.5*sqrt(2mu), width ~1.5/sqrt(2mu)
  const s = Math.sqrt(2 * mu);
  const seed = new Float64Array(g.n);
  const cx = g.Lx / 2, cy = g.Ly / 2;
  for (let j = 0; j < g.ny; j++) for (let i = 0; i < g.nx; i++) {
    const dx = g.wrapX(g.x(i) - cx), dy = g.wrapY(g.y(j) - cy);
    seed[j * g.nx + i] = 1.5 * s * Math.exp((-(dx * dx + dy * dy) * s * s) / 4.5);
  }

  const res = solveProfile(g, {
    Lop, N: nonlinearity[variant].profileN({ q }), gamma: 1.5, init: seed, iters: 800,
  });
  const out = res.phi ? { ...res, phi: upsamplePeriodic(res.phi, n, grid.nx) } : res;
  return cachePut(cache, key, out);
}

function makeModel(variant, id, name, blurb, extraParam, opts = {}) {
  return {
    id, name, blurb, family: 'nls', variant,
    field: { kind: 'intensity', label: '|ψ|²' },
    grid: { n: 128, L: 50 },
    dt: 0.015,
    collapseGuard: opts.collapseGuard ?? 0,
    params: [
      { key: 'mu', label: 'μ (обратная ширина²)', min: 0.15, max: 2.5, step: 0.05, value: opts.mu ?? 0.8,
        hint: 'больше μ — уже и ярче солитон' },
      ...(extraParam ? [extraParam] : []),
      { key: 'speed', label: 'скорость', min: 0.2, max: 4, step: 0.1, value: 1.5,
        hint: 'НУШ галилеевски инвариантно — направление задаёт протяжка мышью' },
      { key: 'boost', label: 'амплитуда ×', min: 0.8, max: 1.3, step: 0.01, value: opts.boost ?? 1,
        hint: 'отклонение от точного профиля' },
    ],
    spawnKinds: [{ id: 'bullet', label: 'солитон' }],
    presets: opts.presets ?? [],
    create(backend, grid, dt, params) {
      const nl = nonlinearity[variant];
      const solver = new NlsSolver(backend, grid, {
        dt, gain: nl.gain(params), hamiltonianDensity: nl.pot(params),
      });
      // The boosted profile is built on the host and handed to the backend as
      // one complex bump; nothing here touches psi's storage directly.
      const bre = new Float64Array(grid.n), bim = new Float64Array(grid.n);
      const put = (x, y, vx, vy, mu, amp) => {
        const { phi, ok } = nlsProfile(grid, variant, mu, params.q ?? 0);
        if (!phi) return false;
        bre.fill(0); bim.fill(0);
        addCenteredBoosted(grid, bre, bim, phi, x, y, vx, vy, amp);
        solver.addComplex(bre, bim);
        return ok;
      };
      return {
        solver,
        step: () => solver.step(),
        clear: () => solver.clear(),
        get time() { return solver.time; },
        field: () => solver.field(),
        target: () => solver.target(),
        peak: () => solver.peak(),
        spawn(x, y, drag) {
          const len = Math.hypot(drag.x, drag.y);
          const sc = len > 1e-6 ? params.speed / Math.max(len, 1) : 0;
          return put(x, y, drag.x * sc, drag.y * sc, params.mu, params.boost);
        },
        applyPreset(p) {
          solver.clear();
          let ok = true;
          for (const s of p.items) {
            const r = put(s.x * grid.Lx, s.y * grid.Ly, s.vx, s.vy,
              s.mu ?? params.mu, s.amp ?? params.boost);
            if (r === false) ok = false;
          }
          // A cubic-quintic soliton simply does not exist for every (mu, q);
          // say so instead of leaving an empty box and no explanation.
          return ok;
        },
        diagnostics() {
          const d = solver.diagnostics();
          return { 'мощность ∫|ψ|²': d.power, 'гамильтониан': d.energy, 'макс |ψ|²': d.peak };
        },
        peakIntensity() { return solver.diagnostics().peak; },
      };
    },
  };
}

export const nlsCubic = makeModel(
  'cubic', 'nlsCubic', 'НУШ кубическое — критический случай',
  'Классическая ловушка 2D. Таунсендовский профиль существует, но неустойчив: '
  + 'амплитуда ×1.05 → коллапс за конечное время, ×0.95 → расплывание. '
  + 'Устойчивого 2D-солитона здесь нет.',
  null,
  {
    mu: 0.8, boost: 1.05, collapseGuard: 300,
    presets: [
      { id: 'collapse', label: 'коллапс (×1.05)', items: [{ x: 0.5, y: 0.5, vx: 0, vy: 0, amp: 1.05 }] },
      { id: 'disperse', label: 'расплывание (×0.95)', items: [{ x: 0.5, y: 0.5, vx: 0, vy: 0, amp: 0.95 }] },
      { id: 'marginal', label: 'точный профиль (на грани)', items: [{ x: 0.5, y: 0.5, vx: 0, vy: 0, amp: 1 }] },
    ],
  },
);

export const nlsCQ = makeModel(
  'cq', 'nlsCQ', 'НУШ кубично-квинтичное — устойчивые пули',
  'Квинтичный член −q|ψ|⁴ψ останавливает коллапс, и 2D-солитоны становятся '
  + 'устойчивыми. Летают под любым углом, сталкиваются почти упруго; при малых '
  + 'углах могут слипаться — неинтегрируемость в действии.',
  { key: 'q', label: 'q (квинтичный член)', min: 0.05, max: 0.6, step: 0.01, value: 0.2 },
  {
    mu: 0.7,
    presets: [
      { id: 'head', label: 'лоб в лоб', items: [
        { x: 0.25, y: 0.5, vx: 1.6, vy: 0 }, { x: 0.75, y: 0.5, vx: -1.6, vy: 0 },
      ] },
      { id: 'oblique', label: 'под углом 90°', items: [
        { x: 0.25, y: 0.25, vx: 1.3, vy: 1.3 }, { x: 0.75, y: 0.25, vx: -1.3, vy: 1.3 },
      ] },
      { id: 'four', label: 'четыре в центр', items: [
        { x: 0.2, y: 0.5, vx: 1.4, vy: 0 }, { x: 0.8, y: 0.5, vx: -1.4, vy: 0 },
        { x: 0.5, y: 0.2, vx: 0, vy: 1.4 }, { x: 0.5, y: 0.8, vx: 0, vy: -1.4 },
      ] },
      { id: 'grazing', label: 'скользящее (слипание)', items: [
        { x: 0.3, y: 0.46, vx: 0.9, vy: 0 }, { x: 0.7, y: 0.54, vx: -0.9, vy: 0 },
      ] },
      // The reference state, and the answer to "why does everything ripple":
      // it does not. Every other preset here is a collision, and a collision in
      // a non-integrable model *must* shed radiation, which then wraps the
      // periodic box and fills the frame. This one just stands there - peak
      // 3.8717 -> 3.8716 and background/peak 5e-13 out to t = 200.
      { id: 'single', label: 'один солитон (эталон)', items: [{ x: 0.5, y: 0.5, vx: 0, vy: 0 }] },
    ],
  },
);

export const nlsSat = makeModel(
  'sat', 'nlsSat', 'НУШ насыщающееся',
  'Насыщающаяся нелинейность |ψ|²/(1+s|ψ|²) — фоторефрактивные кристаллы. '
  + 'Тоже даёт устойчивые 2D-солитоны, но более «мягкие»: сильнее взаимодействуют '
  + 'хвостами и заметно притягиваются друг к другу.',
  { key: 'q', label: 's (насыщение)', min: 0.05, max: 1.5, step: 0.05, value: 0.5 },
  {
    mu: 0.5,
    presets: [
      { id: 'head', label: 'лоб в лоб', items: [
        { x: 0.25, y: 0.5, vx: 1.2, vy: 0 }, { x: 0.75, y: 0.5, vx: -1.2, vy: 0 },
      ] },
      { id: 'orbit', label: 'притяжение', items: [
        { x: 0.42, y: 0.5, vx: 0, vy: 0.35 }, { x: 0.58, y: 0.5, vx: 0, vy: -0.35 },
      ] },
      // Same reference state as in the cubic-quintic model: peak 3.6432 ->
      // 3.6431 over t = 200, background/peak 9e-13. Nothing ripples on its own.
      { id: 'single', label: 'один солитон (эталон)', items: [{ x: 0.5, y: 0.5, vx: 0, vy: 0 }] },
    ],
  },
);
