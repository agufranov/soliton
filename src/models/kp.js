// Kadomtsev-Petviashvili, both branches.
//
//     ( u_t + 6 u u_x + u_xxx )_x  +  3*sigma*u_yy = 0
//
//   sigma = -1  ->  KP-I   (positive / strong-surface-tension dispersion).
//                   Integrable, and the branch that owns *lumps*: rational,
//                   fully 2D localised humps that translate rigidly and pass
//                   through each other elastically.
//   sigma = +1  ->  KP-II  (ordinary shallow water). Also integrable, but it
//                   has NO localised lumps - only line solitons and their
//                   X-shaped resonant interactions. Dropping a hump here just
//                   radiates it away, which is exactly why real ocean solitons
//                   look like long crests rather than travelling bumps.
//
// Sign convention verified numerically in tools/residual.mjs: the classical
// lump solves the sigma = -1 form, and propagates at vx = 3*mu^2.

import { SpectralRealSolver } from '../core/spectralReal.js';
import { addCentered } from '../core/util.js';

function kpLinear(grid, sigma) {
  const L = new Float64Array(grid.n);
  for (let p = 0; p < grid.n; p++) {
    const kx = grid.kx[p], ky = grid.ky[p];
    L[p] = kx ** 3 - (Math.abs(kx) < 1e-12 ? 0 : (3 * sigma * ky * ky) / kx);
  }
  return L;
}

// Manakov-Zakharov-Bordag-Its lump at t = 0, centred on the box middle.
// amplitude 4*mu^2, width ~1/mu, velocity (3*(lambda^2+mu^2), -6*lambda).
function lumpProfile(grid, mu, lambda) {
  const u = new Float64Array(grid.n);
  const inv = 1 / (mu * mu);
  const cx = grid.Lx / 2, cy = grid.Ly / 2;
  for (let j = 0; j < grid.ny; j++) for (let i = 0; i < grid.nx; i++) {
    const dx = grid.wrapX(grid.x(i) - cx), dy = grid.wrapY(grid.y(j) - cy);
    const X = dx + lambda * dy, Y = dy;
    const A = X * X + mu * mu * Y * Y + inv;
    u[j * grid.nx + i] = (4 * (-X * X + mu * mu * Y * Y + inv)) / (A * A);
  }
  return u;
}

// Line soliton 2*k^2 sech^2(k*(x + tan*y)). To stay periodic across the box the
// transverse slope has to be commensurate, so it is quantised here.
function lineProfile(grid, kappa, slopeIndex) {
  const u = new Float64Array(grid.n);
  const tan = (slopeIndex * grid.Lx) / grid.Ly;
  const cx = grid.Lx / 2, cy = grid.Ly / 2;
  for (let j = 0; j < grid.ny; j++) for (let i = 0; i < grid.nx; i++) {
    const dy = grid.y(j) - cy;
    let X = grid.x(i) - cx + tan * dy;
    X = grid.wrapX(X);
    const ch = Math.cosh(kappa * X);
    u[j * grid.nx + i] = (2 * kappa * kappa) / (ch * ch);
  }
  return u;
}

function makeModel(sigma, id, name, blurb, spawnKinds, presets) {
  return {
    id, name, blurb, family: 'kp',
    field: { kind: 'height', label: 'u(x,y)' },
    grid: { n: 128, L: 45 },
    dt: 0.012,           // tuned at n=128; scaled as 128/n for finer grids
    params: [
      { key: 'mu', label: 'μ (масштаб лампа)', min: 0.3, max: 1.3, step: 0.05, value: 0.75,
        hint: 'амплитуда 4μ², скорость 3μ² — связаны, как у KdV' },
      { key: 'slope', label: 'наклон λ', min: -1.2, max: 1.2, step: 0.05, value: 0,
        hint: 'перекос лампа; поперечная скорость −6λ' },
    ],
    spawnKinds,
    presets,
    create(backend, grid, dt, params) {
      const solver = new SpectralRealSolver(backend, grid, {
        Limag: kpLinear(grid, sigma), alpha: 3, dt,
      });
      return {
        solver,
        step: () => solver.step(),
        clear: () => solver.clear(),
        get time() { return solver.time; },
        field: () => solver.physical(),
        target: () => solver.target(),
        peak: () => solver.peak(),
        spawn(x, y, drag, kind) {
          // Drag sets the transverse slope; the forward speed is *not* free —
          // KP-I fixes it at 3(lambda^2 + mu^2) once mu is chosen.
          const lambda = kind === 'line' ? 0 : (params.slope || 0) - drag.y * 0.9;
          const prof = kind === 'line'
            ? lineProfile(grid, Math.max(0.25, params.mu), Math.round(drag.y * 2))
            : lumpProfile(grid, params.mu, lambda);
          const du = new Float64Array(grid.n);
          addCentered(grid, du, prof, x, y);
          solver.addPhysical(du);
        },
        applyPreset(p) {
          solver.clear();
          const du = new Float64Array(grid.n);
          for (const s of p.items) {
            const prof = s.kind === 'line'
              ? lineProfile(grid, s.kappa, s.slopeIndex)
              : lumpProfile(grid, s.mu, s.lambda);
            // A line soliton spans the box, so its preset entries carry no
            // position; default them to the centre rather than producing NaN
            // indices and silently writing nothing.
            addCentered(grid, du, prof, (s.x ?? 0.5) * grid.Lx, (s.y ?? 0.5) * grid.Ly);
          }
          solver.setPhysical(du);
        },
        diagnostics() {
          const d = solver.diagnostics();
          // The L2 norm is an exact KP invariant - a live integrity check on
          // the integrator. Total mass is not useful here: the lump's double
          // integral is only conditionally convergent.
          return { '‖u‖²': d.energy, 'макс |u|': d.peak };
        },
      };
    },
  };
}

export const kpI = makeModel(
  -1, 'kpI', 'KP-I — лампы',
  'Интегрируемо. Рациональные lump-солитоны: локализованные горбы, спад ~1/r², '
  + 'жёсткая связь амплитуда 4μ² ↔ скорость 3μ². Столкновения упругие.',
  [{ id: 'lump', label: 'лампа' }],
  [
    { id: 'pair', label: 'два лампа под углом', items: [
      { kind: 'lump', mu: 0.8, lambda: 0.5, x: 0.25, y: 0.3 },
      { kind: 'lump', mu: 0.8, lambda: -0.5, x: 0.25, y: 0.7 },
    ] },
    // Offset in y on purpose: see the anomalous preset below for why.
    { id: 'chase', label: 'догоняющее столкновение (упругое)', items: [
      { kind: 'lump', mu: 1.0, lambda: 0, x: 0.15, y: 0.42 },
      { kind: 'lump', mu: 0.55, lambda: 0, x: 0.45, y: 0.58 },
    ] },
    // Exactly collinear lumps are a degenerate configuration: instead of
    // passing through, they merge and re-emerge as a symmetric pair moving
    // apart *transversally*. This is the documented anomalous 90-degree
    // scattering of KP-I lumps, not a numerical breakdown - a single lump on
    // the same grid holds its peak to within a few percent over the same time.
    { id: 'anomalous', label: 'аномальное рассеяние на 90°', items: [
      { kind: 'lump', mu: 1.0, lambda: 0, x: 0.15, y: 0.5 },
      { kind: 'lump', mu: 0.55, lambda: 0, x: 0.45, y: 0.5 },
    ] },
    { id: 'three', label: 'три лампа', items: [
      { kind: 'lump', mu: 0.9, lambda: 0.6, x: 0.2, y: 0.25 },
      { kind: 'lump', mu: 0.7, lambda: 0, x: 0.3, y: 0.5 },
      { kind: 'lump', mu: 0.9, lambda: -0.6, x: 0.2, y: 0.75 },
    ] },
  ],
);

export const kpII = makeModel(
  1, 'kpII', 'KP-II — линейные солитоны',
  'Обычная мелкая вода. Тоже интегрируемо, но лампов НЕТ: локальный горб '
  + 'разваливается в излучение. Живут только линейные солитоны и их X-резонансы.',
  [{ id: 'lump', label: 'горб (развалится)' }, { id: 'line', label: 'линейный солитон' }],
  [
    { id: 'x', label: 'X-взаимодействие', items: [
      { kind: 'line', kappa: 0.55, slopeIndex: 1 },
      { kind: 'line', kappa: 0.55, slopeIndex: -1 },
    ] },
    { id: 'decay', label: 'горб разваливается', items: [
      { kind: 'lump', mu: 0.8, lambda: 0, x: 0.3, y: 0.5 },
    ] },
  ],
);
