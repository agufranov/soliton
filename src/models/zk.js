// Zakharov-Kuznetsov, the 2D generalisation of KdV that shows up for
// ion-acoustic waves in a magnetised plasma:
//
//     u_t + 6 u u_x + (u_xx + u_yy)_x = 0
//
// Not integrable, but it does have genuinely 2D, exponentially localised,
// *orbitally stable* solitary waves. Collisions are near-elastic: the humps
// survive and re-emerge, shedding a small dispersive wake.
//
// Unlike KP, propagation is not isotropic - the equation only differentiates
// the dispersive and nonlinear terms along x, so solitary waves travel along
// +x only. Interactions are overtaking collisions, not oblique ones.

import { Grid } from '../core/grid.js';
import { SpectralRealSolver } from '../core/spectralReal.js';
import { solveProfile, upsamplePeriodic, cachePut, PROFILE_N } from '../core/petviashvili.js';
import { addCentered } from '../core/util.js';

const cache = new Map();

// Solitary wave of speed c: solves  laplacian(phi) - c*phi + 3*phi^2 = 0.
// Solved on at most PROFILE_N points and interpolated up; see petviashvili.js.
function zkProfile(grid, c) {
  const key = `${grid.nx}:${grid.Lx}:${c.toFixed(4)}`;
  if (cache.has(key)) return cache.get(key);

  const n = Math.min(grid.nx, PROFILE_N);
  const g = n === grid.nx ? grid : new Grid(n, n, grid.Lx, grid.Ly);

  const Lop = new Float64Array(g.n);
  for (let p = 0; p < g.n; p++) Lop[p] = g.k2[p] + c;

  const seed = new Float64Array(g.n);
  const cx = g.Lx / 2, cy = g.Ly / 2;
  const w = 2 / Math.sqrt(c);
  for (let j = 0; j < g.ny; j++) for (let i = 0; i < g.nx; i++) {
    const dx = g.wrapX(g.x(i) - cx), dy = g.wrapY(g.y(j) - cy);
    const ch = Math.cosh(Math.hypot(dx, dy) / w);
    seed[j * g.nx + i] = (0.9 * c) / (ch * ch);
  }

  const res = solveProfile(g, {
    Lop,
    N: (phi, out) => { for (let p = 0; p < phi.length; p++) out[p] = 3 * phi[p] * phi[p]; },
    gamma: 2,          // quadratic nonlinearity: p/(p-1) = 2
    init: seed,
  });
  const out = res.phi
    ? { ...res, phi: upsamplePeriodic(res.phi, n, grid.nx) }
    : res;
  return cachePut(cache, key, out);
}

export const zk = {
  id: 'zk', name: 'Захаров–Кузнецов', family: 'zk',
  blurb: 'Плазменный 2D-аналог KdV. Неинтегрируемо, но солитоны устойчивы и '
    + 'сталкиваются почти упруго — с небольшим дисперсионным следом. '
    + 'Движение только вдоль +x: уравнение анизотропно.',
  field: { kind: 'height', label: 'u(x,y)' },
  grid: { n: 128, L: 40 },
  dt: 0.01,
  params: [
    { key: 'c', label: 'скорость c', min: 0.4, max: 4, step: 0.1, value: 1.6,
      hint: 'амплитуда растёт вместе с c, ширина ~1/√c' },
  ],
  spawnKinds: [{ id: 'soliton', label: 'солитон' }],
  presets: [
    { id: 'chase', label: 'догоняющее столкновение', items: [
      { c: 3.0, x: 0.15, y: 0.5 }, { c: 1.2, x: 0.5, y: 0.5 },
    ] },
    { id: 'offset', label: 'скользящее столкновение', items: [
      { c: 3.0, x: 0.15, y: 0.42 }, { c: 1.2, x: 0.5, y: 0.58 },
    ] },
    { id: 'row', label: 'ряд из четырёх', items: [
      { c: 3.2, x: 0.1, y: 0.5 }, { c: 2.4, x: 0.32, y: 0.5 },
      { c: 1.7, x: 0.54, y: 0.5 }, { c: 1.1, x: 0.76, y: 0.5 },
    ] },
  ],
  create(backend, grid, dt, params) {
    const L = new Float64Array(grid.n);
    for (let p = 0; p < grid.n; p++) L[p] = grid.kx[p] * grid.k2[p];
    const solver = new SpectralRealSolver(backend, grid, { Limag: L, alpha: 3, dt });
    return {
      solver,
      step: () => solver.step(),
      clear: () => solver.clear(),
      get time() { return solver.time; },
      field: () => solver.physical(),
      target: () => solver.target(),
      peak: () => solver.peak(),
      spawn(x, y, drag) {
        const speed = Math.max(0.4, params.c + drag.x * 1.5);
        const { phi } = zkProfile(grid, speed);
        if (!phi) return false;
        const du = new Float64Array(grid.n);
        addCentered(grid, du, phi, x, y);
        solver.addPhysical(du);
        return true;
      },
      applyPreset(p) {
        const du = new Float64Array(grid.n);
        let ok = true;
        for (const s of p.items) {
          const { phi } = zkProfile(grid, s.c);
          if (phi) addCentered(grid, du, phi, s.x * grid.Lx, s.y * grid.Ly);
          else ok = false;
        }
        solver.setPhysical(du);
        return ok;
      },
      diagnostics() {
        const d = solver.diagnostics();
        return { 'масса ∫u': d.mass, '‖u‖²': d.energy, 'макс |u|': d.peak };
      },
    };
  },
};
