// Numerical sanity checks. Run with:  node tools/verify.mjs
//
// 1. KP-I: evolve the analytic lump and compare against the analytic solution
//    translated to the same time. This checks L(k), the lump formula and the
//    velocity law 3*mu^2 all at once.
// 2. The same lump under KP-II, which must NOT survive - KP-II has no lumps.
// 3. Conservation of the L2 invariant under KP-I and ZK.
//
// sigma = -1 is KP-I. That is the branch the classical lump solves; see
// tools/residual.mjs, which measures the PDE residual for both signs.

import { Grid } from '../src/core/grid.js';
import { CpuBackend } from '../src/core/backend.js';
import { SpectralRealSolver } from '../src/core/spectralReal.js';

function kpLinear(grid, sigma) {
  const L = new Float64Array(grid.n);
  for (let p = 0; p < grid.n; p++) {
    const kx = grid.kx[p], ky = grid.ky[p];
    L[p] = kx * kx * kx - (Math.abs(kx) < 1e-12 ? 0 : (3 * sigma * ky * ky) / kx);
  }
  return L;
}

// Classical KP-I lump (Manakov-Zakharov-Bordag-Its) for
//   (u_t + 6 u u_x + u_xxx)_x - 3 u_yy = 0
function lump(grid, mu, lambda, x0, y0, t) {
  const u = new Float64Array(grid.n);
  const inv = 1 / (mu * mu);
  for (let j = 0; j < grid.ny; j++) {
    for (let i = 0; i < grid.nx; i++) {
      const dx = grid.wrapX(grid.x(i) - x0);
      const dy = grid.wrapY(grid.y(j) - y0);
      const X = dx + lambda * dy + 3 * (lambda * lambda - mu * mu) * t;
      const Y = dy + 6 * lambda * t;
      const A = X * X + mu * mu * Y * Y + inv;
      u[j * grid.nx + i] = (4 * (-X * X + mu * mu * Y * Y + inv)) / (A * A);
    }
  }
  return u;
}

function l2diff(a, b, cell) {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; n += b[i] * b[i]; }
  return Math.sqrt(s / n);
}

function runKp(sigma, label, mu = 0.8, T = 4, dt = 0.002) {
  const grid = new Grid(256, 256, 80, 80);
  const backend = new CpuBackend(grid);
  const solver = new SpectralRealSolver(backend, grid, {
    Limag: kpLinear(grid, sigma), alpha: 3, dt,
  });
  const x0 = 30, y0 = 40;
  solver.setPhysical(lump(grid, mu, 0, x0, y0, 0));
  const d0 = solver.diagnostics();
  const steps = Math.round(T / dt);
  for (let s = 0; s < steps; s++) solver.step();
  const d1 = solver.diagnostics();
  const got = Float64Array.from(solver.physical());
  const want = lump(grid, mu, 0, x0, y0, solver.time);
  console.log(`\n[${label}] sigma=${sigma} mu=${mu} t=${solver.time.toFixed(2)}`);
  console.log(`  relative L2 error vs analytic lump : ${l2diff(got, want, 1).toExponential(3)}`);
  console.log(`  peak  ${d0.peak.toFixed(5)} -> ${d1.peak.toFixed(5)}  (analytic ${(4 * mu * mu).toFixed(5)})`);
  console.log(`  L2^2  ${d0.energy.toFixed(6)} -> ${d1.energy.toFixed(6)}  (drift ${((d1.energy / d0.energy - 1) * 100).toFixed(4)}%)`);
  return l2diff(got, want, 1);
}

// Where did the peak actually end up? Independent check of the speed law.
function measureSpeed(mu = 0.8, T = 4, dt = 0.002) {
  const grid = new Grid(256, 256, 80, 80);
  const backend = new CpuBackend(grid);
  const solver = new SpectralRealSolver(backend, grid, {
    Limag: kpLinear(grid, -1), alpha: 3, dt,
  });
  solver.setPhysical(lump(grid, mu, 0, 30, 40, 0));
  const steps = Math.round(T / dt);
  for (let s = 0; s < steps; s++) solver.step();
  const u = solver.physical();
  let best = -Infinity, bi = 0;
  for (let i = 0; i < u.length; i++) if (u[i] > best) { best = u[i]; bi = i; }
  const px = (bi % grid.nx) * grid.dx, py = Math.floor(bi / grid.nx) * grid.dy;
  console.log(`\n[speed] peak moved (30,40) -> (${px.toFixed(2)},${py.toFixed(2)}) in t=${T}`);
  console.log(`  measured vx = ${((px - 30) / T).toFixed(4)}   predicted 3*mu^2 = ${(3 * mu * mu).toFixed(4)}`);
}

function runZk(T = 2, dt = 0.0015) {
  const grid = new Grid(128, 128, 40, 40);
  const backend = new CpuBackend(grid);
  const L = new Float64Array(grid.n);
  for (let p = 0; p < grid.n; p++) L[p] = grid.kx[p] * grid.k2[p];
  const solver = new SpectralRealSolver(backend, grid, { Limag: L, alpha: 3, dt });
  const u0 = new Float64Array(grid.n);
  for (let j = 0; j < grid.ny; j++) for (let i = 0; i < grid.nx; i++) {
    const dx = grid.wrapX(grid.x(i) - 20), dy = grid.wrapY(grid.y(j) - 20);
    u0[j * grid.nx + i] = 1.0 / Math.cosh(Math.hypot(dx, dy) / 1.6) ** 2;
  }
  solver.setPhysical(u0);
  const d0 = solver.diagnostics();
  for (let s = 0; s < Math.round(T / dt); s++) solver.step();
  const d1 = solver.diagnostics();
  console.log(`\n[ZK] t=${solver.time.toFixed(2)}`);
  console.log(`  mass  ${d0.mass.toFixed(6)} -> ${d1.mass.toFixed(6)}`);
  console.log(`  L2^2  ${d0.energy.toFixed(6)} -> ${d1.energy.toFixed(6)}  (drift ${((d1.energy / d0.energy - 1) * 100).toFixed(4)}%)`);
  console.log(`  peak  ${d0.peak.toFixed(4)} -> ${d1.peak.toFixed(4)}  (relaxes toward the true ZK profile)`);
}

const errI = runKp(-1, 'KP-I  (lump must survive)');
const errII = runKp(1, 'KP-II (lump must NOT survive)');
measureSpeed();
runZk();

console.log('\n--- verdict ---');
console.log(`  KP-I  lump error ${errI.toExponential(2)}  ${errI < 0.05 ? 'OK' : 'TOO LARGE'}`);
console.log(`  KP-II lump error ${errII.toExponential(2)}  ${errII > 0.5 ? 'OK (destroyed, as expected)' : 'SUSPICIOUS'}`);
