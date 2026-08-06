// What does a stretched box cost the KP-I lump?  node tools/aniso.mjs
//
// On a phone the field fills the screen by stretching the *box*, not the grid:
// nx = ny points over Lx x (Lx * aspect), so dy = aspect * dx. The grid stays
// square because the GPU FFT transposes in place and refuses nx != ny. The
// question this answers is whether the coarser y sampling costs anything real.
//
// The same analytic lump is propagated to t=4 in boxes of increasing aspect
// and compared against the analytic solution translated to the same time -
// the identical yardstick tools/verify.mjs uses for the square box.
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
const peak = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
function l2diff(a, b) {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; n += b[i] * b[i]; }
  return Math.sqrt(s / n);
}

function run(label, n, Lx, Ly, mu = 0.75, T = 4, dt = 0.012) {
  const grid = new Grid(n, n, Lx, Ly);
  const solver = new SpectralRealSolver(new CpuBackend(grid), grid, {
    Limag: kpLinear(grid, -1), alpha: 3, dt,
  });
  const x0 = Lx * 0.35, y0 = Ly * 0.5;
  solver.setPhysical(lump(grid, mu, 0, x0, y0, 0));
  const d0 = solver.diagnostics(), p0 = peak(solver.physical());
  for (let s = 0; s < Math.round(T / dt); s++) solver.step();
  const d1 = solver.diagnostics(), p1 = peak(solver.physical());
  const exact = lump(grid, mu, 0, x0, y0, T);
  const drift = Object.keys(d0)
    .filter((k) => Number.isFinite(d0[k]) && d0[k] !== 0)
    .map((k) => `${k} ${(((d1[k] - d0[k]) / d0[k]) * 100).toFixed(4)}%`).join('  ');
  console.log(`${label.padEnd(22)} dy/dx=${(grid.dy / grid.dx).toFixed(2)}  `
    + `пик ${p0.toFixed(4)} → ${p1.toFixed(4)} (точно ${peak(exact).toFixed(4)})  `
    + `отн.L2 ${l2diff(solver.physical(), exact).toExponential(3)}\n`
    + `                       дрейф: ${drift}`);
}

console.log('KP-I, 128², L=45, mu=0.75, dt=0.012, t=4\n');
run('квадрат 45×45', 128, 45, 45);
run('планшет 45×51.7', 128, 45, 51.7);
run('android 45×76.1', 128, 45, 76.1);
run('iPhone Pro 45×82.9', 128, 45, 82.9);
run('вдвое 45×90', 128, 45, 90);
console.log('');
run('квадрат 256²', 256, 45, 45, 0.75, 4, 0.006);
run('iPhone Pro 256²', 256, 45, 82.9, 0.75, 4, 0.006);
