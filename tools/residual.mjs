// Which sign convention does the classical lump actually solve?
// Compute the residual of
//    d/dx (u_t + 6 u u_x + u_xxx) + 3*sigma*u_yy
// on the analytic lump, using spectral space derivatives and a centred
// finite difference in t. Small residual == correct sigma.

import { Grid } from '../src/core/grid.js';
import { fft2 } from '../src/core/fft.js';

const grid = new Grid(256, 256, 60, 60);
const N = grid.n;

function lump(mu, lambda, x0, y0, t) {
  const u = new Float64Array(N);
  const inv = 1 / (mu * mu);
  for (let j = 0; j < grid.ny; j++) for (let i = 0; i < grid.nx; i++) {
    const dx = grid.wrapX(grid.x(i) - x0), dy = grid.wrapY(grid.y(j) - y0);
    const X = dx + lambda * dy + 3 * (lambda * lambda - mu * mu) * t;
    const Y = dy + 6 * lambda * t;
    const A = X * X + mu * mu * Y * Y + inv;
    u[j * grid.nx + i] = (4 * (-X * X + mu * mu * Y * Y + inv)) / (A * A);
  }
  return u;
}

// spectral operator: applies f(kx,ky) (returned as [re,im] multiplier)
function applyOp(u, mult) {
  const re = Float64Array.from(u), im = new Float64Array(N);
  fft2(re, im, grid.nx, grid.ny, false);
  for (let p = 0; p < N; p++) {
    const [mr, mi] = mult(grid.kx[p], grid.ky[p]);
    const r = re[p], m = im[p];
    re[p] = r * mr - m * mi; im[p] = r * mi + m * mr;
  }
  fft2(re, im, grid.nx, grid.ny, true);
  return re;
}

function rms(a) { let s = 0; for (const v of a) s += v * v; return Math.sqrt(s / a.length); }

const mu = 1.0, lambda = 0.0, x0 = 30, y0 = 30;
const dt = 1e-4;
const u = lump(mu, lambda, x0, y0, 0);
const up = lump(mu, lambda, x0, y0, dt);
const um = lump(mu, lambda, x0, y0, -dt);
const ut = new Float64Array(N);
for (let i = 0; i < N; i++) ut[i] = (up[i] - um[i]) / (2 * dt);

const dxOf = (f) => applyOp(f, (kx) => [0, kx]);             // d/dx  -> i kx
const uxt = dxOf(ut);
const u2 = new Float64Array(N); for (let i = 0; i < N; i++) u2[i] = u[i] * u[i];
const term2 = applyOp(u2, (kx) => [-3 * kx * kx, 0]);        // 3 * d2/dx2 (u^2)
const term3 = applyOp(u, (kx) => [kx ** 4, 0]);              // u_xxxx
const uyy = applyOp(u, (kx, ky) => [-ky * ky, 0]);           // u_yy

console.log('scale of individual terms (rms):');
console.log('  d/dx u_t      ', rms(uxt).toExponential(3));
console.log('  3 d2/dx2 u^2  ', rms(term2).toExponential(3));
console.log('  u_xxxx        ', rms(term3).toExponential(3));
console.log('  3 u_yy        ', rms(uyy.map((v) => 3 * v)).toExponential(3));

for (const sigma of [1, -1]) {
  const R = new Float64Array(N);
  for (let i = 0; i < N; i++) R[i] = uxt[i] + term2[i] + term3[i] + 3 * sigma * uyy[i];
  console.log(`residual, sigma=${sigma}:`, rms(R).toExponential(3));
}
