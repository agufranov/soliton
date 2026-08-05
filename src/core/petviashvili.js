// Petviashvili iteration: find the solitary-wave profile phi of
//
//     Lop(k) * phi_hat = FFT( N(phi) )
//
// with Lop real and positive. Plain fixed-point iteration on this diverges;
// the Petviashvili factor M^gamma rescales each iterate to kill exactly that
// instability. gamma = p/(p-1) for a pure power nonlinearity of degree p
// (2 for quadratic / KdV-type, 1.5 for cubic).
//
// Spawning a *converged* profile instead of a Gaussian guess is what makes
// collisions clean: an inexact bump sheds radiation for the rest of the run.

import { fft2 } from './fft.js';

export function solveProfile(grid, { Lop, N, gamma, init, iters = 400, tol = 1e-10 }) {
  const n = grid.n;
  const phi = Float64Array.from(init);
  const re = new Float64Array(n), im = new Float64Array(n);
  const nl = new Float64Array(n);
  const hr = new Float64Array(n), hi = new Float64Array(n);
  let residual = Infinity;

  for (let it = 0; it < iters; it++) {
    // phi_hat
    re.set(phi); im.fill(0);
    fft2(re, im, grid.nx, grid.ny, false);
    // N(phi)_hat
    N(phi, nl);
    hr.set(nl); hi.fill(0);
    fft2(hr, hi, grid.nx, grid.ny, false);

    // M = <phi_hat, Lop phi_hat> / <phi_hat, N_hat>
    let num = 0, den = 0;
    for (let p = 0; p < n; p++) {
      const mag = re[p] * re[p] + im[p] * im[p];
      num += Lop[p] * mag;
      den += re[p] * hr[p] + im[p] * hi[p];
    }
    if (!(Math.abs(den) > 1e-300)) break;
    const M = num / den;
    const f = Math.pow(M, gamma);

    // residual before updating: || Lop*phi_hat - N_hat || / ||Lop*phi_hat||
    let rn = 0, rd = 0;
    for (let p = 0; p < n; p++) {
      const ar = Lop[p] * re[p] - hr[p], ai = Lop[p] * im[p] - hi[p];
      rn += ar * ar + ai * ai;
      rd += (Lop[p] * re[p]) ** 2 + (Lop[p] * im[p]) ** 2;
    }
    residual = Math.sqrt(rn / Math.max(rd, 1e-300));

    for (let p = 0; p < n; p++) { hr[p] = (f * hr[p]) / Lop[p]; hi[p] = (f * hi[p]) / Lop[p]; }
    fft2(hr, hi, grid.nx, grid.ny, true);
    let delta = 0;
    for (let p = 0; p < n; p++) { delta += (hr[p] - phi[p]) ** 2; phi[p] = hr[p]; }
    if (residual < tol) break;
    if (!Number.isFinite(delta)) return { phi: null, residual: Infinity, ok: false };
  }
  return { phi, residual, ok: residual < 1e-6 };
}

// The iteration costs two FFTs per sweep on the host, in f64, whatever the
// compute backend is. At 1024^2 that is a minute of frozen UI for a profile
// that is already resolved to 1e-10 on a far coarser mesh, so profiles are
// solved on at most PROFILE_N points and spectrally interpolated up.
export const PROFILE_N = 256;

// Profile caches hold one full-grid array per entry - 8 MB each at 1024^2 -
// so they are bounded. Insertion-ordered Map, oldest out first.
export function cachePut(cache, key, value, limit = 12) {
  cache.set(key, value);
  while (cache.size > limit) cache.delete(cache.keys().next().value);
  return value;
}

// Band-limited resampling of a periodic field: forward transform, zero-pad the
// spectrum, transform back. Exact for anything the coarse grid resolves, which
// a converged soliton profile is by construction (its spectrum is down at
// 1e-12 by the 2/3 cutoff). Bilinear interpolation would clip the peak instead.
export function upsamplePeriodic(src, nFrom, nTo) {
  if (nFrom === nTo) return Float64Array.from(src);
  const re = Float64Array.from(src), im = new Float64Array(nFrom * nFrom);
  fft2(re, im, nFrom, nFrom, false);

  const bre = new Float64Array(nTo * nTo), bim = new Float64Array(nTo * nTo);
  const put = (i) => {
    const m = i <= nFrom / 2 ? i : i - nFrom;   // same FFT ordering as Grid
    return m >= 0 ? m : m + nTo;
  };
  for (let j = 0; j < nFrom; j++) {
    const tj = put(j);
    for (let i = 0; i < nFrom; i++) {
      const s = j * nFrom + i, d = tj * nTo + put(i);
      bre[d] = re[s]; bim[d] = im[s];
    }
  }
  fft2(bre, bim, nTo, nTo, true);
  // The inverse divides by nTo^2 while the forward summed only nFrom^2 points.
  const g = (nTo / nFrom) ** 2;
  for (let p = 0; p < bre.length; p++) bre[p] *= g;
  return bre;
}

// Convenience: a centred Gaussian seed on the grid, centre at the box middle.
export function gaussianSeed(grid, amp, width) {
  const g = new Float64Array(grid.n);
  const cx = grid.Lx / 2, cy = grid.Ly / 2;
  for (let j = 0; j < grid.ny; j++) for (let i = 0; i < grid.nx; i++) {
    const dx = grid.wrapX(grid.x(i) - cx), dy = grid.wrapY(grid.y(j) - cy);
    g[j * grid.nx + i] = amp * Math.exp(-(dx * dx + dy * dy) / (width * width));
  }
  return g;
}
