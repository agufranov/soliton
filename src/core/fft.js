// Radix-2 Cooley-Tukey FFT on flat typed arrays.
//
// Fields are stored structure-of-arrays: a separate `re` and `im` buffer,
// row-major (index = row * nx + col). This is deliberately the layout a
// WebGPU storage buffer / RG32F texture wants, so the GPU backend can reuse
// the exact same field objects without a repack.
//
// Sizes must be powers of two.

const cache = new Map();

class Fft1d {
  constructor(n) {
    if ((n & (n - 1)) !== 0) throw new Error(`FFT size ${n} is not a power of two`);
    this.n = n;
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let t = 0; t < n / 2; t++) {
      this.cos[t] = Math.cos((2 * Math.PI * t) / n);
      this.sin[t] = Math.sin((2 * Math.PI * t) / n);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n) | 0;
    for (let i = 0; i < n; i++) {
      let x = i, r = 0;
      for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
      this.rev[i] = r;
    }
  }

  // In-place transform of the n samples starting at `offset` with `stride`.
  // sgn = -1 forward (e^{-i2pikn/N}), +1 inverse. No normalisation here.
  run(re, im, offset, stride, sgn) {
    const n = this.n, rev = this.rev, cosT = this.cos, sinT = this.sin;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const a = offset + i * stride, b = offset + j * stride;
        let t = re[a]; re[a] = re[b]; re[b] = t;
        t = im[a]; im[a] = im[b]; im[b] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;
      for (let base = 0; base < n; base += len) {
        for (let k = 0; k < half; k++) {
          const tw = k * step;
          const wr = cosT[tw], wi = sgn * sinT[tw];
          const ia = offset + (base + k) * stride;
          const ib = ia + half * stride;
          const xr = re[ib], xi = im[ib];
          const tr = xr * wr - xi * wi;
          const ti = xr * wi + xi * wr;
          re[ib] = re[ia] - tr; im[ib] = im[ia] - ti;
          re[ia] += tr;         im[ia] += ti;
        }
      }
    }
  }
}

function plan(n) {
  let p = cache.get(n);
  if (!p) { p = new Fft1d(n); cache.set(n, p); }
  return p;
}

// Column scratch. Transforming a column in place would stride by nx and miss
// the cache on every butterfly; gathering it into a contiguous buffer first is
// bit-for-bit identical and ~1.3x faster at 128^2, ~1.4x at 256^2.
let colRe = null, colIm = null;

// In-place 2D FFT. `inverse` also applies the 1/(nx*ny) normalisation.
export function fft2(re, im, nx, ny, inverse) {
  const sgn = inverse ? 1 : -1;
  const px = plan(nx), py = plan(ny);

  for (let row = 0; row < ny; row++) px.run(re, im, row * nx, 1, sgn);

  if (colRe === null || colRe.length < ny) {
    colRe = new Float64Array(ny); colIm = new Float64Array(ny);
  }
  for (let col = 0; col < nx; col++) {
    for (let j = 0; j < ny; j++) { const p = j * nx + col; colRe[j] = re[p]; colIm[j] = im[p]; }
    py.run(colRe, colIm, 0, 1, sgn);
    for (let j = 0; j < ny; j++) { const p = j * nx + col; re[p] = colRe[j]; im[p] = colIm[j]; }
  }

  if (inverse) {
    const s = 1 / (nx * ny);
    for (let i = 0; i < re.length; i++) { re[i] *= s; im[i] *= s; }
  }
}
