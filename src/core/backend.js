// Compute backend.
//
// ---------------------------------------------------------------------------
// GPU PORTABILITY CONTRACT
// ---------------------------------------------------------------------------
// Nothing above this file is allowed to index into a field buffer. Every
// operation on grid data goes through one of the *named* kernels below.
// The set is deliberately finite and every kernel is either
//   (a) elementwise  -> one WGSL compute shader, one thread per cell, or
//   (b) an FFT       -> a Stockham radix-2 pass sequence, or
//   (c) a reduction  -> a two-stage workgroup reduction (diagnostics only).
//
// A WebGpuBackend therefore only has to implement the same method names with
// the same semantics; models and integrators need no changes. Fields are
// {re, im} pairs of flat arrays (structure of arrays) because that maps 1:1
// onto a pair of storage buffers, or onto a single rg32float texture.
//
// The GPU backend (core/webgpuBackend.js) implements exactly this surface.
// Three rules keep the two interchangeable:
//
//   1. Real tables handed to a kernel (grid.mask, grid.kx, phase tables) are
//      treated as IMMUTABLE: the GPU backend uploads each one once and caches
//      it by object identity. Rebuild the table into a new array if it changes.
//   2. Host arrays cross the boundary only through the explicit `adopt` /
//      `setHostReal` / `addHostReal` / `addHostComplex` / `snapshot` calls.
//      Nothing else may touch `field.re` / `field.im`.
//   3. Model-specific pointwise nonlinearities are passed as a {js, wgsl} pair
//      through `pointwise()`; the CPU picks `js`, the GPU compiles `wgsl`.
//
// `snapshot()` and the reductions are the two places where the two backends
// genuinely differ: on the GPU they are one frame stale, because a synchronous
// readback would stall the pipe. Both are used for display only.
//
// Precision: the CPU backend is f64, WebGPU is f32-only. KP/ZK are stiff
// enough that this shortens the usable step; see the dt table in CLAUDE.md.
// ---------------------------------------------------------------------------

import { fft2 } from './fft.js';

export class CpuBackend {
  constructor(grid) {
    this.grid = grid;
    this.n = grid.n;
    this.name = 'cpu-f64';
    this.gpu = false;
  }

  // -- allocation ----------------------------------------------------------
  alloc() { return { re: new Float64Array(this.n), im: new Float64Array(this.n) }; }
  allocReal() { return new Float64Array(this.n); }

  // Take ownership of a host-built {re, im} pair. On the GPU this uploads and
  // drops the host arrays; here it is already the right thing.
  adopt(pair) { return pair; }

  // Compile a pointwise scalar nonlinearity given as {js, wgsl}.
  pointwise(spec) { return spec.js; }

  // -- host <-> field transfers --------------------------------------------
  setHostReal(f, arr) { f.re.set(arr); f.im.fill(0); }

  addHostReal(f, arr) {
    for (let i = 0; i < arr.length; i++) f.re[i] += arr[i];
    f.im.fill(0);
  }

  addHostComplex(f, re, im) {
    for (let i = 0; i < re.length; i++) { f.re[i] += re[i]; f.im[i] += im[i]; }
  }

  // Host-visible copy of a field. On the GPU this is an async download and may
  // lag by a frame; here it is the live buffer.
  snapshot(f) { return f; }

  // -- transforms ----------------------------------------------------------
  forward(f) { fft2(f.re, f.im, this.grid.nx, this.grid.ny, false); }
  inverse(f) { fft2(f.re, f.im, this.grid.nx, this.grid.ny, true); }

  // Fused forms. They exist because on the GPU a pseudospectral step is purely
  // memory bound, and folding these into a transform pass that already touches
  // every element removes a whole sweep of the field each. Here they are just
  // the two operations back to back.
  inverseInto(dst, src) { this.copy(dst, src); this.inverse(dst); }

  forwardSqrReal(f) { this.sqrReal(f, f); this.forward(f); }

  // -- elementwise kernels -------------------------------------------------
  zero(a) { a.re.fill(0); a.im.fill(0); }

  copy(dst, src) { dst.re.set(src.re); dst.im.set(src.im); }

  scale(a, s) {
    const { re, im } = a;
    for (let i = 0; i < re.length; i++) { re[i] *= s; im[i] *= s; }
  }

  // dst = a + s * b   (s real)
  addScaled(dst, a, b, s) {
    for (let i = 0; i < dst.re.length; i++) {
      dst.re[i] = a.re[i] + s * b.re[i];
      dst.im[i] = a.im[i] + s * b.im[i];
    }
  }

  // y += s * x. Separate from addScaled because on the GPU a buffer cannot be
  // bound writable and readable in the same dispatch, so `addScaled(y,y,x,s)`
  // is not expressible there.
  axpy(y, x, s) {
    for (let i = 0; i < y.re.length; i++) {
      y.re[i] += s * x.re[i];
      y.im[i] += s * x.im[i];
    }
  }

  // a *= m, with m a real mask (used for 2/3 dealiasing)
  maskReal(a, m) {
    for (let i = 0; i < a.re.length; i++) { a.re[i] *= m[i]; a.im[i] *= m[i]; }
  }

  // dst = (Re src)^2, imaginary part cleared. The Re-only read is what keeps
  // round-off from seeding a spurious imaginary field in real-valued models.
  sqrReal(dst, src) {
    for (let i = 0; i < dst.re.length; i++) {
      const u = src.re[i];
      dst.re[i] = u * u; dst.im[i] = 0;
    }
  }

  // dst = s * (i * kx) * src   -- the derivative that KP and ZK both need
  mulIkxScaled(dst, src, kx, s) {
    for (let i = 0; i < dst.re.length; i++) {
      const c = s * kx[i];
      dst.re[i] = -c * src.im[i];
      dst.im[i] = c * src.re[i];
    }
  }

  // The same, with the 2/3 dealiasing mask folded in: they are always applied
  // back to back, and on the GPU each costs a full pass over the field.
  mulIkxMasked(dst, src, kx, mask, s) {
    for (let i = 0; i < dst.re.length; i++) {
      const c = s * kx[i] * mask[i];
      dst.re[i] = -c * src.im[i];
      dst.im[i] = c * src.re[i];
    }
  }

  // a *= exp(i * phase[i]) with a precomputed real phase table
  // (the exact linear half-step of split-step NLS).
  rotateByPhase(a, phase) {
    for (let i = 0; i < a.re.length; i++) {
      const c = Math.cos(phase[i]), s = Math.sin(phase[i]);
      const r = a.re[i], m = a.im[i];
      a.re[i] = r * c - m * s;
      a.im[i] = r * s + m * c;
    }
  }

  // dst.re = f(src.re), dst.im = 0. `f` is a compiled pointwise handle -- the
  // sine-Gordon restoring force and its energy density both go through here.
  mapReal(dst, src, f) {
    for (let i = 0; i < dst.re.length; i++) { dst.re[i] = f(src.re[i]); dst.im[i] = 0; }
  }

  // dst.re = f(|src|^2), dst.im = 0. With the identity handle this is just
  // the NLS intensity; with the potential handle it is the energy density.
  abs2Map(dst, src, f) {
    for (let i = 0; i < dst.re.length; i++) {
      dst.re[i] = f(src.re[i] * src.re[i] + src.im[i] * src.im[i]);
      dst.im[i] = 0;
    }
  }

  // a *= exp(i * dt * g(|a|^2)) -- the nonlinear half-step of split-step NLS.
  // `gain` maps intensity to phase rate; it is a tiny pure function so the
  // WGSL version is a literal transcription.
  // A pure phase rotation: |psi| per cell is unchanged, which is why split-step
  // conserves the power exactly. In f32 it does not, and the loss is biased
  // rather than random - see "Дрейф мощности НУШ на GPU" in CLAUDE.md. Do not
  // "fix" that by rescaling onto the incoming modulus: measured, that removes
  // 7% of the kick's drift and none of the FFT's, which is the bigger half.
  nlsPhaseStep(a, dt, gain) {
    for (let i = 0; i < a.re.length; i++) {
      const r = a.re[i], m = a.im[i];
      const th = dt * gain(r * r + m * m);
      const c = Math.cos(th), s = Math.sin(th);
      a.re[i] = r * c - m * s;
      a.im[i] = r * s + m * c;
    }
  }

  // ETDRK4 fused updates. Kept as three named kernels rather than a generic
  // "combine" so each is one shader with a fixed binding list.
  etdStage(out, E2, v, Q, N) {           // out = E2*v + Q*N
    for (let i = 0; i < out.re.length; i++) {
      const ar = E2.re[i] * v.re[i] - E2.im[i] * v.im[i];
      const ai = E2.re[i] * v.im[i] + E2.im[i] * v.re[i];
      const br = Q.re[i] * N.re[i] - Q.im[i] * N.im[i];
      const bi = Q.re[i] * N.im[i] + Q.im[i] * N.re[i];
      out.re[i] = ar + br; out.im[i] = ai + bi;
    }
  }

  etdStageC(out, E2, a, Q, Nb, Nv) {     // out = E2*a + Q*(2*Nb - Nv)
    for (let i = 0; i < out.re.length; i++) {
      const ar = E2.re[i] * a.re[i] - E2.im[i] * a.im[i];
      const ai = E2.re[i] * a.im[i] + E2.im[i] * a.re[i];
      const cr = 2 * Nb.re[i] - Nv.re[i];
      const ci = 2 * Nb.im[i] - Nv.im[i];
      const br = Q.re[i] * cr - Q.im[i] * ci;
      const bi = Q.re[i] * ci + Q.im[i] * cr;
      out.re[i] = ar + br; out.im[i] = ai + bi;
    }
  }

  // v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
  etdFinal(v, E, Nv, f1, Na, Nb, f2, Nc, f3) {
    for (let i = 0; i < v.re.length; i++) {
      const vr = v.re[i], vi = v.im[i];
      let r = E.re[i] * vr - E.im[i] * vi;
      let m = E.re[i] * vi + E.im[i] * vr;
      r += f1.re[i] * Nv.re[i] - f1.im[i] * Nv.im[i];
      m += f1.re[i] * Nv.im[i] + f1.im[i] * Nv.re[i];
      const sr = 2 * (Na.re[i] + Nb.re[i]);
      const si = 2 * (Na.im[i] + Nb.im[i]);
      r += f2.re[i] * sr - f2.im[i] * si;
      m += f2.re[i] * si + f2.im[i] * sr;
      r += f3.re[i] * Nc.re[i] - f3.im[i] * Nc.im[i];
      m += f3.re[i] * Nc.im[i] + f3.im[i] * Nc.re[i];
      v.re[i] = r; v.im[i] = m;
    }
  }

  // Exact linear propagation of a second-order wave field over dt:
  // (u, w=u_t) with u_tt = -omega^2 u.
  waveRotate(u, w, cosw, sinw, invOmega, omega) {
    for (let i = 0; i < u.re.length; i++) {
      const c = cosw[i], s = sinw[i], io = invOmega[i], om = omega[i];
      let a = u.re[i], b = w.re[i];
      u.re[i] = a * c + b * s * io;
      w.re[i] = -a * s * om + b * c;
      a = u.im[i]; b = w.im[i];
      u.im[i] = a * c + b * s * io;
      w.im[i] = -a * s * om + b * c;
    }
  }

  // -- reductions (diagnostics only) ---------------------------------------
  sumRe(a) { let s = 0; for (let i = 0; i < a.re.length; i++) s += a.re[i]; return s; }
  sumSqRe(a) { let s = 0; for (let i = 0; i < a.re.length; i++) s += a.re[i] * a.re[i]; return s; }
  sumAbs2(a) {
    let s = 0;
    for (let i = 0; i < a.re.length; i++) s += a.re[i] * a.re[i] + a.im[i] * a.im[i];
    return s;
  }
  maxAbsRe(a) { let s = 0; for (let i = 0; i < a.re.length; i++) s = Math.max(s, Math.abs(a.re[i])); return s; }

  // [sum w*|a|^2, sum |a|^2] -- the NLS gradient energy needs both.
  weightedSumAbs2(a, w) {
    let ws = 0, s = 0;
    for (let i = 0; i < a.re.length; i++) {
      const m = a.re[i] * a.re[i] + a.im[i] * a.im[i];
      ws += w[i] * m; s += m;
    }
    return [ws, s];
  }

  // -- frame boundary ------------------------------------------------------
  // The GPU backend batches every dispatch of a frame into one submission and
  // flushes here; on the CPU there is nothing to flush.
  flush() {}
  dispose() {}
}
