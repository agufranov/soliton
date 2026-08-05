// Shared engine for the real-valued, KdV-type 2D equations:
//
//     u_t = L(k) u  -  alpha * d/dx (u^2)
//
// KP-I, KP-II and Zakharov-Kuznetsov differ *only* in L(k), so they share
// this whole solver. State is kept in Fourier space; the physical field is
// materialised on demand for rendering and for spawning.
//
// Nothing here indexes a field buffer: every read of the physical field goes
// through a backend reduction or through `snapshot`, and every write goes
// through `setHostReal` / `addHostReal`. That is what lets the same code drive
// the CPU and the WebGPU backend unchanged.

import { Etdrk4 } from './etdrk4.js';

export class SpectralRealSolver {
  constructor(backend, grid, { Limag, alpha, dt }) {
    this.backend = backend;
    this.grid = grid;
    this.alpha = alpha;
    this.dt = dt;
    this.v = backend.alloc();      // spectral state
    this.u = backend.alloc();      // physical scratch / render source
    this.tmp = backend.alloc();
    this.time = 0;
    this.version = 0;              // bumped whenever `v` changes
    this._materialised = -1;
    this._diagAt = -1;
    this._diag = null;

    // N(v) = -alpha * i*kx * FFT((IFFT v)^2), dealiased. Written in fused form:
    // the copy rides on the inverse's first pass, the squaring on the forward's
    // first pass, and the mask on the derivative. On the CPU those fuse into
    // nothing in particular; on the GPU each one removed is a whole sweep of
    // the field, and the step is memory bound.
    this.integrator = new Etdrk4(backend, Limag, dt, (vIn, nOut) => {
      const b = this.backend;
      b.inverseInto(this.tmp, vIn);   // u
      b.forwardSqrReal(this.tmp);     // FFT(u^2)
      b.mulIkxMasked(nOut, this.tmp, grid.kx, grid.mask, -this.alpha);
    });
  }

  step() { this.integrator.step(this.v); this.time += this.dt; this.version++; }

  // u <- inverse FFT of v, at most once per state change. Both the renderer
  // and the diagnostics want it, and on a 1024^2 grid a redundant transform is
  // a quarter of the frame budget.
  target() {
    if (this._materialised !== this.version) {
      this.backend.inverseInto(this.u, this.v);
      this._materialised = this.version;
    }
    return this.u;
  }

  // Physical field, real part only, as a host array. On the GPU backend this
  // is a download and lags a frame; the GPU renderer reads `target()` instead.
  physical() { return this.backend.snapshot(this.target()).re; }

  // Add a physical-space bump (given as a Float64Array of the same length)
  // to the current state.
  addPhysical(du) {
    const b = this.backend;
    b.addHostReal(this.target(), du);
    b.forward(this.u);
    b.maskReal(this.u, this.grid.mask);
    b.copy(this.v, this.u);
    this.version++;
    this._materialised = -1;
  }

  setPhysical(u0) {
    const b = this.backend;
    b.setHostReal(this.u, u0);
    b.forward(this.u);
    b.maskReal(this.u, this.grid.mask);
    b.copy(this.v, this.u);
    this.version++;
    this._materialised = -1;
  }

  clear() { this.backend.zero(this.v); this.time = 0; this.version++; }

  // Cached per state change: the render loop wants the peak for its blow-up
  // guard and the panel wants all three, and on a 1024^2 CPU grid each
  // reduction is a full sweep of a million cells.
  diagnostics() {
    if (this._diagAt !== this.version) {
      const b = this.backend;
      const u = this.target();
      const cell = this.grid.dx * this.grid.dy;
      this._diag = { mass: b.sumRe(u) * cell, energy: b.sumSqRe(u) * cell, peak: b.maxAbsRe(u) };
      this._diagAt = this.version;
    }
    return this._diag;
  }

  peak() { return this.diagnostics().peak; }
}
