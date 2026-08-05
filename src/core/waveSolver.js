// Second-order-in-time nonlinear wave equation
//
//     u_tt = laplacian(u) - V'(u)
//
// split as  u_tt = -(k^2 + m^2) u + R(u),  R(u) = m^2 u - V'(u).
//
// The linear part is a harmonic oscillator per mode, so it is advanced by an
// exact rotation of (u_hat, u_hat_t); the remainder is a Strang velocity kick
// in the middle. Second order, symplectic, and it conserves energy to a
// bounded oscillation rather than drifting.
//
// `restoring` arrives as a {js, wgsl} pair and is compiled by the backend, so
// the kick is a pointwise kernel on both backends and the physical field never
// has to travel to the host mid-step.

export class WaveSolver {
  constructor(backend, grid, { dt, mass = 1, restoring }) {
    this.backend = backend;
    this.grid = grid;
    this.dt = dt;
    this.restoring = backend.pointwise(restoring);
    this.u = backend.alloc();     // spectral
    this.w = backend.alloc();     // spectral, du/dt
    this.tmp = backend.alloc();
    this.phys = backend.alloc();  // materialised u, for render + diagnostics
    this.time = 0;
    this.version = 0;
    this._materialised = -1;

    const n = grid.n;
    this.omega = new Float64Array(n);
    this.invOmega = new Float64Array(n);
    this.cosH = new Float64Array(n);
    this.sinH = new Float64Array(n);
    for (let p = 0; p < n; p++) {
      const om = Math.sqrt(grid.k2[p] + mass * mass);
      this.omega[p] = om;
      this.invOmega[p] = 1 / om;
      this.cosH[p] = Math.cos((om * dt) / 2);
      this.sinH[p] = Math.sin((om * dt) / 2);
    }
  }

  _half() {
    this.backend.waveRotate(this.u, this.w, this.cosH, this.sinH, this.invOmega, this.omega);
  }

  step() {
    const b = this.backend;
    this._half();
    b.inverseInto(this.tmp, this.u);
    b.mapReal(this.tmp, this.tmp, this.restoring);
    b.forward(this.tmp);
    b.maskReal(this.tmp, this.grid.mask);
    b.axpy(this.w, this.tmp, this.dt);
    this._half();
    this.time += this.dt;
    this.version++;
  }

  target() {
    if (this._materialised !== this.version) {
      this.backend.inverseInto(this.phys, this.u);
      this._materialised = this.version;
    }
    return this.phys;
  }

  physical() { return this.backend.snapshot(this.target()).re; }

  setPhysical(u0, w0) {
    const b = this.backend;
    b.setHostReal(this.u, u0);
    b.forward(this.u); b.maskReal(this.u, this.grid.mask);
    b.setHostReal(this.w, w0);
    b.forward(this.w); b.maskReal(this.w, this.grid.mask);
    this.version++;
  }

  // The transform and the mask are both linear, so adding the bump's spectrum
  // to the state's is identical to adding the fields and re-transforming - and
  // it needs no round trip to the host.
  addPhysical(du, dw) {
    const b = this.backend;
    b.setHostReal(this.tmp, du);
    b.forward(this.tmp); b.maskReal(this.tmp, this.grid.mask);
    b.axpy(this.u, this.tmp, 1);
    b.setHostReal(this.tmp, dw);
    b.forward(this.tmp); b.maskReal(this.tmp, this.grid.mask);
    b.axpy(this.w, this.tmp, 1);
    this.version++;
  }

  clear() { this.backend.zero(this.u); this.backend.zero(this.w); this.time = 0; this.version++; }
}
