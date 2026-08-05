// Split-step Fourier (Strang) solver for the 2D nonlinear Schrodinger family
//
//     i psi_t + (1/2) laplacian(psi) + f(|psi|^2) psi = 0
//
// Half linear step in Fourier, full nonlinear phase kick in physical space,
// half linear step again. The linear part is applied *exactly* (it is a pure
// phase), so the only error is the O(dt^2) splitting commutator, and |psi|^2
// integrates to a constant to round-off.
//
// The state lives in physical space between steps, which is also the layout
// the renderer wants.
//
// `gain` and `potential` are pointwise nonlinearities supplied by the model as
// {js, wgsl} pairs and compiled by the backend, so the same solver runs on the
// CPU and on the GPU without a branch.

export class NlsSolver {
  constructor(backend, grid, { dt, gain, hamiltonianDensity }) {
    this.backend = backend;
    this.grid = grid;
    this.dt = dt;
    this.gain = backend.pointwise(gain);
    this.potential = backend.pointwise(hamiltonianDensity);
    this.identity = backend.pointwise({ js: (I) => I, wgsl: 'x' });
    this.psi = backend.alloc();
    this.intensity = backend.alloc();   // |psi|^2 in .re -- render + diagnostics
    this.density = backend.alloc();     // potential energy density
    this.spectrum = backend.alloc();
    this.time = 0;
    this.version = 0;
    this._intensityAt = -1;
    this._diagAt = -1;
    this._diag = null;

    // exp(-i k^2 dt / 4) for the two half steps
    this.halfPhase = new Float64Array(grid.n);
    for (let p = 0; p < grid.n; p++) this.halfPhase[p] = (-grid.k2[p] * dt) / 4;
  }

  step() {
    const b = this.backend;
    b.forward(this.psi);
    b.rotateByPhase(this.psi, this.halfPhase);
    b.inverse(this.psi);
    b.nlsPhaseStep(this.psi, this.dt, this.gain);
    b.forward(this.psi);
    b.rotateByPhase(this.psi, this.halfPhase);
    b.inverse(this.psi);
    this.time += this.dt;
    this.version++;
  }

  target() {
    if (this._intensityAt !== this.version) {
      this.backend.abs2Map(this.intensity, this.psi, this.identity);
      this._intensityAt = this.version;
    }
    return this.intensity;
  }

  field() { return this.backend.snapshot(this.target()).re; }

  // Fold a host-built complex bump (profile times Galilean carrier) into psi.
  addComplex(re, im) {
    this.backend.addHostComplex(this.psi, re, im);
    this.version++;
  }

  clear() { this.backend.zero(this.psi); this.time = 0; this.version++; }

  peak() { return this.diagnostics().peak; }

  // Cached per state change -- see the note on SpectralRealSolver.diagnostics.
  diagnostics() {
    if (this._diagAt === this.version) return this._diag;
    const b = this.backend, cell = this.grid.dx * this.grid.dy;
    const I = this.target();
    const power = b.sumRe(I);
    const peak = b.maxAbsRe(I);

    // Gradient energy from the spectrum: <k^2/2> weighted by |psi_hat|^2,
    // rescaled by the physical-space power so the FFT normalisation drops out.
    b.copy(this.spectrum, this.psi);
    b.forward(this.spectrum);
    const [wk, norm] = b.weightedSumAbs2(this.spectrum, this.grid.k2);
    const grad = norm > 0 ? (0.5 * wk / norm) * power * cell : 0;

    b.abs2Map(this.density, this.psi, this.potential);
    const pot = b.sumRe(this.density);

    this._diag = { power: power * cell, peak, energy: grad - pot * cell };
    this._diagAt = this.version;
    return this._diag;
  }
}
