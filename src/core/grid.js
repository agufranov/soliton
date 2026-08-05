// Periodic square domain [0,Lx) x [0,Ly) sampled on nx * ny points,
// plus the wavenumber tables every spectral model needs.

export class Grid {
  constructor(nx, ny, Lx, Ly) {
    this.nx = nx; this.ny = ny;
    this.Lx = Lx; this.Ly = Ly;
    this.dx = Lx / nx; this.dy = Ly / ny;
    this.n = nx * ny;

    // 1D wavenumbers in FFT order: 0,1,..,N/2-1,-N/2,..,-1 scaled by 2pi/L
    const kx1 = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      const m = i <= nx / 2 ? i : i - nx;
      kx1[i] = (2 * Math.PI * m) / Lx;
    }
    const ky1 = new Float64Array(ny);
    for (let j = 0; j < ny; j++) {
      const m = j <= ny / 2 ? j : j - ny;
      ky1[j] = (2 * Math.PI * m) / Ly;
    }
    this.kx1 = kx1; this.ky1 = ky1;

    // Flattened 2D tables (index = j*nx + i), matching the field layout.
    this.kx = new Float64Array(this.n);
    this.ky = new Float64Array(this.n);
    this.k2 = new Float64Array(this.n);
    // 2/3 dealiasing mask
    this.mask = new Float64Array(this.n);
    const cutX = (2 / 3) * (Math.PI / this.dx);
    const cutY = (2 / 3) * (Math.PI / this.dy);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const p = j * nx + i;
        this.kx[p] = kx1[i];
        this.ky[p] = ky1[j];
        this.k2[p] = kx1[i] * kx1[i] + ky1[j] * ky1[j];
        this.mask[p] = Math.abs(kx1[i]) < cutX && Math.abs(ky1[j]) < cutY ? 1 : 0;
      }
    }
  }

  x(i) { return i * this.dx; }
  y(j) { return j * this.dy; }

  // Shortest periodic offset from a to b, used so a soliton dropped near an
  // edge is still built correctly across the wrap.
  wrapX(d) { const L = this.Lx; d = d % L; if (d > L / 2) d -= L; if (d < -L / 2) d += L; return d; }
  wrapY(d) { const L = this.Ly; d = d % L; if (d > L / 2) d -= L; if (d < -L / 2) d += L; return d; }
}
