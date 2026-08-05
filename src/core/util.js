// Small grid helpers shared by the models.

// Add a profile that is centred on the middle of the box into `dst`, with its
// centre moved to world position (cx, cy). Pure index arithmetic, so it wraps
// correctly around the periodic edges.
export function addCentered(grid, dst, profile, cx, cy, amp = 1) {
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) {
    throw new Error(`addCentered: non-finite centre (${cx}, ${cy})`);
  }
  const nx = grid.nx, ny = grid.ny;
  const si = Math.round(cx / grid.dx) - (nx >> 1);
  const sj = Math.round(cy / grid.dy) - (ny >> 1);
  for (let j = 0; j < ny; j++) {
    const tj = (((j + sj) % ny) + ny) % ny;
    for (let i = 0; i < nx; i++) {
      const ti = (((i + si) % nx) + nx) % nx;
      dst[tj * nx + ti] += amp * profile[j * nx + i];
    }
  }
}

// Same, but for a complex field carrying a Galilean boost exp(i (vx x + vy y)).
// The phase uses *wrapped* offsets from the centre so the carrier never shows a
// seam at the box edge where the envelope is already negligible.
export function addCenteredBoosted(grid, re, im, profile, cx, cy, vx, vy, amp = 1) {
  const nx = grid.nx, ny = grid.ny;
  const si = Math.round(cx / grid.dx) - (nx >> 1);
  const sj = Math.round(cy / grid.dy) - (ny >> 1);
  for (let j = 0; j < ny; j++) {
    const tj = (((j + sj) % ny) + ny) % ny;
    const dy = (j - (ny >> 1)) * grid.dy;
    for (let i = 0; i < nx; i++) {
      const ti = (((i + si) % nx) + nx) % nx;
      const dx = (i - (nx >> 1)) * grid.dx;
      const a = amp * profile[j * nx + i];
      const th = vx * dx + vy * dy;
      re[tj * nx + ti] += a * Math.cos(th);
      im[tj * nx + ti] += a * Math.sin(th);
    }
  }
}

export function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
