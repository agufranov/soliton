// 2D sine-Gordon:   u_tt - laplacian(u) + sin(u) = 0
//
// Included as the honest counterexample. In 1D sine-Gordon is the textbook
// integrable soliton system (kinks, antikinks, breathers, all elastic). In 2D
// it is NOT integrable and there is no stable localised travelling soliton:
//
//   * a circular kink ring ("pulson") contracts under its own line tension,
//     collapses through the centre, re-expands, and loses a slice of its
//     energy to radiation on every bounce until it dies;
//   * a straight kink line is a perfectly good travelling wall, but it is a
//     line, not a hump.
//
// Worth watching precisely because it shows that "1D soliton equation" does
// not generalise to 2D for free - which is the whole reason KP-I and the
// saturable NLS family are interesting.

import { WaveSolver } from '../core/waveSolver.js';

function ringProfile(grid, R0, w, cx, cy, sign) {
  const u = new Float64Array(grid.n);
  for (let j = 0; j < grid.ny; j++) for (let i = 0; i < grid.nx; i++) {
    const dx = grid.wrapX(grid.x(i) - cx), dy = grid.wrapY(grid.y(j) - cy);
    const r = Math.hypot(dx, dy);
    u[j * grid.nx + i] = sign * 4 * Math.atan(Math.exp((R0 - r) / w));
  }
  return u;
}

export const sineGordon = {
  id: 'sg', name: 'Sine-Gordon 2D — контрпример', family: 'sg',
  blurb: 'В 1D это эталонная интегрируемая система (кинки и бризеры, упругие '
    + 'столкновения). В 2D интегрируемость теряется: кольцевой кинк-«пульсон» '
    + 'схлопывается, отскакивает и на каждом отскоке теряет энергию на излучение. '
    + 'Устойчивого 2D-солитона у sine-Gordon нет.',
  field: { kind: 'phase', label: 'u(x,y)' },
  grid: { n: 128, L: 60 },
  dt: 0.03,
  params: [
    { key: 'R0', label: 'радиус кольца', min: 2, max: 18, step: 0.5, value: 8,
      hint: 'кольцо сразу начнёт сжиматься — линейное натяжение кинка' },
    { key: 'w', label: 'толщина стенки', min: 0.6, max: 3, step: 0.1, value: 1,
      hint: 'равновесная толщина кинка равна 1 в этих единицах' },
  ],
  spawnKinds: [
    { id: 'ring', label: 'кольцевой кинк (+2π)' },
    { id: 'antiring', label: 'антикольцо (−2π)' },
  ],
  presets: [
    { id: 'pulson', label: 'один пульсон', items: [{ R0: 9, w: 1, x: 0.5, y: 0.5, sign: 1 }] },
    { id: 'two', label: 'кольцо и антикольцо', items: [
      { R0: 6, w: 1, x: 0.3, y: 0.5, sign: 1 }, { R0: 6, w: 1, x: 0.7, y: 0.5, sign: -1 },
    ] },
  ],
  create(backend, grid, dt, params) {
    const solver = new WaveSolver(backend, grid, {
      dt, mass: 1,
      // R(u) = m^2 u - V'(u) = u - sin(u)
      restoring: { js: (u) => u - Math.sin(u), wgsl: 'x - sin(x)' },
    });
    // Potential energy density 1 - cos(u), as a pointwise kernel so the
    // diagnostic sweep runs wherever the field already lives.
    const density = backend.pointwise({ js: (u) => 1 - Math.cos(u), wgsl: '1.0 - cos(x)' });
    const scratch = backend.alloc();
    const zeros = new Float64Array(grid.n);
    let diagAt = -1, diag = null;
    return {
      solver,
      step: () => solver.step(),
      clear: () => solver.clear(),
      get time() { return solver.time; },
      field: () => solver.physical(),
      target: () => solver.target(),
      peak: () => backend.maxAbsRe(solver.target()),
      spawn(x, y, drag, kind) {
        const sign = kind === 'antiring' ? -1 : 1;
        solver.addPhysical(ringProfile(grid, params.R0, params.w, x, y, sign), zeros);
      },
      applyPreset(p) {
        solver.clear();
        const u = new Float64Array(grid.n);
        for (const s of p.items) {
          const r = ringProfile(grid, s.R0, s.w, s.x * grid.Lx, s.y * grid.Ly, s.sign);
          for (let i = 0; i < u.length; i++) u[i] += r[i];
        }
        solver.setPhysical(u, zeros);
      },
      diagnostics() {
        if (diagAt !== solver.version) {
          const u = solver.target();
          backend.mapReal(scratch, u, density);
          const pot = backend.sumRe(scratch) * grid.dx * grid.dy;
          const peak = backend.maxAbsRe(u);
          diag = { 'потенц. энергия': pot, 'макс |u|': peak, 'в единицах 2π': peak / (2 * Math.PI) };
          diagAt = solver.version;
        }
        return diag;
      },
    };
  },
};
