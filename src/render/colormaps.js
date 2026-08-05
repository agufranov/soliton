// Colour lookup tables, built once as 256-entry Uint8 ramps.
//
// `height`    diverging around zero  - for the real-valued surface models
//             (KP, ZK), so a trough reads differently from a crest.
// `intensity` sequential from black  - for |psi|^2, where zero is "nothing".
// `phase`     cyclic                 - for sine-Gordon, where u and u + 2*pi
//             are physically the same state.

function ramp(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let a = stops[0], b = stops[stops.length - 1];
    for (let s = 0; s < stops.length - 1; s++) {
      if (t >= stops[s][0] && t <= stops[s + 1][0]) { a = stops[s]; b = stops[s + 1]; break; }
    }
    const span = b[0] - a[0];
    const f = span > 0 ? (t - a[0]) / span : 0;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[1][c] + (b[1][c] - a[1][c]) * f;
  }
  return lut;
}

export const COLORMAPS = {
  // Dark at zero, not light. Most of the domain is undisturbed, and a light
  // neutral washes the whole frame out; with a dark rest state the crests and
  // troughs glow out of a quiet background instead.
  height: ramp([
    [0.00, [150, 232, 255]],
    [0.16, [56, 150, 214]],
    [0.34, [24, 58, 110]],
    [0.50, [8, 10, 20]],
    [0.66, [104, 40, 74]],
    [0.84, [216, 98, 52]],
    [1.00, [255, 236, 176]],
  ]),
  intensity: ramp([
    [0.00, [4, 4, 14]],
    [0.18, [42, 16, 82]],
    [0.38, [110, 30, 110]],
    [0.58, [186, 54, 88]],
    [0.76, [238, 118, 40]],
    [0.90, [250, 194, 84]],
    [1.00, [255, 250, 220]],
  ]),
  // Field range is [-2*pi, 2*pi] and u = 0, +-2*pi are all the same vacuum,
  // so the ramp is dark at 0, 0.5 and 1 and glows in between: what lights up
  // is the kink wall itself, which is the only thing carrying energy.
  phase: ramp([
    [0.000, [10, 12, 22]],
    [0.125, [40, 92, 150]],
    [0.250, [150, 214, 240]],
    [0.375, [40, 92, 150]],
    [0.500, [10, 12, 22]],
    [0.625, [150, 70, 60]],
    [0.750, [246, 190, 120]],
    [0.875, [150, 70, 60]],
    [1.000, [10, 12, 22]],
  ]),
};
