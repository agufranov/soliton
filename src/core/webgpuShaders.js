// WGSL sources for the WebGPU backend.
//
// Every kernel here is the literal transcription of one method of CpuBackend.
// Fields are a single storage buffer of vec2<f32> (re, im interleaved) -- the
// "one rg32float texture" half of the storage contract in core/backend.js.
// Real tables (mask, kx, phase, ...) are plain array<f32> of the same length.
//
// Sizes are baked into the source as constants rather than passed as uniforms:
// a grid size change rebuilds the backend anyway, and a compile-time N lets the
// FFT keep the whole row in workgroup memory.
//
// Two conventions worth knowing before editing:
//   * a buffer may not be bound read-only and read-write in the same dispatch,
//     so the two genuinely in-place kernels (sqrReal, mapReal) have their own
//     single-binding variants;
//   * scalars arrive as a 16-byte uniform holding one f32 in .x, cached per
//     distinct value -- every scalar in this app is constant per solver.

export const ELEM_WG = 64;      // threads per workgroup for elementwise kernels
export const RED_WG = 256;      // threads per workgroup for reductions
export const RED_GROUPS = 64;   // first-stage reduction workgroups

// Every generator below is memoised on its arguments. The backend asks for a
// source on *every* dispatch and looks the pipeline up by source string, so
// without this a 128^2 frame rebuilds ~700 multi-kilobyte templates - which
// costs more than the dispatches it is describing.
const memo = new Map();
function once(tag, args, build) {
  const key = tag + '|' + args.join('|');
  let v = memo.get(key);
  if (v === undefined) { v = build(); memo.set(key, v); }
  return v;
}

const head = (n) => `const N : u32 = ${n}u;\n`;

const elem = (n, body, bindings) => `${head(n)}
${bindings}
@compute @workgroup_size(${ELEM_WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
${body}
}`;

const RW = (b, name) => `@group(0) @binding(${b}) var<storage, read_write> ${name} : array<vec2<f32>>;`;
const RO = (b, name) => `@group(0) @binding(${b}) var<storage, read> ${name} : array<vec2<f32>>;`;
const TAB = (b, name) => `@group(0) @binding(${b}) var<storage, read> ${name} : array<f32>;`;
const SCALAR = (b) => `struct Sc { v : f32 };\n@group(0) @binding(${b}) var<uniform> sc : Sc;`;

// --- elementwise -----------------------------------------------------------

const elementwiseRaw = {
  zero: (n) => elem(n, '  a[i] = vec2<f32>(0.0, 0.0);', RW(0, 'a')),

  copy: (n) => elem(n, '  dst[i] = src[i];', `${RW(0, 'dst')}\n${RO(1, 'src')}`),

  scale: (n) => elem(n, '  a[i] = a[i] * sc.v;', `${RW(0, 'a')}\n${SCALAR(1)}`),

  // dst = a + s*b
  addScaled: (n) => elem(n, '  dst[i] = a[i] + sc.v * b[i];',
    `${RW(0, 'dst')}\n${RO(1, 'a')}\n${RO(2, 'b')}\n${SCALAR(3)}`),

  // y += s*x
  axpy: (n) => elem(n, '  y[i] = y[i] + sc.v * x[i];',
    `${RW(0, 'y')}\n${RO(1, 'x')}\n${SCALAR(2)}`),

  // dst += src  (used to fold a host-built bump into a live field)
  addInto: (n) => elem(n, '  dst[i] = dst[i] + src[i];', `${RW(0, 'dst')}\n${RO(1, 'src')}`),

  // dst.re += src.re, dst.im = 0 -- the real-field spawn path, matching
  // CpuBackend.addHostReal exactly (it clears the imaginary part too).
  addRealFrom: (n) => elem(n, '  dst[i] = vec2<f32>(dst[i].x + src[i].x, 0.0);',
    `${RW(0, 'dst')}\n${RO(1, 'src')}`),

  maskReal: (n) => elem(n, '  a[i] = a[i] * m[i];', `${RW(0, 'a')}\n${TAB(1, 'm')}`),

  sqrReal: (n) => elem(n, '  let u = src[i].x;\n  dst[i] = vec2<f32>(u * u, 0.0);',
    `${RW(0, 'dst')}\n${RO(1, 'src')}`),

  sqrRealInPlace: (n) => elem(n, '  let u = a[i].x;\n  a[i] = vec2<f32>(u * u, 0.0);', RW(0, 'a')),

  // dst = s * (i*kx) * src
  mulIkxScaled: (n) => elem(n, `  let c = sc.v * kx[i];
  let s = src[i];
  dst[i] = vec2<f32>(-c * s.y, c * s.x);`,
    `${RW(0, 'dst')}\n${RO(1, 'src')}\n${TAB(2, 'kx')}\n${SCALAR(3)}`),

  // dst = m * s * (i*kx) * src -- the derivative and the 2/3 dealiasing mask in
  // one pass. They are always applied back to back on the KdV-type nonlinear
  // term, and each on its own is a full read-modify-write of the field.
  mulIkxMasked: (n) => elem(n, `  let c = sc.v * kx[i] * m[i];
  let s = src[i];
  dst[i] = vec2<f32>(-c * s.y, c * s.x);`,
    `${RW(0, 'dst')}\n${RO(1, 'src')}\n${TAB(2, 'kx')}\n${TAB(3, 'm')}\n${SCALAR(4)}`),

  rotateByPhase: (n) => elem(n, `  let th = ph[i];
  let c = cos(th); let s = sin(th);
  let v = a[i];
  a[i] = vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);`,
    `${RW(0, 'a')}\n${TAB(1, 'ph')}`),

  // out = E2*v + Q*Nv   (complex products)
  etdStage: (n) => elem(n, `  let e = E2[i]; let vv = v[i]; let q = Q[i]; let nn = Nv[i];
  let ar = e.x * vv.x - e.y * vv.y;
  let ai = e.x * vv.y + e.y * vv.x;
  let br = q.x * nn.x - q.y * nn.y;
  let bi = q.x * nn.y + q.y * nn.x;
  out[i] = vec2<f32>(ar + br, ai + bi);`,
    `${RW(0, 'out')}\n${RO(1, 'E2')}\n${RO(2, 'v')}\n${RO(3, 'Q')}\n${RO(4, 'Nv')}`),

  // out = E2*a + Q*(2*Nb - Nv)
  etdStageC: (n) => elem(n, `  let e = E2[i]; let aa = a[i]; let q = Q[i];
  let ar = e.x * aa.x - e.y * aa.y;
  let ai = e.x * aa.y + e.y * aa.x;
  let c = 2.0 * Nb[i] - Nv[i];
  let br = q.x * c.x - q.y * c.y;
  let bi = q.x * c.y + q.y * c.x;
  out[i] = vec2<f32>(ar + br, ai + bi);`,
    `${RW(0, 'out')}\n${RO(1, 'E2')}\n${RO(2, 'a')}\n${RO(3, 'Q')}\n${RO(4, 'Nb')}\n${RO(5, 'Nv')}`),

  // v = E*v + f1*Nv   -- first half of etdFinal, split so no dispatch needs
  // more than the guaranteed 8 storage bindings.
  etdFinalA: (n) => elem(n, `  let e = E[i]; let vv = v[i]; let a = f1[i]; let nn = Nv[i];
  var r = e.x * vv.x - e.y * vv.y;
  var m = e.x * vv.y + e.y * vv.x;
  r = r + a.x * nn.x - a.y * nn.y;
  m = m + a.x * nn.y + a.y * nn.x;
  v[i] = vec2<f32>(r, m);`,
    `${RW(0, 'v')}\n${RO(1, 'E')}\n${RO(2, 'f1')}\n${RO(3, 'Nv')}`),

  // v += f2*2*(Na+Nb) + f3*Nc
  etdFinalB: (n) => elem(n, `  let s = 2.0 * (Na[i] + Nb[i]);
  let b = f2[i]; let c = f3[i]; let nc = Nc[i];
  var r = v[i].x + b.x * s.x - b.y * s.y;
  var m = v[i].y + b.x * s.y + b.y * s.x;
  r = r + c.x * nc.x - c.y * nc.y;
  m = m + c.x * nc.y + c.y * nc.x;
  v[i] = vec2<f32>(r, m);`,
    `${RW(0, 'v')}\n${RO(1, 'f2')}\n${RO(2, 'Na')}\n${RO(3, 'Nb')}\n${RO(4, 'f3')}\n${RO(5, 'Nc')}`),

  waveRotate: (n) => elem(n, `  let c = cw[i]; let s = sw[i]; let io = iom[i]; let om = omg[i];
  let uu = u[i]; let ww = w[i];
  u[i] = vec2<f32>(uu.x * c + ww.x * s * io, uu.y * c + ww.y * s * io);
  w[i] = vec2<f32>(-uu.x * s * om + ww.x * c, -uu.y * s * om + ww.y * c);`,
    `${RW(0, 'u')}\n${RW(1, 'w')}\n${TAB(2, 'cw')}\n${TAB(3, 'sw')}\n${TAB(4, 'iom')}\n${TAB(5, 'omg')}`),
};

export const elementwise = {};
for (const [name, build] of Object.entries(elementwiseRaw)) {
  elementwise[name] = (n) => once(name, [n], () => build(n));
}

// --- pointwise nonlinearities (model-supplied WGSL) ------------------------
// `expr` is an expression in `x`, e.g. "x - sin(x)" or "x/(1.0 + 0.2*x)".

const pointwiseRaw = {
  // a *= exp(i*dt*gain(|a|^2))
  // Deliberately NOT renormalised onto the incoming modulus; see the note on
  // CpuBackend.nlsPhaseStep. It costs a sqrt per cell and buys 7% of a third of
  // the drift, because the bias is in f32 rounding itself, not in |(cos, sin)|.
  nlsPhaseStep: (n, expr) => elem(n, `  let v = a[i];
  let x = v.x * v.x + v.y * v.y;
  let th = sc.v * (${expr});
  let c = cos(th); let s = sin(th);
  a[i] = vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);`,
    `${RW(0, 'a')}\n${SCALAR(1)}`),

  mapReal: (n, expr) => elem(n, `  let x = src[i].x;
  dst[i] = vec2<f32>(${expr}, 0.0);`, `${RW(0, 'dst')}\n${RO(1, 'src')}`),

  mapRealInPlace: (n, expr) => elem(n, `  let x = a[i].x;
  a[i] = vec2<f32>(${expr}, 0.0);`, RW(0, 'a')),

  abs2Map: (n, expr) => elem(n, `  let v = src[i];
  let x = v.x * v.x + v.y * v.y;
  dst[i] = vec2<f32>(${expr}, 0.0);`, `${RW(0, 'dst')}\n${RO(1, 'src')}`),
};

export const pointwiseSrc = {};
for (const [name, build] of Object.entries(pointwiseRaw)) {
  pointwiseSrc[name] = (n, expr) => once(name, [n, expr], () => build(n, expr));
}

// --- FFT -------------------------------------------------------------------
//
// One workgroup per row, whole row resident in workgroup memory: 1024 complex
// f32 is 8 KB, half of the 16 KB guaranteed workgroup storage. Bit-reversal on
// load, then the same radix-2 Cooley-Tukey butterfly ladder as core/fft.js,
// with the identical twiddle convention (w = cos t + i*sgn*sin t, sgn = -1
// forward) so the two backends agree mode for mode.
//
// Columns are done by transposing and running the row kernel again -- strided
// column access would serialise the memory system. Note there are only THREE
// passes, not four: the spectrum is left transposed rather than being turned
// back, and everything that touches it (the k tables, the ETDRK4 coefficients)
// is stored transposed to match. See the backend for why that is safe.
//
// A pseudospectral step is purely memory bound, so these kernels also absorb
// whatever pointwise work can ride along on a pass that has to touch every
// element regardless:
//   opts.src   read from a second buffer (folds an otherwise separate `copy`)
//   opts.load  expression in `v` applied on load (folds u -> u^2)
//   opts.store expression in `v` applied on store (folds the 1/n of an inverse)

export const fftRowSrc = (n, inverse, opts = {}) =>
  once('fftRow', [n, inverse, opts.src ? 1 : 0, opts.load || '', opts.store || ''],
    () => fftRowRaw(n, inverse, opts));

function fftRowRaw(n, inverse, opts) {
  const bits = Math.log2(n) | 0;
  const wg = Math.min(RED_WG, n / 2);
  const from = opts.src ? 'src' : 'data';
  const loaded = opts.load ? opts.load : 'v';
  const stored = opts.store ? opts.store : 'v';
  // Real and imaginary parts live in SEPARATE f32 arrays, each padded by one
  // slot every 32, and every index goes through ix(). Both details are about
  // LDS bank conflicts, and together they are worth ~1.6x on this kernel:
  //
  //   * as interleaved vec2<f32> an element straddles two banks, so the
  //     stride-2 access of the first butterfly stage collides 4 ways across a
  //     32-lane half-wave. Split into f32 that is only 2 ways.
  //   * the padding makes ix(2t) = 2t + t/16 run through all 32 banks exactly
  //     once as t goes 0..31, so the stride-2 stage collides not at all.
  //
  // Cost is 2*(N + N/32)*4 bytes of workgroup storage - 8448 at N = 1024,
  // still half of the guaranteed 16 KB.
  const pad = n + (n >> 5);
  return `const N : u32 = ${n}u;
const PAD : u32 = ${pad}u;
const BITS : u32 = ${bits}u;
const W : u32 = ${wg}u;
const SGN : f32 = ${inverse ? '1.0' : '-1.0'};
const TAU : f32 = 6.283185307179586;

@group(0) @binding(0) var<storage, read_write> data : array<vec2<f32>>;
${opts.src ? '@group(0) @binding(1) var<storage, read> src : array<vec2<f32>>;' : ''}
var<workgroup> shr : array<f32, PAD>;
var<workgroup> shi : array<f32, PAD>;

fn ix(i : u32) -> u32 { return i + (i >> 5u); }

@compute @workgroup_size(W)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) li : vec3<u32>) {
  let row = wg.x * N;
  let tid = li.x;

  for (var i = tid; i < N; i = i + W) {
    let v = ${from}[row + i];
    let w = ${loaded};
    let d = ix(reverseBits(i) >> (32u - BITS));
    shr[d] = w.x;
    shi[d] = w.y;
  }
  workgroupBarrier();

  var len : u32 = 2u;
  loop {
    if (len > N) { break; }
    let half = len >> 1u;
    for (var b = tid; b < N / 2u; b = b + W) {
      let k = b & (half - 1u);
      let pa = ix((b / half) * len + k);
      let pb = ix((b / half) * len + k + half);
      let ang = TAU * f32(k) / f32(len);
      let wr = cos(ang);
      let wi = SGN * sin(ang);
      let xr = shr[pb];
      let xi = shi[pb];
      let tr = xr * wr - xi * wi;
      let ti = xr * wi + xi * wr;
      let ar = shr[pa];
      let ai = shi[pa];
      shr[pb] = ar - tr;
      shi[pb] = ai - ti;
      shr[pa] = ar + tr;
      shi[pa] = ai + ti;
    }
    workgroupBarrier();
    len = len << 1u;
  }

  for (var i = tid; i < N; i = i + W) {
    let s = ix(i);
    let v = vec2<f32>(shr[s], shi[s]);
    data[row + i] = ${stored};
  }
}`;
}

export const TILE = 16;

// `scale` folds the 1/(nx*ny) of an inverse transform into a pass that is
// already reading and writing every element, instead of a separate sweep.
export const transposeSrc = (nx, ny, scale = 1) =>
  once('transpose', [nx, ny, scale], () => transposeRaw(nx, ny, scale));

function transposeRaw(nx, ny, scale) {
  const s = scale === 1 ? '' : ` * ${scale === Math.round(scale) ? scale.toFixed(1) : scale}`;
  return `const NX : u32 = ${nx}u;
const NY : u32 = ${ny}u;
const T : u32 = ${TILE}u;

@group(0) @binding(0) var<storage, read> src : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst : array<vec2<f32>>;
// Row stride T+1, not T. The whole point of the tile is that one of the two
// accesses is transposed, and with an exact power-of-two stride every lane of
// that access lands in the same LDS bank - a 32-way conflict on the hottest
// line in the kernel. One slot of padding per row skews it so they do not.
var<workgroup> tile : array<vec2<f32>, ${TILE * (TILE + 1)}>;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) li : vec3<u32>) {
  let x = wg.x * T + li.x;
  let y = wg.y * T + li.y;
  if (x < NX && y < NY) { tile[li.y * (T + 1u) + li.x] = src[y * NX + x]; }
  workgroupBarrier();
  let tx = wg.y * T + li.x;
  let ty = wg.x * T + li.y;
  if (tx < NY && ty < NX) { dst[ty * NY + tx] = tile[li.x * (T + 1u) + li.y]${s}; }
}`;
}

// --- reductions ------------------------------------------------------------
//
// Diagnostics only. Two stages: RED_GROUPS workgroups each fold a strided slice
// into one vec4 (sum Re, sum Re^2, sum |a|^2, max |Re|), then a single
// workgroup folds those. The serial slice uses Kahan compensation on the three
// sums -- without it the L2 norm of a 1024^2 field flickers in the fourth
// decimal, which is exactly the digit the invariants panel is there to watch.

const KAHAN = `
fn kadd(acc : vec3<f32>, comp : vec3<f32>, v : vec3<f32>) -> array<vec3<f32>, 2> {
  let y = v - comp;
  let t = acc + y;
  return array<vec3<f32>, 2>(t, (t - acc) - y);
}`;

export const reduceSrc = (n) => once('reduce', [n], () => reduceRaw(n));

function reduceRaw(n) {
  return `const N : u32 = ${n}u;
const WG : u32 = ${RED_WG}u;
const GROUPS : u32 = ${RED_GROUPS}u;

@group(0) @binding(0) var<storage, read> src : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> out : array<vec4<f32>>;
var<workgroup> sh : array<vec4<f32>, WG>;
${KAHAN}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) li : vec3<u32>) {
  var acc = vec3<f32>(0.0, 0.0, 0.0);
  var comp = vec3<f32>(0.0, 0.0, 0.0);
  var mx : f32 = 0.0;
  let stride = WG * GROUPS;
  for (var i = wg.x * WG + li.x; i < N; i = i + stride) {
    let v = src[i];
    let a2 = v.x * v.x + v.y * v.y;
    let r = kadd(acc, comp, vec3<f32>(v.x, v.x * v.x, a2));
    acc = r[0]; comp = r[1];
    mx = max(mx, abs(v.x));
  }
  sh[li.x] = vec4<f32>(acc, mx);
  workgroupBarrier();
  for (var s = WG >> 1u; s > 0u; s = s >> 1u) {
    if (li.x < s) {
      let o = sh[li.x + s];
      let m = sh[li.x];
      sh[li.x] = vec4<f32>(m.xyz + o.xyz, max(m.w, o.w));
    }
    workgroupBarrier();
  }
  if (li.x == 0u) { out[wg.x] = sh[0]; }
}`;
}

export const reduceFinalSrc = () => once('reduceFinal', [], () => reduceFinalRaw());

function reduceFinalRaw() {
  return `const GROUPS : u32 = ${RED_GROUPS}u;

@group(0) @binding(0) var<storage, read> part : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out : array<vec4<f32>>;
var<workgroup> sh : array<vec4<f32>, GROUPS>;

@compute @workgroup_size(GROUPS)
fn main(@builtin(local_invocation_id) li : vec3<u32>) {
  sh[li.x] = part[li.x];
  workgroupBarrier();
  for (var s = GROUPS >> 1u; s > 0u; s = s >> 1u) {
    if (li.x < s) {
      let o = sh[li.x + s];
      let m = sh[li.x];
      sh[li.x] = vec4<f32>(m.xyz + o.xyz, max(m.w, o.w));
    }
    workgroupBarrier();
  }
  if (li.x == 0u) { out[0] = sh[0]; }
}`;
}

// [sum w*|a|^2, sum |a|^2] in .x/.y of the same vec4 slot, so the readback
// path and the staging pool are shared with the plain reduction.
export const reduceWeightedSrc = (n) => once('reduceW', [n], () => reduceWeightedRaw(n));

function reduceWeightedRaw(n) {
  return `const N : u32 = ${n}u;
const WG : u32 = ${RED_WG}u;
const GROUPS : u32 = ${RED_GROUPS}u;

@group(0) @binding(0) var<storage, read> src : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> wt : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<vec4<f32>>;
var<workgroup> sh : array<vec4<f32>, WG>;
${KAHAN}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg : vec3<u32>,
        @builtin(local_invocation_id) li : vec3<u32>) {
  var acc = vec3<f32>(0.0, 0.0, 0.0);
  var comp = vec3<f32>(0.0, 0.0, 0.0);
  let stride = WG * GROUPS;
  for (var i = wg.x * WG + li.x; i < N; i = i + stride) {
    let v = src[i];
    let a2 = v.x * v.x + v.y * v.y;
    let r = kadd(acc, comp, vec3<f32>(wt[i] * a2, a2, 0.0));
    acc = r[0]; comp = r[1];
  }
  sh[li.x] = vec4<f32>(acc, 0.0);
  workgroupBarrier();
  for (var s = WG >> 1u; s > 0u; s = s >> 1u) {
    if (li.x < s) { sh[li.x] = sh[li.x] + sh[li.x + s]; }
    workgroupBarrier();
  }
  if (li.x == 0u) { out[wg.x] = sh[0]; }
}`;
}

export const reduceWeightedFinalSrc = () => once('reduceWFinal', [], () => reduceWeightedFinalRaw());

function reduceWeightedFinalRaw() {
  return `const GROUPS : u32 = ${RED_GROUPS}u;

@group(0) @binding(0) var<storage, read> part : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out : array<vec4<f32>>;
var<workgroup> sh : array<vec4<f32>, GROUPS>;

@compute @workgroup_size(GROUPS)
fn main(@builtin(local_invocation_id) li : vec3<u32>) {
  sh[li.x] = part[li.x];
  workgroupBarrier();
  for (var s = GROUPS >> 1u; s > 0u; s = s >> 1u) {
    if (li.x < s) { sh[li.x] = sh[li.x] + sh[li.x + s]; }
    workgroupBarrier();
  }
  if (li.x == 0u) { out[0] = sh[0]; }
}`;
}

// --- display ---------------------------------------------------------------
//
// Same picture as render/renderer.js -- colour ramp times Lambertian shading
// off the field gradient -- but evaluated per screen pixel instead of per cell,
// so a 1024^2 field is not funnelled through a 1M-iteration JS loop every
// frame. The field is sampled bilinearly (the CPU path lets the browser
// interpolate the finished colours instead; on a hump the difference is not
// visible, on the 2/3 cutoff ripples it is slightly cleaner here).

export const displaySrc = (nx, ny) => once('display', [nx, ny], () => displayRaw(nx, ny));

function displayRaw(nx, ny) {
  return `const NX : i32 = ${nx};
const NY : i32 = ${ny};

struct U {
  scale : f32,
  relief : f32,
  diverging : f32,
  vw : f32,
  vh : f32,
  p0 : f32,
  p1 : f32,
  p2 : f32,
};
struct Lut { c : array<vec4<f32>, 256> };

@group(0) @binding(0) var<storage, read> fld : array<vec2<f32>>;
@group(0) @binding(1) var<uniform> u : U;
@group(0) @binding(2) var<uniform> lut : Lut;

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}

fn at(ix : i32, iy : i32) -> f32 {
  let x = ((ix % NX) + NX) % NX;
  let y = ((iy % NY) + NY) % NY;
  return fld[y * NX + x].x;
}

fn sample(gx : f32, gy : f32) -> f32 {
  let x0 = i32(floor(gx));
  let y0 = i32(floor(gy));
  let fx = gx - f32(x0);
  let fy = gy - f32(y0);
  let a = mix(at(x0, y0), at(x0 + 1, y0), fx);
  let b = mix(at(x0, y0 + 1), at(x0 + 1, y0 + 1), fx);
  return mix(a, b, fy);
}

fn ramp(t : f32) -> vec3<f32> {
  let c = clamp(t, 0.0, 1.0) * 255.0;
  let i = i32(floor(c));
  let f = c - f32(i);
  let lo = lut.c[i].rgb;
  let hi = lut.c[min(i + 1, 255)].rgb;
  return mix(lo, hi, f);
}

@fragment
fn fs(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
  // pixel centre -> continuous cell coordinate, half-cell offset so cell
  // centres land on cell centres exactly as putImageData + bilinear would
  let gx = (pos.x / u.vw) * f32(NX) - 0.5;
  let gy = (pos.y / u.vh) * f32(NY) - 0.5;

  let v = sample(gx, gy);
  let t = select(v / u.scale, 0.5 + 0.5 * v / u.scale, u.diverging > 0.5);
  var rgb = ramp(t);

  if (u.relief > 0.0) {
    let g = u.relief * (2.2 / u.scale);
    let dx = (sample(gx + 1.0, gy) - sample(gx - 1.0, gy)) * 0.5 * g;
    let dy = (sample(gx, gy + 1.0) - sample(gx, gy - 1.0)) * 0.5 * g;
    let inv = inverseSqrt(dx * dx + dy * dy + 1.0);
    let lam = (-dx * (-0.45) - dy * (-0.62) + 0.64) * inv;
    rgb = rgb * (0.62 + 0.72 * max(lam, 0.0));
  }
  return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}`;
}
