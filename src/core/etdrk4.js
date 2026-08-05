// Exponential time differencing, 4th order Runge-Kutta (Cox-Matthews),
// with the phi-functions evaluated by the Kassam-Trefethen contour mean so
// they stay accurate as L -> 0 without catastrophic cancellation.
//
// Solves  v_t = L v + N(v)  in Fourier space, with L diagonal (purely
// imaginary for every dispersive model here, so |exp(hL)| == 1 and the
// scheme is stable at any h the nonlinear term tolerates).

const M = 32;        // points on the unit contour
const DIRECT = 2;    // |hL| above which the closed forms are used directly

// L is given as a real array `Limag` holding the imaginary part of L(k)
// (all our operators are pure imaginary). Returns the six coefficient fields,
// built on the host and then handed to the backend, which is what lets the
// GPU backend keep no host mirror of them.
//
// The contour mean exists to avoid catastrophic cancellation as hL -> 0. Away
// from the origin there is nothing to cancel: the phi-functions are entire, so
// the mean over the unit circle around hL is exactly their value at hL, and the
// closed forms evaluate it to full precision once |hL| >= 2. Most modes of a
// KP or ZK operator are far out (|hL| ~ 9 at the 128^2 cutoff), so taking the
// short path there turns a multi-second rebuild at 1024^2 into a brief one.
// Checked against the all-contour version: tools/verify.mjs is unchanged to
// every printed digit.
export function etdCoefficients(backend, Limag, h) {
  const n = Limag.length;
  const mk = () => ({ re: new Float64Array(n), im: new Float64Array(n) });
  const E = mk(), E2 = mk(), Q = mk(), f1 = mk(), f2 = mk(), f3 = mk();

  // contour points r_j = exp(i*pi*(j+0.5)/M)
  const rr = new Float64Array(M), ri = new Float64Array(M);
  for (let j = 0; j < M; j++) {
    const th = (Math.PI * (j + 0.5)) / M;
    rr[j] = Math.cos(th); ri[j] = Math.sin(th);
  }

  for (let p = 0; p < n; p++) {
    const li = Limag[p] * h;                  // hL = i*li
    E.re[p] = Math.cos(li); E.im[p] = Math.sin(li);
    E2.re[p] = Math.cos(li / 2); E2.im[p] = Math.sin(li / 2);

    const far = Math.abs(li) >= DIRECT;
    const pts = far ? 1 : M;
    let qr = 0, qi = 0, a1r = 0, a1i = 0, a2r = 0, a2i = 0, a3r = 0, a3i = 0;
    for (let j = 0; j < pts; j++) {
      // z = hL + r_j on the contour, z = hL when the closed forms are safe
      const zr = far ? 0 : rr[j], zi = far ? li : li + ri[j];
      const z2r = zr * zr - zi * zi, z2i = 2 * zr * zi;
      const z3r = z2r * zr - z2i * zi, z3i = z2r * zi + z2i * zr;
      const ez = Math.exp(zr);
      const er = ez * Math.cos(zi), ei = ez * Math.sin(zi);
      const eh = Math.exp(zr / 2);
      const hr = eh * Math.cos(zi / 2), hi = eh * Math.sin(zi / 2);

      // Q += (exp(z/2) - 1)/z
      let nr = hr - 1, ni = hi;
      let d = zr * zr + zi * zi;
      qr += (nr * zr + ni * zi) / d; qi += (ni * zr - nr * zi) / d;

      const d3 = z3r * z3r + z3i * z3i;

      // f1 += (-4 - z + e^z (4 - 3z + z^2)) / z^3
      let tr = 4 - 3 * zr + z2r, ti = -3 * zi + z2i;
      nr = -4 - zr + (er * tr - ei * ti);
      ni = -zi + (er * ti + ei * tr);
      a1r += (nr * z3r + ni * z3i) / d3; a1i += (ni * z3r - nr * z3i) / d3;

      // f2 += (2 + z + e^z (-2 + z)) / z^3
      tr = -2 + zr; ti = zi;
      nr = 2 + zr + (er * tr - ei * ti);
      ni = zi + (er * ti + ei * tr);
      a2r += (nr * z3r + ni * z3i) / d3; a2i += (ni * z3r - nr * z3i) / d3;

      // f3 += (-4 - 3z - z^2 + e^z (4 - z)) / z^3
      tr = 4 - zr; ti = -zi;
      nr = -4 - 3 * zr - z2r + (er * tr - ei * ti);
      ni = -3 * zi - z2i + (er * ti + ei * tr);
      a3r += (nr * z3r + ni * z3i) / d3; a3i += (ni * z3r - nr * z3i) / d3;
    }
    const s = h / pts;
    Q.re[p] = s * qr; Q.im[p] = s * qi;
    f1.re[p] = s * a1r; f1.im[p] = s * a1i;
    f2.re[p] = s * a2r; f2.im[p] = s * a2i;
    f3.re[p] = s * a3r; f3.im[p] = s * a3i;
  }
  return {
    E: backend.adopt(E), E2: backend.adopt(E2), Q: backend.adopt(Q),
    f1: backend.adopt(f1), f2: backend.adopt(f2), f3: backend.adopt(f3),
  };
}

export class Etdrk4 {
  // nonlinear(vIn, nOut) must write N(v) into nOut, both in Fourier space.
  constructor(backend, Limag, h, nonlinear) {
    this.backend = backend;
    this.nonlinear = nonlinear;
    this.h = h;
    this.c = etdCoefficients(backend, Limag, h);
    this.Nv = backend.alloc(); this.Na = backend.alloc();
    this.Nb = backend.alloc(); this.Nc = backend.alloc();
    this.a = backend.alloc(); this.b = backend.alloc(); this.cc = backend.alloc();
  }

  step(v) {
    const b = this.backend, c = this.c;
    this.nonlinear(v, this.Nv);
    b.etdStage(this.a, c.E2, v, c.Q, this.Nv);
    this.nonlinear(this.a, this.Na);
    b.etdStage(this.b, c.E2, v, c.Q, this.Na);
    this.nonlinear(this.b, this.Nb);
    b.etdStageC(this.cc, c.E2, this.a, c.Q, this.Nb, this.Nv);
    this.nonlinear(this.cc, this.Nc);
    b.etdFinal(v, c.E, this.Nv, c.f1, this.Na, this.Nb, c.f2, this.Nc, c.f3);
  }
}
