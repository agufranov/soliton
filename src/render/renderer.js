// Field renderer: colour map plus fake directional lighting derived from the
// local gradient, so a hump actually reads as a hump on a surface rather than
// as a flat blob of colour.
//
// The field is painted at grid resolution into an ImageData and then blown up
// with the browser's bilinear scaler - cheap, and smoother than nearest
// neighbour at 128^2.

import { COLORMAPS } from './colormaps.js';

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.buffer = document.createElement('canvas');
    this.bctx = this.buffer.getContext('2d');
    this.scaleSmoothed = 1;
    this.relief = 1;
  }

  resizeFor(grid) {
    if (this.buffer.width !== grid.nx || this.buffer.height !== grid.ny) {
      this.buffer.width = grid.nx;
      this.buffer.height = grid.ny;
      this.image = this.bctx.createImageData(grid.nx, grid.ny);
    }
    this.grid = grid;
  }

  // kind: 'height' | 'intensity' | 'phase'
  draw(field, kind, opts = {}) {
    const grid = this.grid;
    const nx = grid.nx, ny = grid.ny;
    const lut = COLORMAPS[kind] || COLORMAPS.height;
    const px = this.image.data;

    // --- dynamic range ----------------------------------------------------
    let hi = 0;
    for (let i = 0; i < field.length; i++) { const a = Math.abs(field[i]); if (a > hi) hi = a; }
    if (kind === 'phase') hi = 2 * Math.PI;
    hi = Math.max(hi, 1e-6);
    // ease the scale so the picture does not flicker on every collision spike
    const k = opts.autoscale === false ? 1 : 0.06;
    this.scaleSmoothed += (hi - this.scaleSmoothed) * k;
    const scale = Math.max(this.scaleSmoothed, hi * 0.55);

    const diverging = kind !== 'intensity';
    const relief = this.relief;
    // gradient -> normal, in units where one cell is one unit of length
    const gscale = relief * (2.2 / scale);
    const lx = -0.45, ly = -0.62, lz = 0.64;

    for (let j = 0; j < ny; j++) {
      const jm = ((j - 1) + ny) % ny, jp = (j + 1) % ny;
      for (let i = 0; i < nx; i++) {
        const p = j * nx + i;
        const im = ((i - 1) + nx) % nx, ip = (i + 1) % nx;
        const v = field[p];

        const t = diverging
          ? 0.5 + (0.5 * v) / scale
          : v / scale;
        let idx = (t * 255) | 0;
        idx = idx < 0 ? 0 : idx > 255 ? 255 : idx;

        let shade = 1;
        if (relief > 0) {
          const gx = (field[j * nx + ip] - field[j * nx + im]) * 0.5 * gscale;
          const gy = (field[jp * nx + i] - field[jm * nx + i]) * 0.5 * gscale;
          const inv = 1 / Math.sqrt(gx * gx + gy * gy + 1);
          const lam = (-gx * lx - gy * ly + lz) * inv;
          shade = 0.62 + 0.72 * (lam > 0 ? lam : 0);
        }

        const o = p * 4, c = idx * 3;
        px[o] = lut[c] * shade;
        px[o + 1] = lut[c + 1] * shade;
        px[o + 2] = lut[c + 2] * shade;
        px[o + 3] = 255;
      }
    }

    this.bctx.putImageData(this.image, 0, 0);

    const cv = this.canvas, ctx = this.ctx;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(this.buffer, 0, 0, cv.width, cv.height);
    return scale;
  }
}

// The drag vector currently being aimed. It lives on its own transparent
// canvas stacked over the field, because the field canvas underneath is either
// a 2D one (CPU) or a WebGPU one (GPU) and a canvas only ever gets one context
// type in its life.
export class AimOverlay {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  resize() {
    const w = Math.round(this.canvas.clientWidth) || 820;
    const h = Math.round(this.canvas.clientHeight) || 820;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
    }
  }

  clear() { this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); }

  draw(from, to) {
    const ctx = this.ctx;
    this.clear();
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    const a = Math.atan2(to.y - from.y, to.x - from.x);
    const h = 9;
    ctx.beginPath();
    ctx.moveTo(to.x, to.y);
    ctx.lineTo(to.x - h * Math.cos(a - 0.4), to.y - h * Math.sin(a - 0.4));
    ctx.lineTo(to.x - h * Math.cos(a + 0.4), to.y - h * Math.sin(a + 0.4));
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(from.x, from.y, 4, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }
}
