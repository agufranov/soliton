// Display path for the WebGPU backend.
//
// The CPU renderer walks the field once per cell in JavaScript. That is fine at
// 128^2 and it is the whole frame at 1024^2 - a million iterations of colour
// lookup plus lighting, plus a putImageData of 4 MB, every frame. It would have
// made the GPU backend look slow for reasons that have nothing to do with the
// solver.
//
// So in GPU mode the field never leaves the device: a fullscreen fragment
// shader reads the same storage buffer the solver just wrote and evaluates the
// identical colour ramp and Lambertian shading per screen pixel. The only
// number that crosses back is the autoscale peak, which comes from the cached
// reduction and may be a frame old - it is smoothed over ~16 frames anyway.

import { COLORMAPS } from './colormaps.js';
import { displaySrc } from '../core/webgpuShaders.js';

export class GpuRenderer {
  constructor(canvas, ctx) {
    this.canvas = canvas;
    this.device = ctx.device;
    this.context = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.scaleSmoothed = 1;
    this.relief = 1;
    this.pipelines = new Map();   // grid size -> pipeline
    this.luts = new Map();        // colormap kind -> uniform buffer
    this.binds = new Map();
    this.uniform = this.device.createBuffer({
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniformData = new Float32Array(8);
  }

  resizeFor(grid) {
    this.grid = grid;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(this.canvas.clientWidth * dpr) || 820;
    const h = Math.round(this.canvas.clientHeight * dpr) || 820;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
    }
  }

  _pipeline() {
    const key = `${this.grid.nx}x${this.grid.ny}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code: displaySrc(this.grid.nx, this.grid.ny) });
      p = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  // The 256x3 byte ramps become 256 vec4 in a uniform buffer, normalised.
  _lut(kind) {
    let b = this.luts.get(kind);
    if (!b) {
      const src = COLORMAPS[kind] || COLORMAPS.height;
      const data = new Float32Array(256 * 4);
      for (let i = 0; i < 256; i++) {
        data[i * 4] = src[i * 3] / 255;
        data[i * 4 + 1] = src[i * 3 + 1] / 255;
        data[i * 4 + 2] = src[i * 3 + 2] / 255;
        data[i * 4 + 3] = 1;
      }
      b = this.device.createBuffer({
        size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(b, 0, data);
      this.luts.set(kind, b);
    }
    return b;
  }

  // `field` is a backend field whose .re holds the scalar to draw; `hi` is
  // max|field| from the backend reduction.
  draw(field, kind, hi) {
    const pipeline = this._pipeline();
    const lut = this._lut(kind);

    let peak = kind === 'phase' ? 2 * Math.PI : Math.max(hi || 0, 1e-6);
    if (!Number.isFinite(peak)) peak = this.scaleSmoothed;
    this.scaleSmoothed += (peak - this.scaleSmoothed) * 0.06;
    const scale = Math.max(this.scaleSmoothed, peak * 0.55);

    const u = this.uniformData;
    u[0] = scale;
    u[1] = this.relief;
    u[2] = kind !== 'intensity' ? 1 : 0;
    u[3] = this.canvas.width;
    u[4] = this.canvas.height;
    this.device.queue.writeBuffer(this.uniform, 0, u);

    const key = `${this.grid.nx}:${kind}:${field.buf.__id}`;
    let bind = this.binds.get(key);
    if (!bind) {
      bind = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: field.buf } },
          { binding: 1, resource: { buffer: this.uniform } },
          { binding: 2, resource: { buffer: lut } },
        ],
      });
      this.binds.set(key, bind);
    }

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    return scale;
  }

  // Buffers are recreated with every backend; their bind groups must go too.
  invalidate() { this.binds.clear(); }
}
