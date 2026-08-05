// WebGPU compute backend: the same named kernels as CpuBackend, on the device.
//
// Read core/backend.js first -- it defines the contract. What is specific here:
//
//  * A field is one GPUBuffer of vec2<f32> (re, im interleaved). Host mirrors
//    exist only where something explicitly asks for one (`snapshot`).
//  * Everything a frame does is recorded into ONE command encoder with ONE open
//    compute pass and submitted by `flush()`. Per-dispatch submits cost more
//    than the dispatches; WebGPU already inserts the barrier between dispatches
//    inside a pass, so batching is free correctness-wise.
//  * `snapshot` and the reductions are asynchronous. They return the most
//    recently completed readback, which lags the simulation by a frame or two.
//    They feed the invariants panel and the blow-up guard, and nothing else --
//    a synchronous readback would drain the pipeline every frame and would
//    make the GPU path slower than the CPU one at 128^2.
//  * Anything that pushes host bytes at the device flushes first. queue
//    writeBuffer executes at call time, so without that flush it would land
//    *before* the dispatches already recorded in the open encoder.
//
// f32 vs f64: this is the real numerical difference from the CPU backend. See
// "Точность на GPU" in CLAUDE.md for the measured drift.

import * as S from './webgpuShaders.js';

let nextId = 1;
const tag = (o) => { o.__id = nextId++; return o; };

// Ask the browser for a device. Returns null when WebGPU is unavailable, so
// the caller can leave the GPU option disabled instead of throwing.
export async function requestGpuContext() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    // Timestamps are what let the frame loop know the real cost of a step.
    // Wall-clock timing cannot: queue submission is asynchronous, so a
    // stopwatch measures either nothing (before the work runs) or the round
    // trip of the completion signal, which on this machine is ~18 ms whatever
    // the queue holds. Optional - the loop has a fallback.
    const timing = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice(timing ? { requiredFeatures: ['timestamp-query'] } : {});
    const info = adapter.info || {};
    const label = [info.vendor, info.architecture].filter(Boolean).join(' ') || 'webgpu';
    return { adapter, device, label, limits: device.limits, timing };
  } catch {
    return null;
  }
}

export class WebGpuBackend {
  constructor(ctx, grid) {
    if (grid.nx !== grid.ny) {
      throw new Error('WebGpuBackend: the FFT transpose path assumes a square grid');
    }
    this.ctx = ctx;
    this.device = ctx.device;
    this.grid = grid;
    this.n = grid.n;
    this.name = 'gpu-f32';
    this.gpu = true;

    this.pipelines = new Map();     // wgsl source -> GPUComputePipeline
    this.binds = new Map();         // pipeline+buffers -> GPUBindGroup
    this.scalars = new Map();       // f32 value -> 16-byte uniform buffer
    this.tables = new WeakMap();    // Float64Array -> array<f32> storage buffer
    this.owned = [];

    this.scratch = this._field();   // FFT transpose ping-pong
    this.hostBuf = this._buffer(this.n * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.interleave = new Float32Array(this.n * 2);

    this.encoder = null;
    this.pass = null;
    this.pendingCopies = [];

    // GPU-side timing of the compute pass, when the device supports it.
    this.timing = !!ctx.timing;
    if (this.timing) {
      this.querySet = this.device.createQuerySet({ type: 'timestamp', count: 2 });
      this.queryResolve = this._buffer(16,
        GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      this.queryStage = this._buffer(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      this.queryBusy = false;
    }
    this.reductions = new WeakMap();
    this.weighted = new WeakMap();
    this.snaps = new WeakMap();
    this.dispatches = 0;            // per-flush counter, for the stats panel
  }

  // -- device plumbing -----------------------------------------------------
  _buffer(size, usage) {
    const b = tag(this.device.createBuffer({ size, usage }));
    this.owned.push(b);
    return b;
  }

  _field() {
    return this._buffer(this.n * 8,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
  }

  _pipeline(src) {
    let p = this.pipelines.get(src);
    if (!p) {
      const module = this.device.createShaderModule({ code: src });
      p = tag(this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } }));
      this.pipelines.set(src, p);
    }
    return p;
  }

  _bindGroup(pipeline, bufs) {
    let key = String(pipeline.__id);
    for (const b of bufs) key += ':' + b.__id;
    let bg = this.binds.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      this.binds.set(key, bg);
    }
    return bg;
  }

  _pass() {
    if (!this.encoder) this.encoder = this.device.createCommandEncoder();
    if (!this.pass) {
      // Timestamp the pass only when the previous reading has been collected;
      // one in flight is plenty for a frame-rate control loop.
      const stamp = this.timing && !this.queryBusy;
      this.pass = this.encoder.beginComputePass(stamp ? {
        timestampWrites: {
          querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
        },
      } : undefined);
      this.passStamped = stamp;
    }
    return this.pass;
  }

  _run(src, bufs, groups) {
    const p = this._pipeline(src);
    const pass = this._pass();
    pass.setPipeline(p);
    pass.setBindGroup(0, this._bindGroup(p, bufs));
    pass.dispatchWorkgroups(groups[0], groups[1] || 1, groups[2] || 1);
    this.dispatches++;
  }

  // one thread per cell
  _elem(src, bufs) { this._run(src, bufs, [Math.ceil(this.n / S.ELEM_WG)]); }

  _scalar(v) {
    let b = this.scalars.get(v);
    if (!b) {
      b = this._buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(b, 0, new Float32Array([v, 0, 0, 0]));
      this.scalars.set(v, b);
    }
    return b;
  }

  // Real tables are cached by object identity, so they must not be mutated
  // after first use -- see rule 1 in core/backend.js.
  //
  // They are also TRANSPOSED on upload, because every table a kernel takes is
  // indexed in spectral space and this backend keeps the spectrum transposed
  // (see _fft2). That is a total rule, not a case-by-case one: mask, kx, the
  // NLS phase table and the four sine-Gordon rotation tables are all spectral,
  // and nothing physical-space ever arrives as a table.
  _table(arr) {
    let b = this.tables.get(arr);
    if (!b) {
      const nx = this.grid.nx, ny = this.grid.ny;
      const t = new Float32Array(this.n);
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) t[i * ny + j] = arr[j * nx + i];
      }
      b = this._buffer(this.n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(b, 0, t);
      this.tables.set(arr, b);
    }
    return b;
  }

  // -- allocation ----------------------------------------------------------
  alloc() { return { n: this.n, buf: this._field() }; }
  allocReal() { return new Float64Array(this.n); }

  // The only thing anyone adopts is the set of ETDRK4 coefficient fields, and
  // those live in spectral space -- so they transpose, exactly like _table.
  adopt(pair) {
    const f = this.alloc();
    const nx = this.grid.nx, ny = this.grid.ny;
    const re = new Float64Array(this.n), im = new Float64Array(this.n);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        re[i * ny + j] = pair.re[j * nx + i];
        im[i * ny + j] = pair.im[j * nx + i];
      }
    }
    this._writeField(f.buf, re, im);
    return f;
  }

  pointwise(spec) {
    if (!spec || typeof spec.wgsl !== 'string') {
      throw new Error('WebGpuBackend.pointwise: model did not supply a wgsl expression');
    }
    return spec;
  }

  // -- host transfers ------------------------------------------------------
  _writeField(buf, re, im) {
    const t = this.interleave;
    if (im) { for (let i = 0; i < this.n; i++) { t[2 * i] = re[i]; t[2 * i + 1] = im[i]; } }
    else { for (let i = 0; i < this.n; i++) { t[2 * i] = re[i]; t[2 * i + 1] = 0; } }
    this.flush();                     // ordering: see the header note
    this.device.queue.writeBuffer(buf, 0, t);
  }

  setHostReal(f, arr) { this._writeField(f.buf, arr, null); }

  addHostReal(f, arr) {
    this._writeField(this.hostBuf, arr, null);
    this._elem(S.elementwise.addRealFrom(this.n), [f.buf, this.hostBuf]);
    this.flush();                     // hostBuf is shared; do not let two spawns race
  }

  addHostComplex(f, re, im) {
    this._writeField(this.hostBuf, re, im);
    this._elem(S.elementwise.addInto(this.n), [f.buf, this.hostBuf]);
    this.flush();
  }

  // Latest completed download of `f`. One to two frames stale by design.
  snapshot(f) {
    let rec = this.snaps.get(f);
    if (!rec) {
      rec = {
        re: new Float64Array(this.n), im: new Float64Array(this.n),
        stage: this._buffer(this.n * 8, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
        busy: false,
      };
      this.snaps.set(f, rec);
    }
    if (!rec.busy) {
      rec.busy = true;
      // Resolved at flush time, not now: _fft2 swaps a field's buffer with the
      // scratch one, so `f.buf` here may not be the buffer that holds the data
      // by the time the copy is actually encoded.
      this.pendingCopies.push({
        srcOf: () => f.buf, dst: rec.stage, size: this.n * 8, rec,
        take: (a) => {
          for (let i = 0; i < this.n; i++) { rec.re[i] = a[2 * i]; rec.im[i] = a[2 * i + 1]; }
        },
      });
    }
    return rec;
  }

  // -- transforms ----------------------------------------------------------
  //
  // rows -> transpose -> rows. THREE passes, not four: the result is left in
  // transposed layout instead of being transposed back.
  //
  // Writing X[l][m] at index m*N+l is just as good a spectrum as writing it at
  // l*N+m, because every operation performed on a spectrum here is pointwise.
  // What has to match is the k tables and the ETDRK4 coefficients, and those go
  // through _table and adopt, which transpose. The inverse undoes it: rows ->
  // transpose -> rows takes a transposed spectrum back to a normal field.
  //
  // The saved pass is a quarter of the FFT traffic, and a step is memory bound,
  // so it is a quarter of the step. The 1/n of an inverse rides along on the
  // transpose rather than costing its own sweep.
  //
  // `opts.src` reads the first pass from another buffer and `opts.load`
  // transforms it on the way in; both let a caller drop a separate pass.
  _fft2(f, inverse, opts = {}) {
    const n = this.grid.nx, t = n / S.TILE;
    const first = S.fftRowSrc(n, inverse, opts);
    const rest = S.fftRowSrc(n, inverse);
    const tr = S.transposeSrc(n, n, inverse ? 1 / this.n : 1);
    this._run(first, opts.src ? [f.buf, opts.src.buf] : [f.buf], [n]);
    this._run(tr, [f.buf, this.scratch], [t, t]);
    this._run(rest, [this.scratch], [n]);
    // The answer is in the scratch buffer; swap rather than copy it back.
    const held = f.buf;
    f.buf = this.scratch;
    this.scratch = held;
  }

  forward(f) { this._fft2(f, false); }
  inverse(f) { this._fft2(f, true); }

  // inverse(dst <- src) without the caller having to copy first.
  inverseInto(dst, src) { this._fft2(dst, true, { src }); }

  // forward(f) with f <- (Re f)^2 applied first, in physical space.
  forwardSqrReal(f) { this._fft2(f, false, { load: 'vec2<f32>(v.x * v.x, 0.0)' }); }

  // -- elementwise kernels -------------------------------------------------
  zero(a) { this._elem(S.elementwise.zero(this.n), [a.buf]); }

  copy(dst, src) { this._elem(S.elementwise.copy(this.n), [dst.buf, src.buf]); }

  scale(a, s) { this._elem(S.elementwise.scale(this.n), [a.buf, this._scalar(s)]); }

  addScaled(dst, a, b, s) {
    this._elem(S.elementwise.addScaled(this.n), [dst.buf, a.buf, b.buf, this._scalar(s)]);
  }

  axpy(y, x, s) { this._elem(S.elementwise.axpy(this.n), [y.buf, x.buf, this._scalar(s)]); }

  maskReal(a, m) { this._elem(S.elementwise.maskReal(this.n), [a.buf, this._table(m)]); }

  sqrReal(dst, src) {
    if (dst.buf === src.buf) this._elem(S.elementwise.sqrRealInPlace(this.n), [dst.buf]);
    else this._elem(S.elementwise.sqrReal(this.n), [dst.buf, src.buf]);
  }

  mulIkxScaled(dst, src, kx, s) {
    this._elem(S.elementwise.mulIkxScaled(this.n),
      [dst.buf, src.buf, this._table(kx), this._scalar(s)]);
  }

  mulIkxMasked(dst, src, kx, mask, s) {
    this._elem(S.elementwise.mulIkxMasked(this.n),
      [dst.buf, src.buf, this._table(kx), this._table(mask), this._scalar(s)]);
  }

  rotateByPhase(a, phase) {
    this._elem(S.elementwise.rotateByPhase(this.n), [a.buf, this._table(phase)]);
  }

  mapReal(dst, src, f) {
    if (dst.buf === src.buf) this._elem(S.pointwiseSrc.mapRealInPlace(this.n, f.wgsl), [dst.buf]);
    else this._elem(S.pointwiseSrc.mapReal(this.n, f.wgsl), [dst.buf, src.buf]);
  }

  abs2Map(dst, src, f) {
    this._elem(S.pointwiseSrc.abs2Map(this.n, f.wgsl), [dst.buf, src.buf]);
  }

  nlsPhaseStep(a, dt, gain) {
    this._elem(S.pointwiseSrc.nlsPhaseStep(this.n, gain.wgsl), [a.buf, this._scalar(dt)]);
  }

  etdStage(out, E2, v, Q, N) {
    this._elem(S.elementwise.etdStage(this.n), [out.buf, E2.buf, v.buf, Q.buf, N.buf]);
  }

  etdStageC(out, E2, a, Q, Nb, Nv) {
    this._elem(S.elementwise.etdStageC(this.n), [out.buf, E2.buf, a.buf, Q.buf, Nb.buf, Nv.buf]);
  }

  // Split in two so no dispatch needs more than the 8 storage bindings every
  // WebGPU implementation guarantees; the arithmetic order is unchanged.
  etdFinal(v, E, Nv, f1, Na, Nb, f2, Nc, f3) {
    this._elem(S.elementwise.etdFinalA(this.n), [v.buf, E.buf, f1.buf, Nv.buf]);
    this._elem(S.elementwise.etdFinalB(this.n), [v.buf, f2.buf, Na.buf, Nb.buf, f3.buf, Nc.buf]);
  }

  waveRotate(u, w, cosw, sinw, invOmega, omega) {
    this._elem(S.elementwise.waveRotate(this.n),
      [u.buf, w.buf, this._table(cosw), this._table(sinw),
        this._table(invOmega), this._table(omega)]);
  }

  // -- reductions (diagnostics only) ---------------------------------------
  // One dispatch pair produces all four scalars; whichever accessor is called
  // first in a frame schedules it, the rest read the same cached vector.
  _reduce(f) {
    let rec = this.reductions.get(f);
    if (!rec) {
      rec = {
        part: this._buffer(S.RED_GROUPS * 16, GPUBufferUsage.STORAGE),
        res: this._buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
        stage: this._buffer(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
        v: new Float32Array(4), busy: false,
      };
      this.reductions.set(f, rec);
    }
    if (!rec.busy) {
      this._run(S.reduceSrc(this.n), [f.buf, rec.part], [S.RED_GROUPS]);
      this._run(S.reduceFinalSrc(), [rec.part, rec.res], [1]);
      rec.busy = true;
      this.pendingCopies.push({ srcOf: () => rec.res, dst: rec.stage, size: 16, rec, take: (a) => rec.v.set(a) });
    }
    return rec.v;
  }

  sumRe(a) { return this._reduce(a)[0]; }
  sumSqRe(a) { return this._reduce(a)[1]; }
  sumAbs2(a) { return this._reduce(a)[2]; }

  // max() ignores NaN, so a blown-up field would still report a finite peak.
  // The sum of squares does propagate it, and that is what the guard needs.
  maxAbsRe(a) { const v = this._reduce(a); return Number.isFinite(v[1]) ? v[3] : NaN; }

  weightedSumAbs2(a, w) {
    let rec = this.weighted.get(a);
    if (!rec) {
      rec = {
        part: this._buffer(S.RED_GROUPS * 16, GPUBufferUsage.STORAGE),
        res: this._buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
        stage: this._buffer(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
        v: new Float32Array(4), busy: false,
      };
      this.weighted.set(a, rec);
    }
    if (!rec.busy) {
      this._run(S.reduceWeightedSrc(this.n), [a.buf, this._table(w), rec.part], [S.RED_GROUPS]);
      this._run(S.reduceWeightedFinalSrc(), [rec.part, rec.res], [1]);
      rec.busy = true;
      this.pendingCopies.push({ srcOf: () => rec.res, dst: rec.stage, size: 16, rec, take: (x) => rec.v.set(x) });
    }
    return [rec.v[0], rec.v[1]];
  }

  // -- frame boundary ------------------------------------------------------
  // `onTiming(ms)` is called later with the GPU-side duration of this flush's
  // compute pass, if the device supports timestamps and no reading is pending.
  flush(onTiming) {
    if (!this.encoder && this.pendingCopies.length === 0) return;
    let stamped = false;
    if (this.pass) {
      this.pass.end();
      this.pass = null;
      stamped = this.passStamped;
    }
    if (!this.encoder) this.encoder = this.device.createCommandEncoder();
    if (stamped) {
      this.encoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolve, 0);
      this.encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryStage, 0, 16);
      this.queryBusy = true;
    }
    const copies = this.pendingCopies;
    this.pendingCopies = [];
    for (const c of copies) this.encoder.copyBufferToBuffer(c.srcOf(), 0, c.dst, 0, c.size);
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;

    if (stamped) {
      this.queryStage.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(this.queryStage.getMappedRange());
        const ns = Number(t[1] - t[0]);
        this.queryStage.unmap();
        this.queryBusy = false;
        if (onTiming && ns > 0) onTiming(ns / 1e6);
      }).catch(() => { this.queryBusy = false; });
    }

    for (const c of copies) {
      c.dst.mapAsync(GPUMapMode.READ).then(() => {
        c.take(new Float32Array(c.dst.getMappedRange()));
        c.dst.unmap();
        c.rec.busy = false;
      }).catch(() => { c.rec.busy = false; });
    }
  }

  dispose() {
    if (this.pass) { this.pass.end(); this.pass = null; }
    this.encoder = null;
    this.pendingCopies = [];
    for (const b of this.owned) { try { b.destroy(); } catch { /* already gone */ } }
    this.owned = [];
    this.pipelines.clear();
    this.binds.clear();
    this.scalars.clear();
  }
}
