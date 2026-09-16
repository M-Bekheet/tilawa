export interface TensorLike {
  data: Float32Array | BigInt64Array | Int32Array;
  dims: readonly number[];
  type: string;
}

export interface OrtSessionLike {
  run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>>;
}

export interface OrtLike {
  InferenceSession: {
    create(
      model: Uint8Array | ArrayBuffer,
      options?: {
        executionProviders?: string[];
        graphOptimizationLevel?: string;
      },
    ): Promise<OrtSessionLike>;
  };
  Tensor: new (
    type: string,
    data: Float32Array | BigInt64Array | Int32Array,
    dims: readonly number[],
  ) => TensorLike;
}

export interface ZipformerIoInput {
  name: string;
  dims: number[];
  dtype: string;
}

export interface ZipformerIo {
  T: number;
  hop: number;
  featureDim: number;
  vocabSize: number;
  inputs: ZipformerIoInput[];
  outputs?: ZipformerIoInput[];
}

function numel(dims: readonly number[]): number {
  let n = 1;
  for (const d of dims) n *= d;
  return n;
}

export class ZipformerRunner {
  readonly io: ZipformerIo;
  leftoverFrames = 0;
  processedLens = 0n;
  private readonly session: OrtSessionLike;
  private readonly Tensor: OrtLike["Tensor"];
  private readonly buffer: Float32Array[] = [];
  private states = new Map<string, TensorLike>();
  private readonly stateNames: string[] = [];

  private constructor(session: OrtSessionLike, io: ZipformerIo, Tensor: OrtLike["Tensor"]) {
    this.session = session;
    this.io = io;
    this.Tensor = Tensor;
    for (const inp of io.inputs) {
      if (inp.name === "x") continue;
      this.stateNames.push(inp.name);
    }
    this.initStates();
  }

  static async create(
    ort: unknown,
    model: Uint8Array | ArrayBuffer,
    io: ZipformerIo,
    executionProviders: string[] = ["cpu"],
  ): Promise<ZipformerRunner> {
    const runtime = ort as OrtLike;
    const session = await runtime.InferenceSession.create(model, {
      executionProviders,
      graphOptimizationLevel: "all",
    });
    const runner = new ZipformerRunner(session, io, runtime.Tensor);
    return runner;
  }

  static fromSession(
    session: OrtSessionLike,
    io: ZipformerIo,
    Tensor: OrtLike["Tensor"],
  ): ZipformerRunner {
    return new ZipformerRunner(session, io, Tensor);
  }

  reset(): void {
    this.buffer.length = 0;
    this.leftoverFrames = 0;
    this.processedLens = 0n;
    this.initStates();
  }

  async accept(frames: Float32Array[]): Promise<{ logProbs: Float32Array; frames: number }> {
    for (const f of frames) this.buffer.push(f);
    const T = this.io.T;
    const hop = this.io.hop;
    const dim = this.io.featureDim;
    const vocab = this.io.vocabSize;
    const chunks: Float32Array[] = [];
    while (this.buffer.length >= T) {
      const x = new Float32Array(T * dim);
      for (let t = 0; t < T; t++) x.set(this.buffer[t]!, t * dim);
      const feeds: Record<string, TensorLike> = {
        x: new this.Tensor("float32", x, [1, T, dim]),
      };
      for (const name of this.stateNames) feeds[name] = this.states.get(name)!;
      const out = await this.session.run(feeds);
      const lp = out.log_probs;
      if (!lp) throw new Error("model output missing log_probs");
      const data = lp.data instanceof Float32Array ? lp.data : new Float32Array(lp.data as ArrayLike<number>);
      const F = lp.dims.length >= 2 ? Number(lp.dims[1]) : data.length / vocab;
      chunks.push(data.slice(0, F * vocab));
      for (const name of this.stateNames) {
        const neu = out[`new_${name}`];
        if (!neu) throw new Error(`model missing new_${name}`);
        this.states.set(name, neu);
      }
      const pl = this.states.get("processed_lens");
      if (pl && pl.data instanceof BigInt64Array) this.processedLens = pl.data[0] ?? 0n;
      else if (pl) this.processedLens = BigInt(Number(pl.data[0]));
      this.buffer.splice(0, hop);
    }
    this.leftoverFrames = this.buffer.length;
    if (chunks.length === 0) return { logProbs: new Float32Array(0), frames: 0 };
    let total = 0;
    for (const c of chunks) total += c.length;
    const logProbs = new Float32Array(total);
    let o = 0;
    for (const c of chunks) {
      logProbs.set(c, o);
      o += c.length;
    }
    return { logProbs, frames: total / vocab };
  }

  private initStates(): void {
    this.states.clear();
    for (const inp of this.io.inputs) {
      if (inp.name === "x") continue;
      const n = numel(inp.dims);
      if (inp.dtype === "int64") {
        this.states.set(inp.name, new this.Tensor("int64", new BigInt64Array(n), inp.dims));
      } else {
        this.states.set(inp.name, new this.Tensor("float32", new Float32Array(n), inp.dims));
      }
    }
  }
}
