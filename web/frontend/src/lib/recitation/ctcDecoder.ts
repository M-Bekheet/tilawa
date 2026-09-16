import type { CtcToken, HeardChar } from "./types";

export class GreedyCtcDecoder {
  readonly symbols: readonly string[];
  readonly blank: number;
  private previousBest: number;
  private frameIndex = 0;
  private run: { id: number; frame: number; p1: number; p2: number } | null = null;

  constructor(symbols: readonly string[], blank: number) {
    this.symbols = symbols;
    this.blank = blank;
    this.previousBest = blank;
  }

  get framesDecoded(): number {
    return this.frameIndex;
  }

  reset(): void {
    this.previousBest = this.blank;
    this.frameIndex = 0;
    this.run = null;
  }

  consume(logProbs: ArrayLike<number>, frames: number, classes: number): CtcToken[] {
    const out: CtcToken[] = [];
    for (let t = 0; t < frames; t++) {
      const row = t * classes;
      let best = 0;
      let p1 = logProbs[row]!;
      let p2 = -Infinity;
      for (let c = 1; c < classes; c++) {
        const p = logProbs[row + c]!;
        if (p > p1) {
          p2 = p1;
          p1 = p;
          best = c;
        } else if (p > p2) {
          p2 = p;
        }
      }
      this.step(best, p1, p2, out);
    }
    return out;
  }

  flush(): CtcToken[] {
    const out: CtcToken[] = [];
    if (this.run) {
      out.push(this.emit(this.run));
      this.run = null;
    }
    this.previousBest = this.blank;
    return out;
  }

  private step(best: number, p1: number, p2: number, out: CtcToken[]): void {
    const blank = this.blank;
    if (best !== blank && best !== this.previousBest) {
      if (this.run) out.push(this.emit(this.run));
      this.run = { id: best, frame: this.frameIndex, p1, p2 };
    } else if (
      best !== blank &&
      best === this.previousBest &&
      this.run &&
      p1 > this.run.p1
    ) {
      this.run.p1 = p1;
      this.run.p2 = p2;
    } else if (best === blank && this.run) {
      out.push(this.emit(this.run));
      this.run = null;
    }
    this.previousBest = best;
    this.frameIndex++;
  }

  private emit(run: { id: number; frame: number; p1: number; p2: number }): CtcToken {
    return {
      sym: this.symbols[run.id] ?? "",
      frame: run.frame,
      margin: Math.exp(run.p1) - Math.exp(run.p2),
    };
  }
}

export function expandTokens(tokens: readonly CtcToken[]): HeardChar[] {
  const out: HeardChar[] = [];
  for (const t of tokens) {
    for (const ch of t.sym) {
      out.push({ ch, frame: t.frame, margin: t.margin });
    }
  }
  return out;
}
