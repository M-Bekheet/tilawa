import { QuranDB } from "@tilawa/core";
import type { QuranVerse, SurroundingVerse, WorkerOutbound } from "../lib/types";
import { SURROUNDING_CONTEXT } from "../lib/types";
import {
  accumulateSnapshot,
  ayahConfidence,
  ayahKey,
  buildFinalSequence,
  FALLBACK_MAX_DISTANCE,
  mergeTallies,
  newlyEligibleAyahs,
  shouldRunFallback,
  snapshotTallies,
  wordProgressFromCursor,
  type AyahTally,
  type FallbackHit,
  type WordVerdict,
} from "../lib/zipformer-emission";
import {
  BLANK_ID,
  DEFAULT_CONFIG,
  GreedyCtcDecoder,
  KaldiFbank,
  QuranCorpus,
  QuranIndex,
  RecitationEngine,
  TOKENS,
  ZipformerRunner,
  costTable,
  normalizedDistance,
  stripPreambles,
  type EngineConfig,
  type ZipformerIo,
} from "../lib/recitation";

const SAMPLE_RATE = 16000;
const TAIL_SECONDS = 2.0;

export const ZIPFORMER_CACHE_KEY = "zipformer-interp-gentle-a05-int8";
export const ZIPFORMER_MODEL_URL = "/models/zipformer_interp_gentle_a05.int8.onnx";
export const ZIPFORMER_IO_URL = "/models/zipformer_interp_gentle_a05.io.json";
export const ZIPFORMER_QURAN_URL = "/zipformer_quran.json";
export const DISPLAY_QURAN_URL = "/quran.json";

interface EncodedAyah {
  surah: number;
  ayah: number;
  ids: Uint8Array;
}

export function displayQuranFromRaw(raw: unknown): QuranDB {
  if (!Array.isArray(raw)) throw new Error("quran.json must be an array");
  return new QuranDB(
    raw.map((row) => {
      const v = row as QuranVerse;
      return {
        ...v,
        phonemes: v.phonemes ?? "",
        phonemes_joined: v.phonemes_joined ?? "",
        phoneme_words: v.phoneme_words ?? [],
      };
    }),
  );
}

function surroundingVerses(db: QuranDB, surah: number, ayah: number): SurroundingVerse[] {
  return db.getSurah(surah)
    .filter((v) => Math.abs(v.ayah - ayah) <= SURROUNDING_CONTEXT)
    .map((v) => ({
      surah: v.surah,
      ayah: v.ayah,
      text: v.text_uthmani,
      is_current: v.ayah === ayah,
    }));
}

export interface ZipformerHostOptions {
  ort: unknown;
  modelBytes: Uint8Array | ArrayBuffer;
  io: ZipformerIo;
  corpusJson: unknown;
  quranDb: QuranDB;
  executionProviders: string[];
  config?: EngineConfig;
  tailSeconds?: number;
  minWordFraction?: number;
  stayOnSurah?: boolean;
  enableFallback?: boolean;
  fallbackMaxDistance?: number;
}

export class ZipformerHost {
  private readonly corpus: QuranCorpus;
  private readonly index: QuranIndex;
  private readonly table = costTable();
  private readonly ayahIds: EncodedAyah[] = [];
  private readonly quranDb: QuranDB;
  private readonly runner: ZipformerRunner;
  private readonly cfg: EngineConfig;
  private readonly tailSeconds: number;
  private readonly minWordFraction: number;
  private readonly stayOnSurah: boolean;
  private readonly enableFallback: boolean;
  private readonly fallbackMaxDistance: number;
  private fbank = new KaldiFbank();
  private decoder = new GreedyCtcDecoder(TOKENS, BLANK_ID);
  private engine: RecitationEngine;
  private accumulated = new Map<string, AyahTally>();
  private emitted = new Set<string>();
  private transcript: string[] = [];
  private lastCursor: { surah: number; ayah: number; word: number } | null = null;
  lastFallback: FallbackHit | null = null;
  debugEnabled = false;

  private constructor(runner: ZipformerRunner, opts: ZipformerHostOptions) {
    this.runner = runner;
    this.quranDb = opts.quranDb;
    this.cfg = { ...DEFAULT_CONFIG, ...opts.config };
    this.tailSeconds = opts.tailSeconds ?? TAIL_SECONDS;
    this.minWordFraction = opts.minWordFraction ?? 0.5;
    this.stayOnSurah = opts.stayOnSurah ?? false;
    this.enableFallback = opts.enableFallback ?? true;
    this.fallbackMaxDistance = opts.fallbackMaxDistance ?? FALLBACK_MAX_DISTANCE;
    this.corpus = new QuranCorpus(opts.corpusJson);
    this.index = new QuranIndex(this.corpus, this.cfg);
    for (const s of this.corpus.surahs) {
      for (let a = 1; a <= s.ayahCount; a++) {
        const first = this.corpus.ayahFirstWord(s.n, a);
        const end = first + this.corpus.ayahWordCount(s.n, a);
        this.ayahIds.push({
          surah: s.n,
          ayah: a,
          ids: this.table.encode(this.corpus.text.slice(this.corpus.wordStart[first], this.corpus.wordStart[end])),
        });
      }
    }
    this.engine = this.makeEngine();
  }

  static async create(opts: ZipformerHostOptions): Promise<ZipformerHost> {
    const bytes = opts.modelBytes instanceof Uint8Array
      ? opts.modelBytes
      : new Uint8Array(opts.modelBytes);
    const runner = await ZipformerRunner.create(opts.ort, bytes, opts.io, opts.executionProviders);
    return new ZipformerHost(runner, opts);
  }

  get tallies(): AyahTally[] {
    return [...this.accumulated.values()].sort((a, b) => a.firstSeen - b.firstSeen);
  }

  get transcriptText(): string {
    return this.transcript.join("");
  }

  get engineState(): string {
    return this.engine.state;
  }

  reset(): WorkerOutbound[] {
    this.accumulated = new Map();
    this.emitted = new Set();
    this.transcript = [];
    this.lastCursor = null;
    this.lastFallback = null;
    this.resetDecoder();
    this.engine = this.makeEngine();
    return [];
  }

  async feed(samples: Float32Array): Promise<WorkerOutbound[]> {
    return this.feedSamples(samples);
  }

  async stop(): Promise<WorkerOutbound[]> {
    const out: WorkerOutbound[] = [];
    const silence = new Float32Array(Math.round(this.tailSeconds * SAMPLE_RATE));
    out.push(...await this.feedSamples(silence));

    const frames = this.fbank.inputFinished();
    if (frames.length) {
      out.push(...await this.runFrames(frames));
    }
    const flushed = this.decoder.flush();
    if (flushed.length) {
      out.push(...this.consumeTokens(flushed));
    }

    this.dumpTallies();
    const live = newlyEligibleAyahs(this.accumulated, this.emitted, this.minWordFraction);
    for (const t of live) {
      this.emitted.add(ayahKey(t));
      out.push(this.toVerseMatch(t));
    }

    let fallback: FallbackHit | null = null;
    if (this.enableFallback && shouldRunFallback([...this.emitted])) {
      fallback = this.fallbackSearch(this.transcript.join(""));
      this.lastFallback = fallback;
      if (fallback) {
        const words = this.corpus.ayahWordCount(fallback.surah, fallback.ayah);
        const tally: AyahTally = {
          surah: fallback.surah,
          ayah: fallback.ayah,
          ok: words,
          unsure: 0,
          wrong: 0,
          skipped: 0,
          pending: 0,
          words,
          firstSeen: this.accumulated.size,
        };
        this.accumulated.set(ayahKey(tally), tally);
        if (!this.emitted.has(ayahKey(tally))) {
          this.emitted.add(ayahKey(tally));
          out.push(this.toVerseMatch(tally));
        }
        if (this.debugEnabled) {
          out.push({
            type: "debug",
            event: "fallback",
            at: Date.now(),
            data: { ...fallback },
          });
        }
      }
    }

    const seq = buildFinalSequence([...this.accumulated.values()], fallback, this.minWordFraction);
    out.push({
      type: "final_sequence",
      verses: seq.verses,
      confidence: Math.round(seq.confidence * 100) / 100,
    });
    out.push({
      type: "raw_transcript",
      text: this.transcript.join(""),
      confidence: seq.confidence,
    });
    return out;
  }

  private makeEngine(): RecitationEngine {
    const engine = new RecitationEngine(this.corpus, this.index, this.cfg);
    engine.setStayOnSurah(this.stayOnSurah);
    engine.startSearch();
    engine.onBeforeRelocate = () => this.dumpTallies();
    return engine;
  }

  private resetDecoder(): void {
    this.fbank.reset();
    this.decoder.reset();
    this.runner.reset();
  }

  wordCount = (surah: number, ayah: number): number =>
    this.corpus.ayahWordCount(surah, ayah);

  private dumpTallies(): void {
    const tracer = this.engine.tracer;
    if (!tracer) return;
    const snap = snapshotTallies(tracer.verdicts(true) as WordVerdict[], this.wordCount);
    accumulateSnapshot(this.accumulated, snap);
  }

  private currentSnapshot(): Map<string, AyahTally> {
    const tracer = this.engine.tracer;
    if (!tracer) return new Map();
    return snapshotTallies(tracer.verdicts(true) as WordVerdict[], this.wordCount);
  }

  private async feedSamples(samples: Float32Array): Promise<WorkerOutbound[]> {
    const frames = this.fbank.acceptWaveform(samples);
    return this.runFrames(frames);
  }

  private async runFrames(frames: Float32Array[]): Promise<WorkerOutbound[]> {
    if (!frames.length) return [];
    const { logProbs, frames: out } = await this.runner.accept(frames);
    if (out === 0) return [];
    const tokens = this.decoder.consume(logProbs, out, this.runner.io.vocabSize);
    return this.consumeTokens(tokens);
  }

  private consumeTokens(tokens: Array<{ sym: string; frame: number; margin: number }>): WorkerOutbound[] {
    const out: WorkerOutbound[] = [];
    for (const t of tokens) this.transcript.push(t.sym);
    for (const ev of this.engine.feed(tokens, this.decoder.framesDecoded)) {
      out.push(...this.handle(ev));
    }
    out.push(...this.emitNewMatches());
    if (tokens.length) {
      out.push({
        type: "raw_transcript",
        text: this.transcript.join(""),
        confidence: 1,
      });
    }
    return out;
  }

  private handle(ev: { type: string; surah?: number; ayah?: number; word?: number }): WorkerOutbound[] {
    const out: WorkerOutbound[] = [];
    switch (ev.type) {
      case "cursor": {
        if (ev.surah != null && ev.ayah != null && ev.word != null) {
          this.lastCursor = { surah: ev.surah, ayah: ev.ayah, word: ev.word };
          out.push(this.wordProgress());
        }
        break;
      }
      case "verdicts": {
        if (this.lastCursor) out.push(this.wordProgress());
        break;
      }
      case "located": {
        if (ev.surah != null && ev.ayah != null) {
          out.push({
            type: "verse_candidate",
            candidates: [{
              surah: ev.surah,
              ayah: ev.ayah,
              confidence: 0.5,
              rank: 0,
              source: "discovery",
            }],
            stable: false,
            final_flush: false,
          });
        }
        break;
      }
      case "idle":
      case "completed": {
        this.dumpTallies();
        out.push(...this.emitNewMatches(this.accumulated));
        this.engine.startSearch();
        this.resetDecoder();
        this.lastCursor = null;
        break;
      }
      default:
        break;
    }
    if (this.debugEnabled && ev.type !== "cursor" && ev.type !== "verdicts") {
      out.push({
        type: "debug",
        event: ev.type,
        at: Date.now(),
        data: ev as unknown as Record<string, unknown>,
      });
    }
    return out;
  }

  private wordProgress(): WorkerOutbound {
    const cursor = this.lastCursor!;
    const tracer = this.engine.tracer;
    const verdicts = (tracer ? tracer.verdicts(true) : []) as WordVerdict[];
    return wordProgressFromCursor(cursor, verdicts, this.wordCount(cursor.surah, cursor.ayah));
  }

  private emitNewMatches(source?: Map<string, AyahTally>): WorkerOutbound[] {
    const tallies = source ?? mergeTallies(this.accumulated, this.currentSnapshot());
    const out: WorkerOutbound[] = [];
    for (const t of newlyEligibleAyahs(tallies, this.emitted, this.minWordFraction)) {
      this.emitted.add(ayahKey(t));
      out.push(this.toVerseMatch(t));
    }
    return out;
  }

  private toVerseMatch(t: AyahTally): WorkerOutbound {
    const verse = this.quranDb.getVerse(t.surah, t.ayah);
    return {
      type: "verse_match",
      surah: t.surah,
      ayah: t.ayah,
      verse_text: verse?.text_uthmani ?? "",
      surah_name: verse?.surah_name ?? "",
      confidence: Math.round(ayahConfidence(t) * 100) / 100,
      surrounding_verses: surroundingVerses(this.quranDb, t.surah, t.ayah),
    };
  }

  private fallbackSearch(text: string): FallbackHit | null {
    if (!text) return null;
    const stripped = stripPreambles(text, this.table);
    const rest = text.slice(stripped.offset);
    if (stripped.basmala && rest.length < this.cfg.searchMinChars) {
      return { surah: 1, ayah: 1, distance: 0, how: "basmala" };
    }
    const q = this.table.encode(rest.length >= 3 ? rest : text);
    let best: FallbackHit | null = null;
    for (const a of this.ayahIds) {
      if (a.ids.length > 2.5 * q.length + 8 || q.length > 2.5 * a.ids.length + 8) continue;
      const d = normalizedDistance(q, a.ids, this.table);
      if (!best || d < best.distance) best = { surah: a.surah, ayah: a.ayah, distance: d, how: "whole-ayah" };
    }
    if (!best || best.distance > this.fallbackMaxDistance) return null;
    return best;
  }
}
