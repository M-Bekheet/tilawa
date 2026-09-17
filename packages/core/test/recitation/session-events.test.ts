/**
 * `ZipformerSession` end to end with a scripted ONNX session: fbank -> runner
 * -> greedy CTC -> engine -> emission -> SDK verse events. No model file.
 *
 * The stub ignores the audio and plays back a phoneme timeline taken from the
 * corpus itself (one token every other frame, blank in between so repeated
 * letters survive CTC collapsing), which is what a perfect decode of that
 * recitation would look like.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { QuranCorpus } from "../../src/recitation/corpus";
import {
  DEFAULT_ZIPFORMER_IO,
  ZipformerSession,
  createZipformerSession,
} from "../../src/recitation/session";
import { BLANK_ID, TOKENS, VOCAB_SIZE } from "../../src/recitation/tokens";
import type { OrtSessionLike, TensorLike } from "../../src/recitation/zipformerRunner";
import type { WorkerOutbound } from "../../src/types";
import { findModelIo, requireCorpus } from "./paths";

const SAMPLE_RATE = 16000;
const CHUNK = 7680; // 480 ms — one Zipformer hop worth of fbank frames
const FRAMES_PER_RUN = 12; // 48 fbank frames, subsampling factor 4
const HIGH = Math.log(0.9);
const LOW = Math.log(0.001);

const corpusJson = JSON.parse(readFileSync(requireCorpus(), "utf8"));
const corpus = new QuranCorpus(corpusJson);

const TOKEN_ID = new Map<string, number>();
for (let id = 0; id < TOKENS.length; id++) {
  const sym = TOKENS[id]!;
  if (sym.length === 1 && !TOKEN_ID.has(sym)) TOKEN_ID.set(sym, id);
}

function tokenIds(phonemes: string): number[] {
  return [...phonemes].map((ch) => {
    const id = TOKEN_ID.get(ch);
    if (id === undefined) throw new Error(`no CTC token for phoneme ${JSON.stringify(ch)}`);
    return id;
  });
}

class StubTensor implements TensorLike {
  readonly type: string;
  readonly data: Float32Array | BigInt64Array | Int32Array;
  readonly dims: readonly number[];

  constructor(
    type: string,
    data: Float32Array | BigInt64Array | Int32Array,
    dims: readonly number[],
  ) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

/** Replays a fixed token queue, one token per two output frames. */
class ScriptedOrtSession implements OrtSessionLike {
  private readonly queue: number[];
  runs = 0;

  constructor(ids: readonly number[]) {
    this.queue = [...ids];
  }

  get pending(): number {
    return this.queue.length;
  }

  async run(feeds: Record<string, TensorLike>): Promise<Record<string, TensorLike>> {
    this.runs++;
    const out: Record<string, TensorLike> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      if (name !== "x") out[`new_${name}`] = tensor;
    }
    const data = new Float32Array(FRAMES_PER_RUN * VOCAB_SIZE).fill(LOW);
    for (let frame = 0; frame < FRAMES_PER_RUN; frame++) {
      const id = frame % 2 === 0 ? this.queue.shift() ?? BLANK_ID : BLANK_ID;
      data[frame * VOCAB_SIZE + id] = HIGH;
    }
    out.log_probs = new StubTensor("float32", data, [1, FRAMES_PER_RUN, VOCAB_SIZE]);
    return out;
  }
}

const TensorCtor = StubTensor as unknown as new (
  type: string,
  data: Float32Array | BigInt64Array | Int32Array,
  dims: readonly number[],
) => TensorLike;

const FATIHA = [1, 2, 3, 4, 5, 6, 7].map((ayah) => corpus.ayahPhonemes(1, ayah)).join("");

async function runFatiha(): Promise<{
  session: ZipformerSession;
  seen: WorkerOutbound[];
  returned: WorkerOutbound[];
  ort: ScriptedOrtSession;
}> {
  const ort = new ScriptedOrtSession(tokenIds(FATIHA));
  const seen: WorkerOutbound[] = [];
  const session = await createZipformerSession({
    session: ort,
    Tensor: TensorCtor,
    corpus: corpusJson,
    quran: [
      {
        surah: 1,
        ayah: 2,
        text_uthmani: "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ",
        surah_name: "الفاتحة",
        surah_name_en: "Al-Fatihah",
      },
    ],
    onEvent: (msg) => seen.push(msg),
  });

  // 2 frames per token, 12 frames per run: enough audio to drain the queue.
  const chunks = Math.ceil((tokenIds(FATIHA).length * 2) / FRAMES_PER_RUN) + 3;
  const returned: WorkerOutbound[] = [];
  for (let i = 0; i < chunks; i++) {
    returned.push(...(await session.feed(new Float32Array(CHUNK))));
  }
  returned.push(...(await session.stop()));
  return { session, seen, returned, ort };
}

describe("ZipformerSession", () => {
  it("bundles the shipped model's I/O manifest", () => {
    expect(DEFAULT_ZIPFORMER_IO.T).toBe(61);
    expect(DEFAULT_ZIPFORMER_IO.hop).toBe(48);
    expect(DEFAULT_ZIPFORMER_IO.featureDim).toBe(80);
    expect(DEFAULT_ZIPFORMER_IO.vocabSize).toBe(VOCAB_SIZE);
    expect(DEFAULT_ZIPFORMER_IO.inputs[0]?.name).toBe("x");

    const shipped = findModelIo();
    if (shipped) {
      expect(DEFAULT_ZIPFORMER_IO).toEqual(JSON.parse(readFileSync(shipped, "utf8")));
    }
  });

  it("rejects an incomplete runtime seam", async () => {
    await expect(
      createZipformerSession({ corpus: corpusJson } as never),
    ).rejects.toThrow(/pass `ort` \+ `model`/);
    await expect(
      createZipformerSession({ session: new ScriptedOrtSession([]), corpus: corpusJson }),
    ).rejects.toThrow(/`Tensor`/);
  });

  it("turns a scripted decode of al-Fatiha into SDK verse events", async () => {
    const { session, seen, returned, ort } = await runFatiha();

    expect(ort.runs).toBeGreaterThan(10);
    expect(ort.pending).toBe(0);
    expect(seen).toEqual(returned);

    expect(session.transcript).toBe(FATIHA);
    // Finishing a surah drops back to searching, ready for the next one.
    expect(session.engineState).toBe("searching");

    const types = new Set(returned.map((m) => m.type));
    expect(types).toContain("verse_candidate");
    expect(types).toContain("word_progress");
    expect(types).toContain("raw_transcript");
    expect(types).toContain("verse_match");
    expect(types).toContain("final_sequence");

    const candidate = returned.find((m) => m.type === "verse_candidate");
    expect(candidate).toMatchObject({
      stable: false,
      final_flush: false,
      candidates: [{ surah: 1, rank: 0, source: "discovery" }],
    });

    const progress = returned.filter((m) => m.type === "word_progress");
    for (const p of progress) {
      if (p.type !== "word_progress") continue;
      expect(p.surah).toBe(1);
      expect(p.total_words).toBe(corpus.ayahWordCount(p.surah, p.ayah));
      expect(p.word_index).toBeLessThan(p.total_words);
    }

    const matches = returned.filter((m) => m.type === "verse_match");
    const refs = matches.map((m) => (m.type === "verse_match" ? `${m.surah}:${m.ayah}` : ""));
    expect(refs).toContain("1:2");
    expect(refs).toContain("1:7");
    expect(new Set(refs).size).toBe(refs.length); // each ayah emitted once
    for (const m of matches) {
      if (m.type !== "verse_match") continue;
      expect(m.surah).toBe(1);
      expect(m.confidence).toBeGreaterThanOrEqual(0.5);
    }

    // Display text comes from the `quran` source; only 1:2 was provided.
    const fatiha2 = matches.find((m) => m.type === "verse_match" && m.ayah === 2);
    expect(fatiha2).toMatchObject({ surah_name: "الفاتحة" });

    const final = returned.filter((m) => m.type === "final_sequence").at(-1);
    expect(final?.type).toBe("final_sequence");
    if (final?.type !== "final_sequence") throw new Error("unreachable");
    const sequence = final.verses.map((v) => `${v.surah}:${v.ayah}`);
    expect(sequence).toEqual([...sequence].sort(byAyah));
    expect(sequence).toContain("1:2");
    expect(final.confidence).toBeGreaterThan(0.5);

    expect(session.verses.map((t) => `${t.surah}:${t.ayah}`)).toEqual(
      expect.arrayContaining(["1:2", "1:7"]),
    );
  }, 120_000);

  it("reset() clears the transcript and the emitted set", async () => {
    const { session } = await runFatiha();
    expect(session.transcript.length).toBeGreaterThan(0);

    session.reset();
    expect(session.transcript).toBe("");
    expect(session.tallies).toEqual([]);
    expect(session.verses).toEqual([]);
    expect(session.engineState).toBe("searching");
    expect(session.lastFallback).toBeNull();
  }, 120_000);
});

function byAyah(a: string, b: string): number {
  const [as, aa] = a.split(":").map(Number) as [number, number];
  const [bs, ba] = b.split(":").map(Number) as [number, number];
  return as - bs || aa - ba;
}
