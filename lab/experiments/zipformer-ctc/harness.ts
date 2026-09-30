// Node harness around the native MIT recitation engine.
//
// Protocol: one JSON request per stdin line, one JSON response per stdout line.
//   {"id": 1, "pcm": "/path/to/float32le-16k.bin", "mode": "tracking" | "correction"}
//   -> {"id": 1, "verses": [{surah, ayah, ok, unsure, words}], "transcript": "...",
//       "events": [...], "decodeMs": n}
// Host loop is @tilawa/core's ZipformerSession + emission (same as the browser worker).

import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_CONFIG,
  displayQuranFromRaw,
  MIN_WORD_FRACTION,
  ZipformerSession,
  type BridgedAyahTally,
  type EngineConfig,
  type ZipformerIo,
} from "@tilawa/core";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..", "..", "..");
const FRONTEND = path.resolve(ROOT, "web", "frontend");

function camelToZipformerEnv(key: string): string {
  return "ZIPFORMER_" + key.replace(/[A-Z]/g, (c) => "_" + c).toUpperCase();
}

function configFromEnv(): EngineConfig {
  const cfg: EngineConfig = { ...DEFAULT_CONFIG };
  for (const key of Object.keys(DEFAULT_CONFIG) as (keyof EngineConfig)[]) {
    const raw = process.env[camelToZipformerEnv(key)];
    if (raw == null || raw === "") continue;
    const n = Number(raw);
    (cfg as unknown as Record<string, number | string>)[key] = Number.isFinite(n) ? n : raw;
  }
  return cfg;
}

const CONFIG = configFromEnv();

const MODEL = process.env.ZIPFORMER_MODEL ?? path.join(ROOT, "data", "zipformer", "zipformer_interp_gentle_a05.int8.onnx");
const CORPUS = process.env.ZIPFORMER_CORPUS ?? path.join(ROOT, "data", "zipformer", "zipformer_quran.json");
const ORT_DIR = process.env.ZIPFORMER_ORT_DIR ?? path.join(FRONTEND, "node_modules");
const IO_PATH =
  process.env.ZIPFORMER_IO ??
  path.join(HERE, "zipformer-io.json");
const DISPLAY_QURAN =
  process.env.ZIPFORMER_DISPLAY_QURAN ?? path.join(FRONTEND, "public", "quran.json");
const CHUNK = Number(process.env.ZIPFORMER_CHUNK ?? 7680);
const TAIL_SECONDS = Number(process.env.ZIPFORMER_TAIL_SECONDS ?? 2.0);
const MIN_FRAC = Number(process.env.ZIPFORMER_MIN_WORD_FRACTION ?? MIN_WORD_FRACTION);
const ALLOW_GAPS = process.env.ZIPFORMER_ALLOW_GAPS === "1";
const GAP_MAX_WORDS = Number(process.env.ZIPFORMER_GAP_MAX_WORDS ?? 3);
const MODE = process.env.ZIPFORMER_MODE ?? "recognize";
const FALLBACK = process.env.ZIPFORMER_FALLBACK !== "0";
const FALLBACK_MAX_DISTANCE = Number(process.env.ZIPFORMER_FALLBACK_MAX_DISTANCE ?? 0.5);

const require = createRequire(path.join(ORT_DIR, "/"));
const ort = require("onnxruntime-node");

const io = JSON.parse(readFileSync(IO_PATH, "utf8")) as ZipformerIo;
const corpusJson = JSON.parse(readFileSync(CORPUS, "utf8"));
const quranDb = existsSync(DISPLAY_QURAN)
  ? displayQuranFromRaw(JSON.parse(readFileSync(DISPLAY_QURAN, "utf8")))
  : displayQuranFromRaw([]);

async function createHost(): Promise<ZipformerSession> {
  return ZipformerSession.create({
    ort,
    model: new Uint8Array(readFileSync(MODEL)),
    io,
    corpus: corpusJson,
    quran: quranDb,
    executionProviders: ["cpu"],
    config: CONFIG,
    tailSeconds: TAIL_SECONDS,
    minWordFraction: MIN_FRAC,
    stayOnSurah: MODE === "stay",
    enableFallback: FALLBACK,
    fallbackMaxDistance: FALLBACK_MAX_DISTANCE,
    allowGaps: ALLOW_GAPS,
    gapMaxWords: GAP_MAX_WORDS,
    debug: true,
  });
}

function ayahWordCount(host: ZipformerSession, surah: number, ayah: number): number {
  try {
    return host.wordCount(surah, ayah);
  } catch {
    return 99;
  }
}

async function recognize(host: ZipformerSession, pcm: Float32Array, mode = "tracking") {
  const t0 = performance.now();
  host.reset();
  host.setMode(mode === "correction" ? "correction" : "tracking");
  const events: Array<Record<string, unknown>> = [];
  const cursorOrder: string[] = [];
  // Correction issues are dismissed on sight so the rest of the clip is still
  // decoded (the session drops audio while an issue is open).
  const corrections: Array<Record<string, unknown>> = [];
  let offset = 0;

  const collect = (msgs: Array<{ type: string; [k: string]: unknown }>): void => {
    for (const ev of msgs) {
      if (ev.type === "correction") {
        const state = ev.state as { phase: string; issue: Record<string, unknown> | null };
        if (state.phase === "error" && state.issue) {
          corrections.push({ ...state.issue, atSeconds: offset / 16000 });
          collect(host.correct("dismiss") as Array<{ type: string; [k: string]: unknown }>);
        }
      } else if (ev.type === "word_progress") {
        const key = `${ev.surah}:${ev.ayah}`;
        if (cursorOrder[cursorOrder.length - 1] !== key) cursorOrder.push(key);
      } else if (ev.type === "debug") {
        const data = (ev.data ?? {}) as Record<string, unknown>;
        events.push({ type: ev.event, ...data });
      }
    }
  };

  for (let i = 0; i < pcm.length; i += CHUNK) {
    offset = Math.min(pcm.length, i + CHUNK);
    collect(await host.feed(pcm.subarray(i, offset)));
  }
  collect(await host.stop());

  const tallies = host.tallies;
  const verses: BridgedAyahTally[] = host.verses;

  let fallback = host.lastFallback;
  if (FALLBACK && verses.length === 0 && fallback) {
    const words = ayahWordCount(host, fallback.surah, fallback.ayah);
    verses.push({
      surah: fallback.surah,
      ayah: fallback.ayah,
      ok: words,
      unsure: 0,
      wrong: 0,
      skipped: 0,
      pending: 0,
      words,
      firstSeen: 0,
    });
    events.push({ type: "fallback", ...fallback });
  }

  return {
    verses,
    fallback,
    all: tallies,
    cursorOrder,
    corrections,
    transcript: host.transcript,
    events,
    state: host.engineState,
    decodeMs: Math.round(performance.now() - t0),
  };
}

async function main(): Promise<void> {
  const host = await createHost();
  const rl = createInterface({ input: process.stdin });
  process.stdout.write(JSON.stringify({
    ready: true,
    model: MODEL,
    minWordFraction: MIN_FRAC,
    allowGaps: ALLOW_GAPS,
    gapMaxWords: GAP_MAX_WORDS,
    tailSeconds: TAIL_SECONDS,
    okDistance: CONFIG.okDistance,
    unsureDistance: CONFIG.unsureDistance,
    searchDecisiveDistance: CONFIG.searchDecisiveDistance,
  }) + "\n");

  for await (const line of rl) {
    if (!line.trim()) continue;
    let req: { id?: number; pcm: string; mode?: string };
    try {
      req = JSON.parse(line) as { id?: number; pcm: string; mode?: string };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      process.stdout.write(JSON.stringify({ error: `bad json: ${msg}` }) + "\n");
      continue;
    }
    try {
      const buf = readFileSync(req.pcm);
      const pcm = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
      const res = await recognize(host, pcm, req.mode);
      process.stdout.write(JSON.stringify({ id: req.id, ...res }) + "\n");
    } catch (e) {
      process.stdout.write(JSON.stringify({
        id: req.id,
        error: e instanceof Error ? e.stack ?? e.message : String(e),
      }) + "\n");
    }
  }
}

void main();
