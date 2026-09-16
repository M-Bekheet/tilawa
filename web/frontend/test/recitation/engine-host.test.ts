import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { pausalPhonemes } from "../../src/lib/recitation/verdicts";
import { QuranCorpus } from "../../src/lib/recitation/corpus";
import { QuranIndex } from "../../src/lib/recitation/search";
import { RecitationEngine } from "../../src/lib/recitation/engine";
import { wholeAyahFallback } from "../../src/lib/recitation/fallback";
import { DEFAULT_CONFIG } from "../../src/lib/recitation/config";
import { costTable } from "../../src/lib/recitation/phonemeCost";
import {
  ayahMeetsGate,
  snapshotTallies,
  type WordVerdict as HostVerdict,
} from "../../src/lib/zipformer-emission";
import type { CtcToken, EngineEvent } from "../../src/lib/recitation/types";

const HERE = dirname(fileURLToPath(import.meta.url));
const VECTORS = resolve(HERE, "../../../../docs/specs/vectors");
const CORPUS_PATH =
  process.env.PROMPTER_CORPUS ??
  resolve(HERE, "../../public/prompter_quran.json");

function load<T>(name: string): T {
  return JSON.parse(readFileSync(resolve(VECTORS, name), "utf8")) as T;
}

const corpus = new QuranCorpus(JSON.parse(readFileSync(CORPUS_PATH, "utf8")));
const index = new QuranIndex(corpus, DEFAULT_CONFIG);
const table = costTable();

describe("waqf", () => {
  const vec = load<{
    examples: {
      phonemes: string;
      plain: string;
      atAyahEnd: boolean;
      pausal: string | null;
      pausalIfForcedNotAyahEnd: string | null;
    }[];
  }>("waqf.json");
  it("pausal derivations", () => {
    for (const ex of vec.examples) {
      expect(pausalPhonemes(ex.phonemes, ex.plain, ex.atAyahEnd)).toBe(ex.pausal);
      expect(pausalPhonemes(ex.phonemes, ex.plain, false)).toBe(ex.pausalIfForcedNotAyahEnd);
    }
  });
});

interface EventVec {
  finalState: string;
  cursor: {
    wordIndex: number;
    surah: number;
    ayah: number;
    word: number;
    cost: number;
  } | null;
  settledVerdicts: unknown[];
  transcript: string;
  framesDecoded: number;
  chunks: Array<{
    tokens: CtcToken[];
    framesDecoded: number;
    events: EngineEvent[];
  }>;
}

function runEvents(name: string) {
  const vec = load<EventVec>(name);
  const engine = new RecitationEngine(corpus, index, DEFAULT_CONFIG);
  engine.setStayOnSurah(false);
  engine.startSearch();
  const all: EngineEvent[] = [];
  for (let i = 0; i < vec.chunks.length; i++) {
    const ch = vec.chunks[i]!;
    const ev = engine.feed(ch.tokens, ch.framesDecoded);
    expect(ev, `chunk ${i} kind frames=${ch.framesDecoded}`).toEqual(ch.events);
    all.push(...ev);
  }
  expect(engine.state).toBe(vec.finalState);
  expect(engine.framesDecoded).toBe(vec.framesDecoded);
  if (vec.cursor) {
    expect(engine.tracker).not.toBeNull();
    expect(engine.tracker!.cursorWordIndex).toBe(vec.cursor.wordIndex);
    expect(engine.tracker!.surah).toBe(vec.cursor.surah);
    expect(engine.corpus.wordAyah[engine.tracker!.cursorWordIndex]).toBe(vec.cursor.ayah);
    expect(engine.corpus.wordInAyah[engine.tracker!.cursorWordIndex]).toBe(vec.cursor.word);
    expect(engine.tracker!.cursorCost).toBe(vec.cursor.cost);
  } else {
    expect(engine.tracker).toBeNull();
  }
  if (vec.settledVerdicts.length) {
    const got = engine.tracer!.verdicts(true);
    expect(got).toEqual(vec.settledVerdicts);
  }
  return { engine, vec, all };
}

describe("engine events", () => {
  it("events_001002.json stays searching with zero events", () => {
    runEvents("events_001002.json");
  });
  it("events_001001.json stays searching with zero events", () => {
    runEvents("events_001001.json");
  });
  it("events_ea_alafasy_multi_001_001_007.json locates Fatiha", () => {
    runEvents("events_ea_alafasy_multi_001_001_007.json");
  });
});

describe("host policy", () => {
  function hostOf(eventFile: string, hostFile: string) {
    const { engine, vec } = runEvents(eventFile);
    const host = load<{
      tallies: Array<{
        surah: number;
        ayah: number;
        ok: number;
        unsure: number;
        wrong: number;
        skipped: number;
        pending: number;
        words: number;
        firstSeen: number;
      }>;
      verses: unknown[];
      fallback: { surah: number; ayah: number; distance: number; how: string } | null;
      engineEvents: string[];
    }>(hostFile);
    const fallback = wholeAyahFallback(vec.transcript, corpus, table);
    const wordCount = (s: number, a: number) => corpus.ayahWordCount(s, a);
    const snap = engine.tracer
      ? snapshotTallies(engine.tracer.verdicts(true) as HostVerdict[], wordCount)
      : new Map();
    const tallies = [...snap.values()].filter((t) => ayahMeetsGate(t));
    return { host, fallback, tallies, engine };
  }

  it("host_001002.json whole-ayah fallback", () => {
    const { host, fallback, tallies } = hostOf("events_001002.json", "host_001002.json");
    expect(tallies).toEqual(host.tallies);
    expect(fallback).toEqual(host.fallback);
  });

  it("host_001001.json basmala fallback", () => {
    const { host, fallback, tallies } = hostOf("events_001001.json", "host_001001.json");
    expect(tallies).toEqual(host.tallies);
    expect(fallback).toEqual(host.fallback);
  });

  it("host_ea_alafasy_multi gated ayahs, no fallback", () => {
    const { host, fallback, tallies } = hostOf(
      "events_ea_alafasy_multi_001_001_007.json",
      "host_ea_alafasy_multi_001_001_007.json",
    );
    expect(tallies).toEqual(host.tallies);
    expect(fallback).toBeNull();
    expect(host.fallback).toBeNull();
  });
});
