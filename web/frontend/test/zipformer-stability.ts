/**
 * Zipformer corpus eval through ZipformerHost (the swapped worker core).
 *
 * Invoked from stability-report.ts via --engine=zipformer.
 */
import { createRequire } from "node:module";
import { execSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { WorkerOutbound } from "../src/lib/types.ts";
import { displayQuranFromRaw, ZipformerHost } from "../src/worker/zipformer-session.ts";
import type { ZipformerIo } from "../src/lib/recitation/zipformerRunner.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND = resolve(HERE, "..");
const SAMPLE_RATE = 16000;
const CHUNK = 7680;

const MODEL =
  process.env.PROMPTER_MODEL ??
  resolve(FRONTEND, "public/models/zipformer_interp_gentle_a05.int8.onnx");
const IO_PATH =
  process.env.PROMPTER_IO ??
  resolve(FRONTEND, "public/models/zipformer_interp_gentle_a05.io.json");
const CORPUS =
  process.env.PROMPTER_CORPUS ??
  resolve(FRONTEND, "public/prompter_quran.json");
const QURAN = resolve(FRONTEND, "public/quran.json");
const ORT_DIR = process.env.PROMPTER_ORT_DIR ?? resolve(FRONTEND, "node_modules");

export interface ZipformerStabilityOpts {
  repeats: number;
  corpusName: string;
  jsonOutPath: string | null;
  sampleLimit: number | null;
}

interface Sample {
  id: string;
  file: string;
  category: string;
  expected_verses: { surah: number; ayah: number }[];
}

function loadAudio(filePath: string): Float32Array {
  const buf = execSync(
    `ffmpeg -hide_banner -loglevel error -i "${filePath}" -f f32le -ar ${SAMPLE_RATE} -ac 1 pipe:1`,
    { maxBuffer: 50 * 1024 * 1024 },
  );
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function verseKey(surah: number, ayah: number): string {
  return `${surah}:${ayah}`;
}

function sequenceEqual(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

function setEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const s = new Set(a);
  return b.every((v) => s.has(v));
}

function recallOf(expected: string[], got: string[]): number {
  if (expected.length === 0) return 1;
  const s = new Set(got);
  return expected.filter((v) => s.has(v)).length / expected.length;
}

function precisionOf(expected: string[], got: string[]): number {
  if (got.length === 0) return expected.length === 0 ? 1 : 0;
  const s = new Set(expected);
  return got.filter((v) => s.has(v)).length / got.length;
}

async function runClip(host: ZipformerHost, pcm: Float32Array): Promise<{
  matches: string[];
  final: string[];
}> {
  host.reset();
  const events: WorkerOutbound[] = [];
  for (let i = 0; i < pcm.length; i += CHUNK) {
    events.push(...await host.feed(pcm.subarray(i, Math.min(pcm.length, i + CHUNK))));
  }
  events.push(...await host.stop());
  const matches = events
    .filter((e) => e.type === "verse_match")
    .map((e) => (e.type === "verse_match" ? verseKey(e.surah, e.ayah) : ""));
  const lastFinal = [...events].reverse().find((e) => e.type === "final_sequence");
  const final = lastFinal && lastFinal.type === "final_sequence"
    ? lastFinal.verses.map((v) => verseKey(v.surah, v.ayah))
    : matches;
  return { matches, final };
}

export async function runZipformerStability(opts: ZipformerStabilityOpts): Promise<void> {
  const { repeats, corpusName, jsonOutPath, sampleLimit } = opts;
  for (const [path, hint] of [
    [MODEL, "missing zipformer onnx"],
    [IO_PATH, "missing zipformer io json"],
    [CORPUS, "missing prompter_quran.json"],
    [QURAN, "missing quran.json"],
  ] as const) {
    if (!existsSync(path)) throw new Error(`${hint}: ${path}`);
  }

  const benchmark = resolve(FRONTEND, `../../benchmark/${corpusName}`);
  const manifest = JSON.parse(readFileSync(resolve(benchmark, "manifest.json"), "utf8")) as {
    samples: Sample[];
  };
  const samples = sampleLimit && sampleLimit > 0
    ? manifest.samples.slice(0, sampleLimit)
    : manifest.samples;

  console.log(`=== ZIPFORMER STABILITY (${repeats} repeats, corpus: ${corpusName}) ===\n`);
  console.log(`Samples: ${samples.length}`);

  const require = createRequire(`${ORT_DIR}/`);
  const ort = require("onnxruntime-node");
  const io = JSON.parse(readFileSync(IO_PATH, "utf8")) as ZipformerIo;
  const host = await ZipformerHost.create({
    ort,
    modelBytes: new Uint8Array(readFileSync(MODEL)),
    io,
    corpusJson: JSON.parse(readFileSync(CORPUS, "utf8")),
    quranDb: displayQuranFromRaw(JSON.parse(readFileSync(QURAN, "utf8"))),
    executionProviders: ["cpu"],
  });

  const audioCache = new Map<string, Float32Array>();
  for (const sample of samples) {
    audioCache.set(sample.id, loadAudio(resolve(benchmark, sample.file)));
  }

  const rows: Array<{
    id: string;
    category: string;
    expected: string[];
    runs: Array<{ matches: string[]; final: string[]; exact: boolean; ordered: boolean }>;
  }> = [];

  let lastOrdered = 0;
  for (let run = 0; run < repeats; run++) {
    console.log(`--- Run ${run + 1}/${repeats} ---`);
    let ordered = 0;
    let exactSet = 0;
    for (const sample of samples) {
      const expected = sample.expected_verses.map((v) => verseKey(v.surah, v.ayah));
      const got = await runClip(host, audioCache.get(sample.id)!);
      const orderedOk = sequenceEqual(expected, got.final);
      const exactOk = setEqual(expected, got.final);
      if (orderedOk) ordered++;
      if (exactOk) exactSet++;
      const status = orderedOk ? "EXACT" : exactOk ? "SET  " : "FAIL ";
      process.stdout.write(
        `  ${status}  ${sample.id} expected=[${expected.join(", ")}] final=[${got.final.join(", ")}]\n`,
      );
      let row = rows.find((r) => r.id === sample.id);
      if (!row) {
        row = { id: sample.id, category: sample.category, expected, runs: [] };
        rows.push(row);
      }
      row.runs.push({
        matches: got.matches,
        final: got.final,
        exact: exactOk,
        ordered: orderedOk,
      });
    }
    lastOrdered = ordered;
    console.log(
      `  Run ${run + 1}: ordered ${ordered}/${samples.length}, exact-set ${exactSet}/${samples.length}\n`,
    );
  }

  const orderedPass = rows.filter((r) => r.runs.every((x) => x.ordered)).length;
  const exactPass = rows.filter((r) => r.runs.every((x) => x.exact)).length;
  const rec = rows.map((r) => recallOf(r.expected, r.runs[r.runs.length - 1]!.final));
  const prec = rows.map((r) => precisionOf(r.expected, r.runs[r.runs.length - 1]!.final));
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);

  const report = {
    engine: "zipformer",
    corpus: corpusName,
    repeats,
    timestamp: new Date().toISOString(),
    aggregate: {
      totalSamples: samples.length,
      orderedStablePass: orderedPass,
      exactStablePass: exactPass,
      lastRunOrdered: lastOrdered,
      finalSequence: {
        meanRecall: mean(rec),
        meanPrecision: mean(prec),
      },
    },
    samples: rows,
  };

  console.log("=".repeat(60));
  console.log("ZIPFORMER STABILITY SUMMARY");
  console.log("=".repeat(60));
  console.log(`Corpus: ${corpusName} | Repeats: ${repeats}`);
  console.log(`Ordered sequence: ${orderedPass}/${samples.length}`);
  console.log(`Exact set:        ${exactPass}/${samples.length}`);
  console.log(
    `Mean recall ${(mean(rec) * 100).toFixed(1)}%  precision ${(mean(prec) * 100).toFixed(1)}%`,
  );

  const fails = rows.filter((r) => !r.runs.every((x) => x.ordered));
  if (fails.length) {
    console.log(`\nOrdered failures (${fails.length}):`);
    for (const s of fails) {
      const last = s.runs[s.runs.length - 1]!;
      console.log(`  ${s.id} — expected [${s.expected.join(", ")}] final [${last.final.join(", ")}]`);
    }
  }

  if (jsonOutPath) {
    const outPath = resolve(process.cwd(), jsonOutPath);
    writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.log(`\nJSON report saved to: ${outPath}`);
  }
}
