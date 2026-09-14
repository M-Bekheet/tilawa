#!/usr/bin/env node
// Dump kaldiFbank.js frames for a float32-LE 16 kHz PCM file.
// Usage: node fbank_dump.mjs [--chunk N] <pcm.f32le> <out.f32le>
// Writes [nFrames, 80] float32-LE to <out> and prints nFrames on stdout.

import { readFileSync, writeFileSync } from "node:fs";
import { KaldiFbank } from "./engine/browser/kaldiFbank.js";

function parseArgs(argv) {
  let chunk = 0;
  const pos = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--chunk") {
      chunk = Number(argv[++i]);
      if (!Number.isFinite(chunk) || chunk <= 0) {
        console.error("bad --chunk");
        process.exit(2);
      }
    } else if (a.startsWith("--chunk=")) {
      chunk = Number(a.slice("--chunk=".length));
      if (!Number.isFinite(chunk) || chunk <= 0) {
        console.error("bad --chunk");
        process.exit(2);
      }
    } else if (a.startsWith("-")) {
      console.error(`unknown flag ${a}`);
      process.exit(2);
    } else {
      pos.push(a);
    }
  }
  if (pos.length < 2) {
    console.error("usage: fbank_dump.mjs [--chunk N] <pcm.f32le> <out.f32le>");
    process.exit(2);
  }
  return { pcmPath: pos[0], outPath: pos[1], chunk };
}

const { pcmPath, outPath, chunk } = parseArgs(process.argv.slice(2));
const buf = readFileSync(pcmPath);
const copy = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const pcm = new Float32Array(copy);

const fbank = new KaldiFbank();
const frames = [];
if (chunk > 0) {
  for (let i = 0; i < pcm.length; i += chunk) {
    frames.push(...fbank.acceptWaveform(pcm.subarray(i, Math.min(i + chunk, pcm.length))));
  }
} else {
  frames.push(...fbank.acceptWaveform(pcm));
}
frames.push(...fbank.inputFinished());

const packed = new Float32Array(frames.length * KaldiFbank.numBins);
for (let i = 0; i < frames.length; i++) packed.set(frames[i], i * KaldiFbank.numBins);
writeFileSync(outPath, Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength));
process.stdout.write(`${frames.length}\n`);
