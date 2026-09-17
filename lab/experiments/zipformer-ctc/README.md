# zipformer-ctc

Benchmark wrapper over the native MIT recitation engine
(`web/frontend/src/lib/recitation/`) plus Quran-Lab Zipformer2-CTC ONNX.
Registered as `zipformer-ctc`.

This directory used to vendor the recovered alketab JS engine. That tree was
removed; the host loop is now `ZipformerHost` / `zipformer-emission.ts` (same
as the browser worker). The dump-vector oracles in `docs/specs/vectors/` were
generated from the original engine before its removal at `e172b79`.

## What it is

| Stage | Implementation |
|---|---|
| Features | Kaldi-style fbank in TS: 25 ms / 10 ms, 80 mel, pre-emphasis 0.97, povey window, 512 FFT (`web/frontend/src/lib/recitation/fbank.ts`) |
| Acoustic model | **Streaming Zipformer2-CTC** (k2/icefall export), int8 ONNX, window T=61 / hop 48. Default fetch is Quran-Lab `zipformer_p_arabic_v3.1.int8.onnx` (NPL-1.2). The shipped demo uses `interp-gentle-a0.5`. See EXPERIMENTS.md. |
| Vocab | 251 tokens: Arabic letters *with* harakat, shadda-as-doubling, madd-length-as-repetition + `<blank>` (`tokens.txt`) |
| Decode | Greedy CTC with per-token margin = p(top1) − p(top2) |
| Corpus | `quran.json` v2: every word as `[mushaf glyphs, phoneme string, plain text]` |
| Locate / track / verdicts | Native MIT engine (`web/frontend/src/lib/recitation/`) |

`reference_tools/` and `reference_io/` are Quran-Lab NPL-1.2 eval/export copies
plus our ONNX PER wrapper. Keep them.

## How the wrapper works

`run.py` → `shared.audio.load_audio` → float32 file → `npx tsx harness.ts`
(one long-lived process, stdin/stdout JSON lines). The harness is
`ZipformerHost`: 480 ms chunks, `completed`/`idle` → snapshot verdicts and
search again; at end of audio it appends 2 s of silence to drain the
streaming encoder, flushes the CTC run, and takes settled verdicts. An ayah
is emitted when ≥ 50 % of its words are `ok`/`unsure` and `wrong` does not
outnumber them.

**Fallback (on by default, `ZIPFORMER_FALLBACK=0` to disable):** when no ayah
was emitted, the transcript is matched whole against all 6,236 ayah phoneme
strings (`wholeAyahFallback`).

Env knobs: `ZIPFORMER_MODE=recognize|stay`, `ZIPFORMER_CHUNK`,
`ZIPFORMER_TAIL_SECONDS`, `ZIPFORMER_MIN_WORD_FRACTION`, `ZIPFORMER_ALLOW_GAPS`,
`ZIPFORMER_FALLBACK_MAX_DISTANCE`, `ZIPFORMER_DATA_DIR`, `ZIPFORMER_ORT_DIR`,
`ZIPFORMER_MODEL`, `ZIPFORMER_IO`, `ZIPFORMER_CORPUS`.

## Requirements

- Node ≥ 22, `tsx` and `onnxruntime-node` (uses `web/frontend/node_modules`;
  override with `ZIPFORMER_ORT_DIR`).
- Model + corpus are downloaded on first use into `data/zipformer/`
  (gitignored) unless `ZIPFORMER_MODEL` already points at an existing file.

```bash
.venv/bin/python -m benchmark.runner --experiment zipformer-ctc
.venv/bin/python -m benchmark.runner --experiment zipformer-ctc --corpus test_corpus_v2
.venv/bin/python experiments/zipformer-ctc/run.py benchmark/test_corpus/001002.mp3
```

Third-party notices and NPL-1.2 terms for Zipformer-derived artefacts: see
[`NOTICE.md`](../../NOTICE.md).
