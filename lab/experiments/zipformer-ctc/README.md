# prompter-zipformer

The engine behind https://prompter.alketab.app/ ("ملقّن القرآن", a live Quran
teleprompter), run unmodified under Node and wrapped in the benchmark's
`predict()` contract.

## What it is

| Stage | Implementation |
|---|---|
| Features | Kaldi-style fbank in JS: 25 ms / 10 ms, 80 mel, pre-emphasis 0.97, povey window, 512 FFT (`engine/browser/kaldiFbank.js`) |
| Acoustic model | **Streaming Zipformer2-CTC** (k2/icefall export), **int8** ONNX (onnxruntime dynamic quant), **72.7 MB**, 16 encoder-state tensors carried across chunks, window T=61 frames / hop 48 (`quran_phoneme_zipformer.onnx`). Byte-identical to Quran-Lab `zipformer_p_arabic_v3.1.int8.onnx` — see EXPERIMENTS.md § "Zipformer2-CTC (Quran-Lab v3 reference + fine-tunes)". |
| Vocab | 251 tokens: Arabic letters *with* harakat, shadda-as-doubling, madd-length-as-repetition (`ككِ`, `ممم`, `اااا`, `ۦۦۦۦ`, sukun/qalqalah `ڇ`, ghunna `ۜ` …) + `<blank>` (`engine/model/tokens.js`) |
| Decode | Greedy CTC with per-token margin = p(top1) − p(top2) (`engine/browser/ctcDecoder.js`) |
| Corpus | `quran.json` v2: every word as `[mushaf glyphs, phoneme string, plain text]`; the whole Quran is one phoneme string (`engine/core/corpus.js`) |
| Locate | 5-gram FNV hash index over the phoneme corpus → 32-char window votes → top-24 windows verified by semi-global alignment with graded substitution costs; "decisive" if distance ≤ 0.35, ≥ 20 chars aligned, runner-up ≥ 0.1 worse. Strips استعاذة/basmala heads (`engine/core/search.js`, `phonemeCost.js`, `alignment.js`) |
| Track | Per-surah online DP column (insert/delete 1, graded substitution, `jumpCost` 12 to any word start, `repeatCost` 10 within the current ayah); lost/held detection from a rolling cost rate; relocation to another surah only when the global search agrees twice (`engine/core/tracker.js`, `engine.js`) |
| Verdicts | Trail traced back into per-word `ok / unsure / wrong / skipped / pending`, with pausal-form (waqf) allowance (`engine/core/verdicts.js`, `waqf.js`) |

All JS under `engine/` was recovered verbatim from the site's published source
maps (`main-*.js.map`, `decoder.worker-*.js.map`; package name
`@alketab/quran-engine`). The only edit is `ZipformerRunner.create` taking an
execution-provider list so onnxruntime-node can use `cpu` instead of `wasm`.
The UI, mic capture, PWA and page-index code were not copied.

## How the wrapper works

`run.py` → `shared.audio.load_audio` → float32 file → `node harness.mjs`
(one long-lived process, stdin/stdout JSON lines). The harness replays the
app's **Recognize mode** host loop: 480 ms chunks, `completed`/`idle` →
snapshot verdicts and search again; at end of audio it appends 2 s of silence
to drain the streaming encoder, flushes the CTC run, and takes settled
verdicts. An ayah is emitted when ≥ 50 % of its words are `ok`/`unsure` and
`wrong` does not outnumber them.

**Fallback (on by default, `PROMPTER_FALLBACK=0` to disable):** the live
engine deliberately refuses to lock on short/ambiguous audio (it is a
prompter with a mic that keeps running). When no ayah was emitted, the
transcript is matched whole against all 6,236 ayah phoneme strings with the
engine's own graded distance (basmala-only → 1:1). Pure engine scores 74 % on
v1; the fallback recovers the 14 short one-ayah clips.

Env knobs: `PROMPTER_MODE=recognize|stay`, `PROMPTER_CHUNK`,
`PROMPTER_TAIL_SECONDS`, `PROMPTER_MIN_WORD_FRACTION`,
`PROMPTER_FALLBACK_MAX_DISTANCE`, `PROMPTER_DATA_DIR`, `PROMPTER_ORT_DIR`.

## Requirements

- Node ≥ 22 and `onnxruntime-node` (uses `web/frontend/node_modules`; override
  with `PROMPTER_ORT_DIR`).
- Model + corpus are downloaded on first use into `data/prompter/`
  (gitignored): `quran_phoneme_zipformer.onnx` (72.7 MB **int8**), `quran.json` (5.5 MB).
  Skipped when `PROMPTER_MODEL` already points at an existing file.

```bash
.venv/bin/python -m benchmark.runner --experiment prompter-zipformer
.venv/bin/python -m benchmark.runner --experiment prompter-zipformer --corpus test_corpus_v3
.venv/bin/python experiments/prompter-zipformer/run.py benchmark/test_corpus/001002.mp3   # debug one file
```

## Results (2026-09-14, deterministic across 3 runs)

| Corpus | Recall | Precision | SeqAcc | Correct | Latency | Champion `c2c-direct-mixed-tta` |
|---|---|---|---|---|---|---|
| v1 (53) | 100 % | 100 % | 100 % | 53/53 | 0.99 s | 100 / 100 / 100 |
| v2 (43) | 97.7 % | 97.7 % | 97.7 % | 42/43 | 1.04 s | 97.7 / 97.7 / 97.7 |
| v3 (256) | **96.9 %** | **96.7 %** | **96.5 %** | **247/256** | 0.77 s | 94.8 / 94.9 / 94.1 (241/256) |

v3 misses: `55:53→55:13`, `81:19→69:40`, `37:82→26:66`, `30:1→2:1`,
`26:122→26:9`, `10:43→10:42` ×2 — all textually identical or near-identical
ayahs (same set that blocks `w2v-phonemes`); `107:1→106:4` (short crowd
clip); `100:1` emitted `100:1,100:2` (over-run). It gets all 8 Husary
multi-verse samples that the champion loses.

License of the vendored engine is not stated on the site; treat as
reference-only until clarified. Third-party notices and NPL-1.2 terms for
Zipformer-derived artefacts: see [`NOTICE.md`](../../NOTICE.md).
