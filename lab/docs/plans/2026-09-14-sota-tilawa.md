# Plan: a SOTA Tilawa acoustic model (match or beat the alketab prompter engine)

Status: Done 2026-09-16 — see EXPERIMENTS.md; Track C complete.
Owner: Tilawa (offline-tarteel).
Companion: `experiments/zipformer-ctc/` (historical runner alias
`prompter-zipformer`) — native MIT engine harness over Zipformer2-CTC, plus
Quran-Lab reference-model tooling.

**Decision (2026-09-14 16:05):** the public checkpoint is Quran-Lab
`zipformer_p-arabic-v3` / `zipformer_p_arabic_v3.1` (NPL-1.2). Do not train
Zipformer CTC from scratch. Evaluate v3/v3.1 through the vendored tracker, then
fine-tune v3.1 (`--pos-dim 192`).

## 0. Target and definition of done

Reference (alketab `quran_phoneme_zipformer.onnx`, **72.7 MB int8**, 64.7M params,
run through its own tracker). Earlier drafts called this "72.7 MB fp32"; the
artefact is `onnxruntime.quant` int8. fp32 of the same graph is ~258 MB.

| Corpus | Correct | Rec / Prec / SeqAcc |
|---|---|---|
| v1 (53) | 53/53 | 100 / 100 / 100 |
| v2 (43) | 42/43 | 97.7 / 97.7 / 97.7 |
| v3 (256) | 247/256 | 96.9 / 96.7 / 96.5 |

Of the 9 v3 misses, 7 are textually identical/near-identical ayahs
(55:53/55:13, 81:19/69:40, 37:82/26:66, 30:1/2:1, 26:122/26:9, 10:43/10:42 x2).
No acoustic model fixes those. The benchmark ceiling is therefore ~249/256.

**Done when**, with our own model swapped into the same tracker:

1. v3 >= 247/256 and v1 = 53/53 (ties the reference on the saturated public
   sets), measured as median of 3 runs.
2. Strictly beats the reference on the two sets where there is headroom:
   - **`Quran-Lab/quranic-asr-benchmark`** via `benchmark/test_corpus_qlab`
     (~583 samples after mapping drops; 599 metadata rows minus 16 low-match
     everyayah_heldout and 1 tlog dup). `everyayah_heldout` reciters cannot be
     excluded from EveryAyah train+val, so treat that sub-source as soft.
   - **crowd/phone slice** of v3 + v2 (RetaSy + TLOG rows): fewer misses than
     the reference at equal or lower latency.
3. Ships smaller: **int8 ONNX <= 25 MB**, streaming (cache-carrying), RTF <= 0.1
   single-thread WASM on a mid-range phone.
4. Clean license: every training clip is CC-BY / MIT / CC0 / Apache or our own.
   (Tadabur is CC-BY-NC — allowed for an internal ablation run only, never for
   the shipped checkpoint.)

## 1. What we already know about the reference (recovered, exact)

From the site's published source maps and I/O manifest:

| Component | Value | Source |
|---|---|---|
| Features | Kaldi fbank: 25 ms / 10 ms, 80 mel, pre-emphasis 0.97, povey window, 512-FFT, no CMVN, dither off | `kaldiFbank.js` |
| Model | icefall **streaming Zipformer2 CTC**, 6 stacks, layers `2,2,3,4,3,2`, dims `192,256,384,512,384,256`, downsampling `1,2,4,8,4,2`, conv kernels **`31,31,15,15,15,31`**, heads `4,4,4,8,4,4`, left-context 256 frames, window **T=61 / hop 48** (= 480 ms chunk), 16 cache tensors + `embed_states` + `processed_lens`. Reference ONNX is **int8**, 64.7M params. Cache last-dim = `kernel//2`. The plan's original "72.7 MB fp32" and "kernels `15,15,15,7,7,7`" were wrong (those kernels are icefall's librispeech *pruned* recipe). | ONNX `metadata_props` + `zipformer-io.json` (icefall `export-onnx-streaming-ctc.py` naming) |
| Vocab | **251 tokens** = base letters, letter+haraka, doubled letter (shadda) + haraka, madd-length as repetition (`اا`…`اااااا`, `ۦۦۦۦ`, `ۥۥۥۥ`), sukun/qalqalah `ڇ`, ghunna `ۜ`, wasl `ٲ`, `<blank>` | `tokens.js` |
| Targets | phoneme string per word for all 77k words (`quran.json` v2) — i.e. the G2P output is published, we do not need to write the rules | `data/zipformer/quran.json` |
| Decoder | greedy CTC, per-token margin p1-p2 | `ctcDecoder.js` |
| Matcher | 5-gram phoneme index + graded-cost semi-global alignment + per-surah DP tracker + verdicts | `engine/core/*` (vendored) |

Unknowns: training data, epochs/LR, augmentation, whether labels used flowing
or pausal forms. Everything else is fixed by the artefacts above.

## 2. Strategy

Two tracks, run in parallel, converging on the same ONNX I/O contract so every
checkpoint drops into `experiments/prompter-zipformer/harness.mjs` unchanged:

- **Track A — Replicate.** Same architecture, same vocab, same features,
  same tracker. Only the data is ours. Goal: prove the recipe, get a clean-license
  checkpoint that ties the reference. Low risk.
- **Track B — Beat.** Improvements the reference demonstrably lacks (phone-mic
  robustness, waqf-aware labels, repeat/hesitation handling, better ambiguity
  behaviour, smaller footprint). Each is one ablation on top of the Track A
  recipe, one change per run, kept only if it wins on the held-out set.

Matcher/tracker work is deliberately out of scope for the model runs: we hold
the tracker fixed (the vendored one) so acoustic deltas are measurable. A
separate line (Track C) re-implements the matcher ideas natively in
`web/frontend/src/lib/` for shipping.

## 3. Data

### 3.1 Sources (all verified on HF, 2026-09-14)

| Source | Hours | Labels | Condition | License | Role |
|---|---|---|---|---|---|
| `tarteel-ai/everyayah` (have) | ~600 | text -> map to ayah via `fawazahmed0/quran-audio` or `greentechapps/everyayah_curated_1s_20s` which carry `chapter/verse` | studio, ~30 reciters | MIT | core |
| `hetchyy/quranic-universal-ayahs` (QUA) | ~1,350 | surah/ayah, **exact recited text with repeats**, word/letter timestamps, waqf segments, `recording_context` | studio + taraweeh + YouTube, 45 recitations, 4 riwayat | CC-BY-4.0 | core; drop `*_tarteel` slugs (EveryAyah dupes); Hafs-only for v1 |
| `Quran-Lab/QuranTTS` v4 | 301 | surah/ayah, tajweed phoneme tokens, alignment score | clean studio, 16 reciters | **NPL-1.2 (No-Profit)** | **excluded from Track A**. Staging still has `--sources qurantts` for ablation only. Charging for the Work or a model trained on it is prohibited. |
| `IqraEval/Iqra_train` (have) | ~79 | text + phonemes (no ayah id -> match text to `quran.json`) | non-professional readers | unstated | crowd |
| `RetaSy/quranic_audio_dataset` (have) | ~7 | ayah, correctness label | 1,287 non-Arab phone users | unstated | crowd (only `correct` rows) |
| `tarteel-ai/tlog` (have slice) | large | ayah | Tarteel app phone recordings | unstated, gated | crowd; request full access |
| `MoneerProject/warsh_*` | ~30k clips | ayah | Warsh | unstated | Track B riwayah ablation only |
| `zaibihassan/Quranic-Recitation-Data` | 135 reciters x 114 surahs, word timestamps | full-surah opus + protobuf timings | studio | Apache-2.0 | **multi-ayah window synthesis** (see 3.4) |
| `FaisaI/tadabur` | 1,400+ | surah/ayah, word alignments, 600+ reciters | mixed/YouTube | **CC-BY-NC** | internal ablation only |
| `Quran-Lab/quranic-asr-benchmark` | 599 clips | ayah | includes phone-mic | other, auto-gated | **held-out test only** — never train |

Skip as duplicates: FaresElmenshawi, rabah2026, Buraaq, SLR132/deepdml,
MohamedRashad, dev-ahmedhany, `thethanksforthegod/*mega*`.

Clean-license v1 mix: EveryAyah + QUA (Hafs, non-Tarteel) + Iqra +
RetaSy + TLOG slice. QuranTTS is **out** (NPL-1.2). Hours: see the Task 5
staging summary / `EXPERIMENTS.md` Track A section (not the ~2,000 h sketch).

### 3.2 Labels

- Target string per clip = concatenation of `quran.json` word phonemes for the
  labelled ayah span, space-free (the reference corpus is one contiguous
  string; word boundaries live in the tracker, not the model).
- Tokenisation of the target string into the 251-token vocab: greedy
  longest-match over `tokens.js` (the vocab is closed under the corpus by
  construction; assert zero OOV on all 77k words).
- **Basmala**: prepend the basmala phoneme string when the clip audibly starts
  with it (EveryAyah/QUA ayah-1 files usually include it; QUA marks it). Detect
  with a forced-alignment pass (icefall CTC alignment against both
  hypotheses, keep the lower loss).
- **Waqf**: the reference tracker forgives pausal endings (`waqf.js`), so train
  on **flowing forms** (what `quran.json` contains) and let the tracker handle
  stops. Track B ablation: add pausal-form targets for clip-final words using
  `waqf.js` rules ported to Python.
- **Repeats / mistakes**: QUA's "exact recited text" gives real labels for
  repeats. For all other sources, run a first-epoch CTC-loss filter and drop the
  top 2 % loss clips (label noise: wrong ayah, truncated audio, repeats).

### 3.3 Segmentation

- Keep 1–20 s clips for the main mix (EveryAyah curated 1–20 s cut exists).
- Long ayahs (>20 s, e.g. 2:282) are split at QUA/QuranTTS word timestamps
  into 8–15 s windows; targets sliced by word.

### 3.4 Multi-ayah streaming windows (Track B, the biggest expected win)

The reference and our champion are both trained on isolated ayahs; the
benchmark's multi-verse failures come from that. Synthesise long-context
windows:

- From `zaibihassan` full-surah files + word timestamps and QUA taraweeh
  recordings: cut random 10–25 s windows that start/stop at word boundaries,
  possibly mid-ayah, spanning 2–5 ayahs; target = exact phoneme slice.
- From EveryAyah: concatenate 2–4 consecutive ayahs of the same reciter with
  0–800 ms silence, 30 % of the time insert a breath/room-noise gap.
- Ratio: 25 % of training minutes from these windows.

### 3.5 Augmentation (Track A defaults, Track B knobs)

- Speed perturb 0.9 / 1.0 / 1.1 (lhotse standard).
- SpecAugment (icefall defaults).
- **Phone-mic simulation** (Track B): MUSAN noise + RIR (already on Modal:
  `scripts/download_musan_rir_modal.py`), codec passes (Opus 16–32 kbps, AAC
  64 kbps — matches RetaSy/TLOG/app capture), random band-limiting to 4–6 kHz,
  gain jitter. Apply to 40 % of studio clips.
- Lhotse cut mixing of two reciters at −15 to −25 dB (background recitation
  from a neighbouring room / TV — real failure mode in masjid use).

## 4. Model and training (Track A)

- Framework: **icefall** `egs/librispeech/ASR/zipformer`, `--causal 1`,
  CTC head only (`--use-ctc 1 --use-transducer 0`). Reuse recipe, swap the
  data module + tokenizer (custom 251-token `tokens.txt`; bypass BPE).
- Config: `--num-encoder-layers 2,2,3,4,3,2 --feedforward-dim 512,768,1024,1536,1024,768 --encoder-dim 192,256,384,512,384,256 --encoder-unmasked-dim 192,192,256,256,256,192 --cnn-module-kernel 31,31,15,15,15,31 --num-heads 4,4,4,8,4,4 --pos-dim 192 --chunk-size 16,32,64,-1 --left-context-frames 64,128,256,-1` (multi-chunk training; export at chunk 24 / left 256 → T=61/hop 48). Fine-tune uses train chunks `8,16,24` / left `128,256` to match v3.1 `context_profiles`.
- Features: lhotse `Fbank(num_mel_bins=80)`, dither 0, matching `kaldiFbank.js`
  (verify numerically: run one WAV through both, max abs diff < 1e-3).
- Optimiser: ScaledAdam, base LR 0.045, Eden schedule, 40 epochs over 2,000 h,
  max-duration 1,200 s/batch on H100-80 GB, fp16. ~1.5–2 GPU-days.
- Checkpoint averaging: last 10 epochs.
- Modal: `scripts/train_zipformer_ctc_modal.py`, `gpu="H100"`, `cpu=16`,
  volume `zipformer-ctc-training` (manifests, fbank shards as lhotse `LilcomChunkyWriter`, checkpoints, exports). Detached run; write metrics per epoch to the volume.
- Data staging: stream HF -> 16 kHz FLAC -> lhotse cuts on a CPU job first
  (`scripts/prepare_zipformer_data_modal.py`); never mount `data/`.

## 5. Export, quantise, evaluate

- `export-onnx-streaming.py` -> fp32 ONNX with the reference's exact I/O names.
  Diff `inputNames`/`outputNames` against `engine/model/zipformer-io.json`;
  regenerate the JSON from the export if shapes differ.
- Quantise: `onnxruntime.quantization.quantize_dynamic` int8 on MatMul (Zipformer
  is attention/FFN dominated; conv left fp32). Expect ~20 MB. Validate int8 vs
  fp32 on v3: <= 1 sample delta.
- Evaluate with **the same tracker**: `ZIPFORMER_MODEL=<ours>.onnx .venv/bin/python -m benchmark.runner --experiment zipformer-ctc --corpus test_corpus_v3` (3 runs, median) plus the held-out `quranic-asr-benchmark` via a manifest under `benchmark/test_corpus_qlab/`.
- Per-sample diff vs reference (`benchmark/results/*.json` `per_sample`) is the
  review artefact for each run — record in `EXPERIMENTS.md` and the ledger.
- Latency: `harness.mjs` `decodeMs` and a WASM run in the web worker on a
  throttled Chrome profile.

## 6. Track B ablations (one change per run, in priority order)

| # | Change | Hypothesis | Accept if |
|---|---|---|---|
| B1 | +multi-ayah windows (3.4) | fixes over-run/under-run at ayah boundaries, better relocate | fewer multi misses on v3 + q-lab, no single-ayah regression |
| B2 | +phone-mic augmentation (3.5) | crowd slice | fewer RetaSy/TLOG/q-lab phone misses |
| B3 | +QUA exact-recited labels incl. repeats | tracker `repeatCost` path gets cleaner input | fewer `wrong` verdicts on real repeats (measure on QUA held-out) |
| B4 | pausal-form clip-final targets | better `ok` verdicts at stops | verdict precision up on v3 |
| B5 | **small** config (`1,1,2,3,2,1`, dims `128,192,256,384,256,192`) | ≤ 12 MB int8 | ≤ 2 samples worse on v3 |
| B6 | intermediate-layer CTC (`--use-ctc` at stack 4) for lower latency emission | faster `located` | equal accuracy, lower first-lock time |
| B7 | Tadabur 300 h ablation (internal only) | does reciter diversity matter beyond 40 | if yes, hunt a CC-BY equivalent (QUA YouTube configs) |

Ambiguity (identical ayahs) is not a model problem; handle in Track C with
context priors (previous ayah, surah continuity, Fatiha hint) — the reference
already has the `hint` mechanism.

## 7. Track C — ship path (parallel, not blocking A/B)

1. Port `engine/core/*` (corpus, phonemeCost, alignment, search, tracker,
   verdicts, waqf) into `web/frontend/src/lib/` as our own TS implementation
   under our license (algorithms, not the vendored files; the reference code is
   unlicensed).
2. Swap the worker to fbank + streaming Zipformer session with cache tensors;
   keep the FastConformer path behind a flag until parity.
3. Ayah emission from verdicts (≥ 50 % ok/unsure) replaces `VerseTracker`
   commits; port the whole-ayah fallback for short clips.
4. Stability report on v2/v3 (`stability-report.ts`) — must be deterministic like
   the Node harness is.

## 8. Timeline (calendar, one person + Modal)

| Week | Deliverable |
|---|---|
| 1 | Data staging job: manifests for EveryAyah (+ayah ids), QUA, QuranTTS, Iqra, RetaSy, TLOG; label generator + zero-OOV assert; fbank parity test; q-lab test manifest |
| 2 | Track A run 1 (40 ep). Export, quantise, A/B vs reference on v1/v2/v3/q-lab. Write-up |
| 3 | B1 + B2 (parallel H100s). Pick best; int8 export; decide small vs medium |
| 4 | Track C worker swap behind flag; browser stability report; PR |
| 5+ | B3–B6 as time allows; TLOG full-access request outcome; riwayah ablation |

Budget: ~6 H100-days ≈ $300–400 on Modal for A + B1 + B2 + B5.

## 9. Risks

- **HF download volume** (QUA 75 GB, EveryAyah 117 GB): stage once to a Modal
  volume, not locally.
- **Label mismatch between sources' text and `quran.json` spelling** (hamzat
  wasl, alef variants): always key by (surah, ayah) not text; for Iqra map text
  -> ayah with the existing `shared.quran_db` matcher and drop < 0.95 matches.
- **T/hop mismatch on export**: the reference's 61/48 may come from a
  non-default chunk size; treat io.json as generated, not copied.
- **Tracker over-fitted to reference's error profile** (its cost table and
  thresholds were tuned on its model's confusions): expect to re-tune
  `okDistance`/`unsureDistance`/`searchDecisiveDistance` for our model; do it
  on v2 only, verify on v3 + q-lab.
- **Licence of the reference engine** unknown: Track C re-implements; nothing
  vendored ships.

## 10. Immediate next actions

1. ~~Request access: `Quran-Lab/quranic-asr-benchmark` (auto), `tarteel-ai/tlog` full (manual).~~ **Done** — q-lab manifest at `benchmark/test_corpus_qlab` (583 samples). TLOG uses the public `clean` split capped at 100 h.
2. ~~`scripts/prepare_zipformer_data_modal.py`: stage EveryAyah + QUA + QuranTTS to volume, emit lhotse cuts with phoneme targets; print OOV report and hours per source.~~ **Done** (QuranTTS excluded by NPL-1.2; mix is everyayah, qua, iqra, retasy, tlog). Full ingest launched Task 5.
3. ~~fbank parity test (`tests/test_fbank_parity.py`) between lhotse and `kaldiFbank.js` via the Node harness.~~ **Done** — max abs diff **5.96e-4** on real `001002.mp3` (synthetic 1.77e-4), under the 1e-3 budget.
4. ~~`scripts/train_zipformer_ctc_modal.py` from icefall zipformer recipe; smoke-train 1 h subset for 200 steps; export; run through `harness.mjs` to prove the I/O contract before the real run.~~ **Done** — synthetic H100 smoke + fused streaming CTC ONNX; io.json names+dims match reference. 40-epoch `trackA-v1` is the Task 5 launch (after staging gates).
