# Benchmark results

Three test corpora: **v1** (53 samples: user recordings, EveryAyah reference, RetaSy crowdsourced), **v2** (43 samples: RetaSy expanded + EveryAyah multi-verse), and **v3** (256 samples: EveryAyah Alafasy+Husary singles/multis + TLOG-clean crowd-sourced filtered through shipped ONNX + user recordings). v3 exists to reduce the per-sample noise floor: on v1 a one-sample swing is ±1.9pp recall, whereas on v3 the same swing is ±0.4pp.

Metrics: **Recall** = fraction of expected verses found. **Precision** = fraction of emitted verses that were expected. **ExactSetAcc** = deduped emitted set exactly matches expected set, order ignored. **OrderedSeqAcc** = deduped emitted sequence exactly equals the expected ordered sequence. Older changelog entries and tables that say "SeqAcc" used the pre-rename set metric, so read them as ExactSetAcc unless a row explicitly says ordered.

Browser streaming reports now separate durable raw `verse_match` commits from silence-time `final_sequence`. Raw commit metrics remain the engineering guardrail for bad visible emissions. Final-sequence metrics measure the product contract after accumulated streaming evidence has had a chance to smooth or repair early hypotheses.

ONNX inference is non-deterministic at **±3–6 samples per run** on v1 — streaming numbers below are medians over 3 runs (except the deferred-emission changelog entry, which was measured at 5 runs).

## Shipped model

Current browser/runtime model: Cyberistic's full-mixed text CTC FastConformer, `web/frontend/public/fastconformer_full_mixed.onnx` (88 MB, mixed int4 MatMul + int8 Conv/LayerNorm). It takes raw 16 kHz waveform input (`audio_signal`, `length`), emits 1025-token Arabic BPE CTC logprobs, then uses greedy text retrieval plus CTC reranking against `quran_ctc_tokens.json`.

| Mode | Corpus | Recall | Precision | ExactSetAcc | Notes |
|---|---|---|---|---|---|
| **`c2c-direct-mixed-tta` full-file batch** | v1 | **100%** | **100%** | **100%** | Cyberistic champion, median across 3 reproduced runs |
| **`c2c-direct-mixed` full-file batch** | v1 | 98% | 98% | 98% | Same ONNX without 0.9x/1.1x TTA |

The browser worker now uses the non-TTA full-mixed ONNX (`fastconformer_full_mixed.onnx`) because TTA is too expensive for live streaming. Run the v2/v3 stability reports before replacing the historical streaming rows below with post-swap numbers.

Historical pre-Cyberistic browser/RN streaming baseline, using `fastconformer-phoneme v4-tlog` (131 MB quantized ONNX):

| Mode | Corpus | Recall | Precision | ExactSetAcc | Correct |
|---|---|---|---|---|---|
| **Browser/RN streaming** (300ms chunks, `RecitationTracker`) | v2 | **87.9%** | **68.9%** | **55.8%** | 37/43 |
| **Browser/RN streaming** | v3 | **89.3%** | **73.4%** | **58.2%** | 223–225/256 |
| Non-streaming (full-file, single `matchVerse()`) | v1 | 84.1% | 84.9% | 81.1% | 43/53 |
| Non-streaming (full-file, single `matchVerse()`) | v2 | 78.1% | 79.1% | 74.4% | 32/43 |

### Streaming changelog

**2026-04-25 — decode-stability gate on single-cycle commits** (file: `web/frontend/src/lib/tracker.ts`)
A context-sweep diagnostic (`web/frontend/test/diagnose-context-sweep.ts`) measured how the model's CTC greedy decode of audio prefixes compares to its decode of the full audio. On v1 the result was striking: across prefix lengths from 1s to 5s, **~50% of every prefix-decode token gets revised** when full audio context arrives (median LCP / |prefix-decode| ≈ 0.50). Full-audio WER vs the expected phoneme reference is 14%, so the offline ceiling is fine — but every short-prefix decode sits in a regime where half its emissions are non-final because the FastConformer encoder uses bidirectional attention to refine early frames once more audio is in.

The browser's `RecitationTracker` was committing `verse_match` on single-cycle `clearMargin` paths — riding those unstable predictions. The fix gates that one path: track `lastRawPhonemes`, and require the current decode's Levenshtein ratio to the previous cycle's decode be ≥ 0.70 before allowing a single-cycle clearMargin commit. Repeated-leader and finalFlush commits are not gated (they have their own multi-cycle protection). Commit is denied with no diagnostic noise — the tracker either commits in this cycle, defers to the next, or eventually fires via the existing `repeatedLeader` path. Continuation jumps (the next ayah of the verse currently being tracked) are not gated either, since they're not the "early-frame instability" failure mode.

The gate is on by default; set `DECODE_STABILITY_GATE_OFF=1` in env to disable for benchmarking.

Numbers (3-repeat median):
- v3 (256 samples): recall 82.1% → **89.3%** (+7.2pp), precision 64.1% → **73.4%** (+9.3pp), SeqAcc 46.1% → **58.2%** (+12.1pp). Per-run correct [213, 204, 204] → [223, 225, 224]. **Stable-pass 186 → 216 (+30), stable-fail 30 → 25 (−5), flaky 40 → 15 (−25)** — the gate doesn't just lift the median, it makes the pipeline noticeably more deterministic.
- v2 blind check: recall 85.6% → **87.9%** (+2.3pp), precision 66.6% → **68.9%** (+2.3pp), SeqAcc 53.5% → **55.8%** (+2.3pp). Per-run correct [37, 36, 36] → [37, 37, 37]. Smaller gain than v3, consistent with v2 being mostly clean professional recitations where short-prefix decodes are less ambiguous to begin with — the bigger v3 win comes from the 80 noisier TLOG-clean samples where decode stability matters more.

The improvement is roughly an order of magnitude bigger than the prior streaming experiments because it targets a different failure class: not "score threshold tuning" (which the matcher/tracker attempts on 2026-04-21 exhausted) but "the upstream signal that scores are computed from is unreliable until enough context arrives." Three matcher tweaks moved nothing measurable on v1; this one moved v3 SeqAcc 12pp.

Targets specifically: long single-verse and multi-verse samples where an early streaming chunk happened to score well against a wrong verse and got committed before the correct verse's evidence accumulated. On v3 baseline-vs-gated diffs, samples like `tlog_m020_010_105` (got `[20:34]` baseline, suppressed and recovered with gate), `ea_alafasy_034005` (`[22:51, 22:52, 22:53]` → `[22:51]`), `multi_055_001_004` (`[20:5, 55:2, 55:3, 55:4]` → correct on gated runs in v1) flip from stable-fail to stable-pass.

The diagnostic that motivated this: `npx tsx test/diagnose-context-sweep.ts` — for each test sample, runs inference on prefixes [1, 2, 3, 5, 10]s of audio and reports phoneme WER vs the expected reference plus prefix-vs-full-decode stability. Reproduces the ~50% instability finding in ~2 min on a Mac.

Measurement commands:
```
DECODE_STABILITY_GATE_OFF=1 npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v3 --json=test/stab-gate-baseline-v3.json
                            npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v3 --json=test/stab-gate-on-v3.json
DECODE_STABILITY_GATE_OFF=1 npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/stab-gate-baseline-v2.json
                            npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/stab-gate-on-v2.json
```
Raw JSON at `web/frontend/test/stab-gate-{baseline,on}-{v2,v3}.json`. 38 vitest cases pass (no new cases — existing coverage exercises the unchanged commit paths; the gated path is exercised by `stability-report` on real audio because mocked tests run a single cycle and never hit the multi-cycle stability comparison).

**2026-04-22 — silence-flush pending emission on final flush** (commit `508844b`)
When the utterance ends and the tracker has auto-advanced to a pending next-verse emission that never got fresh-audio confirmation, emit the pending message instead of rolling it back — but only when the advance had strong acoustic evidence at the time. Specifically, capture `prefixScore - suffixScore` as `pendingEmissionMargin` at advance time (from the existing `ADVANCE_RELATIVE_MARGIN < 3.0` gate). On `finalFlush`, emit the pending message only when `pendingEmissionMargin < ADVANCE_FLUSH_STRICT_MARGIN` (0.5, much tighter than the normal advance gate). The tighter threshold prevents one-verse overshoot when the reciter actually stopped at the penultimate verse.

Numbers (3-repeat median):
- v3 (256 samples): recall 83.4% → 83.7% (+0.3pp), precision 63.5% → 64.4% (+0.9pp), SeqAcc 44.1% → **46.1%** (+2.0pp). Per-run correct [204, 207, 207] → [209, 212, 208]. **Stable-fail 34 → 28 (−6)** — the six samples gained are the structural win, not variance.
- v2 blind check: recall 82.7% → **85.6%** (+2.9pp), precision 63.7% → 68.1% (+4.4pp), SeqAcc 46.5% → 48.8% (+2.3pp). Same-direction movement on v2 confirms it's not an overfit to v3.

Targets specifically: `multi_114_001_006` (Al-Nas 1-6 dropping verse 6 on silence), `user_ikhlas_2_3` (Al-Ikhlas verses 2-3 dropping verse 3), and similar last-verse-of-span cases where utterance ends before the pending emission could be confirmed by fresh audio. SeqAcc gains more than recall because this fix specifically repairs the last component of ordered sequences, which is exactly what exact-match SeqAcc weighs.

Measurement commands:
```
npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v3 --json=test/silence-flush-v3-stability.json
npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/silence-flush-v2-stability.json
```
Raw JSON at `web/frontend/test/silence-flush-v3-stability.json` and `…-v2-stability.json`. 38 vitest cases pass (including 2 new coverage cases for strict-margin-emits and loose-margin-suppresses).

**2026-04-22 — v3 benchmark corpus (256 samples)** (scripts: `benchmark/build_v3_corpus.py`, `benchmark/augment_v3_corpus.py`, `benchmark/tlog_filter_v3.py`)
After four consecutive falsified streaming experiments (three matcher/tracker + v7 streaming-aug training) all landing inside or just outside the ±3–6-sample v1 variance envelope, the bottleneck became measurement fidelity rather than idea generation. Rebuilt the corpus at ~5× the size.

Sources and composition:
- **EveryAyah singles (140)**: 80 short + 60 medium + 20 long, drawn by reciter-alternating across `Alafasy_128kbps` and `Husary_128kbps`, picked from shuffled Quran pools that don't overlap (surah, ayah) with v1/v2.
- **EveryAyah multi-ayah (20)**: 29 hand-picked 3–6-ayah sequences concatenated via ffmpeg pipe→f32le with 0.5s silence gaps and written at 16 kHz mono. 10 Shatri sequences 404'd (reciter dir not on everyayah.com); Alafasy + Husary sequences all succeeded. `Shatri_128kbps` does not exist on the CDN; six stale 404-HTML files landed on disk as `.mp3` and were removed during pruning, cutting 6 samples.
- **TLOG-clean 80**: replaces RetaSy entirely. TLOG's `clean` split is still noisy at the transcription level, so each candidate was streamed with `Audio(decode=False)`, ffmpeg-decoded to 16 kHz mono, greedy-CTC transcribed through the shipped FastConformer phoneme ONNX, and phoneme-compared against the canonical phoneme string for the filename-referenced (surah, ayah) in `quran_phonemes.json`. Only samples with Levenshtein ratio ≥ 0.75 pass; target 60 medium + 20 long. Hit the target in 12,180 scans (10 s median duration, median ratio 0.95, 2,129 rejected for ratio below threshold + 29 for decode failure).
- **User recordings (2)**: imran_23 + ikhlas_2_3 copied from v1.

Baseline at the time (3-repeat streaming, then-shipped v4-tlog phoneme ONNX):
- Per-run correct [204, 207, 207] / 256
- **Median recall 83.4%**, **precision 63.5%**, **SeqAcc 44.1%**
- Stable-pass 187, flaky 35, stable-fail 34

Compared to v1's 80.9% median recall, v3 shows the shipped pipeline at ~same headline recall but with **ten times the statistical power** per metric (σ of "correct" across the 3 runs is 1.7 samples on v3 vs 0.7 samples on v1, but in relative terms that's ±0.7pp vs ±1.3pp). This means a 3pp streaming improvement is now cleanly visible above noise, where on v1 it was indistinguishable from per-run jitter. Future attempts that were rejected as noise on v1 can be re-measured on v3; narrow tracker experiments (silence-flush final emission, late-verse stitching) now have a realistic path to acceptance.

Tracker raw results JSON: `web/frontend/test/v3-baseline-stability.json`. Both `stability-report.ts` and `benchmark/runner.py` already accept `--corpus=test_corpus_v3` without code changes.

**2026-04-22 — v7 streaming-aug training, falsified** (scaffold kept at `497fc91`, checkpoint discarded)
Curriculum-style fine-tune: start from v4-tlog weights, reuse v5-robust-u6 data (v4-tlog audio was not on the volume anymore), apply streaming-like augmentation (silence prob 0.2→0.6 + range 0.4s→1.5s, shift ±200→±400ms, white_noise prob 0.3→0.5, gain range widened). 3000 steps at LR 2e-5, best `val_loss=14.01` at step 3000 (still decreasing, but training completed as configured).

Ran 3-repeat v1 stability report against the shipped pipeline with the v7 ONNX swapped in. Per-run correct [35, 37, 35] vs v4 baseline [40, 39]; median **recall 71.7% (−9.2pp)**, **precision 60.6% (−6.2pp)**, **SeqAcc 43.4% (−3.8pp)**. Stable-pass 27 (−8), flaky samples 17 (+8). Outside ONNX variance — real regression.

Hypothesized cause: the expanded silence / shift windows shifted the model's output distribution such that in-distribution samples (v5-robust-u6 training data) got noisier CTC decodes at 300ms browser-streaming chunk sizes, not cleaner. The training signal optimizes full-utterance val_loss on full-audio inputs; streaming chunks see more of the augmentation than the full 10–30s training clip does (relative to its content). Put differently: the augmentor perturbs *seconds* of silence on a clip whose content is also seconds long, but at 300ms streaming that same perturbation is a qualitatively different signal.

Shipped ONNX restored to v4-tlog; v7 checkpoint discarded (stays on `fastconformer-phoneme-training` volume for possible revisit). Scaffold code (streaming-aug flag, init-from-checkpoint, manifest-reuse) kept in `scripts/train_fastconformer_phoneme_modal.py` for future training experiments. Raw stability JSON at `web/frontend/test/streaming-attempts-2026-04-21/v7-stream-aug-v1.json`.

**Takeaway:** data-augmentation matching the inference-time distribution is not obviously CTC-safe, even when the transcript is unchanged. A streaming-aware loss (e.g. compute CTC on random sub-windows of the clip) or direct streaming inference during training would be a more faithful approach.

**2026-04-21 — three matcher/tracker attempts, all falsified** (no commit — worktrees discarded)
Three narrow attempts to close the streaming-vs-batch gap. All landed **inside** the ±3–6 sample ONNX variance envelope on 2-run v1; none shipped. Baseline: 35 stable-pass / 9 stable-fail / 9 flaky, medianRecall 80.9%, medianSeqAcc 47.2%, per-run [40, 39] correct.

1. **Rare-phoneme n-gram surah expansion (always-on)** — ported the w2v-phonemes 5-gram rarity vote into `QuranDB.retrieveCandidates` to broaden the Pass 2 surah set when Levenshtein alone put the right surah outside the top-N. Result: seqAcc +0.9pp, **recall −2.2pp, precision −1.6pp**. The extra surah candidates surfaced spans whose coincidental ratio() beat the correct verse. `retasy_024` recovered but other samples regressed in compensation.
2. **Rare-phoneme n-gram, gated to low-text-confidence paths only** — same mechanism, activated only when the primary match is weak. Ship-blocking variance: per-run correct swung **[44, 34]**. One run at 44/53 was the best observed sample count across all attempts, but seqAcc dropped −5.7pp in the median.
3. **Short-text first-match gate** — raise `FIRST_MATCH_THRESHOLD` to 0.82/0.9 when decoded phoneme text is < 15/10 chars, to suppress the "short ambiguous chunk latches onto a distant verse" failure class (retasy_024/025: 1:7 → 82:11; multi_055: 55:1 → 20:5). Result: recall −1.7pp, seqAcc −1.0pp. Target failures still failed identically — the wrong verse wins at a cycle when text is already long enough to clear the gate.

Raw per-sample JSON lives in `web/frontend/test/streaming-attempts-2026-04-21/{baseline-main,ngram-always-on,ngram-gated,short-text-gate}-v1.json`. The working hypothesis is that these nine stable-fail samples are at the **ASR quality floor** (CTC decoded phonemes that genuinely look more like the wrong verse than the right one) and cannot be fixed by matcher/tracker tuning alone. The productive next lever is training-side (v7 streaming-aug fine-tune).

**2026-04-11 — deferred emission** (commit `63774dc`)  
Auto-advanced `verse_match` messages are now held as *pending* until fresh audio produces primary word alignment on the next verse; if tracking stales, the pending emission is silently dropped with full state rollback. This prevents cascades where verse N completing triggers emission of N+1, N+2, … without audio evidence.

v1: precision **53.8% → 66.8%** (+13.0pp), SeqAcc **26.4% → 47.2%** (+20.8pp), recall **78.9% → 78.6%** (−0.3pp). Same pattern on v2 blind check. 0 stable-pass → stable-fail regressions across 5 runs.

Measurement tool: `npx tsx web/frontend/test/stability-report.ts --repeats=5 [--corpus=test_corpus_v2]` produces per-sample stability classification + JSON.

**2026-04-03 — Phase A fixes**  
Short-utterance CTC rescue, span-aware commit, acoustic-dominant override. Also widened our understanding of variance: ONNX is ±3–6 samples/run on v1 (not ±2–3 as previously assumed). Earlier one-shot 45/53 and 50/53 figures sat at the high end of that distribution; the realistic pre-deferred-emission streaming baseline was 40–44/53.

## All experiments — streaming (Python, 3s chunks)

`StreamingPipeline` feeds 3s audio segments to each model, accumulates text into `VerseTracker` for progressive matching. Mirrors the browser pattern but with larger chunks.

| Experiment | Base model | FT | Type | Size | v1 Rec | v1 Prec | v1 Seq | v1 Lat | v2 Rec | v2 Prec | v2 Seq | v2 Lat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **tadabur-whisper-small** | FaisaI/tadabur-Whisper-Small | ✓ | arabic | 461 MB | **87%** | 58% | 42% | 3.3s | **84%** | 58% | 47% | 3.8s |
| **fastconformer-lm-fusion** | nvidia FastConformer | — | arabic | 115 MB | 82% | **66%** | **55%** | **0.8s** | 74% | **59%** | **53%** | **1.0s** |
| fastconformer-ctc-rescore | nvidia FastConformer | ✓ | arabic | 260 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| fastconformer-phoneme | nvidia FastConformer | ✓ | phoneme | 436 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| nvidia-fastconformer | nvidia FastConformer | — | arabic | 115 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| fastconformer-nbest-bruteforce | nvidia FastConformer | — | arabic | 550 MB | 80% | 61% | 49% | 0.8s | 77% | 60% | 51% | 1.0s |
| rabah-pruned-ctc/8L-ft-fn | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 71% | 55% | 42% | 2.7s | 65% | 49% | 40% | 3.4s |
| whisper-lora | whisper-small + LoRA | ✓ | arabic | 485 MB | 64% | 40% | 19% | 5.6s | 72% | 49% | 37% | 6.3s |
| whisper-small | whisper-small | — | arabic | 461 MB | 63% | 42% | 26% | 3.8s | 53% | 33% | 21% | 6.0s |
| rabah-pruned-ctc/12L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 193 MB | 61% | 41% | 25% | 3.4s | 56% | 40% | 33% | 4.4s |
| two-stage | moonshine-tiny + wav2vec2 | ✓ | arabic | 463 MB | 47% | 23% | 13% | 3.7s | 38% | 24% | 19% | 5.8s |
| distilled-ctc | wav2vec2-base (distilled) | ✓ | arabic | 360 MB | 7% | 7% | 6% | 0.5s | 5% | 3% | 2% | 0.5s |

`tadabur-whisper-small` has the highest raw streaming recall but at 3–5× FastConformer latency. FastConformer variants dominate the speed/accuracy/size frontier. `w2v-phonemes` cannot stream — no chunked `transcribe()` path.

## All experiments — batch (Python, full-file)

Full-file transcription then single `matchVerse()` call.

| Experiment | Base model | FT | Type | Size | v1 Rec | v1 Prec | v1 Seq | v1 Lat | v2 Rec | v2 Prec | v2 Seq | v2 Lat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **c2c-direct-mixed-tta** (Cyberistic winning entry) | nvidia FastConformer | — | arabic | **88 MB** | **100%** | **100%** | **100%** | **0.84s** | — | — | — | — |
| **c2c-direct-mixed** | nvidia FastConformer | — | arabic | **88 MB** | 98% | 98% | 98% | **0.72s** | — | — | — | — |
| **prompter-zipformer** (alketab Quran Prompter engine, Node) | streaming Zipformer2-CTC (k2) | — | tajweed-phoneme | 73 MB | **100%** | **100%** | **100%** | 0.99s | **98%** | **98%** | **98%** | 1.04s |
| **w2v-phonemes/large** | hetchyy/r7 | — | phoneme | 970 MB | **100%** | **100%** | **100%** | 15.2s | **95%** | **95%** | **95%** | 30.4s |
| **w2v-phonemes/base** | hetchyy/r15_95m | — | phoneme | 388 MB | — | — | — | — | — | — | — | — |
| **w2v-phonemes/base-local-int8** | hetchyy/r15_95m | — | phoneme | 118 MB | — | — | — | — | — | — | — | — |
| **fastconformer-lm-fusion** | nvidia FastConformer | — | arabic | 115 MB | 95% | 96% | **94%** | 7.2s | **95%** | **95%** | **95%** | 6.6s |
| **nvidia-fastconformer** | nvidia FastConformer | — | arabic | 115 MB | 95% | 95% | 92% | **0.7s** | 93% | 90% | 86% | **0.9s** |
| fastconformer-phoneme | nvidia FastConformer | ✓ | phoneme | 436 MB | 95% | 95% | 92% | 7.9s | 93% | 90% | 86% | 7.1s |
| fastconformer-ctc-rescore | nvidia FastConformer | ✓ | arabic | 260 MB | 95% | 95% | 92% | 7.3s | 93% | 90% | 86% | 6.7s |
| fastconformer-nbest-bruteforce | nvidia FastConformer | — | arabic | 550 MB | 95% | 95% | 92% | 0.6s | 93% | 90% | 86% | 0.9s |
| tadabur-whisper-small | FaisaI/tadabur-Whisper-Small | ✓ | arabic | 461 MB | 86% | 88% | 79% | 1.3s | 87% | 87% | 81% | 1.4s |
| whisper-lora | whisper-small + LoRA | ✓ | arabic | 485 MB | 82% | 86% | 77% | 2.3s | 81% | 84% | 79% | 2.1s |
| rabah-pruned-ctc/8L-ft-fn | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 75% | 75% | 74% | 3.7s | 77% | 77% | 77% | 3.9s |
| whisper-small | whisper-small | — | arabic | 461 MB | 73% | 76% | 68% | 1.0s | 50% | 50% | 47% | 1.1s |
| two-stage | moonshine-tiny + wav2vec2 | ✓ | arabic | 463 MB | 69% | 69% | 66% | 2.3s | 56% | 56% | 51% | 2.2s |
| rabah-pruned-ctc/12L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 193 MB | 63% | 63% | 60% | 5.3s | 67% | 67% | 67% | 5.2s |
| rabah-pruned-ctc/8L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 55% | 55% | 55% | 4.0s | 47% | 47% | 47% | 4.0s |
| rabah-pruned-ctc/6L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 121 MB | 54% | 54% | 51% | 3.3s | 56% | 56% | 56% | 3.1s |
| distilled-ctc | wav2vec2-base (distilled) | ✓ | arabic | 360 MB | 30% | 29% | 26% | 0.6s | 26% | 26% | 26% | 6.2s |

### Phoneme matcher: strategy comparison

Historical ONNX phoneme model via Python `predict()`, swapping out the matching strategy:

| Matching strategy | v1 Recall | v1 SeqAcc | v2 Recall | v2 SeqAcc |
|---|---|---|---|---|
| Simple `ratio()` | 79% | 75% | 87% | 86% |
| **Multi-pass (fragment + span)** | **90%** | **87%** | **87%** | **84%** |

The multi-pass matcher (ported from the browser's `quran-db.ts` — fragment scoring, short-query boost, bismillah stripping, multi-verse spans) adds +11pp v1 recall at zero decode cost. Matching quality was the bottleneck, not decoding.

### 0% recall — broken or inapplicable

| Experiment | Base model | Type | Size | Reason |
|---|---|---|---|---|
| contrastive | HuBERT + AraBERT | embedding | 900 MB | English encoder → useless Arabic features |
| contrastive-v2 | HuBERT + AraBERT | embedding | 367 MB | Same fundamental issue as v1 |
| embedding-search | HuBERT + FAISS | embedding | 397 MB | HuBERT encodes speaker identity, not content |
| ctc-alignment | wav2vec2-xlsr-53-arabic | arabic | 1.2 GB | `transcribe()` path broken; runner uses it |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran | arabic | 290 MB | Model loading errors on all samples |
| streaming-asr | mlx-whisper base | arabic | 145 MB | Needs mlx-whisper (not installed) |
| two-stage-faster-whisper-pruned | faster-whisper + pruned CTC | arabic | — | Needs faster-whisper (not installed) |

## Deep dive: Rabah pruned CTC variants

Layer pruning + optional fine-tuning applied to `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final`.

| Variant | Layers | Pruning | FT | v1 Rec | v1 Seq | v2 Rec | v2 Seq | Lat | Size |
|---|---|---|---|---|---|---|---|---|---|
| 8L-ft-fn-int8 | 8 | first_n | ✓ | **75%** | **74%** | **77%** | **77%** | 3.7s | 145 MB |
| 12L-ft-es-int8 | 12 | evenly_spaced | ✓ | 63% | 60% | 67% | 67% | 5.3s | 193 MB |
| 12L-int8 | 12 | evenly_spaced | — | 62% | 62% | 51% | 51% | 5.5s | 193 MB |
| 8L-ft-es-int8 | 8 | evenly_spaced | ✓ | 55% | 55% | 47% | 47% | 4.0s | 145 MB |
| 6L-ft-es-int8 | 6 | evenly_spaced | ✓ | 54% | 51% | 56% | 56% | 3.3s | 121 MB |
| 8L-int8 | 8 | evenly_spaced | — | 2% | 2% | 0% | 0% | 4.0s | 145 MB |
| 6L-int8 | 6 | evenly_spaced | — | 0% | 0% | 0% | 0% | 3.2s | 121 MB |

`first_n` pruning (keep layers 0–7) beats `evenly_spaced` by ~20pp at the same layer count. Fine-tuning the CTC head is non-optional — unfinetuned pruned models score near 0%.

## Deep dive: TLOG data-mix fine-tunes

Fine-tuning the phoneme CTC head with varying amounts of TLOG (phone-recorded recitation).

| Model | TLOG | Filter | Streaming v1 | Streaming v2 | Notes |
|---|---|---|---|---|---|
| **v4-tlog** (shipped) | ~18K (5/verse) | 0.3 | **45/53 (85%)** † | **32/43 (74%)** † | best checkpoint |
| v5-robust-u6 | 0 (no TLOG) | — | 43/53 (81%) | 33/43 (77%) | removing TLOG also hurts |
| v4-tlog-heavy | ~53K (15/verse) | 0.3 | 36–38/53 (70%) | 25/43 (58%) | regression |
| v4-tlog-hq | ~74K (30/verse) | 0.5 | 29–31/53 (56%) | 23–24/43 (54%) | bigger regression |
| v6-augmented | ~29K (5/verse) | none | 26/53 (49%) NS | — | +MUSAN +teacher relabel, worst |

† v4-tlog figures are single-run; the post-Phase-A median was 40–44/53 v1.

**Takeaways:** ~18K TLOG at filter=0.3 is a genuine sweet spot. Scaling up volume regresses; removing TLOG also regresses; combining multiple data-side changes (v6) makes attribution impossible. **Rule: one data change per training run.**

**v6-augmented failure detail:** unfiltered TLOG (29K) + teacher pseudo-labels on 75% of samples + MUSAN noise aug, all together. Training metrics looked healthy (val_loss=58.39 at step 6500) but downstream accuracy collapsed. Unfiltered TLOG alone contains ~38% bad samples per the quality filter; the teacher relabeler added an unknown additional error rate on the rest. Streaming export also crashed with an ONNX mutex error (NeMo <2.7 compat).

## Zipformer2-CTC (Quran-Lab v3 reference + fine-tunes)

**2026-09-14 owner decision:** the model we were replicating is public `Quran-Lab/zipformer_p-arabic-v3` (NPL-1.2, licence accepted that day). From-scratch Track A training is cancelled; evaluate the published checkpoints through our tracker, then fine-tune them (separate task). HF names are `zipformer_p_arabic_v3{,.1}{.onnx,.int8.onnx,.pt}` (not the short `v3.onnx` from the brief). Modal snapshot `ap-VfKhHbddvJEY8DbR36LckL` → volume `zipformer-ctc-training` `/vol/reference/`. Local copy (gitignored) `data/prompter/reference/`.

**Identity.** Alketab's vendored `quran_phoneme_zipformer.onnx` (72,705,392 B, site URL `?v=31755836`) is **byte- and param-identical** to `zipformer_p_arabic_v3.1.int8.onnx` (sha256 `31755836528da336…`). It is onnxruntime dynamic-int8 (`onnx.infer=onnxruntime.quant`, MatMul QInt8) of the v3.1 madd fine-tune, not v3 and not fp32. Closest non-identical: v3.1 fp32 shares all 555 unquantized initializer hashes (int8 adds 870 quant/scale/zp tensors). All four ONNX exports match default `zipformer-io.json` (T=61, hop=48, left_context 256, 99 in/99 out) — `PROMPTER_IO` not needed. `tokens.txt` has the same 251 ids; Quran-Lab puts `<blank> 250` on line 1, ours on the last line.

**Their card / export.** ~5,400 effective hours/epoch × 10 epochs; train chunk mix `1000:0.5,640:0.35,320:0.15` (chunk frames 8/16/24); export `chunk_size=max(mix)=24`, `left_context_frames=256`, opset 13, then `quantize_dynamic(..., QInt8, op_types=["MatMul"])`. Published PER on quranic-asr-benchmark v1.1: 1.43% held-out studio / 3.65% phone / 9.10% unseen reciter. `quran_per_eval.py` wants a JSONL of `{audio_filepath, text, source?}`, gold via `quran_text2phoneme.json`, and their private icefall `build_model` — it does not run here. Copies under `experiments/prompter-zipformer/reference_tools/`.

### Tracker eval (our harness; median; scores identical across repeats)

| Model | Size | v1 (53) | v2 (43) | v3 (256) Rec/Prec/Seq | qlab (583) | qlab EA / nufais / tlog | lat v3 / qlab |
|---|---|---|---|---|---|---|---|
| **v3.1 fp32** | 251 MB | **53/53** | 42/43 | 96.9 / 96.7 / 96.5 **(247/256)** | **571/583** (97.9%) | 184/184, 193/200, 194/199 | 0.93 s / 0.58 s |
| v3.1 int8 (vendored) | 69 MB | **53/53** | 42/43 | same 247/256 | same 571/583 | same split | 0.75 s / 0.48 s |
| v3 fp32 | 251 MB | 52/53 | **43/43** | same 247/256 | same 571/583 | same split | 0.88 s / 0.58 s |
| v3 int8 | 69 MB | 52/53 | **43/43** | same 247/256 | same 571/583 | same split | 0.75 s / 0.48 s |
| ft-v31 fp32 | 248 MB | 45/53 | 39/43 | 92.3 / 94.5 / 90.2 **(231/256)** | 566/583 (97.1%) | 184/184, 194/200, 188/199 | 0.85 s / 0.56 s |
| ft-v31 int8 | 66 MB | 45/53 | 39/43 | same 231/256 | same 566/583 | same split | 0.68 s / 0.44 s |
| ft-v31 fp32 + `ALLOW_GAPS=1` | 248 MB | 45/53 | 39/43 | 92.3 / 94.5 / 90.2 **(231/256)** | 566/583 (97.1%) | 184/184, 194/200, 188/199 | 1.48 s / — |

v3 vs v3.1 swap one crowd clip: v3 misses `retasy_012` (114:2→114:3); v3.1 misses `retasy_v2_012` (1:3→55:1). v3-corpus and qlab miss *sets* are identical across all four ONNX files. Repeats never differed in correct-count (latency only). Grid: v3.1 fp32+int8 all corpora ×3; v3 fp32 on v3/qlab ×3; v3 fp32 v1/v2 and v3 int8 all ×1. ft-v31 fp32+int8 all corpora ×3 (scores identical across repeats).

**PER (ONNX streaming greedy, v3.1 fp32, qlab):** overall **5.56%** (exact 49.1%); everyayah_heldout 2.39%, qul_alnufais 7.78%, tlog_holdout 7.06%. Not comparable 1:1 to the card (different decode, gold is `ordered_quran_phonemes.json` by surah:ayah, torchaudio kaldi fbank). Wrapper: `experiments/prompter-zipformer/reference_tools/per_onnx_wrapper.py`.

**PER (ft-v31 fp32, same wrapper):** overall **4.37%** (exact 61.6%); EA 1.69%, nufais 4.59%, tlog 8.18%. int8 4.35%. Acoustic PER improved vs v3.1; tracker SeqAcc did not.

Reproduction:

```bash
cd .worktrees/sota-tilawa
PROMPTER_DATA_DIR=/Users/rock/ai/projects/offline-tarteel/data/prompter \
PROMPTER_MODEL=/Users/rock/ai/projects/offline-tarteel/data/prompter/reference/zipformer_p_arabic_v3.1.onnx \
PROMPTER_CORPUS=/Users/rock/ai/projects/offline-tarteel/data/prompter/quran.json \
PROMPTER_ORT_DIR=/Users/rock/ai/projects/offline-tarteel/web/frontend/node_modules \
/Users/rock/ai/projects/offline-tarteel/.venv/bin/python -m benchmark.runner --experiment prompter-zipformer --corpus test_corpus_v3
# full grid: .venv/bin/python experiments/prompter-zipformer/eval_reference_grid.py
# fetch: modal run scripts/fetch_reference_zipformer_modal.py
# identity: .venv/bin/python scripts/onnx_compare.py --vendored data/prompter/quran_phoneme_zipformer.onnx --candidates data/prompter/reference/zipformer_p_arabic_v3*.onnx
```

Raw JSON: `benchmark/results/2026-09-14_16*.json` / `_17*.json` / `_18*.json`; ledger `benchmark/results/qlab_v3_eval_ledger.json`; PER `benchmark/results/v31_fp32_qlab_per.json`.

### Fine-tune data mix (staged 2026-09-15)

Volume `zipformer-ctc-training` `/manifests/`. All five `*_cuts_fbank.jsonl.gz` exist. OOV 0. Speed perturb ×3 at train time.

| source | clips | hours | notes |
|---|---|---|---|
| everyayah | 128,204 | 358.0 | train+validation, 1 skip_short |
| qua | 239,975 | 747.2 | fbank 11/12 shards ≈ 685 h perturbed; shard 11 hung twice, skipped |
| iqra | 16,070 | 21.5 | 55,321 low_match dropped (MSA+Quran mix) |
| retasy | 357 | 0.36 | `correct` only; icefall drops ~3 s clips labelled 2:255 |
| tlog | 41,083 | 100.0 | qlab tlog_holdout ids excluded |
| **total (raw)** | **425,689** | **1,227** | ~2,700 h with ×3 perturb |

### ft-v31 (NOT PROMOTED)

Fine-tune from Quran-Lab v3.1 `.pt` (`--init-from`, inverse-permute CTC blank 250→0, `--pos-dim 192`). Config: 5 epochs, `--max-duration 1200`, `--avg 3` (epochs 2–5), `--base-lr 0.005`, `--warmup-batches 500`, chunks `8,16,24`, left `128,256`. GPU `H100:4` DDP after `201f8d1` (worker `world_size=torch.cuda.device_count()`). Train app `ap-6In0UqeDBVXxeLVFXw1aUe` (cancelled first client `ap-a4vuTLIWmrU9rOEoVMrX3g` had `world_size=1` because `ZIPFORMER_GPU` was unset on the worker). Volume `/vol/exp/ft-v31`, export `/vol/exports/ft-v31` (`model.onnx` 259.6 MB, `model.int8.onnx` 69.2 MB, T=61 hop=48, `io_diff=[]`). Trained on the 11/12 QUA fbank merge (shard 11 later finished; this run did not see it).

CTC first-batch 0.6411 (pretrained, <1.0). `metrics.jsonl`: ep1 train 0.1066 / valid 0.0884; ep5 train 0.0796 / valid 0.0324. Wall 9536 s (~2.65 h). Max GPU mem ~18 GB / 80 GB. Epoch-3 valid 0.0342 is worse than ep4/5 — no extra `--epoch 3 --avg 1` export.

Promotion bar (strict): qlab ≥ 572 AND v1 = 53 AND v3 ≥ 247 vs v3.1 baseline 571 / 53 / 247. **NOT PROMOTED** — qlab 566/583, v1 45/53, v3 231/256. tlog_holdout 188/199 vs 194; nufais 194/200 vs 193 (the only slice that improved). int8 matches fp32 scores, ~20% faster.

v1/v3 regressions are mostly multi-ayah truncation (first 1–2 ayahs only). qlab: −6 tlog_holdout, +1 nufais (`qul_alnufais__37_43`). ONNX PER **4.37%** (EA 1.69 / nufais 4.59 / tlog 8.18; exact 61.6%) vs v3.1 **5.56%** — acoustics improved, tracker SeqAcc did not.

Epoch sweep (fp32 `--avg 1` except ep5 avg 3; one deterministic harness run):

| ckpt | v1 | v3 |
|---|---|---|
| v3.1 init | **53/53** | **247/256** |
| ep1 avg1 | 46/53 | 225/256 |
| ep2 avg1 | 44/53 | 214/256 |
| ep5 avg3 | 45/53 | 231/256 |

Most of the damage is epoch 1; epoch 2 is the trough; epoch 5 recovers some v3 but not v1. Not a clean “more FT → worse” slope.

**Failure mode (ep5, 5 v3 multi clips):** mixed acoustic + matcher, acoustic first. CTC transcript **drops short connecting ayahs** (Fatiha 1:3/1:4/1:6 absent; Fil 105:2/105:4 absent; 25:66 head absent). Later ayahs that *are* in the transcript often still get tracker tallies (`ok` full), but (1) `MIN_WORD_FRACTION=0.5` rejects partials (109:4 ok+unsure=2/5 words; 25:66 unsure=1/4) and (2) `predict()` `_contiguous_head` stops at the first gap so SeqAcc looks like prefix truncation (1:1–7→1:1–2 even though 1:5 and 1:7 were emitted). Needs B1 multi-ayah windows and/or tracker re-tune (`okDistance`/`unsureDistance`/word-fraction + don't truncate at holes).

Raw JSON: `benchmark/results/2026-09-15_18*.json` / `_19*.json`; ledger `benchmark/results/ft_v31_eval_ledger.json`; PER `benchmark/results/ft_v31_fp32_qlab_per.json`. Epoch-1/2: `2026-09-15_194659.json` (v1), `_195041.json` (v3), `_195131.json` (v1), `_195511.json` (v3). Export apps `ap-6aM9VJL2SXHIT54dfZMBMr` (ep1), `ap-MAxnoFmA6Ahh017w6RDpeu` (ep2).

### E3 tracker re-tune (diagnostic, NOT PROMOTED)

Matcher-only probe on ft-v31 ep5 avg-3 (`sha256` `7c7f0f4d…`). v2 grid of 12 configs: only `PROMPTER_ALLOW_GAPS=1` appeared to move v2 (40/43), via `_contiguous_head` inventing a corpus-short hole. Fix round 1 requires both neighbours already emitted and `ok+unsure≥1`, and Python skips a hole only if that ayah is already in `verses`. After that, ft-v31+gaps = default: **45/53, 39/43, 231/256, 566/583**. The three “recovered” clips (`multi_036_001_005`, `ea_multi_056_001_004`, `ea_alafasy_multi_095_001_005`) revert. Reference + gaps stays **53/247/571**. Not a promotion candidate — the loss is acoustic, not a prefix-fill matcher bug. Raw: grid `2026-09-15_205959.json`–`_211248.json`; honest re-verify `_220245` (v1), `_222224` (v2), `_220911` (v3), ref `_221000`/`_221313`/`_222037`.

## Per-experiment notes

**c2c-direct-mixed-tta** — Cyberistic's winning entry and current champion. It runs the mixed int4+int8 FastConformer ONNX once at 1.0x speed, skips augmentation for confident predictions, and only runs 0.9x/1.1x speed-perturbed passes on low-confidence samples. Reproduced locally over 3 runs at 100% recall, 100% precision, and 100% sequence accuracy on v1 (53 samples), with 0.84s average latency.

**c2c-direct-mixed** — Same CTC re-rank algorithm without TTA, using `web/frontend/public/fastconformer_full_mixed.onnx` (88 MB). This is the model now loaded by the browser worker. Reproduced at 98% recall / 98% precision / 98% sequence accuracy on v1 at 0.72s average latency; TTA recovers the remaining miss.

**prompter-zipformer** — The engine behind https://prompter.alketab.app/ (`@alketab/quran-engine`, recovered from its published source maps and run unmodified under Node via onnxruntime-node; see `experiments/prompter-zipformer/README.md`). Streaming Zipformer2-CTC over a 251-token tajweed-phoneme vocab (letters+harakat, madd length as repetition), greedy CTC, then a whole-Quran 5-gram phoneme index + graded-cost semi-global alignment to locate, and a per-surah online DP tracker with per-word verdicts. The wrapper replays the app's Recognize-mode host loop and emits ayahs with ≥50% ok/unsure words; because the live engine refuses to lock on short clips, a whole-ayah nearest-match fallback over the same phoneme distance handles clips where no lock happened (pure engine: 74% v1; with fallback: 100%). Deterministic across runs (no ±3–6 sample jitter). **v1 53/53, v2 42/43, v3 247/256 (96.9% / 96.7% / 96.5%), qlab 571/583 (97.9%)**. The 72.7 MB ONNX is **Quran-Lab `zipformer_p_arabic_v3.1.int8.onnx`** (byte-identical, NPL-1.2 accepted 2026-09-14), not fp32 — see `## Zipformer2-CTC (Quran-Lab v3 reference + fine-tunes)`. vs champion `c2c-direct-mixed-tta` at 241/256 (94.8% / 94.9% / 94.1%) on the same v3 run — it takes all 8 Husary multi-verse samples the champion loses. Remaining v3 misses are textually identical/near-identical ayahs (`55:53→55:13`, `81:19→69:40`, `37:82→26:66`, `30:1→2:1`, `26:122→26:9`, `10:43→10:42`), plus one short crowd clip (`107:1→106:4`) and one over-run (`100:1` → `100:1,100:2`). Takeaways for our stack: (1) a phoneme alphabet that encodes harakat + madd length gives the matcher far more discriminative chars per second than BPE text; (2) locate-then-track with graded substitution costs beats per-chunk `matchVerse()` on multi-verse; (3) the shipped 69 MB int8 is already the reference checkpoint — fine-tune that, don't train from scratch. Runs at ~5% RTF single-threaded CPU. Engine JS license still unstated; the acoustic model is NPL-1.2.

**ctc-alignment** — CTC forced alignment with `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB). Scores verses directly against frame-level logits via the CTC forward algorithm, skipping greedy-decode information loss. Too large (6×) and too slow (5×) for on-device.

**nvidia-fastconformer** — `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`. Best speed/accuracy/size balance for streaming. A fine-tune sweep (v1, v2a, v2b, v3c) failed to beat the zero-shot baseline.

**fastconformer-ctc-rescore** — Two-stage: FastConformer ASR + CTC re-score top-50 candidates with the fine-tuned 8L Rabah head. Re-scoring doesn't recover failures — both models miss the same hard cases (short isolated letters, multi-verse).

**fastconformer-nbest-bruteforce** — N-best beam search + CTC brute-force. Regressed vs baseline: beam candidates without an LM are near-identical. A Quran-specific LM or constrained decode would be needed.

**fastconformer-lm-fusion** — FastConformer + pyctcdecode Quran LM. Best batch SeqAcc (94% v1, 95% v2) but too much added latency for streaming and awkward in-browser.

**fastconformer-phoneme** — Fine-tuned FastConformer CTC head on a 69-phoneme Buckwalter vocab. Former shipped ONNX model (`fastconformer_phoneme_q8.onnx`, 131 MB), now kept for historical experiments and streaming-regression comparisons. Trained on 71K Iqra + 55K TTS + 1.8K RetaSy + ~18K filtered TLOG.

**w2v-phonemes** — Phoneme CTC + Levenshtein matching. `large-int8` (r7, 970 MB INT8 ONNX) hits **100% batch on v1 and 96.1% / 96.1% / 96.1% (recall/precision/SeqAcc) on v3** — the strongest batch oracle we have, but 1 GB is too large to ship to browser. `base` (r15_95m, 388 MB fp32) is now accessible with Ahmed's read token and hit **97.1% / 97.1% / 97.1%** on the downloadable EveryAyah slice of v3 (174 samples, avg 0.90s CPU, result `benchmark/results/2026-04-29_091708.json`). A Modal-exported local dynamic-int8 ONNX (`base-local-int8`, 118 MB, artifacts in `data/r15-onnx/` or Modal volume `w2v-phonemes-r15`) preserved the same **97.1% / 97.1% / 97.1%** on that slice, with avg CPU latency 1.11s (result `benchmark/results/2026-04-29_100633.json`), and scored **96.0% / 96.1% / 95.7%** on full v3 (256 samples, avg 0.89s CPU, result `benchmark/results/2026-04-29_103225.json`). Both fp32 and int8 fail the same five EveryAyah short/repeated-phrase collisions (`55:53→55:13`, `81:19→69:40`, `37:82→26:66`, `30:1→2:1`, `26:122→26:9`), so the remaining batch error is mostly context/ambiguity rather than acoustic quality. A phoneme-aware naive chunked baseline (`predict_streaming`, 3s independent chunks) scored only **20.6% / 12.2% / 3.9%** on full v3 (result `benchmark/results/2026-04-29_103627.json`), confirming r15 is a batch/verifier model, not a true streaming model. O(T²) wav2vec2 attention and independent chunk CTC collapse are the blockers; use r15/r7 as verifier/teacher while true streaming should be cache-aware FastConformer RNNT/CTC. As of 2026-04-22 `_decode_phonemes` chunks audio >25s into 25s windows with 1s overlap, each independently CTC-collapsed then concatenated — without chunking, a single 200s sample bloats memory to 22 GB and effectively hangs on Apple Silicon's ArmKleidiAI MatMul path. Upstream `base-int8` (`hetchyy/r15_95m_onnx_int8`) still returns 404 on HF; our local `base-local-int8` entry is shown only when `data/r15-onnx/model_int8.onnx` or `R15_ONNX_DIR/model_int8.onnx` exists. HF token required for fp32.

Use case: r7 remains the highest-accuracy distillation teacher; r15 is now a plausible server-side/batch verifier and quantization candidate if a real int8 export can be produced.

**tadabur-whisper-small** — Best Whisper fine-tune we tested. Highest streaming recall (87% v1) at 3× FastConformer latency.

**rabah-pruned-ctc** — Layer-pruned Rabah CTC; see deep-dive above.

**two-stage** — Moonshine Tiny Arabic (103 MB) for fast ASR + CTC re-score on top 50 candidates, falling back to a large CTC. Blocked on the small CTC model.

**whisper-lora / whisper-small** — Whisper-small base + optional LoRA. LoRA helps vs base; both trail FastConformer, especially streaming.

**distilled-ctc (failed)** — wav2vec2-base knowledge-distilled from a large CTC teacher. English-only pretraining means no usable Arabic speech features.

**contrastive / contrastive-v2 / embedding-search (failed)** — All three failed for the same reason: English-pretrained audio encoders (HuBERT, wav2vec2-base) don't produce useful features for Arabic.

## Key findings

1. **FastConformer dominates for streaming.** Best speed/accuracy/size tradeoff across every viable experiment.
2. **CTC forced alignment is the most accurate batch approach**, but too large (1.2 GB) for on-device.
3. **ASR quality is the bottleneck.** All ASR-based approaches fail on the same samples.
4. **English-pretrained audio encoders fail on Arabic.** wav2vec2-base, HuBERT, Moonshine can't produce useful features.
5. **Pruning + fine-tuning works.** 24→8 layers with `first_n` pruning + CTC fine-tuning recovers most accuracy (75% at 145 MB).
6. **Short verses are hard across all approaches** — under 3–4 words doesn't give enough signal.
7. **Matching quality matters more than decode strategy.** Multi-pass phoneme matching takes Python batch from 79%→90% v1. pyctcdecode beam is worse than greedy for this model.
8. **Beam-candidate injection into the tracker regressed.** The verse/span trie (1.7M nodes, 2.2ms decode) works correctly, but beam-matched verses override correct greedy results. Surah-level expansion is the safer next step.
9. **TLOG: one quality-filtered bucket wins.** ~18K filtered at 0.3 is the sweet spot; more volume, lower filter, no TLOG, or combined data changes all regress.
10. **Streaming precision had a cascade bug.** Auto-advanced `verse_match` messages emitted without audio evidence. Deferred emission (2026-04-11) fixes it: +13pp precision, +20.8pp SeqAcc on v1.
11. **Cyberistic's text CTC rerank moved the batch ceiling.** `c2c-direct-mixed-tta` is now the v1 champion at 100% / 100% / 100% with an 88 MB ONNX. The previous v4-tlog phoneme model remains useful as a historical streaming baseline, but new shipped runtime work should start from `fastconformer_full_mixed.onnx`, `vocab.json`, and `quran_ctc_tokens.json`.
12. **r7 (Ahmed's 1B wav2vec2 phoneme CTC) is still a strong v3 batch oracle.** 96.1% / 96.1% / 96.1% on v3 (256 samples, full-file batch), but 1 GB is too large to ship and wav2vec2 attention is not streaming-friendly. Use r7/r15 as teacher/verifier candidates, not the browser runtime.
13. **v3 SeqAcc is mostly a tracker state problem, not a recognizability problem.** Exact-match diagnostics (`web/frontend/test/analyze-v3-stability.ts`) show the v3 gap is dominated by extra emissions: cached streaming exact-fail runs include 124 `extra_after_expected` and 29 `wrong_surah_jump` cases across 768 runs. Comparing those cached streaming outputs against the r7 batch oracle (`web/frontend/test/compare-streaming-oracle.ts --stability-json=... --oracle-results=benchmark/results/r7-v3-batch.json`) shows the first long/medium exact-fail samples are `streaming_tracker_loss`: r7 predicts the exact expected verse while streaming emits expected+extras. The old phoneme ONNX full-file path was too weak to serve as this oracle; it often missed the expected verse on those same long clips. Two tempting runtime invariants were falsified and reverted: consuming the buffer after evidence-backed stale exits, and blocking selected candidates dominated by the current fusion leader. The next tracker attempt needs explicit segment ownership / active-hypothesis comparison, not score-threshold or rank gates.
14. **Alketab ships Quran-Lab v3.1 int8.** `quran_phoneme_zipformer.onnx` is byte-identical to `Quran-Lab/zipformer_p-arabic-v3` `zipformer_p_arabic_v3.1.int8.onnx` (NPL-1.2). Tracker scores are deterministic; v3.1 vs v3 only swap one v1/v2 crowd clip. Fine-tune v3.1 rather than training Zipformer CTC from scratch.
15. **A 5-epoch full-mix fine-tune of v3.1 at lr 0.005 dropped tracker SeqAcc even as CTC valid loss fell 0.088→0.032 and ONNX PER 5.56%→4.37%.** ft-v31: v1 53→45, v3 247→231, qlab 571→566 (tlog −6, nufais +1). Failures are mostly multi-ayah truncations, not wrong-surah. Lower LR / fewer epochs / freeze encoder next — do not ship this checkpoint.
16. **Fine-tuning v3.1 on isolated-ayah data lowers PER but breaks multi-ayah tracking; needs B1 multi-ayah windows and/or tracker re-tune.** Epoch sweep (fp32): v1 53→46→44→45 and v3 247→225→214→231 at init/ep1/ep2/ep5. CTC skips short connecting ayahs (acoustic); `MIN_WORD_FRACTION=0.5` plus `_contiguous_head` then report only the prefix (matcher).

## Methodology

- **Batch:** experiment's `transcribe()` processes the full audio file. `StreamingPipeline` matches transcript against all 6,236 verses via Levenshtein. Per-sample R/P/SeqAcc, averaged.
- **Python streaming:** 3s chunks, independent transcription per chunk, accumulated text fed to `VerseTracker` for progressive matching.
- **Browser/RN streaming:** `RecitationTracker` feeds 300ms chunks through ONNX with a 4s silence tail to flush discovery. Current runtime uses Cyberistic's raw-audio `fastconformer_full_mixed.onnx`; older stability artifacts before the swap used `fastconformer_phoneme_q8.onnx`.
- **Latency:** wall-clock per sample, excluding first-sample warmup. Apple Silicon (CPU).
- **Variance:** ONNX inference is non-deterministic at ±3–6 samples/run on v1. Always report medians over 3 runs (max).

Raw JSON results live in `benchmark/results/`. Stability JSON from streaming runs lives in `web/frontend/test/*-stability.json`.

## Roadmap

Designs in `docs/plans/` for the work remaining between 78.6% streaming recall and the 95% target:

- **Curriculum / hard-example fine-tune (v7)** — start from v4-tlog, short low-LR second stage weighted by current failure buckets: short/noisy RetaSy, huruf-muqatta'at openers, clipped-start TLOG.
- **Streaming-like augmentation** — explicit start/end truncation, mild reverb, random short-window crops, adjacent-ayah concatenation. Current augmentor only has speed/gain/noise/shift/silence; the model never sees what streaming actually produces.
- **Phoneme n-gram anchoring in the browser matcher** — port rare-phoneme voting from `experiments/w2v-phonemes/` into `quran-db.ts` for surah-level expansion when `ratio()` is weak.
- **Teacher distillation (w2v-phonemes/large → FastConformer)** — use the 100%-batch teacher to generate soft labels. The earlier failed distillation used English wav2vec2-base as the student; that's what falsified, not the distillation idea.
- **Segment-aware tracker state** — replace implicit "current rolling buffer" ownership with explicit audio segments / active verse hypotheses. Diagnostics show stale exits after real word/acoustic progress can replay already-assigned audio through open discovery, causing expected+extra cascades. A safe fix should compare rediscovery candidates against the active verse/segment hypothesis before emitting, rather than relying on elapsed-time lockouts, rank gates, or buffer clearing.
- **Deferred A4 — gated trie beam candidate expansion** — expand the candidate surah set (don't inject direct candidates). Beam infrastructure already wired in `inference.ts`. A partial beam-derived surah-expansion probe was reverted because uncalibrated beam hints still pushed wrong-initial/wrong-surah paths; any future beam hint must first prove calibration against diagnostics.
