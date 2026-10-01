"""Synthetic word-level mistake set for the correction engine (plan Phase 0 step 6a).

Base audio: single ayahs (and 3-ayah windows) from everyayah.com reciters that
no training manifest contains — the three q-lab reciters for the test split,
the six dev reciters for the dev split. Word boundaries come from a CTC Viterbi
alignment of the v3 ONNX against the per-word phonemes of the tracker corpus;
cuts sit at the midpoint of the gap between words, joined with a 30 ms crossfade.

Kinds (one edit per clip, internal word so both neighbours exist):
  omission      drop word j                      -> expect possible_omission @ j
  substitution  word j <- a different word of the same reciter from another ayah
                                                  -> expect possible_substitution @ j
  repetition    word j said twice                -> no rule exists yet; recorded
  skip_ayah     3-ayah window minus the middle   -> expect possible_skipped_ayah
  clean         unmodified control

Audio + manifest go to --out-dir (outside the repo).

  ../.venv/bin/python benchmark/build_correction_synth.py --split test --model /tmp/models/v3.onnx \\
      --out-dir /tmp/phase0/correction_test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "benchmark"))

from build_heldout_multi import DEV_RECITERS, RECITERS, SR, fetch_pcm  # noqa: E402

FRAME_S = 0.04
XFADE_S = 0.03
KINDS = ("omission", "substitution", "repetition")
EXPECTED_FLAG = {"omission": "possible_omission", "substitution": "possible_substitution",
                 "skip_ayah": "possible_skipped_ayah", "repetition": None, "clean": None}


def word_bounds(model, corpus, tok, pcm: np.ndarray, surah: int, ayah: int) -> list[tuple[float, float]] | None:
    """(start_s, end_s) per word, cut points at mid-gaps between words."""
    from shared.fbank import compute_fbank
    from shared.zipformer_score import ctc_viterbi_spans

    words = corpus.word_phonemes(surah, ayah)
    ids_per_word = [tok.encode(w) for w in words]
    ref = [i for w in ids_per_word for i in w]
    lp = model.log_probs(compute_fbank(pcm, sr=SR))
    spans = ctc_viterbi_spans(lp, ref)
    if spans is None:
        return None
    firsts, lasts, k = [], [], 0
    for w in ids_per_word:
        firsts.append(spans[k][0])
        lasts.append(spans[k + len(w) - 1][1])
        k += len(w)
    dur = len(pcm) / SR
    cuts = [0.0] + [((lasts[i] + firsts[i + 1]) / 2 + 0.5) * FRAME_S for i in range(len(words) - 1)] + [dur]
    return [(max(0.0, cuts[i]), min(dur, cuts[i + 1])) for i in range(len(words))]


def seg(pcm, a, b):
    return pcm[int(a * SR): int(b * SR)]


def join(parts: list[np.ndarray]) -> np.ndarray:
    n = int(XFADE_S * SR)
    out = parts[0].copy()
    for p in parts[1:]:
        if len(out) > n and len(p) > n:
            fade = np.linspace(0, 1, n, dtype=np.float32)
            mixed = out[-n:] * (1 - fade) + p[:n] * fade
            out = np.concatenate([out[:-n], mixed, p[n:]])
        else:
            out = np.concatenate([out, p])
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("test", "dev"), required=True)
    ap.add_argument("--model", required=True, help="ONNX used only for word alignment")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-reciter", type=int, default=70, help="single-ayah base clips per reciter")
    ap.add_argument("--skip-per-reciter", type=int, default=15)
    ap.add_argument("--clean-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf

    from shared.phoneme_labels import PhonemeCorpus, PhonemeTokenizer, load_tokens
    from shared.paths import resolve_data_file
    from shared.zipformer_score import StreamingZipformer

    out = Path(args.out_dir).resolve()
    if str(out).startswith(str(LAB.parent.resolve())):
        raise SystemExit("--out-dir must be outside the repo")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".mp3"
    reciters = RECITERS if args.split == "test" else DEV_RECITERS
    corpus = PhonemeCorpus(resolve_data_file("zipformer/quran.json"))
    tok = PhonemeTokenizer(load_tokens())
    model = StreamingZipformer(args.model)
    rng = random.Random(args.seed)
    nwords = {k: len(v) for k, v in corpus._ayah_words.items()}
    pool = [k for k, n in nwords.items() if 6 <= n <= 20]
    samples = []

    def write(sid, pcm, meta):
        sf.write(out / f"{sid}.wav", pcm, SR, subtype="PCM_16")
        samples.append({"id": sid, "file": f"{sid}.wav", **meta})

    for reciter in reciters:
        tag = reciter.split("_")[0].lower()
        bases = []
        for s, a in rng.sample(pool, args.per_reciter * 2):
            if len(bases) >= args.per_reciter:
                break
            pcm = fetch_pcm(reciter, s, a, cache)
            if pcm is None or len(pcm) / SR > 25:
                continue
            wb = word_bounds(model, corpus, tok, pcm, s, a)
            if wb is None or min(b - a_ for a_, b in wb) < 0.08:
                continue
            bases.append((s, a, pcm, wb))
        for i, (s, a, pcm, wb) in enumerate(bases):
            n = len(wb)
            kind = "clean" if rng.random() < args.clean_frac else KINDS[i % len(KINDS)]
            j = rng.randrange(1, n - 1)
            parts = [seg(pcm, *wb[w]) for w in range(n)]
            meta = {"surah": s, "ayah": a, "ayah_end": None, "reciter": reciter, "kind": kind,
                    "word": None if kind == "clean" else j, "n_words": n,
                    "expected_flag": EXPECTED_FLAG[kind], "expected_verses": [{"surah": s, "ayah": a}]}
            if kind == "omission":
                parts = parts[:j] + parts[j + 1:]
                meta["edit_at_s"] = round(wb[j][0], 3)
            elif kind == "repetition":
                parts = parts[: j + 1] + [parts[j]] + parts[j + 1:]
                meta["edit_at_s"] = round(wb[j][1], 3)
            elif kind == "substitution":
                donors = [b for b in bases if (b[0], b[1]) != (s, a)]
                ds, da, dpcm, dwb = rng.choice(donors)
                target = corpus.word_phonemes(s, a)[j]
                cands = [k for k in range(1, len(dwb) - 1) if corpus.word_phonemes(ds, da)[k] != target]
                k = min(cands, key=lambda k: abs((dwb[k][1] - dwb[k][0]) - (wb[j][1] - wb[j][0])))
                parts[j] = seg(dpcm, *dwb[k])
                meta.update(edit_at_s=round(wb[j][0], 3), donor=f"{ds}:{da}:{k}")
            write(f"corr_{args.split}_{tag}_{kind}_{s:03d}_{a:03d}_{j}", join(parts), meta)
        # skip-ayah: 3 consecutive ayahs, middle one removed
        made = 0
        for s, a in rng.sample(list(nwords), 400):
            if made >= args.skip_per_reciter:
                break
            if (s, a + 2) not in nwords or any(nwords[(s, a + d)] > 15 for d in range(3)):
                continue
            pcms = [fetch_pcm(reciter, s, a + d, cache) for d in (0, 2)]
            if any(p is None for p in pcms):
                continue
            gap = np.zeros(int(0.5 * SR), dtype=np.float32)
            write(f"corr_{args.split}_{tag}_skip_ayah_{s:03d}_{a:03d}", np.concatenate([pcms[0], gap, pcms[1]]),
                  {"surah": s, "ayah": a, "ayah_end": a + 2, "reciter": reciter, "kind": "skip_ayah",
                   "skipped": {"surah": s, "ayah": a + 1}, "expected_flag": EXPECTED_FLAG["skip_ayah"],
                   "expected_verses": [{"surah": s, "ayah": a}, {"surah": s, "ayah": a + 2}]})
            made += 1
    (out / "manifest.json").write_text(json.dumps({"samples": samples}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from collections import Counter

    print(f"wrote {len(samples)} clips to {out}: {dict(Counter(s['kind'] for s in samples))}")


if __name__ == "__main__":
    main()
