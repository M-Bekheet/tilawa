"""Per-clip q-lab PER for streaming Zipformer ONNX exports, two gold variants.

- ``ordered``: `ordered_quran_phonemes.json[surah:ayah]` (what
  `per_onnx_wrapper.py` has always reported; surah:ayah from our
  `test_corpus_qlab` manifest, so everyayah_heldout rows below the 0.95
  match score are absent).
- ``text``: `quran_text2phoneme.json[norm(reference_text)]` over all 600
  `benchmark.jsonl` rows — the Quran-Lab `quran_per_eval.py` gold, including
  the v1.1 corrected references (repeated-ayah qul_alnufais clips).

Decode is ONNX streaming greedy (chunk 24 / left 256), not their batched
PyTorch `model.encode`, so it is still not 1:1 with the published 1.43/3.65/9.10.

Per-clip JSONL (clip ids) goes to --out-dir, which must be outside the repo;
only aggregates belong in EXPERIMENTS.md.

  TILAWA_DATA_ROOT=/path/to/data ../.venv/bin/python scripts/qlab_per_eval.py \\
      --qlab-dir /tmp/qlab --out-dir /tmp/phase0/qlab_per \\
      --model v3=/tmp/models/v3.onnx --model v31=/tmp/models/v31.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

# Escaped: the literal form in quran_per_eval.py reorders under bidi rendering.
_DIAC = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")


def qlab_norm(s: str) -> str:
    """`quran_per_eval.norm`: the key normalisation of quran_text2phoneme.json."""
    s = unicodedata.normalize("NFC", str(s))
    s = _DIAC.sub("", s)
    for a, b in [("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي"), ("ة", "ه"), ("ؤ", "و"), ("ئ", "ي")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def build_rows(qlab_dir: Path, manifest: Path, ref_dir: Path, tok) -> list[dict]:
    ordered = json.loads((ref_dir / "ordered_quran_phonemes.json").read_text(encoding="utf-8"))
    raw = json.loads((ref_dir / "quran_text2phoneme.json").read_text(encoding="utf-8"))
    t2p = {qlab_norm(k): v for k, v in raw.items()}
    ours = json.loads(manifest.read_text(encoding="utf-8"))
    ours = ours["samples"] if isinstance(ours, dict) else ours
    sa_by_id = {s["id"].replace("__", "/"): (s["surah"], s["ayah"]) for s in ours}

    def enc(ph):
        try:
            return tok.encode(str(ph).replace(" ", ""))
        except Exception:
            return None

    rows = []
    for line in (qlab_dir / "benchmark.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        wav = qlab_dir / o["audio"]
        if not wav.is_file():
            continue
        sa = sa_by_id.get(o["id"])
        g_ord = None
        if sa is not None:
            e = ordered.get(f"{sa[0]}:{sa[1]}")
            if e is not None:
                g_ord = enc(e["aya_phoneme"] if isinstance(e, dict) else e)
        ph = t2p.get(qlab_norm(o["reference_text"]))
        rows.append(
            {
                "id": o["id"],
                "source": o["source"],
                "wav": str(wav),
                "surah": sa[0] if sa else 0,
                "ayah": sa[1] if sa else 0,
                "gold_ordered": g_ord,
                "gold_text": enc(ph) if ph is not None else None,
            }
        )
    return rows


_W: dict = {}


def _init(model_path: str) -> None:
    from shared.zipformer_score import StreamingZipformer

    _W["m"] = StreamingZipformer(model_path, threads=1)


def _run(row: dict) -> dict:
    from shared.audio import load_audio
    from shared.fbank import compute_fbank
    from shared.zipformer_score import best_path_logprob, ctc_forced_logprob, edit_counts, greedy_ids

    audio = load_audio(row["wav"], sr=16000)
    lp = _W["m"].log_probs(compute_fbank(audio, sr=16000))
    hyp = greedy_ids(lp)
    out = {k: row[k] for k in ("id", "source", "surah", "ayah")}
    out["duration"] = round(len(audio) / 16000, 3)
    out["hyp_len"] = len(hyp)
    out["hyp"] = hyp
    for g in ("ordered", "text"):
        ref = row[f"gold_{g}"]
        if ref is None:
            out[g] = None
            continue
        ec = edit_counts(ref, hyp)
        flp = ctc_forced_logprob(lp, ref)
        out[g] = {
            "ref_len": ec.ref_len,
            "sub": ec.sub,
            "ins": ec.ins,
            "del": ec.dele,
            "forced_lp_per_token": round(flp / max(len(ref), 1), 5) if flp > -1e30 else None,
            "gap_per_frame": round((best_path_logprob(lp) - flp) / max(lp.shape[0], 1), 5) if flp > -1e30 else None,
        }
    return out


def summarize(per_clip: list[dict]) -> dict:
    res: dict = {}
    for g in ("ordered", "text"):
        agg = defaultdict(lambda: defaultdict(int))
        for r in per_clip:
            s = r.get(g)
            if not s:
                continue
            for src in (r["source"], "ALL"):
                a = agg[src]
                a["clips"] += 1
                a["ref_len"] += s["ref_len"]
                a["sub"] += s["sub"]
                a["ins"] += s["ins"]
                a["del"] += s["del"]
                a["exact"] += int(s["sub"] + s["ins"] + s["del"] == 0)
        res[g] = {}
        for src, a in sorted(agg.items()):
            n = max(a["ref_len"], 1)
            err = a["sub"] + a["ins"] + a["del"]
            res[g][src] = {
                "clips": a["clips"],
                "per": round(100 * err / n, 3),
                "sub": round(100 * a["sub"] / n, 3),
                "ins": round(100 * a["ins"] / n, 3),
                "del": round(100 * a["del"] / n, 3),
                "exact": round(100 * a["exact"] / max(a["clips"], 1), 1),
            }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlab-dir", required=True)
    ap.add_argument("--manifest", default=str(LAB / "benchmark" / "test_corpus_qlab" / "manifest.json"))
    ap.add_argument("--ref-dir", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=path.onnx (repeatable)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from shared.paths import resolve_data_file
    from shared.phoneme_labels import PhonemeTokenizer, load_tokens

    out_dir = Path(args.out_dir).resolve()
    if str(out_dir).startswith(str(LAB.parent.resolve())):
        raise SystemExit("--out-dir must be outside the repo (per-clip ids are not committed)")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(args.ref_dir) if args.ref_dir else resolve_data_file("zipformer/reference/quran_text2phoneme.json").parent
    rows = build_rows(Path(args.qlab_dir), Path(args.manifest), ref_dir, PhonemeTokenizer(load_tokens()))
    if args.limit:
        rows = rows[: args.limit]
    print(f"rows={len(rows)} ordered_gold={sum(r['gold_ordered'] is not None for r in rows)} "
          f"text_gold={sum(r['gold_text'] is not None for r in rows)}", flush=True)

    summaries = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(path,)) as ex:
            per_clip = list(ex.map(_run, rows, chunksize=4))
        (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_clip), encoding="utf-8")
        summaries[name] = summarize(per_clip)
        for g in ("ordered", "text"):
            line = "  ".join(f"{s}={v['per']:.2f}(S{v['sub']:.2f}/I{v['ins']:.2f}/D{v['del']:.2f})" for s, v in summaries[name][g].items())
            print(f"[{name}] {g}: {line}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
