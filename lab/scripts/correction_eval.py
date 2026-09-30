"""Score the SDK correction engine on a synthetic mistake set (build_correction_synth.py).

Runs every clip through the zipformer-ctc harness in correction mode (issues
dismissed on sight), then reports per kind: recall (flag on the edited ayah at
word j±1, any word kind), kind-correct recall, exact-word localisation, median
flag latency, precision per flag kind, and false flags per clean minute.

  ZIPFORMER_MODEL=/tmp/models/v3.onnx ZIPFORMER_ORT_DIR=../web/frontend/node_modules \\
    ../.venv/bin/python scripts/correction_eval.py --corpus /tmp/phase0/correction_test --out /tmp/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "scripts"))

WORD_KINDS = {"possible_omission", "possible_substitution", "possible_vowel"}
AYAH_KINDS = {"possible_skipped_ayah", "unclear_ayah"}


def match(sample: dict, flag: dict) -> tuple[bool, bool, bool]:
    """(hit, kind_ok, exact_word) for one flag against one edited clip."""
    kind = sample["kind"]
    if kind == "skip_ayah":
        sk = sample["skipped"]
        hit = flag.get("kind") in AYAH_KINDS and (flag.get("surah"), flag.get("ayah")) == (sk["surah"], sk["ayah"])
        return hit, hit and flag.get("kind") == "possible_skipped_ayah", hit
    if kind in ("clean", "repetition") or sample.get("word") is None:
        return False, False, False
    same_ayah = (flag.get("surah"), flag.get("ayah")) == (sample["surah"], sample["ayah"])
    j = sample["word"]
    hit = same_ayah and flag.get("kind") in WORD_KINDS and abs(int(flag.get("word", -99)) - j) <= 1
    return hit, hit and flag.get("kind") == sample["expected_flag"], hit and int(flag.get("word", -99)) == j


def summarize(rows: list[dict]) -> dict:
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    out = {}
    for kind, rs in sorted(by_kind.items()):
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        out[kind] = {
            "n": len(rs),
            "recall": round(sum(r["hit"] for r in rs) / len(rs), 3),
            "kind_recall": round(sum(r["kind_ok"] for r in rs) / len(rs), 3),
            "exact_word": round(sum(r["exact"] for r in rs) / max(sum(r["hit"] for r in rs), 1), 3),
            "median_latency_s": round(statistics.median(lat), 2) if lat else None,
            "clips_with_any_flag": sum(bool(r["flags"]) for r in rs),
            "spurious_flags": sum(r["spurious"] for r in rs),
        }
    clean = by_kind.get("clean", [])
    clean_min = sum(r["duration"] for r in clean) / 60
    out["clean_false_flags_per_min"] = round(sum(len(r["flags"]) for r in clean) / max(clean_min, 1e-9), 3)
    pred = Counter(f["kind"] for r in rows for f in r["flags"])
    tp = Counter(f["kind"] for r in rows for f, ok in zip(r["flags"], r["flag_ok"]) if ok)
    out["precision_by_flag_kind"] = {k: round(tp[k] / n, 3) for k, n in pred.items()}
    edits = [r for r in rows if r["kind"] not in ("clean", "repetition")]
    out["overall"] = {"recall": round(sum(r["hit"] for r in edits) / max(len(edits), 1), 3),
                      "precision": round(sum(tp.values()) / max(sum(pred.values()), 1), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if str(out).startswith(str(LAB.parent.resolve())):
        raise SystemExit("--out must be outside the repo")
    from tracker_corpus_eval import load_run

    import soundfile as sf

    run = load_run()
    corpus = Path(args.corpus)
    samples = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))["samples"]
    if args.limit:
        samples = samples[: args.limit]
    rows = []
    for smp in samples:
        wav = corpus / smp["file"]
        res = run.recognize(str(wav), mode="correction")
        flags = [{k: f.get(k) for k in ("kind", "surah", "ayah", "word", "wordIndex", "atSeconds")}
                 for f in res.get("corrections") or []]
        m = [match(smp, f) for f in flags]
        hit_idx = next((i for i, x in enumerate(m) if x[0]), None)
        latency = None
        if hit_idx is not None and smp.get("edit_at_s") is not None:
            latency = round(flags[hit_idx]["atSeconds"] - smp["edit_at_s"], 2)
        rows.append({
            "id": smp["id"], "kind": smp["kind"], "duration": sf.info(str(wav)).duration, "flags": flags,
            "flag_ok": [x[0] for x in m], "hit": hit_idx is not None, "kind_ok": any(x[1] for x in m),
            "exact": any(x[2] for x in m), "latency_s": latency, "spurious": sum(1 for x in m if not x[0]),
            "verses": [(v["surah"], v["ayah"]) for v in res.get("verses") or []],
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summ = summarize(rows)
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
