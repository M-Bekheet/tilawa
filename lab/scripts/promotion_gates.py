"""Promotion gates for Beyond-QLab v3 candidates vs the v3 control.

Inputs are the per-clip outputs of `eval_modal.py` (mirrored from
/vol/phase0/eval/<model>/), `tracker_corpus_eval.py` and `correction_eval.py`.

Hard gates (all must pass):
  1. headline PER (q-lab v1.1 refs, all 600 clips) strictly below v3
  2. held-out multi-ayah windows: 0 ayahs >= 50% deleted
  3. non-clean TLOG insertion rate >= INS_FLOOR x v3, on both the q-lab
     tlog_holdout non-clean clips and the TLOG dev non-clean slice
  4. held-out multi-ayah tracker SeqAcc >= v3
  5. correction false flags per clean minute <= v3 (when both provided)
Reported, not gated: madd-free headline, per-source PER, dev PER, dev drops.

  ../.venv/bin/python scripts/promotion_gates.py --eval-dir /tmp/phase0/eval --cand a0-ep2 --base v3 \\
      --holdout-scores /tmp/phase0/holdout_v3.jsonl --tracker-dir /tmp/phase0/tracker --correction-dir /tmp/phase0/correction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INS_FLOOR = 0.85


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.is_file() else []


def rate(rows: list[dict], gold: str, keep=lambda r: True) -> dict:
    n = s = i = d = c = 0
    for r in rows:
        g = r.get(gold)
        if not g or not keep(r):
            continue
        n += g["ref_len"]; s += g["sub"]; i += g["ins"]; d += g["del"]; c += 1
    n = max(n, 1)
    return {"clips": c, "per": round(100 * (s + i + d) / n, 3), "sub": round(100 * s / n, 3),
            "ins": round(100 * i / n, 3), "del": round(100 * d / n, 3)}


def drops(rows: list[dict]) -> int:
    return sum(f >= 0.5 for r in rows for f in r.get("ayah_del_frac", []))


def model_view(eval_dir: Path, name: str, holdout_nonclean: set[str]) -> dict:
    d = eval_dir / name
    q, hm, dev = load(d / "qlab.jsonl"), load(d / "heldout_multi.jsonl"), load(d / "dev_everyayah.jsonl")
    tdev, tnc = load(d / "tlog_dev.jsonl"), load(d / "tlog_dev_nonclean.jsonl")
    v = {"headline": rate(q, "headline")}
    for src in ("everyayah_heldout", "qul_alnufais", "tlog_holdout"):
        v[f"headline_{src}"] = rate(q, "headline", lambda r, s=src: r["source"] == s)
    v["holdout_nonclean"] = rate(q, "ordered", lambda r: r["id"] in holdout_nonclean)
    v["heldout_multi"] = {**rate(hm, "ordered"), "ayahs_dropped": drops(hm)}
    v["dev_single"] = rate(dev, "ordered", lambda r: "ayah_lens" not in r or len(r["ayah_lens"]) == 1)
    v["dev_multi"] = {**rate(dev, "ordered", lambda r: len(r.get("ayah_lens", [1])) > 1),
                      "ayahs_dropped": drops([r for r in dev if len(r.get("ayah_lens", [1])) > 1])}
    v["tlog_dev"] = rate(tdev, "ordered")
    v["tlog_dev_nonclean"] = rate(tnc, "ordered")
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", default="v3")
    ap.add_argument("--holdout-scores", required=True, help="tlog_holdout per-clip filter scores (v3)")
    ap.add_argument("--tracker-dir", default="")
    ap.add_argument("--correction-dir", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    nonclean = {r["id"] for r in load(Path(args.holdout_scores)) if r.get("bucket") != "clean"}
    ev = Path(args.eval_dir)
    base, cand = model_view(ev, args.base, nonclean), model_view(ev, args.cand, nonclean)

    def tracker(name):
        p = Path(args.tracker_dir) / f"{name}_heldout_multi.summary.json" if args.tracker_dir else None
        return json.loads(p.read_text()) if p and p.is_file() else None

    def correction(name):
        p = Path(args.correction_dir) / f"{name}_test.summary.json" if args.correction_dir else None
        return json.loads(p.read_text()) if p and p.is_file() else None

    gates = {
        "headline_per_below_v3": cand["headline"]["per"] < base["headline"]["per"],
        "heldout_multi_zero_drops": cand["heldout_multi"]["ayahs_dropped"] == 0,
        "nonclean_ins_holdout": cand["holdout_nonclean"]["ins"] >= INS_FLOOR * base["holdout_nonclean"]["ins"],
        "nonclean_ins_tlog_dev": cand["tlog_dev_nonclean"]["ins"] >= INS_FLOOR * base["tlog_dev_nonclean"]["ins"],
    }
    tb, tc = tracker(args.base), tracker(args.cand)
    if tb and tc:
        gates["heldout_multi_tracker_seqacc"] = tc["seq_acc"] >= tb["seq_acc"]
    cb, cc = correction(args.base), correction(args.cand)
    if cb and cc:
        gates["correction_false_flags"] = cc["clean_false_flags_per_min"] <= cb["clean_false_flags_per_min"]
    result = {"cand": args.cand, "base": args.base, "promote": all(gates.values()), "gates": gates,
              "metrics": {"base": base, "cand": cand}, "tracker": {"base": tb, "cand": tc},
              "correction": {"base": cb, "cand": cc}}
    print(json.dumps({"promote": result["promote"], "gates": gates}, indent=2))
    for k in base:
        print(f"{k:28s} base {json.dumps(base[k])}\n{'':28s} cand {json.dumps(cand[k])}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
