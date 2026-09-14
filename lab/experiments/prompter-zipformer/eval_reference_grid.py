"""Run Quran-Lab zipformer ONNX checkpoints through benchmark.runner.

Priority (brief): v3.1 fp32+int8 all corpora ×3, then v3 fp32 on v3/qlab ×3,
then the rest ×1. Resumable via a JSON ledger.

Usage:
  .venv/bin/python experiments/prompter-zipformer/eval_reference_grid.py
  .venv/bin/python experiments/prompter-zipformer/eval_reference_grid.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path("/Users/rock/ai/projects/offline-tarteel")
PY = MAIN / ".venv" / "bin" / "python"
REF = MAIN / "data" / "prompter" / "reference"
LEDGER = ROOT / "benchmark" / "results" / "qlab_v3_eval_ledger.json"
SAVED_RE = re.compile(r"Results saved to (.+\.json)")

MODELS = {
    "v3.1-fp32": REF / "zipformer_p_arabic_v3.1.onnx",
    "v3.1-int8": REF / "zipformer_p_arabic_v3.1.int8.onnx",
    "v3-fp32": REF / "zipformer_p_arabic_v3.onnx",
    "v3-int8": REF / "zipformer_p_arabic_v3.int8.onnx",
}
CORPORA = ("test_corpus", "test_corpus_v2", "test_corpus_v3", "test_corpus_qlab")


def plan(smoke: bool) -> list[tuple[str, str, int]]:
    """Return (model_key, corpus, repeats)."""
    if smoke:
        return [("v3.1-fp32", "test_corpus", 1)]
    jobs: list[tuple[str, str, int]] = []
    for mk in ("v3.1-fp32", "v3.1-int8"):
        for c in CORPORA:
            jobs.append((mk, c, 3))
    jobs.append(("v3-fp32", "test_corpus_v3", 3))
    jobs.append(("v3-fp32", "test_corpus_qlab", 3))
    jobs.append(("v3-fp32", "test_corpus", 1))
    jobs.append(("v3-fp32", "test_corpus_v2", 1))
    for c in CORPORA:
        jobs.append(("v3-int8", c, 1))
    return jobs


def load_ledger() -> dict:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"runs": []}


def save_ledger(data: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def already_done(ledger: dict, model: str, corpus: str, n: int) -> bool:
    hits = [
        r
        for r in ledger["runs"]
        if r.get("model") == model and r.get("corpus") == corpus and r.get("ok")
    ]
    return len(hits) >= n


def run_one(model_key: str, corpus: str) -> dict:
    model = MODELS[model_key]
    env = os.environ.copy()
    env["PROMPTER_DATA_DIR"] = str(MAIN / "data" / "prompter")
    env["PROMPTER_MODEL"] = str(model)
    env["PROMPTER_CORPUS"] = str(MAIN / "data" / "prompter" / "quran.json")
    env["PROMPTER_ORT_DIR"] = str(MAIN / "web" / "frontend" / "node_modules")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        str(PY),
        "-u",
        "-m",
        "benchmark.runner",
        "--experiment",
        "prompter-zipformer",
        "--corpus",
        corpus,
    ]
    print(f"\n>>> {model_key} {corpus}  {datetime.now().isoformat(timespec='seconds')}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    rc = proc.wait()
    out = "".join(chunks)
    m = SAVED_RE.search(out)
    result_path = m.group(1).strip() if m else ""
    metrics = {}
    if result_path and Path(result_path).is_file():
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        row = payload[0] if isinstance(payload, list) else payload
        n = int(row.get("total") or 0)
        seq = [s for s in row.get("per_sample") or [] if s.get("sequence_accuracy") == 1.0]
        metrics = {
            "recall": row.get("recall"),
            "precision": row.get("precision"),
            "sequence_accuracy": row.get("sequence_accuracy"),
            "avg_latency": row.get("avg_latency"),
            "total": n,
            "correct": len(seq),
        }
    return {
        "model": model_key,
        "corpus": corpus,
        "ok": rc == 0 and bool(result_path),
        "returncode": rc,
        "result_path": result_path,
        "metrics": metrics,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ledger = load_ledger()
    jobs = plan(args.smoke)
    for model, corpus, repeats in jobs:
        for i in range(repeats):
            if already_done(ledger, model, corpus, i + 1):
                print(f"skip {model} {corpus} run {i+1}/{repeats} (ledger)", flush=True)
                continue
            rec = run_one(model, corpus)
            rec["repeat_index"] = i + 1
            rec["repeats_planned"] = repeats
            ledger["runs"].append(rec)
            save_ledger(ledger)
            if not rec["ok"]:
                print(f"FAILED {model} {corpus} rc={rec['returncode']}", flush=True)
                sys.exit(rec["returncode"] or 1)
    print("grid done", LEDGER)


if __name__ == "__main__":
    main()
