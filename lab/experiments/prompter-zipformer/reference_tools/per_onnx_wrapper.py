"""PER for Quran-Lab zipformer ONNX on test_corpus_qlab.

Their `quran_per_eval.py` needs the private icefall training package. This
wrapper is the ONNX equivalent: kaldi fbank → cache-aware streaming Zipformer
→ greedy CTC (blank=250) → unit Levenshtein vs `ordered_quran_phonemes.json`.

Usage:
  .venv/bin/python experiments/prompter-zipformer/reference_tools/per_onnx_wrapper.py \\
      --model /Users/rock/ai/projects/offline-tarteel/data/prompter/reference/zipformer_p_arabic_v3.1.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from Levenshtein import distance as lev_distance

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from shared.audio import load_audio  # noqa: E402
from shared.fbank import compute_fbank  # noqa: E402
from shared.prompter_labels import PhonemeTokenizer, load_tokens  # noqa: E402
from zipformer_ctc_utils import io_json_from_session  # noqa: E402

BLANK = 250
MAIN = Path("/Users/rock/ai/projects/offline-tarteel")
REF_DIR = MAIN / "data" / "prompter" / "reference"
DEFAULT_IO = (
    ROOT
    / "experiments"
    / "prompter-zipformer"
    / "engine"
    / "model"
    / "zipformer-io.json"
)


def _norm(s: str) -> str:
    import re

    diac = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
    s = unicodedata.normalize("NFC", str(s))
    s = diac.sub("", s)
    for a, b in [
        ("أ", "ا"),
        ("إ", "ا"),
        ("آ", "ا"),
        ("ٱ", "ا"),
        ("ى", "ي"),
        ("ة", "ه"),
        ("ؤ", "و"),
        ("ئ", "ي"),
    ]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def greedy_ctc(ids: list[int], blank: int = BLANK) -> list[int]:
    out: list[int] = []
    prev = None
    for u in ids:
        if u == blank:
            prev = u
            continue
        if u != prev:
            out.append(u)
        prev = u
    return out


def _init_states(io: dict) -> dict[str, np.ndarray]:
    states = {}
    for inp in io["inputs"]:
        if inp["name"] == "x":
            continue
        dims = [int(d) for d in inp["dims"]]
        dtype = np.int64 if inp.get("dtype") == "int64" else np.float32
        states[inp["name"]] = np.zeros(dims, dtype=dtype)
    return states


def decode_streaming(sess: ort.InferenceSession, io: dict, frames: np.ndarray) -> list[int]:
    t_win = int(io["T"])
    hop = int(io["hop"])
    states = _init_states(io)
    out_names = [o.name for o in sess.get_outputs()]
    ids: list[int] = []
    i = 0
    n = frames.shape[0]
    # 2 s of zero frames (~200 at 10 ms) to drain the encoder, matching the harness.
    pad = np.zeros((200, frames.shape[1]), dtype=np.float32)
    frames = np.concatenate([frames, pad], axis=0)
    n = frames.shape[0]
    while i + t_win <= n:
        x = frames[i : i + t_win][None, ...].astype(np.float32, copy=False)
        feeds = {"x": x, **states}
        outs = sess.run(out_names, feeds)
        by_name = dict(zip(out_names, outs))
        lp = by_name["log_probs"]
        # lp: [1, t, vocab]
        arg = lp.reshape(lp.shape[1], lp.shape[2]).argmax(-1).tolist()
        ids.extend(int(u) for u in arg)
        for name in list(states):
            states[name] = by_name[f"new_{name}"]
        i += hop
    return greedy_ctc(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=str(REF_DIR / "zipformer_p_arabic_v3.1.onnx"),
    )
    ap.add_argument(
        "--corpus",
        default=str(ROOT / "benchmark" / "test_corpus_qlab"),
    )
    ap.add_argument("--io", default=str(DEFAULT_IO))
    ap.add_argument("--gold", default=str(REF_DIR / "ordered_quran_phonemes.json"))
    ap.add_argument("--text2ph", default=str(REF_DIR / "quran_text2phoneme.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"] if isinstance(manifest, dict) else manifest
    if args.limit:
        samples = samples[: args.limit]

    gold_by_key = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    tok = PhonemeTokenizer(load_tokens())

    io = json.loads(Path(args.io).read_text(encoding="utf-8"))
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    built = io_json_from_session(
        sess.get_inputs(),
        sess.get_outputs(),
        model=Path(args.model).name,
        T=io["T"],
        hop=io["hop"],
    )
    io["inputs"] = built["inputs"]

    tot_err = tot_len = seen = perfect = 0
    src_err: dict[str, int] = defaultdict(int)
    src_len: dict[str, int] = defaultdict(int)
    src_n: dict[str, int] = defaultdict(int)
    src_perf: dict[str, int] = defaultdict(int)
    misses = 0
    rows_out = []

    for n, sample in enumerate(samples, 1):
        key = f"{sample['surah']}:{sample['ayah']}"
        entry = gold_by_key.get(key)
        if not entry:
            misses += 1
            continue
        ph = entry["aya_phoneme"] if isinstance(entry, dict) else entry
        ph = str(ph).replace(" ", "")
        try:
            gold_ids = tok.encode(ph)
        except Exception as e:
            misses += 1
            if misses <= 5:
                print(f"gold encode fail {key}: {e}", flush=True)
            continue
        wav = corpus / sample["file"]
        audio = load_audio(str(wav), sr=16000)
        frames = compute_fbank(audio, sr=16000)
        pred_ids = decode_streaming(sess, io, frames)
        err = lev_distance(
            "".join(chr(i + 1) for i in gold_ids),
            "".join(chr(i + 1) for i in pred_ids),
        )
        tot_err += err
        tot_len += len(gold_ids)
        seen += 1
        src = sample.get("source", "ALL")
        src_err[src] += err
        src_len[src] += len(gold_ids)
        src_n[src] += 1
        if err == 0:
            perfect += 1
            src_perf[src] += 1
        rows_out.append(
            {
                "id": sample["id"],
                "source": src,
                "surah": sample["surah"],
                "ayah": sample["ayah"],
                "err": err,
                "gold_len": len(gold_ids),
                "pred_len": len(pred_ids),
            }
        )
        if n % 40 == 0 or n == len(samples):
            per = 100 * tot_err / max(tot_len, 1)
            print(f"  {n}/{len(samples)} running PER={per:.2f}% n={seen}", flush=True)

    print(f"\n=== ONNX PER {Path(args.model).name} on {corpus.name} ===", flush=True)
    print(f"{'source':20s} {'clips':>5s} {'PER%':>7s} {'exact%':>7s}", flush=True)
    summary = {}
    for s in sorted(src_err):
        per = 100 * src_err[s] / max(src_len[s], 1)
        exact = 100 * src_perf[s] / max(src_n[s], 1)
        print(f"{s:20s} {src_n[s]:5d} {per:7.2f} {exact:7.1f}", flush=True)
        summary[s] = {"clips": src_n[s], "per": per, "exact": exact}
    all_per = 100 * tot_err / max(tot_len, 1)
    all_exact = 100 * perfect / max(seen, 1)
    print(f"{'ALL':20s} {seen:5d} {all_per:7.2f} {all_exact:7.1f}", flush=True)
    print(f"gold misses: {misses}", flush=True)
    summary["ALL"] = {"clips": seen, "per": all_per, "exact": all_exact, "misses": misses}
    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "per_sample": rows_out}, indent=2) + "\n",
            encoding="utf-8",
        )
        print("wrote", args.json)


if __name__ == "__main__":
    main()
