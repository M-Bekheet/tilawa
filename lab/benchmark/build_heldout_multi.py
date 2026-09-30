"""Held-out-reciter multi-ayah windows for the multi-ayah PER / bridge gate.

The v3 corpus multi clips (Alafasy / Husary) are in the fine-tune mix and in
Quran-Lab's own training, so they cannot gate promotion. This builds 2–4-ayah
windows from the three q-lab held-out EveryAyah reciters (absent from v3's
training per the model card, and from our *_rx manifests), concatenated with
0.5 s silence like `build_v3_corpus.py`. Half the windows contain a short
(<= 4 word) "bridge" ayah, the class FT dropped (EXPERIMENTS.md, ft-v31
failure mode).

Output (audio + manifest.json in the v3 schema) goes to --out-dir, outside the
repo: the clip list is eval data and is not committed.

  ../.venv/bin/python benchmark/build_heldout_multi.py --out-dir /tmp/phase0/heldout_multi
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))

RECITERS = ("Sahl_Yassin_128kbps", "Akram_AlAlaqimy_128kbps", "Muhsin_Al_Qasim_192kbps")
BASE_URL = "https://everyayah.com/data"
SR = 16000
GAP_S = 0.5
MAX_WINDOW_S = 30.0
BRIDGE_MAX_WORDS = 4


def ayah_words(db) -> dict[tuple[int, int], int]:
    return {(v["surah"], v["ayah"]): len(v["text_clean"].split()) for v in db.verses}


def candidate_windows(words: dict[tuple[int, int], int], n_min=2, n_max=4, max_words=40):
    by_surah: dict[int, int] = {}
    for s, a in words:
        by_surah[s] = max(by_surah.get(s, 0), a)
    bridge, plain = [], []
    for s, n_ayahs in by_surah.items():
        for a0 in range(1, n_ayahs + 1):
            for k in range(n_min, n_max + 1):
                a1 = a0 + k - 1
                if a1 > n_ayahs:
                    break
                ws = [words[(s, a)] for a in range(a0, a1 + 1)]
                if sum(ws) > max_words:
                    continue
                (bridge if min(ws) <= BRIDGE_MAX_WORDS else plain).append((s, a0, a1))
    return bridge, plain


def fetch_pcm(reciter: str, surah: int, ayah: int, cache: Path) -> np.ndarray | None:
    mp3 = cache / reciter / f"{surah:03d}{ayah:03d}.mp3"
    mp3.parent.mkdir(parents=True, exist_ok=True)
    if not mp3.is_file() or mp3.stat().st_size == 0:
        r = subprocess.run(["curl", "-sfL", "-m", "30", f"{BASE_URL}/{reciter}/{mp3.name}", "-o", str(mp3)])
        if r.returncode != 0 or not mp3.is_file() or mp3.stat().st_size < 1000:
            mp3.unlink(missing_ok=True)
            return None
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(mp3), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    return np.frombuffer(out.stdout, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-reciter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf

    from shared.quran_db import QuranDB

    out = Path(args.out_dir).resolve()
    if str(out).startswith(str(LAB.parent.resolve())):
        raise SystemExit("--out-dir must be outside the repo")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / ".mp3"
    words = ayah_words(QuranDB(LAB / "data" / "quran.json"))
    bridge, plain = candidate_windows(words)
    rng = random.Random(args.seed)
    samples = []
    for reciter in RECITERS:
        picks = rng.sample(bridge, args.per_reciter) + rng.sample(plain, args.per_reciter)
        kept_b = kept_p = 0
        for s, a0, a1 in picks:
            is_bridge = min(words[(s, a)] for a in range(a0, a1 + 1)) <= BRIDGE_MAX_WORDS
            if (kept_b if is_bridge else kept_p) >= args.per_reciter // 2:
                continue
            parts = [fetch_pcm(reciter, s, a, cache) for a in range(a0, a1 + 1)]
            if any(p is None for p in parts):
                continue
            gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
            wav = np.concatenate([x for p in parts for x in (p, gap)][:-1])
            if len(wav) / SR > MAX_WINDOW_S:
                continue
            sid = f"heldout_multi_{reciter.split('_')[0].lower()}_{s:03d}_{a0:03d}_{a1:03d}"
            sf.write(out / f"{sid}.wav", wav, SR, subtype="PCM_16")
            samples.append({
                "id": sid, "file": f"{sid}.wav", "surah": s, "ayah": a0, "ayah_end": a1,
                "category": "multi", "source": "everyayah_heldout_multi", "reciter": reciter,
                "bridge": is_bridge,
                "expected_verses": [{"surah": s, "ayah": a} for a in range(a0, a1 + 1)],
            })
            if is_bridge:
                kept_b += 1
            else:
                kept_p += 1
    (out / "manifest.json").write_text(json.dumps({"samples": samples}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(samples)} windows to {out} "
          f"(bridge={sum(s['bridge'] for s in samples)}, per reciter={args.per_reciter})")


if __name__ == "__main__":
    main()
