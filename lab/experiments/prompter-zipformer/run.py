"""prompter-zipformer -- the alketab "Quran Prompter" engine (prompter.alketab.app)
run as-is under Node, wrapped in the benchmark's predict() contract.

Pipeline (all vendored JS, recovered from the site's source maps -- see README.md):
  16 kHz PCM -> Kaldi fbank (80 mel) -> streaming Zipformer2-CTC ONNX (251
  tajweed-phoneme tokens, 72 MB fp32) -> greedy CTC -> whole-Quran 5-gram
  search + per-surah online DP tracker -> per-word verdicts.

This file only: decodes audio with shared.audio, ships raw float32 to a
long-lived `node harness.mjs` over stdin/stdout, and turns the harness's
per-ayah verdict tallies into {surah, ayah, ayah_end}.

Model + corpus are fetched on first use into data/prompter/ (env override:
PROMPTER_DATA_DIR).
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PROMPTER_DATA_DIR", PROJECT_ROOT / "data" / "prompter"))
MODEL_PATH = DATA_DIR / "quran_phoneme_zipformer.onnx"
CORPUS_PATH = DATA_DIR / "quran.json"
ORT_DIR = Path(
    os.environ.get("PROMPTER_ORT_DIR", PROJECT_ROOT / "web" / "frontend" / "node_modules")
)

SITE = "https://prompter.alketab.app"
MODEL_URL = f"{SITE}/models/quran_phoneme_zipformer.onnx?v=31755836"
CORPUS_URL = f"{SITE}/data/quran.json?v=24360c05"

_proc: subprocess.Popen | None = None
_req_id = 0


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[prompter-zipformer] downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _ensure_assets() -> None:
    if not MODEL_PATH.exists():
        _fetch(MODEL_URL, MODEL_PATH)
    if not CORPUS_PATH.exists():
        _fetch(CORPUS_URL, CORPUS_PATH)
    if not (ORT_DIR / "onnxruntime-node").exists():
        raise FileNotFoundError(
            f"onnxruntime-node not found under {ORT_DIR}; run `npm install` in web/frontend "
            "or set PROMPTER_ORT_DIR"
        )


def _ensure_proc() -> subprocess.Popen:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return _proc
    _ensure_assets()
    env = dict(os.environ)
    env.setdefault("PROMPTER_MODEL", str(MODEL_PATH))
    env.setdefault("PROMPTER_CORPUS", str(CORPUS_PATH))
    env.setdefault("PROMPTER_ORT_DIR", str(ORT_DIR))
    _proc = subprocess.Popen(
        ["node", str(HERE / "harness.mjs")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
        text=True,
        bufsize=1,
    )
    ready = _proc.stdout.readline()
    if not ready or not json.loads(ready).get("ready"):
        raise RuntimeError(f"harness failed to start: {ready!r}")
    atexit.register(_shutdown)
    return _proc


def _shutdown() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.stdin.close()
            _proc.wait(timeout=5)
        except Exception:
            _proc.kill()
    _proc = None


def recognize(audio_path: str) -> dict:
    """Raw harness result: verses (accepted ayahs), all tallies, transcript, events."""
    global _req_id
    proc = _ensure_proc()
    audio = load_audio(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as f:
        audio.astype("float32").tofile(f)
        pcm_path = f.name
    try:
        _req_id += 1
        proc.stdin.write(json.dumps({"id": _req_id, "pcm": pcm_path}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
    finally:
        os.unlink(pcm_path)
    if not line:
        _shutdown()
        raise RuntimeError("harness died")
    res = json.loads(line)
    if "error" in res:
        raise RuntimeError(res["error"])
    return res


def _contiguous_head(verses: list[dict]) -> tuple[int, int, int | None]:
    """First verse plus the longest run of consecutive ayahs in the same surah."""
    first = verses[0]
    surah, ayah = first["surah"], first["ayah"]
    end = ayah
    for v in verses[1:]:
        if v["surah"] == surah and v["ayah"] == end + 1:
            end = v["ayah"]
        else:
            break
    return surah, ayah, (end if end != ayah else None)


def predict(audio_path: str) -> dict:
    res = recognize(audio_path)
    verses = res["verses"]
    if not verses:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": res["transcript"],
        }
    surah, ayah, ayah_end = _contiguous_head(verses)
    ok = sum(v["ok"] for v in verses)
    words = sum(v["words"] for v in verses)
    return {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": ayah_end,
        "score": ok / max(1, words),
        "transcript": res["transcript"],
        "verses": [(v["surah"], v["ayah"]) for v in verses],
    }


def transcribe(audio_path: str) -> str:
    return recognize(audio_path)["transcript"]


def model_size() -> int:
    try:
        return MODEL_PATH.stat().st_size
    except OSError:
        return 72_705_392


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = recognize(p)
        print(
            Path(p).name,
            f"{r['decodeMs']}ms",
            r["state"],
            [(v["surah"], v["ayah"], v["ok"], v["unsure"], v["words"]) for v in r["verses"]],
            [e["type"] for e in r["events"]],
        )
        print("   ", r["transcript"])
