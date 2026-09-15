"""prompter-zipformer -- the alketab "Quran Prompter" engine (prompter.alketab.app)
run as-is under Node, wrapped in the benchmark's predict() contract.

Pipeline (all vendored JS, recovered from the site's source maps -- see README.md):
  16 kHz PCM -> Kaldi fbank (80 mel) -> streaming Zipformer2-CTC ONNX (251
  tajweed-phoneme tokens, 72.7 MB int8; onnxruntime dynamic quant, byte-identical
  to Quran-Lab zipformer_p_arabic_v3.1.int8.onnx — see EXPERIMENTS.md
  "Zipformer2-CTC (Quran-Lab v3 reference + fine-tunes)") -> greedy CTC ->
  whole-Quran 5-gram search + per-surah online DP tracker -> per-word verdicts.

This file only: decodes audio with shared.audio, ships raw float32 to a
long-lived `node harness.mjs` over stdin/stdout, and turns the harness's
per-ayah verdict tallies into {surah, ayah, ayah_end}.

Model + corpus are fetched on first use into data/prompter/ (env override:
PROMPTER_DATA_DIR) unless PROMPTER_MODEL already points at an existing file.
"""

from __future__ import annotations

import atexit
import hashlib
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
_model_sha_cache: dict[str, str] = {}


def resolved_model_path() -> Path:
    override = os.environ.get("PROMPTER_MODEL")
    if override:
        return Path(override)
    return MODEL_PATH


def benchmark_name(base: str = "prompter-zipformer") -> str:
    """Result JSON `name`; suffix only when PROMPTER_MODEL is set."""
    override = os.environ.get("PROMPTER_MODEL")
    if override:
        return f"{base}[{Path(override).name}]"
    return base


def model_sha256_prefix(path: Path | None = None) -> str:
    """First 8 hex chars of the ONNX file sha256; cached per resolved path."""
    p = path if path is not None else resolved_model_path()
    key = str(p.resolve()) if p.is_file() else str(p)
    cached = _model_sha_cache.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    prefix = h.hexdigest()[:8]
    _model_sha_cache[key] = prefix
    return prefix


def _provenance(path: Path | None = None) -> dict:
    p = path if path is not None else resolved_model_path()
    out = {"model": p.name, "model_sha256_prefix": ""}
    try:
        out["model_sha256_prefix"] = model_sha256_prefix(p)
    except OSError:
        pass
    return out


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[prompter-zipformer] downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _ensure_assets() -> None:
    override = os.environ.get("PROMPTER_MODEL")
    if override:
        model = Path(override)
        if not model.is_file():
            raise FileNotFoundError(
                f"PROMPTER_MODEL={override!r} is not an existing file"
            )
    elif not MODEL_PATH.exists():
        _fetch(MODEL_URL, MODEL_PATH)
    if not CORPUS_PATH.exists():
        _fetch(CORPUS_URL, CORPUS_PATH)
    if not (ORT_DIR / "onnxruntime-node").exists():
        raise FileNotFoundError(
            f"onnxruntime-node not found under {ORT_DIR}; run `npm install` in web/frontend "
            "or set PROMPTER_ORT_DIR"
        )


def _ensure_proc() -> subprocess.Popen:
    global _proc, _HARNESS_GAP_MAX_WORDS
    if _proc is not None and _proc.poll() is None:
        return _proc
    _ensure_assets()
    env = dict(os.environ)
    env.setdefault("PROMPTER_MODEL", str(MODEL_PATH))
    env.setdefault("PROMPTER_CORPUS", str(CORPUS_PATH))
    env.setdefault("PROMPTER_ORT_DIR", str(ORT_DIR))
    env.setdefault("PROMPTER_GAP_MAX_WORDS", str(_gap_max_words()))
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
    parsed = json.loads(ready) if ready else {}
    if not parsed.get("ready"):
        raise RuntimeError(f"harness failed to start: {ready!r}")
    if "gapMaxWords" in parsed:
        _HARNESS_GAP_MAX_WORDS = int(parsed["gapMaxWords"])
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


GAP_MAX_WORDS = 3
_HARNESS_GAP_MAX_WORDS: int | None = None
_ayah_words_cache: dict[tuple[int, int], int] | None = None


def _allow_gaps() -> bool:
    return os.environ.get("PROMPTER_ALLOW_GAPS") == "1"


def _gap_max_words() -> int:
    raw = os.environ.get("PROMPTER_GAP_MAX_WORDS")
    if raw not in (None, ""):
        return int(raw)
    if _HARNESS_GAP_MAX_WORDS is not None:
        return _HARNESS_GAP_MAX_WORDS
    return GAP_MAX_WORDS


def _ayah_word_count(surah: int, ayah: int) -> int:
    """Word count from the prompter corpus; 99 (do not bridge) if unknown."""
    global _ayah_words_cache
    if _ayah_words_cache is None:
        _ayah_words_cache = {}
        path = Path(os.environ.get("PROMPTER_CORPUS", CORPUS_PATH))
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            for s in data.get("surahs", []):
                n = int(s["n"])
                for i, a in enumerate(s.get("ayahs", [])):
                    _ayah_words_cache[(n, i + 1)] = len(a.get("w", []))
    return _ayah_words_cache.get((surah, ayah), 99)


def _bridge_extras(
    accepted: list[dict],
    tallies: list[dict],
    *,
    gap_max_words: int | None = None,
) -> list[dict]:
    """Inject a below-threshold short ayah only when both neighbours already emit.

    Prefix fill is forbidden: a tally for ayah 3 with accepted [4, 5] stays out.
    Requires ok+unsure >= 1.
    """
    limit = _gap_max_words() if gap_max_words is None else gap_max_words
    have = {(t["surah"], t["ayah"]) for t in accepted}
    extra = []
    for t in tallies:
        key = (t["surah"], t["ayah"])
        if key in have:
            continue
        if t.get("words", 99) > limit:
            continue
        if t.get("ok", 0) + t.get("unsure", 0) < 1:
            continue
        if t.get("wrong", 0) > t.get("ok", 0) + t.get("unsure", 0):
            continue
        if (t["surah"], t["ayah"] - 1) not in have:
            continue
        if (t["surah"], t["ayah"] + 1) not in have:
            continue
        extra.append({**t, "bridged": True})
        have.add(key)
    if not extra:
        return accepted
    out = list(accepted) + extra
    out.sort(key=lambda t: t.get("firstSeen", 0))
    return out


def _contiguous_head(
    verses: list[dict],
    *,
    allow_gaps: bool | None = None,
    word_count=None,
) -> tuple[int, int, int | None]:
    """First verse plus the longest run of consecutive ayahs in the same surah.

    When allow_gaps (PROMPTER_ALLOW_GAPS=1), skip a single missing ayah of
    ≤ gap-max words *only if that ayah is already in `verses`* (harness-bridged).
    Does not invent a hole from the corpus, and does not walk backward: [4, 5]
    stays start=4.
    """
    if allow_gaps is None:
        allow_gaps = _allow_gaps()
    count_fn = word_count or _ayah_word_count
    present = {(v["surah"], v["ayah"]) for v in verses}
    first = verses[0]
    surah, ayah = first["surah"], first["ayah"]
    end = ayah
    for v in verses[1:]:
        if v["surah"] != surah:
            break
        if v["ayah"] == end + 1:
            end = v["ayah"]
            continue
        if allow_gaps and v["ayah"] == end + 2:
            skipped = end + 1
            if (surah, skipped) in present and count_fn(surah, skipped) <= _gap_max_words():
                end = v["ayah"]
                continue
        break
    return surah, ayah, (end if end != ayah else None)


def predict(audio_path: str) -> dict:
    res = recognize(audio_path)
    provenance = _provenance()
    verses = res["verses"]
    if _allow_gaps():
        accepted = [v for v in verses if not v.get("bridged")]
        tallies = res.get("all") or verses
        verses = _bridge_extras(accepted, tallies)
    if not verses:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": res["transcript"],
            **provenance,
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
        **provenance,
    }


def transcribe(audio_path: str) -> str:
    return recognize(audio_path)["transcript"]


def model_size() -> int:
    p = resolved_model_path()
    try:
        return p.stat().st_size
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
