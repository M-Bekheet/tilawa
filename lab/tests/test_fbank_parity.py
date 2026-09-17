import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.audio import load_audio
from shared.fbank import compute_fbank

NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "experiments" / "zipformer-ctc" / "fbank_dump.ts"
_ORT = Path(os.environ.get(
    "ZIPFORMER_ORT_DIR",
    ROOT.parent / "web" / "frontend" / "node_modules",
))
TSX = _ORT / ".bin" / "tsx"
_CLIP = "001002.mp3"
_CORPUS_CANDIDATES = (
    ROOT / "benchmark" / "test_corpus",
    Path("/Users/rock/ai/projects/offline-tarteel/lab/benchmark/test_corpus"),
)


def _corpus_dir() -> Path:
    for d in _CORPUS_CANDIDATES:
        if (d / _CLIP).is_file():
            return d
    for d in _CORPUS_CANDIDATES:
        if d.is_dir():
            return d
    return _CORPUS_CANDIDATES[0]


CORPUS_DIR = _corpus_dir()

N_BINS = 80
MAX_ABS = 1e-3
SR = 16_000
CHUNK = 7680

pytestmark = pytest.mark.skipif(
    NODE is None or not TSX.is_file(),
    reason="node/tsx not available (need web/frontend/node_modules/.bin/tsx)",
)


def _synth_wave(duration_s: float = 3.7, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * SR))
    t = np.arange(n, dtype=np.float64) / SR
    wave = (
        0.25 * np.sin(2 * np.pi * 220.0 * t)
        + 0.15 * np.sin(2 * np.pi * 440.0 * t)
        + 0.10 * np.sin(2 * np.pi * 880.0 * t)
        + 0.05 * rng.standard_normal(n)
    )
    return np.clip(wave, -0.5, 0.5).astype(np.float32)


def _real_clip() -> Path | None:
    preferred = CORPUS_DIR / "001002.mp3"
    if preferred.is_file():
        return preferred
    manifest = CORPUS_DIR / "manifest.json"
    if not manifest.is_file():
        return None
    import json

    samples = json.loads(manifest.read_text())["samples"]
    if not samples:
        return None
    cand = CORPUS_DIR / samples[0]["file"]
    return cand if cand.is_file() else None


def _write_pcm(path: Path, wave: np.ndarray) -> None:
    np.ascontiguousarray(wave, dtype="<f4").tofile(path)


def _read_frames(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    assert raw.size % N_BINS == 0, f"output size {raw.size} not divisible by {N_BINS}"
    return raw.reshape(-1, N_BINS)


def _run_js(pcm: Path, out: Path, chunk: int | None = None) -> int:
    cmd = [str(TSX), str(DUMP), str(pcm), str(out)]
    if chunk is not None:
        cmd = [str(TSX), str(DUMP), "--chunk", str(chunk), str(pcm), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"fbank_dump failed rc={r.returncode}: {r.stderr}\n{r.stdout}"
    return int(r.stdout.strip().splitlines()[-1])


def _assert_close(py: np.ndarray, js: np.ndarray, label: str) -> tuple[float, float]:
    assert py.shape == js.shape, f"{label}: frame/bin shape py {py.shape} vs js {js.shape}"
    abs_diff = np.abs(py.astype(np.float64) - js.astype(np.float64))
    max_d = float(abs_diff.max()) if abs_diff.size else 0.0
    mean_d = float(abs_diff.mean()) if abs_diff.size else 0.0
    assert max_d < MAX_ABS, (
        f"{label}: max abs diff {max_d:.6e} mean {mean_d:.6e} "
        f"(nFrames={py.shape[0]}, nBins={py.shape[1]}) exceeds {MAX_ABS}"
    )
    return max_d, mean_d


def test_parity_synthetic():
    wave = _synth_wave()
    py = compute_fbank(wave, sr=SR)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pcm = td / "synth.f32"
        out = td / "js.f32"
        _write_pcm(pcm, wave)
        n_js = _run_js(pcm, out)
        js = _read_frames(out)
    assert n_js == js.shape[0] == py.shape[0], (
        f"synthetic frame count py={py.shape[0]} js_stdout={n_js} js_file={js.shape[0]}"
    )
    max_d, mean_d = _assert_close(py, js, "synthetic")
    print(f"synthetic nFrames={py.shape[0]} max_abs={max_d:.6e} mean_abs={mean_d:.6e}")


def test_parity_real_audio():
    clip = _real_clip()
    if clip is None:
        pytest.skip("benchmark corpus clip missing")
    wave = load_audio(str(clip), sr=SR)
    py = compute_fbank(wave, sr=SR)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pcm = td / "real.f32"
        out = td / "js.f32"
        _write_pcm(pcm, wave)
        n_js = _run_js(pcm, out)
        js = _read_frames(out)
    assert n_js == js.shape[0] == py.shape[0], (
        f"real ({clip.name}) frame count py={py.shape[0]} js_stdout={n_js} js_file={js.shape[0]}"
    )
    max_d, mean_d = _assert_close(py, js, f"real {clip.name}")
    print(f"real {clip.name} nFrames={py.shape[0]} max_abs={max_d:.6e} mean_abs={mean_d:.6e}")


def test_streaming_chunk_equals_oneshot():
    wave = _synth_wave()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pcm = td / "synth.f32"
        oneshot = td / "oneshot.f32"
        chunked = td / "chunked.f32"
        _write_pcm(pcm, wave)
        n_one = _run_js(pcm, oneshot)
        n_chunk = _run_js(pcm, chunked, chunk=CHUNK)
        a = _read_frames(oneshot)
        b = _read_frames(chunked)
    assert n_one == n_chunk == a.shape[0] == b.shape[0]
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a, b), (
        f"streaming --chunk {CHUNK} is not bitwise equal to one-shot "
        f"(max abs {float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))):.6e})"
    )
