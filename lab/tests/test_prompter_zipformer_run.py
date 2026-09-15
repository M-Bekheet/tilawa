"""prompter-zipformer run.py helpers (no Node / ONNX)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_MOD_PATH = ROOT / "experiments" / "prompter-zipformer" / "run.py"
_SPEC = importlib.util.spec_from_file_location("prompter_zipformer_run", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run)


def test_benchmark_name_stable_by_default(monkeypatch):
    monkeypatch.delenv("PROMPTER_MODEL", raising=False)
    assert run.benchmark_name() == "prompter-zipformer"


def test_benchmark_name_suffix_when_prompter_model_set(monkeypatch):
    monkeypatch.setenv("PROMPTER_MODEL", "/tmp/ft-v31/model.onnx")
    assert run.benchmark_name() == "prompter-zipformer[model.onnx]"


def test_model_sha256_prefix_cached(tmp_path: Path):
    p = tmp_path / "m.onnx"
    payload = b"zipformer-int8-bytes"
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()[:8]
    assert run.model_sha256_prefix(p) == expected
    p.write_bytes(b"changed")
    assert run.model_sha256_prefix(p) == expected  # cached once


def test_ensure_assets_skips_default_download(monkeypatch, tmp_path: Path):
    model = tmp_path / "custom.onnx"
    model.write_bytes(b"onnx")
    corpus = tmp_path / "quran.json"
    corpus.write_text("{}", encoding="utf-8")
    ort = tmp_path / "node_modules" / "onnxruntime-node"
    ort.mkdir(parents=True)
    monkeypatch.setenv("PROMPTER_MODEL", str(model))
    run.CORPUS_PATH = corpus
    run.ORT_DIR = tmp_path / "node_modules"
    run.MODEL_PATH = tmp_path / "missing" / "quran_phoneme_zipformer.onnx"

    def _boom(*_a, **_k):
        raise AssertionError("must not download when PROMPTER_MODEL exists")

    monkeypatch.setattr(run, "_fetch", _boom)
    run._ensure_assets()


def test_allow_gaps_reads_env(monkeypatch):
    monkeypatch.delenv("PROMPTER_ALLOW_GAPS", raising=False)
    assert run._allow_gaps() is False
    monkeypatch.setenv("PROMPTER_ALLOW_GAPS", "1")
    assert run._allow_gaps() is True
    monkeypatch.setenv("PROMPTER_ALLOW_GAPS", "0")
    assert run._allow_gaps() is False


def test_contiguous_head_default_stops_at_gap():
    verses = [
        {"surah": 1, "ayah": 1},
        {"surah": 1, "ayah": 2},
        {"surah": 1, "ayah": 5},
    ]
    assert run._contiguous_head(verses, allow_gaps=False) == (1, 1, 2)


def test_contiguous_head_allow_gaps_bridges_short_ayah():
    verses = [
        {"surah": 1, "ayah": 5},
        {"surah": 1, "ayah": 7},
    ]
    wc = {(1, 6): 3}

    def count(s, a):
        return wc.get((s, a), 99)

    assert run._contiguous_head(verses, allow_gaps=True, word_count=count) == (1, 5, 7)


def test_contiguous_head_allow_gaps_skips_long_hole():
    verses = [
        {"surah": 105, "ayah": 1},
        {"surah": 105, "ayah": 3},
    ]
    wc = {(105, 2): 5}

    def count(s, a):
        return wc.get((s, a), 99)

    assert run._contiguous_head(verses, allow_gaps=True, word_count=count) == (105, 1, None)


def test_contiguous_head_allow_gaps_does_not_skip_two_ayahs():
    verses = [
        {"surah": 1, "ayah": 1},
        {"surah": 1, "ayah": 2},
        {"surah": 1, "ayah": 5},
    ]
    wc = {(1, 3): 2, (1, 4): 3}

    def count(s, a):
        return wc.get((s, a), 99)

    assert run._contiguous_head(verses, allow_gaps=True, word_count=count) == (1, 1, 2)


def test_contiguous_head_allow_gaps_follows_env(monkeypatch):
    monkeypatch.setenv("PROMPTER_ALLOW_GAPS", "1")
    verses = [{"surah": 1, "ayah": 2}, {"surah": 1, "ayah": 4}]
    wc = {(1, 3): 2}

    def count(s, a):
        return wc.get((s, a), 99)

    assert run._contiguous_head(verses, word_count=count) == (1, 2, 4)


def test_ensure_proc_forwards_prompter_knobs(monkeypatch):
    knobs = {
        "PROMPTER_MIN_WORD_FRACTION": "0.3",
        "PROMPTER_ALLOW_GAPS": "1",
        "PROMPTER_TAIL_SECONDS": "3.0",
        "PROMPTER_OK_DISTANCE": "0.2",
        "PROMPTER_UNSURE_DISTANCE": "0.5",
        "PROMPTER_SEARCH_DECISIVE_DISTANCE": "0.45",
    }
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)

    captured: dict = {}

    class _Stdout:
        def readline(self):
            return '{"ready": true}\n'

    class FakeProc:
        def __init__(self, *args, **kwargs):
            captured["env"] = kwargs["env"]
            self.stdin = None
            self.stdout = _Stdout()

        def poll(self):
            return None

    monkeypatch.setattr(run, "_ensure_assets", lambda: None)
    monkeypatch.setattr(run.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(run.atexit, "register", lambda *_a, **_k: None)
    run._proc = None
    proc = run._ensure_proc()
    assert proc is not None
    env = captured["env"]
    for k, v in knobs.items():
        assert env[k] == v, k
    run._proc = None
