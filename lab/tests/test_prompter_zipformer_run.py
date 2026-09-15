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
