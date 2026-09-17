"""zipformer-ctc run.py helpers (no Node / ONNX)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_MOD_PATH = ROOT / "experiments" / "zipformer-ctc" / "run.py"
_SPEC = importlib.util.spec_from_file_location("zipformer_ctc_run", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run)


def test_benchmark_name_stable_by_default(monkeypatch):
    monkeypatch.delenv("ZIPFORMER_MODEL", raising=False)
    assert run.benchmark_name() == "zipformer-ctc"


def test_benchmark_name_suffix_when_zipformer_model_set(monkeypatch):
    monkeypatch.setenv("ZIPFORMER_MODEL", "/tmp/ft-v31/model.onnx")
    assert run.benchmark_name() == "zipformer-ctc[model.onnx]"


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
    monkeypatch.setenv("ZIPFORMER_MODEL", str(model))
    monkeypatch.setattr(run, "CORPUS_PATH", corpus)
    monkeypatch.setattr(run, "ORT_DIR", tmp_path / "node_modules")
    monkeypatch.setattr(run, "MODEL_PATH", tmp_path / "missing" / "zipformer_interp_gentle_a05.int8.onnx")

    def _boom(*_a, **_k):
        raise AssertionError("must not download when ZIPFORMER_MODEL exists")

    monkeypatch.setattr(run, "_fetch", _boom)
    run._ensure_assets()


def test_allow_gaps_reads_env(monkeypatch):
    monkeypatch.delenv("ZIPFORMER_ALLOW_GAPS", raising=False)
    assert run._allow_gaps() is False
    monkeypatch.setenv("ZIPFORMER_ALLOW_GAPS", "1")
    assert run._allow_gaps() is True
    monkeypatch.setenv("ZIPFORMER_ALLOW_GAPS", "0")
    assert run._allow_gaps() is False


def test_contiguous_head_default_stops_at_gap():
    verses = [
        {"surah": 1, "ayah": 1},
        {"surah": 1, "ayah": 2},
        {"surah": 1, "ayah": 5},
    ]
    assert run._contiguous_head(verses, allow_gaps=False) == (1, 1, 2)


def test_contiguous_head_allow_gaps_bridges_short_ayah_only_if_present():
    verses = [
        {"surah": 1, "ayah": 5},
        {"surah": 1, "ayah": 7},
        {"surah": 1, "ayah": 6, "words": 3},
    ]
    wc = {(1, 6): 3}

    def count(s, a):
        return wc.get((s, a), 99)

    assert run._contiguous_head(verses, allow_gaps=True, word_count=count) == (1, 5, 7)


def test_contiguous_head_allow_gaps_does_not_invent_missing_ayah():
    verses = [
        {"surah": 1, "ayah": 5},
        {"surah": 1, "ayah": 7},
    ]
    wc = {(1, 6): 3}

    def count(s, a):
        return wc.get((s, a), 99)

    assert run._contiguous_head(verses, allow_gaps=True, word_count=count) == (1, 5, None)


def test_contiguous_head_four_five_does_not_start_at_three():
    verses = [
        {"surah": 36, "ayah": 4},
        {"surah": 36, "ayah": 5},
    ]
    assert run._contiguous_head(verses, allow_gaps=True, word_count=lambda *_: 3) == (
        36,
        4,
        5,
    )
    bridged = run._bridge_extras(
        verses,
        verses
        + [{"surah": 36, "ayah": 3, "ok": 1, "unsure": 0, "wrong": 0, "words": 3, "firstSeen": 0}],
        gap_max_words=3,
    )
    assert [v["ayah"] for v in bridged] == [4, 5]
    surah, ayah, end = run._contiguous_head(bridged, allow_gaps=True, word_count=lambda *_: 3)
    assert (surah, ayah, end) == (36, 4, 5)


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
    monkeypatch.setenv("ZIPFORMER_ALLOW_GAPS", "1")
    verses = [
        {"surah": 1, "ayah": 2},
        {"surah": 1, "ayah": 4},
        {"surah": 1, "ayah": 3, "words": 2},
    ]
    wc = {(1, 3): 2}

    def count(s, a):
        return wc.get((s, a), 99)

    assert run._contiguous_head(verses, word_count=count) == (1, 2, 4)


def test_bridge_extras_between_neighbours_not_prefix():
    accepted = [
        {"surah": 1, "ayah": 2, "ok": 2, "unsure": 0, "wrong": 0, "words": 4, "firstSeen": 0},
        {"surah": 1, "ayah": 4, "ok": 2, "unsure": 0, "wrong": 0, "words": 4, "firstSeen": 2},
    ]
    mid = {"surah": 1, "ayah": 3, "ok": 1, "unsure": 0, "wrong": 0, "words": 2, "firstSeen": 1}
    out = run._bridge_extras(accepted, accepted + [mid], gap_max_words=3)
    assert {(v["surah"], v["ayah"]) for v in out} == {(1, 2), (1, 3), (1, 4)}


def test_bridge_extras_requires_evidence():
    accepted = [
        {"surah": 1, "ayah": 2, "ok": 2, "unsure": 0, "wrong": 0, "words": 4, "firstSeen": 0},
        {"surah": 1, "ayah": 4, "ok": 2, "unsure": 0, "wrong": 0, "words": 4, "firstSeen": 2},
    ]
    ghost = {"surah": 1, "ayah": 3, "ok": 0, "unsure": 0, "wrong": 0, "words": 2, "firstSeen": 1}
    out = run._bridge_extras(accepted, accepted + [ghost], gap_max_words=3)
    assert {(v["surah"], v["ayah"]) for v in out} == {(1, 2), (1, 4)}


def test_gap_max_words_reads_env(monkeypatch):
    monkeypatch.delenv("ZIPFORMER_GAP_MAX_WORDS", raising=False)
    run._HARNESS_GAP_MAX_WORDS = None
    assert run._gap_max_words() == 3
    monkeypatch.setenv("ZIPFORMER_GAP_MAX_WORDS", "2")
    assert run._gap_max_words() == 2


def test_ensure_proc_forwards_zipformer_knobs(monkeypatch):
    knobs = {
        "ZIPFORMER_MIN_WORD_FRACTION": "0.3",
        "ZIPFORMER_ALLOW_GAPS": "1",
        "ZIPFORMER_TAIL_SECONDS": "3.0",
        "ZIPFORMER_OK_DISTANCE": "0.2",
        "ZIPFORMER_UNSURE_DISTANCE": "0.5",
        "ZIPFORMER_SEARCH_DECISIVE_DISTANCE": "0.45",
        "ZIPFORMER_GAP_MAX_WORDS": "3",
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
