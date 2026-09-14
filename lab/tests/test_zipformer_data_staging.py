"""Deterministic helpers for Zipformer CTC data staging (no Modal, no lhotse)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_MOD_PATH = ROOT / "scripts" / "prepare_zipformer_data_modal.py"
_SPEC = importlib.util.spec_from_file_location("prepare_zipformer_data_modal", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)

TOKENS_TXT = (
    ROOT
    / "experiments"
    / "prompter-zipformer"
    / "engine"
    / "model"
    / "tokens.txt"
)


def test_parse_qul_and_tlog_filenames():
    assert prep.parse_surah_ayah_filename("1_1.wav") == (1, 1)
    assert prep.parse_surah_ayah_filename("002_255.wav") == (2, 255)
    assert prep.parse_surah_ayah_filename("18_10_abc123.wav") == (18, 10)
    assert prep.parse_surah_ayah_filename("audio/tlog_holdout/3_4_xyz.flac") == (3, 4)
    assert prep.parse_surah_ayah_filename("hf://tarteel-ai/tlog/1_7_99.wav?download=1") == (1, 7)
    assert prep.parse_surah_ayah_filename("not-a-verse.wav") is None
    assert prep.parse_surah_ayah_filename("") is None


def test_duration_policy():
    assert prep.duration_decision(0.5) == "skip_short"
    assert prep.duration_decision(1.0) == "keep"
    assert prep.duration_decision(20.0) == "keep"
    assert prep.duration_decision(20.01) == "long"
    assert prep.duration_decision(60.0) == "long"
    assert prep.duration_decision(60.01) == "skip_long"


def test_retasy_keep_only_correct():
    assert prep.retasy_keep("correct") is True
    assert prep.retasy_keep("in_correct") is False
    assert prep.retasy_keep("not_related_quran") is False
    assert prep.retasy_keep("not_match_aya") is False
    assert prep.retasy_keep(None) is False
    assert prep.retasy_keep("multiple_aya") is False
    assert prep.BAD_RETASY_LABELS == {"in_correct", "not_related_quran", "not_match_aya"}


def test_tarteel_and_hafs_filters():
    assert prep.is_tarteel_dupe(slug="khalifa_al_tunaiji_tarteel") is True
    assert prep.is_tarteel_dupe(reciter="Tarteel EveryAyah") is True
    assert prep.is_tarteel_dupe(slug="maher_al_muaiqly_qdc") is False
    assert prep.is_hafs_riwayah("Hafs") is True
    assert prep.is_hafs_riwayah("hafs") is True
    assert prep.is_hafs_riwayah("Warsh") is False
    assert prep.is_hafs_riwayah(None) is True


def test_basmala_candidate_ayah1():
    assert prep.basmala_candidate("everyayah", 2, 1) is True
    assert prep.basmala_candidate("tlog", 1, 1) is False
    assert prep.basmala_candidate("tlog", 9, 1) is False
    assert prep.basmala_candidate("everyayah", 2, 2) is False
    assert prep.basmala_candidate("qua", 2, 1, {"recording_context": "Studio"}) is True


def test_license_npl_not_permissive():
    npl = "Quran-Lab No-Profit License\nThe Work is not for sale.\n"
    assert prep.license_is_permissive(npl) is False
    assert prep.license_is_permissive("MIT License\nPermission is hereby granted") is True
    assert prep.license_is_permissive("Creative Commons Attribution 4.0 International") is True


def test_categorize_matches_v3():
    assert prep.categorize_word_count(1) == "short"
    assert prep.categorize_word_count(5) == "short"
    assert prep.categorize_word_count(6) == "medium"
    assert prep.categorize_word_count(15) == "medium"
    assert prep.categorize_word_count(16) == "long"


def test_qlab_reciter_and_flat_name():
    assert prep.qlab_reciter("qul_alnufais") == "alnufais"
    assert prep.qlab_reciter("everyayah_heldout") == "everyayah_heldout"
    assert prep.qlab_reciter("tlog_holdout") == "tlog"
    assert prep.qlab_flat_filename("qul_alnufais", "1_1.wav") == "qul_alnufais__1_1.wav"


def test_parse_sources_csv():
    assert prep.parse_sources("everyayah,retasy") == ["everyayah", "retasy"]
    assert prep.parse_sources("qurantts") == ["qurantts"]
    assert "qurantts" not in prep.DEFAULT_SOURCES
    assert prep.parse_sources("") == list(prep.DEFAULT_SOURCES)
    assert prep.DEFAULT_SOURCES == ("everyayah", "qua", "iqra", "retasy", "tlog")
    with pytest.raises(ValueError):
        prep.parse_sources("nope")


def test_load_tokens_txt_roundtrip():
    tokens = prep.load_token_inventory(TOKENS_TXT)
    assert len(tokens) == 251
    assert tokens[-1] == "<blank>"
    from shared.prompter_labels import PhonemeTokenizer

    tok = PhonemeTokenizer(tokens)
    ids = tok.encode(tokens[0])
    assert ids == [0]


def test_pick_surah_ayah_column_aliases():
    assert prep.pick_surah_ayah({"sura": 2, "ayah": 255}) == (2, 255)
    assert prep.pick_surah_ayah({"chapter": 1, "verse": 7}) == (1, 7)
    assert prep.pick_surah_ayah({"surah": 114, "ayah": 6}) == (114, 6)
    assert prep.pick_surah_ayah({"text": "nope"}) is None


def test_everyayah_splits_never_test():
    assert prep.EVERYAYAH_SPLITS == ("train", "validation")
    assert "test" not in prep.EVERYAYAH_SPLITS


def test_qlab_exclusion_set_from_fake_manifest():
    samples = [
        {"source": "tlog_holdout", "file": "tlog_holdout__1_2_abc.wav"},
        {"source": "tlog_holdout", "id": "tlog_holdout__3_4_xyz"},
        {"source": "tlog_holdout", "file": "18_10_deadbeef.wav"},
        {"source": "everyayah_heldout", "file": "everyayah_heldout__test-00009-of-00013_332.wav"},
        {"source": "qul_alnufais", "file": "qul_alnufais__2_255.wav"},
    ]
    excl = prep.qlab_exclusions_from_samples(samples)
    assert excl["tlog_ids"] == {"1_2_abc", "3_4_xyz", "18_10_deadbeef"}
    assert prep.tlog_holdout_key("audio/11_90_2170378964.wav") == "11_90_2170378964"
    assert prep.tlog_holdout_key("tlog_holdout__11_90_2170378964.wav") == "11_90_2170378964"
    assert prep.is_nufais_holdout(slug="yasser_al_nufais_qul") is True
    assert prep.is_nufais_holdout(name_en="Mishary Alnufais") is True
    assert prep.is_nufais_holdout(slug="maher_al_muaiqly_qdc") is False


def test_qlab_exclusion_load_tmp_manifest(tmp_path):
    payload = {
        "samples": [
            {"source": "tlog_holdout", "file": "tlog_holdout__5_6_id1.wav"},
            {"source": "qul_alnufais", "file": "qul_alnufais__1_1.wav"},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    excl = prep.load_qlab_exclusions(path)
    assert excl["tlog_ids"] == {"5_6_id1"}


def test_existing_flac_skip_path(tmp_path):
    import numpy as np
    import soundfile as sf

    audio_root = tmp_path / "audio"
    clip_id = prep.make_clip_id("iqra", 0, 1, 1)
    path = prep.flac_clip_path("iqra", clip_id, audio_root=audio_root)
    path.parent.mkdir(parents=True)
    wav = np.zeros(16000, dtype=np.float32)
    wav[100:200] = 0.05
    sf.write(str(path), wav, 16000, format="FLAC")
    dur = prep.existing_flac_duration(path)
    assert dur is not None
    assert abs(dur - 1.0) < 0.05

    empty = tmp_path / "empty.flac"
    empty.write_bytes(b"")
    assert prep.existing_flac_duration(empty) is None
    assert prep.existing_flac_duration(tmp_path / "missing.flac") is None

    stats = prep._empty_stats("iqra")
    decoded, reused_dur = prep._audio_for_clip(
        source="iqra",
        idx=0,
        surah=1,
        ayah=1,
        audio_obj={"array": np.ones(8000, dtype=np.float32), "sampling_rate": 16000},
        force=False,
        stats=stats,
        audio_root=audio_root,
    )
    assert decoded is None
    assert abs(reused_dur - 1.0) < 0.05
    assert stats.get("skipped", {}).get("audio_error") is None


def test_iqra_keep_flags_threshold():
    flags = prep.iqra_row_keep_flags(
        {
            "sentence_match": 0.4,
            "tashkeel_match": 0.7,
            "sentence_search": 0.96,
            "tashkeel_search": 0.5,
            "sentence_hamza": 0.2,
            "tashkeel_hamza": 0.94,
        }
    )
    assert flags["baseline"] is False
    assert flags["search"] is True
    assert flags["hamza"] is False
    assert flags["baseline_score"] == 0.7
    assert flags["search_score"] == 0.96


def test_skip_hf_stream_uses_skip_when_present():
    class _DS:
        def skip(self, n):
            self.n = n
            return self

    ds = _DS()
    out = prep.skip_hf_stream(ds, 12, "tlog")
    assert out is ds
    assert ds.n == 12
    assert prep.skip_hf_stream(ds, 0) is ds
    assert prep.skip_hf_stream([1, 2, 3], 5) == [1, 2, 3]
