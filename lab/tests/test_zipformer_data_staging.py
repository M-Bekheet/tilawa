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
