"""B1 multi-ayah window selection + target text (no lhotse, no Modal)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_MOD_PATH = ROOT / "scripts" / "prepare_zipformer_data_modal.py"
_SPEC = importlib.util.spec_from_file_location("prepare_zipformer_data_modal", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


class StubCorpus:
    def ayah_phonemes(self, surah: int, ayah: int) -> str:
        return f"S{surah}A{ayah}"

    def span_phonemes(self, surah: int, ayah_start: int, ayah_end: int) -> str:
        return "".join(self.ayah_phonemes(surah, a) for a in range(ayah_start, ayah_end + 1))


def _rec(speaker: str, surah: int, ayah: int, duration: float, idx: int = 0) -> dict:
    cid = f"{speaker}_{surah}_{ayah}_{idx}"
    return {
        "id": cid,
        "speaker": speaker,
        "surah": surah,
        "ayah": ayah,
        "duration": duration,
        "text": f"ph{ayah}",
    }


def _cut_dict(speaker: str, surah: int, ayah: int, duration: float, ayah_end=None, cid=None):
    end = ayah if ayah_end is None else ayah_end
    return {
        "id": cid or f"{speaker}_{surah}_{ayah}",
        "duration": duration,
        "supervisions": [
            {
                "speaker": speaker,
                "text": "abc",
                "custom": {"surah": surah, "ayah": ayah, "ayah_end": end, "source": "everyayah"},
            }
        ],
    }


def test_consecutive_spans_skip_gaps_and_bound_length():
    ayahs = [1, 2, 3, 4, 6, 7]
    spans = prep.consecutive_ayah_spans(ayahs, min_len=2, max_len=4)
    assert (1, 2) in spans
    assert (2, 3) in spans
    assert (3, 4) in spans
    assert (6, 7) in spans
    assert (1, 3) in spans
    assert (2, 4) in spans
    assert (1, 4) in spans
    assert (5, 6) not in spans
    assert (4, 6) not in spans
    assert (6, 8) not in spans
    assert (1, 5) not in spans
    assert all(2 <= (e - s + 1) <= 4 for s, e in spans)
    assert all(e == s + (e - s) for s, e in spans)
    for s, e in spans:
        for a in range(s, e):
            assert a + 1 in ayahs or (s, e) not in spans
            assert a in ayahs and (a + 1) in ayahs


def test_consecutive_spans_empty_on_gaps_and_singletons():
    assert prep.consecutive_ayah_spans([1, 3, 5]) == []
    assert prep.consecutive_ayah_spans([7]) == []
    assert prep.consecutive_ayah_spans([]) == []
    assert prep.consecutive_ayah_spans([10, 11]) == [(10, 11)]


def test_cut_dict_skips_already_span_and_bad():
    ok = prep.cut_dict_window_record(_cut_dict("r1", 1, 2, 3.0))
    assert ok is not None
    assert ok["speaker"] == "r1"
    assert ok["ayah"] == 2
    assert prep.cut_dict_window_record(_cut_dict("r1", 1, 1, 3.0, ayah_end=3)) is None
    assert prep.cut_dict_window_record({"id": "x", "duration": 1.0, "supervisions": []}) is None
    assert prep.cut_dict_window_record({"id": "x", "duration": 0.0, "supervisions": [{}]}) is None


def test_group_first_duplicate_wins():
    recs = [
        _rec("r1", 2, 1, 1.0, idx=0),
        _rec("r1", 2, 1, 9.0, idx=1),
        _rec("r1", 2, 2, 1.0),
    ]
    groups = prep.group_window_records(recs)
    assert groups[("r1", 2)][1]["id"] == "r1_2_1_0"
    assert groups[("r1", 2)][1]["duration"] == 1.0


def test_candidates_drop_overlong_and_keep_2_to_4():
    recs = [
        _rec("r1", 1, 1, 1.0),
        _rec("r1", 1, 2, 1.0),
        _rec("r1", 1, 3, 1.0),
        _rec("r1", 1, 4, 1.0),
        _rec("r1", 1, 5, 20.0),
        _rec("r1", 1, 6, 20.0),
    ]
    groups = prep.group_window_records(recs)
    picks = prep.candidate_multi_ayah_windows(groups)
    pairs = {(p.ayah, p.ayah_end, p.n_ayahs) for p in picks}
    assert (1, 2, 2) in pairs
    assert (1, 4, 4) in pairs
    assert (5, 6, 2) not in pairs
    assert all(p.n_ayahs in (2, 3, 4) for p in picks)
    assert all(p.max_duration_s() <= prep.MULTI_MAX_DURATION_S for p in picks)


def test_select_windows_seed_deterministic_and_stratified():
    recs = []
    for a in range(1, 21):
        recs.append(_rec("alice", 2, a, 1.0))
    for a in range(1, 6):
        recs.append(_rec("bob", 3, a, 1.0))
    groups = prep.group_window_records(recs)
    cands = prep.candidate_multi_ayah_windows(groups)
    a = prep.select_windows_stratified(cands, 10, seed=0)
    b = prep.select_windows_stratified(cands, 10, seed=0)
    c = prep.select_windows_stratified(cands, 10, seed=1)
    assert [(w.speaker, w.surah, w.ayah, w.ayah_end) for w in a] == [
        (w.speaker, w.surah, w.ayah, w.ayah_end) for w in b
    ]
    assert [(w.speaker, w.surah, w.ayah, w.ayah_end) for w in a] != [
        (w.speaker, w.surah, w.ayah, w.ayah_end) for w in c
    ]
    assert len(a) == 10
    n_alice = sum(1 for w in a if w.speaker == "alice")
    n_bob = sum(1 for w in a if w.speaker == "bob")
    assert n_alice > n_bob >= 1


def test_allocate_stratified_largest_remainder():
    # 10 from sizes 7 and 3 → 7 and 3. 5 from 7 and 3 → 4 and 1 (7/10*5=3.5, 3/10*5=1.5).
    assert prep.allocate_stratified_counts({"a": 7, "b": 3}, 10) == {"a": 7, "b": 3}
    got = prep.allocate_stratified_counts({"a": 7, "b": 3}, 5)
    assert got["a"] + got["b"] == 5
    assert got["a"] == 4 and got["b"] == 1
    assert prep.allocate_stratified_counts({"a": 2, "b": 2}, 0) == {"a": 0, "b": 0}


def test_multi_ayah_window_text_no_spaces_and_concat():
    corpus = StubCorpus()
    text = prep.multi_ayah_window_text(corpus, 2, 1, 3)
    assert text == "S2A1S2A2S2A3"
    assert " " not in text
    assert text == "".join(corpus.ayah_phonemes(2, a) for a in range(1, 4))


def test_n_ayahs_histogram():
    picks = [
        prep.MultiAyahWindowPick("r", 1, 1, 2, ("a", "b"), (1.0, 1.0), 2),
        prep.MultiAyahWindowPick("r", 1, 1, 3, ("a", "b", "c"), (1.0, 1.0, 1.0), 3),
        prep.MultiAyahWindowPick("r", 1, 2, 3, ("b", "c"), (1.0, 1.0), 2),
    ]
    assert prep.n_ayahs_histogram(picks) == {"2": 2, "3": 1, "4": 0}


def test_sample_window_gaps_seed_and_bounds():
    rng = __import__("random").Random(0)
    gaps = prep.sample_window_gaps(4, rng)
    assert len(gaps) == 3
    assert all(0.0 <= g <= prep.MULTI_GAP_MAX_S for g in gaps)
    rng2 = __import__("random").Random(0)
    assert prep.sample_window_gaps(4, rng2) == gaps
    assert prep.sample_window_gaps(1, rng) == ()


def test_supervision_covers_window_rejects_first_ayah_only():
    assert prep.supervision_covers_window(10.0, 0.0, 10.0)
    assert prep.supervision_covers_window(10.0, 0.0, 9.96)
    assert not prep.supervision_covers_window(10.0, 0.0, 3.5)
    assert not prep.supervision_covers_window(10.0, 0.2, 10.0)


def test_whole_cut_supervision_fields_span_the_window():
    d = prep.whole_cut_supervision_fields(
        cut_id="w1",
        duration=12.5,
        recording_id="r",
        text="abc",
        speaker="s",
        custom={"n_ayahs": 3, "source": "everyayah_multi"},
    )
    assert d["id"] == "w1"
    assert d["start"] == 0.0
    assert d["duration"] == 12.5
    assert d["text"] == "abc"
    assert d["custom"]["n_ayahs"] == 3
    assert prep.supervision_covers_window(12.5, d["start"], d["duration"])
