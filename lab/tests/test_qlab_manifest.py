"""Q-lab held-out corpus manifest checks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmark" / "test_corpus_qlab" / "manifest.json"

REQUIRED_SOURCES = {"everyayah_heldout", "qul_alnufais", "tlog_holdout"}
REQUIRED_RECITERS = {"everyayah_heldout", "alnufais", "tlog"}
REQUIRED_FIELDS = {
    "id",
    "file",
    "surah",
    "ayah",
    "ayah_end",
    "category",
    "source",
    "reciter",
    "expected_verses",
}


def test_qlab_manifest_exists_and_loads():
    assert MANIFEST.is_file(), f"missing {MANIFEST}"
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(data.get("samples"), list)
    assert data["samples"], "manifest has no samples"


def test_qlab_samples_schema_unique_ids_and_sources():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    samples = data["samples"]
    ids = [s["id"] for s in samples]
    assert len(ids) == len(set(ids))
    sources = {s["source"] for s in samples}
    assert sources == REQUIRED_SOURCES
    reciters = {s["reciter"] for s in samples}
    assert reciters == REQUIRED_RECITERS
    for s in samples:
        missing = REQUIRED_FIELDS - set(s)
        assert not missing, f"{s.get('id')}: missing {missing}"
        assert s["ayah_end"] is None
        verses = s["expected_verses"]
        assert verses, f"{s['id']}: empty expected_verses"
        assert verses[0]["surah"] == s["surah"]
        assert verses[0]["ayah"] == s["ayah"]
        assert s["category"] in {"short", "medium", "long"}
        assert s["file"].startswith(f"{s['source']}__")
        assert not s["id"].startswith(f"{s['source']}__{s['source']}__")
        assert isinstance(s["surah"], int) and 1 <= s["surah"] <= 114
        assert isinstance(s["ayah"], int) and s["ayah"] >= 1
