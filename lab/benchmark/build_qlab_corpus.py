"""
Build test_corpus_qlab from Quran-Lab/quranic-asr-benchmark.

Downloads audio + metadata via huggingface_hub.snapshot_download, copies wavs
flat as `<source>__<file>.wav`, and writes manifest.json in the v3 schema.

everyayah_heldout rows have no (surah, ayah) in the filename — they are mapped
with QuranDB.match_verse (score ≥ 0.95) and dropped otherwise.

Usage:
    .venv/bin/python benchmark/build_qlab_corpus.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.quran_db import QuranDB  # noqa: E402
from shared.paths import data_root  # noqa: E402

# Helpers live next to the Modal staging job so thresholds stay in one place.
_STAGING = PROJECT_ROOT / "scripts" / "prepare_zipformer_data_modal.py"


def _load_staging():
    import importlib.util

    spec = importlib.util.spec_from_file_location("prepare_zipformer_data_modal", _STAGING)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load_staging()

CORPUS_DIR = Path(__file__).parent / "test_corpus_qlab"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
HF_REPO = "Quran-Lab/quranic-asr-benchmark"
SOURCES = ("everyayah_heldout", "qul_alnufais", "tlog_holdout")
MATCH_MIN_SCORE = 0.95


def _quran_path() -> Path:
    wt = PROJECT_ROOT / "data" / "quran.json"
    if wt.is_file():
        return wt
    return data_root() / "quran.json"


def _word_count(db: QuranDB, surah: int, ayah: int) -> int:
    verse = db.get_verse(surah, ayah)
    if not verse:
        return 0
    return len(verse["text_clean"].split())


def _copy_wav(src: Path, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    if not src.is_file() or src.stat().st_size == 0:
        return False
    dest.write_bytes(src.read_bytes())
    return dest.stat().st_size > 0


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    hf_dir = Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            allow_patterns=["audio/**", "benchmark.jsonl", "LICENSE"],
            local_dir=str(CORPUS_DIR / ".hf"),
        )
    )
    print(f"snapshot: {hf_dir}")

    db = QuranDB(_quran_path())
    samples: list[dict] = []
    drops = Counter()
    source_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for source in SOURCES:
        meta_path = hf_dir / "audio" / source / "metadata.jsonl"
        if not meta_path.is_file():
            raise FileNotFoundError(f"missing {meta_path}")
        reciter = prep.qlab_reciter(source)
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                raw_id = str(row.get("id") or "").strip()
                sample_id = raw_id.replace("/", "__") if raw_id else ""
                if sample_id and sample_id in seen_ids:
                    drops[f"{source}_dup_id"] += 1
                    continue
                file_name = row.get("file_name") or ""
                src_wav = hf_dir / "audio" / source / Path(file_name).name
                if not src_wav.is_file():
                    # metadata file_name may already include audio/<source>/
                    alt = hf_dir / file_name
                    src_wav = alt if alt.is_file() else src_wav

                if source == "everyayah_heldout":
                    hit = db.match_verse(row.get("text") or "")
                    if hit is None or float(hit.get("score") or 0) < MATCH_MIN_SCORE:
                        drops["everyayah_heldout_low_match"] += 1
                        continue
                    surah, ayah = int(hit["surah"]), int(hit["ayah"])
                else:
                    parsed = prep.parse_surah_ayah_filename(file_name) or prep.parse_surah_ayah_filename(
                        src_wav.name
                    )
                    if parsed is None:
                        drops[f"{source}_bad_filename"] += 1
                        continue
                    surah, ayah = parsed

                wav_name = src_wav.name if src_wav.is_file() else Path(file_name).name
                flat = prep.qlab_flat_filename(source, wav_name)
                dest = CORPUS_DIR / flat
                if src_wav.is_file():
                    if not _copy_wav(src_wav, dest):
                        drops[f"{source}_copy_fail"] += 1
                        continue
                else:
                    drops[f"{source}_missing_wav"] += 1
                    continue

                if not sample_id:
                    sample_id = Path(flat).stem
                cat = prep.categorize_word_count(_word_count(db, surah, ayah))
                samples.append(
                    {
                        "id": sample_id,
                        "file": flat,
                        "surah": surah,
                        "ayah": ayah,
                        "ayah_end": None,
                        "category": cat,
                        "source": source,
                        "reciter": reciter,
                        "expected_verses": [{"surah": surah, "ayah": ayah}],
                    }
                )
                seen_ids.add(sample_id)
                source_counts[source] += 1
                category_counts[cat] += 1

    MANIFEST_PATH.write_text(
        json.dumps({"samples": samples}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MANIFEST_PATH} ({len(samples)} samples)")
    print("per source:", dict(source_counts))
    print("per category:", dict(category_counts))
    print("drops:", dict(drops))
    print(f"everyayah_heldout mapping drop count: {drops['everyayah_heldout_low_match']}")


if __name__ == "__main__":
    main()
