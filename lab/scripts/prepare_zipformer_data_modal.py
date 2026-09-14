"""Stage clean-license Quran recitation audio to lhotse CutSets on Modal.

Writes 16 kHz FLAC + phoneme-string supervisions to volume `zipformer-ctc-training`
(`/vol/audio/<source>/`, `/vol/manifests/<source>_cuts.jsonl.gz`). Optional Kaldi
fbank features land at `/vol/fbank/<source>/` and
`/vol/manifests/<source>_cuts_fbank.jsonl.gz`.

Usage:
  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --sources everyayah,retasy --limit 20 --skip-fbank

  modal run scripts/prepare_zipformer_data_modal.py --summary-only

  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --sources everyayah --limit 20
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Pure helpers (imported by tests; no lhotse)
# ---------------------------------------------------------------------------

ALL_SOURCES = ("everyayah", "qua", "qurantts", "iqra", "retasy", "tlog")
BAD_RETASY_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}
MIN_DURATION_S = 1.0
LONG_DURATION_S = 20.0
MAX_DURATION_S = 60.0
MATCH_MIN_SCORE = 0.95
HF_EVERYAYAH_CURATED = "greentechapps/everyayah_curated_1s_20s"
HF_EVERYAYAH = "tarteel-ai/everyayah"
HF_QUA = "hetchyy/quranic-universal-ayahs"
HF_QURANTTS = "Quran-Lab/QuranTTS"
HF_IQRA = "IqraEval/Iqra_train"
HF_RETASY = "RetaSy/quranic_audio_dataset"
HF_TLOG = "tarteel-ai/tlog"

_SA_NAME_RE = re.compile(
    r"^(\d+)_(\d+)(?:_[^.]+)?\.(?:wav|flac|mp3)$",
    re.IGNORECASE,
)
_SURA_AYAH_KEYS = (
    ("surah", "ayah"),
    ("sura", "ayah"),
    ("chapter", "verse"),
    ("sura_number", "aya_number"),
    ("surah_id", "ayah_id"),
    ("chapter_number", "verse_number"),
)
QLAB_RECITER = {
    "qul_alnufais": "alnufais",
    "everyayah_heldout": "everyayah_heldout",
    "tlog_holdout": "tlog",
}
_PERMISSIVE_LICENSE_MARKERS = (
    "mit license",
    "apache license",
    "bsd license",
    "cc-by",
    "cc by",
    "creative commons attribution",
    "cc0",
    "isc license",
    "unlicense",
)
_NONPERMISSIVE_MARKERS = (
    "no-profit",
    "npl-1",
    "not for sale",
    "commercial use prohibited",
    "non-profit license",
)


def parse_surah_ayah_filename(name: str) -> tuple[int, int] | None:
    """Parse `S_A.wav` / `S_A_id.wav` (optionally with a directory prefix)."""
    if not name:
        return None
    base = Path(str(name)).name
    m = _SA_NAME_RE.match(base)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def duration_decision(duration_s: float) -> str:
    """`keep` (1–20s), `long` (20–60s, still written), or skip_*."""
    if duration_s < MIN_DURATION_S:
        return "skip_short"
    if duration_s > MAX_DURATION_S:
        return "skip_long"
    if duration_s > LONG_DURATION_S:
        return "long"
    return "keep"


def retasy_keep(label: str | None) -> bool:
    return label == "correct"


def is_tarteel_dupe(**fields: object) -> bool:
    blob = " ".join(str(v).lower() for v in fields.values() if v not in (None, ""))
    return "tarteel" in blob


def is_hafs_riwayah(value: object | None) -> bool:
    if value is None or str(value).strip() == "":
        return True
    return "hafs" in str(value).lower()


def basmala_candidate(
    source: str,
    surah: int,
    ayah: int,
    row: dict | None = None,
) -> bool:
    if ayah == 1 and surah not in (1, 9):
        return True
    if source == "qua" and row:
        ctx = str(row.get("recording_context") or "").lower()
        if "basmala" in ctx:
            return True
    return False


def license_is_permissive(text: str) -> bool:
    t = text.lower()
    if any(m in t for m in _NONPERMISSIVE_MARKERS):
        return False
    return any(m in t for m in _PERMISSIVE_LICENSE_MARKERS)


def categorize_word_count(word_count: int) -> str:
    """Same thresholds as `benchmark/build_v3_corpus.py`."""
    if word_count <= 5:
        return "short"
    if word_count <= 15:
        return "medium"
    return "long"


def qlab_reciter(source: str) -> str:
    return QLAB_RECITER[source]


def qlab_flat_filename(source: str, file_name: str) -> str:
    return f"{source}__{Path(file_name).name}"


def parse_sources(csv: str) -> list[str]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    unknown = [p for p in parts if p not in ALL_SOURCES]
    if unknown:
        raise ValueError(f"unknown sources {unknown}; expected subset of {ALL_SOURCES}")
    return parts


def pick_surah_ayah(row: dict) -> tuple[int, int] | None:
    for s_key, a_key in _SURA_AYAH_KEYS:
        if s_key in row and a_key in row and row[s_key] is not None and row[a_key] is not None:
            try:
                return int(row[s_key]), int(row[a_key])
            except (TypeError, ValueError):
                return None
    return None


def load_token_inventory(path: str | Path) -> list[str]:
    """Load icefall `tokens.txt` or Task-1 `tokens.js`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if "export const TOKENS" in text:
        from shared.prompter_labels import load_tokens

        return load_tokens(path)
    by_id: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sym, sep, idx_s = line.rpartition(" ")
        if not sep:
            raise ValueError(f"bad tokens.txt line: {line!r}")
        by_id[int(idx_s)] = sym
    if not by_id:
        raise ValueError(f"no tokens in {path}")
    n = max(by_id) + 1
    missing = [i for i in range(n) if i not in by_id]
    if missing:
        raise ValueError(f"token id gaps in {path}: {missing[:8]}")
    return [by_id[i] for i in range(n)]


def data_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    if (project_root / "data" / "prompter" / "quran.json").is_file():
        return project_root
    return Path(
        os.environ.get("TILAWA_DATA_ROOT", "/Users/rock/ai/projects/offline-tarteel")
    )


def local_quran_json() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    wt = project_root / "data" / "quran.json"
    if wt.is_file():
        return wt
    return data_root() / "data" / "quran.json"


# ---------------------------------------------------------------------------
# Modal image / volume
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = data_root()
_PROMPTER_QURAN = DATA_ROOT / "data" / "prompter" / "quran.json"
_QURAN_JSON = local_quran_json()
_TOKENS_TXT = (
    PROJECT_ROOT
    / "experiments"
    / "prompter-zipformer"
    / "engine"
    / "model"
    / "tokens.txt"
)
_SHARED = PROJECT_ROOT / "shared"

app = modal.App("zipformer-ctc-data")
vol = modal.Volume.from_name("zipformer-ctc-training", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "lhotse",
        "datasets>=3.0,<4.0",
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
        "soundfile",
        "numpy",
        "python-Levenshtein",
    )
    .pip_install(
        "torch",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("lilcom", "kaldi-native-fbank")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(str(_SHARED / "prompter_labels.py"), remote_path="/app/shared/prompter_labels.py")
    .add_local_file(str(_SHARED / "normalizer.py"), remote_path="/app/shared/normalizer.py")
    .add_local_file(str(_SHARED / "quran_db.py"), remote_path="/app/shared/quran_db.py")
    .add_local_file(str(_SHARED / "fbank.py"), remote_path="/app/shared/fbank.py")
    .add_local_file(str(_PROMPTER_QURAN), remote_path="/app/data/prompter/quran.json")
    .add_local_file(str(_QURAN_JSON), remote_path="/app/data/quran.json")
    .add_local_file(str(_TOKENS_TXT), remote_path="/app/tokens.txt")
)

_FN_KW = dict(
    image=image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=16,
    memory=32768,
    timeout=12 * 3600,
)


def _bump_skip(stats: dict, reason: str) -> None:
    skipped = stats.setdefault("skipped", {})
    skipped[reason] = int(skipped.get(reason, 0)) + 1


def _empty_stats(source: str) -> dict:
    return {
        "source": source,
        "clips": 0,
        "hours": 0.0,
        "oov": 0,
        "skipped": {},
        "license_ok": True,
        "features": None,
        "hf_repo": None,
        "split": None,
    }


def _boot_remote() -> None:
    sys.path.insert(0, "/app")
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    Path("/vol/hf_cache").mkdir(parents=True, exist_ok=True)
    Path("/vol/audio").mkdir(parents=True, exist_ok=True)
    Path("/vol/manifests").mkdir(parents=True, exist_ok=True)
    Path("/vol/licenses").mkdir(parents=True, exist_ok=True)
    Path("/vol/fbank").mkdir(parents=True, exist_ok=True)


def _print_features(source: str, repo: str, features, splits) -> None:
    print(f"[{source}] repo={repo}")
    print(f"[{source}] features: {features}")
    print(f"[{source}] splits: {splits}")


def _load_labelers():
    from shared.prompter_labels import OOVError, PhonemeCorpus, PhonemeTokenizer
    from shared.quran_db import QuranDB

    corpus = PhonemeCorpus("/app/data/prompter/quran.json")
    tokenizer = PhonemeTokenizer(load_token_inventory("/app/tokens.txt"))
    db = QuranDB(Path("/app/data/quran.json"))
    return corpus, tokenizer, db, OOVError


def _stream_ds(repo: str, split: str, name: str | None = None, audio_col: str = "audio"):
    """Streaming HF split with Audio(decode=False) so we never need torchcodec."""
    from datasets import Audio, load_dataset

    ds = (
        load_dataset(repo, name, split=split, streaming=True)
        if name
        else load_dataset(repo, split=split, streaming=True)
    )
    feats = getattr(ds, "features", None) or {}
    if audio_col in feats:
        ds = ds.cast_column(audio_col, Audio(sampling_rate=16000, decode=False))
    return ds


def _ffmpeg_pcm16k(src: bytes | str):
    import subprocess
    import numpy as np

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        "pipe:0" if isinstance(src, (bytes, bytearray)) else str(src),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    if isinstance(src, (bytes, bytearray)):
        proc = subprocess.run(cmd, input=src, capture_output=True, check=True)
    else:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"ffmpeg empty audio: {err[:200]}")
    return wav, float(wav.shape[0] / 16000.0)


def _audio_to_16k_mono(audio_obj) -> tuple["object", float]:
    import io

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def _resample(arr, sr: int):
        wav = np.asarray(arr, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1 if wav.shape[-1] <= 8 else 0)
        wav = np.ascontiguousarray(wav.reshape(-1))
        if int(sr) != 16000:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), int(sr), 16000
            ).numpy()
        return wav, float(wav.shape[0] / 16000.0)

    if audio_obj is None:
        raise ValueError("missing audio")
    if isinstance(audio_obj, dict):
        if audio_obj.get("array") is not None:
            return _resample(audio_obj["array"], int(audio_obj.get("sampling_rate") or 16000))
        blob = audio_obj.get("bytes")
        path = audio_obj.get("path")
        if blob is not None:
            raw = bytes(blob)
            try:
                arr, sr = sf.read(io.BytesIO(raw), always_2d=False)
                return _resample(arr, sr)
            except Exception:
                return _ffmpeg_pcm16k(raw)
        if path:
            try:
                arr, sr = sf.read(str(path), always_2d=False)
                return _resample(arr, sr)
            except Exception:
                return _ffmpeg_pcm16k(str(path))
        raise ValueError("audio dict has no array/bytes/path")
    return _resample(audio_obj, 16000)


def _wav_from_file(path: str | Path) -> tuple["object", float]:
    import soundfile as sf

    arr, sr = sf.read(str(path), always_2d=False)
    return _audio_to_16k_mono({"array": arr, "sampling_rate": sr})


def _write_flac(path: Path, wav) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, 16000, format="FLAC")


def _match_ayah(db, text: str):
    if not text or not str(text).strip():
        return None
    hit = db.match_verse(str(text))
    if hit is None or float(hit.get("score") or 0) < MATCH_MIN_SCORE:
        return None
    return hit


def _phoneme_text(corpus, tokenizer, OOVError, surah: int, ayah: int, ayah_end: int | None):
    end = ayah_end if ayah_end is not None else ayah
    phonemes = corpus.span_phonemes(surah, ayah, end)
    tokenizer.encode(phonemes)
    return phonemes


def _save_cuts(source: str, recordings, supervisions) -> None:
    from lhotse import CutSet, RecordingSet, SupervisionSet

    recs = RecordingSet.from_recordings(recordings)
    sups = SupervisionSet.from_segments(supervisions)
    cuts = CutSet.from_manifests(recordings=recs, supervisions=sups)
    out = Path(f"/vol/manifests/{source}_cuts.jsonl.gz")
    cuts.to_file(str(out))
    print(f"[{source}] wrote {out} ({len(cuts)} cuts)")


def _write_stats(stats: dict) -> None:
    path = Path(f"/vol/manifests/{stats['source']}_stats.json")
    path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"[{stats['source']}] stats: {json.dumps(stats, default=str)}")


def _maybe_skip_existing(source: str, force: bool, stats: dict) -> dict | None:
    cuts = Path(f"/vol/manifests/{source}_cuts.jsonl.gz")
    if cuts.is_file() and not force:
        print(f"[{source}] {cuts} exists; skip (pass --force to redo)")
        existing = Path(f"/vol/manifests/{source}_stats.json")
        if existing.is_file():
            try:
                loaded = json.loads(existing.read_text(encoding="utf-8"))
                loaded["skipped_existing"] = True
                _write_stats(loaded)
                return loaded
            except json.JSONDecodeError:
                pass
        stats["skipped_existing"] = True
        _write_stats(stats)
        return stats
    return None


def _commit(every_n: int, n: int) -> None:
    if n > 0 and n % every_n == 0:
        vol.commit()
        print(f"  volume commit at {n} clips")


def _append_cut(recordings, supervisions, *, clip_id, flac_path, phonemes, speaker, custom):
    from lhotse import Recording, SupervisionSegment

    rec = Recording.from_file(str(flac_path), recording_id=clip_id)
    recordings.append(rec)
    supervisions.append(
        SupervisionSegment(
            id=clip_id,
            recording_id=clip_id,
            start=0.0,
            duration=rec.duration,
            channel=0,
            text=phonemes,
            language="quran-phonemes",
            speaker=speaker or "unknown",
            custom=custom,
        )
    )


def _ingest_clip(
    *,
    source: str,
    idx: int,
    wav,
    duration: float,
    surah: int,
    ayah: int,
    ayah_end: int | None,
    speaker: str,
    condition: str,
    stats: dict,
    corpus,
    tokenizer,
    OOVError,
    recordings,
    supervisions,
    extra_custom: dict | None = None,
) -> bool:
    decision = duration_decision(duration)
    if decision.startswith("skip_"):
        _bump_skip(stats, decision)
        return False
    try:
        phonemes = _phoneme_text(corpus, tokenizer, OOVError, surah, ayah, ayah_end)
    except OOVError:
        stats["oov"] += 1
        _bump_skip(stats, "oov")
        return False
    except ValueError:
        _bump_skip(stats, "missing_ayah")
        return False
    clip_id = f"{source}_{idx:08d}_{surah}_{ayah}"
    flac_path = Path("/vol/audio") / source / f"{clip_id}.flac"
    _write_flac(flac_path, wav)
    custom = {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": ayah_end if ayah_end is not None else ayah,
        "source": source,
        "basmala_candidate": basmala_candidate(source, surah, ayah, extra_custom),
        "condition": condition,
        "long": decision == "long",
    }
    if extra_custom:
        custom.update(extra_custom)
    _append_cut(
        recordings,
        supervisions,
        clip_id=clip_id,
        flac_path=flac_path,
        phonemes=phonemes,
        speaker=str(speaker or "unknown"),
        custom=custom,
    )
    stats["clips"] += 1
    stats["hours"] += duration / 3600.0
    return True


# ---------------------------------------------------------------------------
# Per-source prepare functions
# ---------------------------------------------------------------------------


def _prepare_everyayah(limit: int, force: bool) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("everyayah")
    skipped = _maybe_skip_existing("everyayah", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()
    recordings, supervisions = [], []

    curated_ok = False
    try:
        builder = load_dataset_builder(HF_EVERYAYAH_CURATED)
        feats = builder.info.features or {}
        names = set(feats.keys())
        _print_features("everyayah", HF_EVERYAYAH_CURATED, feats, builder.info.splits)
        curated_ok = bool({"sura", "ayah"} <= names or {"chapter", "verse"} <= names)
    except Exception as e:
        print(f"[everyayah] curated builder failed: {type(e).__name__}: {e}")

    if curated_ok:
        stats["hf_repo"] = HF_EVERYAYAH_CURATED
        stats["split"] = "train"
        stats["features"] = str(feats)
        ds = _stream_ds(HF_EVERYAYAH_CURATED, "train")
        for idx, row in enumerate(ds):
            if limit and stats["clips"] >= limit:
                break
            sa = pick_surah_ayah(row)
            if sa is None:
                _bump_skip(stats, "no_surah_ayah")
                continue
            surah, ayah = sa
            try:
                wav, dur = _audio_to_16k_mono(row.get("audio"))
            except Exception:
                _bump_skip(stats, "audio_error")
                continue
            _ingest_clip(
                source="everyayah",
                idx=idx,
                wav=wav,
                duration=dur,
                surah=surah,
                ayah=ayah,
                ayah_end=ayah,
                speaker=str(row.get("qari") or row.get("reciter") or "everyayah"),
                condition="studio",
                stats=stats,
                corpus=corpus,
                tokenizer=tokenizer,
                OOVError=OOVError,
                recordings=recordings,
                supervisions=supervisions,
            )
            _commit(25, stats["clips"])
    else:
        builder = load_dataset_builder(HF_EVERYAYAH)
        _print_features("everyayah", HF_EVERYAYAH, builder.info.features, builder.info.splits)
        stats["hf_repo"] = HF_EVERYAYAH
        stats["split"] = "train"
        stats["features"] = str(builder.info.features)
        ds = _stream_ds(HF_EVERYAYAH, "train")
        for idx, row in enumerate(ds):
            if limit and stats["clips"] >= limit:
                break
            hit = _match_ayah(db, row.get("text") or "")
            if hit is None:
                _bump_skip(stats, "low_match")
                continue
            try:
                wav, dur = _audio_to_16k_mono(row.get("audio"))
            except Exception:
                _bump_skip(stats, "audio_error")
                continue
            _ingest_clip(
                source="everyayah",
                idx=idx,
                wav=wav,
                duration=dur,
                surah=int(hit["surah"]),
                ayah=int(hit["ayah"]),
                ayah_end=hit.get("ayah_end") or int(hit["ayah"]),
                speaker=str(row.get("reciter") or "everyayah"),
                condition="studio",
                stats=stats,
                corpus=corpus,
                tokenizer=tokenizer,
                OOVError=OOVError,
                recordings=recordings,
                supervisions=supervisions,
            )
            _commit(25, stats["clips"])

    if recordings:
        _save_cuts("everyayah", recordings, supervisions)
    _write_stats(stats)
    vol.commit()
    return stats


def _prepare_qua(limit: int, force: bool) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("qua")
    stats["hf_repo"] = HF_QUA
    skipped = _maybe_skip_existing("qua", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()
    recordings, supervisions = [], []

    catalog_builder = load_dataset_builder(HF_QUA, "mushafs")
    _print_features("qua/mushafs", HF_QUA, catalog_builder.info.features, catalog_builder.info.splits)
    catalog = load_dataset(HF_QUA, "mushafs", split="all")
    kept = []
    for row in catalog:
        slug = str(row.get("slug") or "")
        if is_tarteel_dupe(
            slug=slug,
            reciter=row.get("name_en") or "",
            reciter_id=row.get("reciter_id") or "",
            channel=row.get("channel") or "",
        ):
            _bump_skip(stats, "tarteel_dupe")
            continue
        if not is_hafs_riwayah(row.get("riwayah")):
            _bump_skip(stats, "not_hafs")
            continue
        kept.append(row)
    print(f"[qua] kept {len(kept)} hafs non-tarteel mushafs")

    printed_ayah_schema = False
    idx = 0
    for mushaf in kept:
        slug = str(mushaf["slug"])
        if limit and stats["clips"] >= limit:
            break
        if not printed_ayah_schema:
            b = load_dataset_builder(HF_QUA, slug)
            _print_features(f"qua/{slug}", HF_QUA, b.info.features, b.info.splits)
            stats["features"] = str(b.info.features)
            printed_ayah_schema = True
        ctx = str(mushaf.get("recording_context") or "")
        condition = "studio" if "studio" in ctx.lower() else "crowd"
        speaker = str(mushaf.get("name_en") or slug)
        ds = _stream_ds(HF_QUA, "train", name=slug)
        for row in ds:
            if limit and stats["clips"] >= limit:
                break
            sa = pick_surah_ayah(row)
            if sa is None:
                _bump_skip(stats, "no_surah_ayah")
                continue
            surah, ayah = sa
            try:
                wav, dur = _audio_to_16k_mono(row.get("audio"))
            except Exception:
                _bump_skip(stats, "audio_error")
                continue
            extra = {
                "recording_context": ctx,
                "slug": slug,
                "riwayah": mushaf.get("riwayah"),
            }
            _ingest_clip(
                source="qua",
                idx=idx,
                wav=wav,
                duration=dur,
                surah=surah,
                ayah=ayah,
                ayah_end=ayah,
                speaker=speaker,
                condition=condition,
                stats=stats,
                corpus=corpus,
                tokenizer=tokenizer,
                OOVError=OOVError,
                recordings=recordings,
                supervisions=supervisions,
                extra_custom=extra,
            )
            idx += 1
            _commit(50, stats["clips"])

    if recordings:
        _save_cuts("qua", recordings, supervisions)
    _write_stats(stats)
    vol.commit()
    return stats


def _prepare_qurantts(limit: int, force: bool) -> dict:
    from datasets import load_dataset, load_dataset_builder
    from huggingface_hub import hf_hub_download

    stats = _empty_stats("qurantts")
    stats["hf_repo"] = HF_QURANTTS
    skipped = _maybe_skip_existing("qurantts", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()
    recordings, supervisions = [], []

    lic_path = hf_hub_download(HF_QURANTTS, "LICENSE", repo_type="dataset")
    lic_text = Path(lic_path).read_text(encoding="utf-8")
    Path("/vol/licenses/qurantts.txt").write_text(lic_text, encoding="utf-8")
    stats["license_ok"] = license_is_permissive(lic_text)
    print(f"[qurantts] license_ok={stats['license_ok']}")

    from datasets import get_dataset_config_names

    configs = get_dataset_config_names(HF_QURANTTS)
    cfg = "metadata" if "metadata" in configs else (configs[0] if configs else None)
    print(f"[qurantts] configs={configs} using={cfg}")
    builder = load_dataset_builder(HF_QURANTTS, cfg) if cfg else load_dataset_builder(HF_QURANTTS)
    _print_features("qurantts", HF_QURANTTS, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    split_names = list(builder.info.splits or {"train": None})
    split = "train" if "train" in split_names else split_names[0]
    stats["split"] = f"{cfg}:{split}" if cfg else split
    ds = (
        load_dataset(HF_QURANTTS, cfg, split=split, streaming=True)
        if cfg
        else load_dataset(HF_QURANTTS, split=split, streaming=True)
    )
    for idx, row in enumerate(ds):
        if limit and stats["clips"] >= limit:
            break
        fn = row.get("file_name") or row.get("path") or ""
        sa = pick_surah_ayah(row) or parse_surah_ayah_filename(fn)
        if sa is None:
            _bump_skip(stats, "no_surah_ayah")
            continue
        surah, ayah = sa
        listed = float(row.get("duration_s") or 0)
        if listed > MAX_DURATION_S:
            _bump_skip(stats, "skip_long")
            continue
        if not fn:
            _bump_skip(stats, "no_file_name")
            continue
        try:
            local = hf_hub_download(HF_QURANTTS, fn, repo_type="dataset")
            wav, dur = _wav_from_file(local)
        except Exception as e:
            print(f"[qurantts] audio fail {fn}: {e}")
            _bump_skip(stats, "audio_error")
            continue
        _ingest_clip(
            source="qurantts",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=surah,
            ayah=ayah,
            ayah_end=ayah,
            speaker=str(row.get("reciter") or "qurantts"),
            condition="studio",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            recordings=recordings,
            supervisions=supervisions,
            extra_custom={"riwaya": row.get("riwaya"), "file_name": fn},
        )
        _commit(25, stats["clips"])

    if recordings:
        _save_cuts("qurantts", recordings, supervisions)
    _write_stats(stats)
    vol.commit()
    return stats


def _prepare_iqra(limit: int, force: bool) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("iqra")
    stats["hf_repo"] = HF_IQRA
    skipped = _maybe_skip_existing("iqra", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()
    recordings, supervisions = [], []
    builder = load_dataset_builder(HF_IQRA)
    _print_features("iqra", HF_IQRA, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "train"
    ds = _stream_ds(HF_IQRA, "train")
    for idx, row in enumerate(ds):
        if limit and stats["clips"] >= limit:
            break
        hit = _match_ayah(db, row.get("sentence") or row.get("tashkeel_sentence") or "")
        if hit is None:
            _bump_skip(stats, "low_match")
            continue
        try:
            wav, dur = _audio_to_16k_mono(row.get("audio"))
        except Exception:
            _bump_skip(stats, "audio_error")
            continue
        _ingest_clip(
            source="iqra",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            ayah_end=hit.get("ayah_end") or int(hit["ayah"]),
            speaker=str(row.get("id") or "iqra"),
            condition="crowd",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            recordings=recordings,
            supervisions=supervisions,
        )
        _commit(25, stats["clips"])
    if recordings:
        _save_cuts("iqra", recordings, supervisions)
    _write_stats(stats)
    vol.commit()
    return stats


def _prepare_retasy(limit: int, force: bool) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("retasy")
    stats["hf_repo"] = HF_RETASY
    skipped = _maybe_skip_existing("retasy", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()
    recordings, supervisions = [], []
    builder = load_dataset_builder(HF_RETASY)
    _print_features("retasy", HF_RETASY, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "train"
    ds = _stream_ds(HF_RETASY, "train")
    for idx, row in enumerate(ds):
        if limit and stats["clips"] >= limit:
            break
        label = row.get("final_label")
        if not retasy_keep(label):
            reason = "bad_label" if label in BAD_RETASY_LABELS else "not_correct"
            _bump_skip(stats, reason)
            continue
        hit = _match_ayah(db, row.get("Aya") or "")
        if hit is None:
            _bump_skip(stats, "low_match")
            continue
        try:
            wav, dur = _audio_to_16k_mono(row.get("audio"))
        except Exception:
            _bump_skip(stats, "audio_error")
            continue
        _ingest_clip(
            source="retasy",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            ayah_end=hit.get("ayah_end") or int(hit["ayah"]),
            speaker=str(row.get("reciter_id") or "retasy"),
            condition="crowd",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            recordings=recordings,
            supervisions=supervisions,
            extra_custom={"final_label": label},
        )
        _commit(25, stats["clips"])
    if recordings:
        _save_cuts("retasy", recordings, supervisions)
    _write_stats(stats)
    vol.commit()
    return stats


def _prepare_tlog(limit: int, force: bool, tlog_max_hours: float) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("tlog")
    stats["hf_repo"] = HF_TLOG
    stats["tlog_max_hours"] = tlog_max_hours
    skipped = _maybe_skip_existing("tlog", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()
    recordings, supervisions = [], []
    builder = load_dataset_builder(HF_TLOG)
    _print_features("tlog", HF_TLOG, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "clean"
    ds = _stream_ds(HF_TLOG, "clean")
    for idx, row in enumerate(ds):
        if limit and stats["clips"] >= limit:
            break
        if tlog_max_hours > 0 and stats["hours"] >= tlog_max_hours:
            _bump_skip(stats, "hours_cap")
            break
        if not row.get("is_clean", True):
            _bump_skip(stats, "unclean")
            continue
        audio = row.get("audio") or {}
        path = ""
        if isinstance(audio, dict):
            path = str(audio.get("path") or "")
        path = path or str(row.get("label") or "")
        parsed = parse_surah_ayah_filename(path)
        if parsed is None:
            _bump_skip(stats, "unmapped")
            continue
        surah, ayah = parsed
        try:
            wav, dur = _audio_to_16k_mono(audio)
        except Exception:
            _bump_skip(stats, "audio_error")
            continue
        _ingest_clip(
            source="tlog",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=surah,
            ayah=ayah,
            ayah_end=ayah,
            speaker="tlog",
            condition="crowd",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            recordings=recordings,
            supervisions=supervisions,
        )
        _commit(50, stats["clips"])
    if recordings:
        _save_cuts("tlog", recordings, supervisions)
    _write_stats(stats)
    vol.commit()
    return stats


@app.function(**_FN_KW)
def prepare_everyayah(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0):
    _boot_remote()
    return _prepare_everyayah(limit, force)


@app.function(**_FN_KW)
def prepare_qua(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0):
    _boot_remote()
    return _prepare_qua(limit, force)


@app.function(**_FN_KW)
def prepare_qurantts(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0):
    _boot_remote()
    return _prepare_qurantts(limit, force)


@app.function(**_FN_KW)
def prepare_iqra(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0):
    _boot_remote()
    return _prepare_iqra(limit, force)


@app.function(**_FN_KW)
def prepare_retasy(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0):
    _boot_remote()
    return _prepare_retasy(limit, force)


@app.function(**_FN_KW)
def prepare_tlog(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0):
    _boot_remote()
    return _prepare_tlog(limit, force, tlog_max_hours)


@app.function(**_FN_KW)
def compute_fbank(source: str, no_speed_perturb: bool = False):
    """Extract lhotse fbanks (+ optional 0.9/1.1 speed copies) for one source."""
    _boot_remote()
    import torch
    from lhotse import CutSet, Fbank, FbankConfig
    from lhotse.features.io import LilcomChunkyWriter
    from shared.fbank import LHOTSE_FBANK_CONFIG

    torch.set_num_threads(1)

    cuts_path = Path(f"/vol/manifests/{source}_cuts.jsonl.gz")
    if not cuts_path.is_file():
        raise FileNotFoundError(f"missing {cuts_path}; run prepare first")
    out_path = Path(f"/vol/manifests/{source}_cuts_fbank.jsonl.gz")
    print(f"[fbank/{source}] load {cuts_path}")
    cuts = CutSet.from_file(str(cuts_path))
    if not no_speed_perturb:
        print(f"[fbank/{source}] speed perturb 0.9/1.1")
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
    storage = Path(f"/vol/fbank/{source}")
    storage.mkdir(parents=True, exist_ok=True)
    extractor = Fbank(FbankConfig(**LHOTSE_FBANK_CONFIG))
    print(f"[fbank/{source}] extract → {storage} ({len(cuts)} cuts, num_jobs=16)")
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=str(storage),
        storage_type=LilcomChunkyWriter,
        num_jobs=16,
    )
    cuts.to_file(str(out_path))
    print(f"[fbank/{source}] wrote {out_path}")
    vol.commit()
    return {"source": source, "cuts": len(cuts), "path": str(out_path)}


@app.function(**_FN_KW)
def summarize():
    _boot_remote()
    rows = []
    for src in ALL_SOURCES:
        p = Path(f"/vol/manifests/{src}_stats.json")
        if p.is_file():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    summary = {r["source"]: r for r in rows}
    Path("/vol/manifests/summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    vol.commit()
    print()
    print(f"{'source':<12} {'clips':>8} {'hours':>10} {'oov':>6} {'license_ok':>12}  skipped")
    print("-" * 80)
    for src in ALL_SOURCES:
        r = summary.get(src)
        if not r:
            print(f"{src:<12} {'—':>8}")
            continue
        skipped = r.get("skipped") or {}
        skip_s = ",".join(f"{k}={v}" for k, v in skipped.items()) or "—"
        print(
            f"{src:<12} {r.get('clips', 0):8d} {float(r.get('hours') or 0):10.3f} "
            f"{r.get('oov', 0):6d} {str(r.get('license_ok', True)):>12}  {skip_s}"
        )
    print()
    print("wrote /vol/manifests/summary.json")
    return summary


PREPARE_FNS = {
    "everyayah": prepare_everyayah,
    "qua": prepare_qua,
    "qurantts": prepare_qurantts,
    "iqra": prepare_iqra,
    "retasy": prepare_retasy,
    "tlog": prepare_tlog,
}


@app.local_entrypoint()
def main(
    sources: str = "everyayah,qua,qurantts,iqra,retasy,tlog",
    limit: int = 0,
    force: bool = False,
    skip_fbank: bool = False,
    tlog_max_hours: float = 100.0,
    summary_only: bool = False,
    no_speed_perturb: bool = False,
):
    if summary_only:
        summarize.remote()
        return
    selected = parse_sources(sources)
    print(f"staging sources={selected} limit={limit} force={force} skip_fbank={skip_fbank}")
    handles = {
        src: PREPARE_FNS[src].spawn(limit, force, tlog_max_hours) for src in selected
    }
    for src, handle in handles.items():
        try:
            stats = handle.get()
            print(f"DONE {src}: clips={stats.get('clips')} hours={stats.get('hours')}")
        except Exception as e:
            print(f"FAILED {src}: {type(e).__name__}: {e}")
    if not skip_fbank:
        fbank_handles = {
            src: compute_fbank.spawn(src, no_speed_perturb) for src in selected
        }
        for src, handle in fbank_handles.items():
            try:
                print(f"FBANK {src}: {handle.get()}")
            except Exception as e:
                print(f"FBANK FAILED {src}: {type(e).__name__}: {e}")
    summarize.remote()
