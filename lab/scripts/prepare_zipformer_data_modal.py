"""Stage clean-license Quran recitation audio to lhotse CutSets on Modal.

Writes 16 kHz FLAC + phoneme-string supervisions to volume `zipformer-ctc-training`
(`/vol/audio/<source>/`, `/vol/manifests/<source>_cuts.jsonl.gz`). Optional Kaldi
fbank features land at `/vol/fbank/<source>/` and
`/vol/manifests/<source>_cuts_fbank.jsonl.gz`.

Default `--sources` is the shipped mix (everyayah, qua, iqra, retasy, tlog).
QuranTTS is NPL-1.2 and is excluded from that mix; pass `--sources qurantts`
explicitly only for an internal ablation.

EveryAyah uses `greentechapps/everyayah_curated_1s_20s` **train + validation**
only. The curated **test** split is the leak into q-lab `everyayah_heldout`
(wav names like `test-00009-of-00013_332.wav`) and is never ingested.

Tlog clips whose filename stem is in q-lab `tlog_holdout`, and QUA catalog
rows whose slug/name contains `nufais`, are dropped as held-out leaks.

Usage:
  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --sources everyayah,retasy --limit 20 --skip-fbank

  modal run scripts/prepare_zipformer_data_modal.py --summary-only

  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --sources everyayah --limit 20

  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --sources qurantts --limit 20 --skip-fbank   # NPL-1.2 ablation, not shipped
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Pure helpers (imported by tests; no lhotse)
# ---------------------------------------------------------------------------

ALL_SOURCES = ("everyayah", "qua", "qurantts", "iqra", "retasy", "tlog")
# QuranTTS is NPL-1.2 — keep the prepare function for opt-in ablation, but it
# is not part of the shipped training mix.
DEFAULT_SOURCES = ("everyayah", "qua", "iqra", "retasy", "tlog")
# q-lab everyayah_heldout wavs are curated *test* shards (test-000NN-of-00013_*).
# Ingest train + validation only — never test — so the held-out set stays unseen.
EVERYAYAH_SPLITS = ("train", "validation")
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
_SA_SEARCH_RE = re.compile(
    r"(?<![0-9])(\d{1,3})_(\d{1,3})(?:_[^/]+)?\.(?:wav|flac|mp3)",
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
    text = str(name).replace("\\", "/")
    base = Path(text).name.split("?")[0]
    m = _SA_NAME_RE.match(base)
    if m is None:
        m = _SA_SEARCH_RE.search(base) or _SA_SEARCH_RE.search(text)
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


def tlog_holdout_key(name: str) -> str:
    """Stem used to match tlog training files against q-lab `tlog_holdout`."""
    base = Path(str(name).replace("\\", "/")).name.split("?")[0]
    if base.startswith("tlog_holdout__"):
        base = base[len("tlog_holdout__") :]
    return Path(base).stem


def is_nufais_holdout(**fields: object) -> bool:
    blob = " ".join(str(v).lower() for v in fields.values() if v not in (None, ""))
    return "nufais" in blob


def qlab_exclusions_from_samples(samples: list[dict]) -> dict:
    """Build held-out exclusion sets from a q-lab-style samples list."""
    tlog_ids: set[str] = set()
    for s in samples:
        if (s.get("source") or "") != "tlog_holdout":
            continue
        key = tlog_holdout_key(s.get("file") or s.get("id") or "")
        if key:
            tlog_ids.add(key)
    return {"tlog_ids": tlog_ids}


def qlab_manifest_path() -> Path:
    here = Path(__file__).resolve().parent.parent
    for p in (
        Path("/app/benchmark/test_corpus_qlab/manifest.json"),
        here / "benchmark" / "test_corpus_qlab" / "manifest.json",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("q-lab manifest.json not found (ship via add_local_file)")


def load_qlab_exclusions(path: str | Path | None = None) -> dict:
    manifest = Path(path) if path is not None else qlab_manifest_path()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    samples = data.get("samples") if isinstance(data, dict) else data
    excl = qlab_exclusions_from_samples(list(samples or []))
    print(f"[qlab] excluding {len(excl['tlog_ids'])} tlog_holdout ids from {manifest}")
    return excl


def make_clip_id(source: str, idx: int, surah: int, ayah: int, split: str | None = None) -> str:
    if split:
        return f"{source}_{split}_{idx:08d}_{surah}_{ayah}"
    return f"{source}_{idx:08d}_{surah}_{ayah}"


def flac_clip_path(source: str, clip_id: str, audio_root: Path | str | None = None) -> Path:
    root = Path(audio_root) if audio_root is not None else Path("/vol/audio")
    return root / source / f"{clip_id}.flac"


def existing_flac_duration(path: Path | str) -> float | None:
    """Duration in seconds if `path` is a non-empty FLAC; else None."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        sr = int(f.samplerate or 16000)
        if sr <= 0:
            return None
        return float(len(f) / sr)


def progress_path(source: str, manifest_root: Path | str | None = None) -> Path:
    root = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    return root / f"{source}_progress.json"


def partial_cuts_path(source: str, manifest_root: Path | str | None = None) -> Path:
    root = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    return root / f"{source}_cuts.partial.jsonl"


def final_cuts_path(source: str, manifest_root: Path | str | None = None) -> Path:
    root = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    return root / f"{source}_cuts.jsonl.gz"


def empty_progress() -> dict:
    return {"rows_seen": 0, "hours_kept": 0.0, "clips_kept": 0, "split_rows": {}}


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write `text` via `<path>.tmp` + fsync + `os.replace` (no truncate-in-place)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_progress(source: str, manifest_root: Path | str | None = None) -> dict | None:
    """Return parsed progress, or None if missing/corrupt (never silently empty)."""
    p = progress_path(source, manifest_root)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[{source}] corrupt progress {p}; ignoring")
        return None
    if not isinstance(data, dict):
        print(f"[{source}] non-object progress {p}; ignoring")
        return None
    data.setdefault("rows_seen", 0)
    data.setdefault("hours_kept", 0.0)
    data.setdefault("clips_kept", 0)
    data.setdefault("split_rows", {})
    return data


def save_progress(source: str, payload: dict, manifest_root: Path | str | None = None) -> None:
    p = progress_path(source, manifest_root)
    merged = empty_progress()
    merged.update(payload)
    merged["rows_seen"] = int(merged.get("rows_seen") or 0)
    merged["hours_kept"] = float(merged.get("hours_kept") or 0.0)
    merged["clips_kept"] = int(merged.get("clips_kept") or 0)
    merged["split_rows"] = dict(merged.get("split_rows") or {})
    atomic_write_text(p, json.dumps(merged, indent=2) + "\n")


def append_cut_dict(path: Path | str, cut_dict: dict) -> None:
    """Append one lhotse cut-dict JSON line and flush (no lhotse required)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(cut_dict, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def hours_from_cut_dicts(cut_dicts: list[dict]) -> float:
    return sum(float(d.get("duration") or 0.0) for d in cut_dicts) / 3600.0


def load_cut_dicts(path: Path | str) -> list[dict]:
    """Load JSONL cut dicts. A torn *final* line is dropped and the file repaired.

    JSONDecodeError or UnicodeDecodeError (mid-codepoint crash on
    ensure_ascii=False Arabic) on the last non-empty line: drop, warn, truncate.
    The same errors on any earlier line still raise.
    """
    path = Path(path)
    if not path.is_file():
        return []
    data = path.read_bytes()
    records: list[tuple[int, bytes, bool]] = []
    i = 0
    while i < len(data):
        nl = data.find(b"\n", i)
        if nl == -1:
            records.append((i, data[i:], False))
            break
        records.append((i, data[i:nl], True))
        i = nl + 1
    nonempty = [k for k, (_, chunk, _) in enumerate(records) if chunk.strip()]
    out: list[dict] = []
    last_good_end = 0
    for k, (start, chunk, had_nl) in enumerate(records):
        if not chunk.strip():
            if had_nl:
                last_good_end = start + len(chunk) + 1
            continue
        try:
            out.append(json.loads(chunk.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_last = bool(nonempty) and k == nonempty[-1]
            if is_last:
                print(
                    f"[{path.name}] torn JSONL line at byte offset {start} "
                    f"({type(exc).__name__}); dropping last line and truncating to {last_good_end}"
                )
                with path.open("r+b") as f:
                    f.truncate(last_good_end)
                    f.flush()
                    os.fsync(f.fileno())
                return out
            raise
        last_good_end = start + len(chunk) + (1 if had_nl else 0)
    return out


def cut_id_of(cut_dict: dict) -> str:
    return str(cut_dict.get("id") or cut_dict.get("cut_id") or "")


def finalize_cut_dicts(cut_dicts: list[dict]) -> list[dict]:
    """Prefix+suffix in order; first id wins (no duplicates)."""
    seen: set[str] = set()
    merged: list[dict] = []
    for d in cut_dicts:
        cid = cut_id_of(d)
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        merged.append(d)
    return merged


def remove_partial_cuts(source: str, manifest_root: Path | str | None = None) -> None:
    p = partial_cuts_path(source, manifest_root)
    if p.is_file():
        p.unlink()


def consume_row(state: dict, split: str | None = None) -> int:
    """Mark one HF row as seen. Always increment, including skipped rows.

    Returns the clip-id idx for this row: per-`split` count when `split` is
    set (everyayah split / QUA mushaf slug), else global `rows_seen`.
    """
    if split is not None:
        sr = state.setdefault("split_rows", {})
        key = str(split)
        idx = int(sr.get(key) or 0)
        sr[key] = idx + 1
        state["rows_seen"] = int(state.get("rows_seen") or 0) + 1
        return idx
    idx = int(state.get("rows_seen") or 0)
    state["rows_seen"] = idx + 1
    return idx


def persist_progress(source: str, state: dict, manifest_root: Path | str | None = None) -> None:
    extra = {
        k: state[k]
        for k in state
        if k not in {"cut_dicts", "rows_seen", "hours_kept", "clips_kept", "split_rows"}
    }
    payload = {
        "rows_seen": int(state.get("rows_seen") or 0),
        "hours_kept": float(state.get("hours_kept") or 0.0),
        "clips_kept": int(state.get("clips_kept") or 0),
        "split_rows": dict(state.get("split_rows") or {}),
    }
    payload.update(extra)
    save_progress(source, payload, manifest_root)


def restore_partial_state(
    source: str,
    manifest_root: Path | str | None = None,
    force: bool = False,
) -> dict:
    """Load partial JSONL + progress, or wipe both (and the final cuts) if force.

    `hours_kept` is always recomputed from cut durations (never summed on top of
    a stale progress value). Missing/corrupt progress with leftover cuts falls
    back to `rows_seen = clips_kept = len(cuts)`.
    """
    man = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    partial = partial_cuts_path(source, man)
    prog = progress_path(source, man)
    final = final_cuts_path(source, man)
    if force:
        for p in (
            partial,
            prog,
            Path(str(prog) + ".tmp"),
            final,
            man / f"{source}_stats.json",
            man / f"{source}_cuts_fbank.jsonl.gz",
        ):
            if p.is_file():
                p.unlink()
        state = empty_progress()
        state["cut_dicts"] = []
        return state
    cut_dicts = load_cut_dicts(partial)
    progress = load_progress(source, man)
    n = len(cut_dicts)
    hours = hours_from_cut_dicts(cut_dicts)
    state = empty_progress()
    if progress is None:
        if n:
            print(
                f"[{source}] progress missing/corrupt with {n} partial cuts; "
                f"rows_seen fallback to clips_kept={n} "
                f"(skip may be incomplete; duplicates de-duped by id)"
            )
        state["cut_dicts"] = cut_dicts
        state["rows_seen"] = n
        state["clips_kept"] = n
        state["hours_kept"] = hours
        return state
    state.update(progress)
    state["cut_dicts"] = cut_dicts
    rows_seen = int(progress.get("rows_seen") or 0)
    if n and rows_seen < n:
        print(
            f"[{source}] progress rows_seen={rows_seen} < {n} cuts; "
            f"raising rows_seen to {n} (skip may be incomplete)"
        )
        rows_seen = n
    state["rows_seen"] = rows_seen
    state["hours_kept"] = hours
    state["clips_kept"] = n
    state["split_rows"] = dict(progress.get("split_rows") or {})
    return state


def skip_hf_stream(ds, n: int, label: str = ""):
    """Advance a streaming HF dataset by n rows when `.skip` exists."""
    if n <= 0:
        return ds
    skip = getattr(ds, "skip", None)
    if callable(skip):
        print(f"[{label or 'stream'}] checkpoint skip({n})")
        return skip(n)
    print(f"[{label or 'stream'}] no .skip(); scanning from 0, reusing existing FLACs")
    return ds


def parse_sources(csv: str) -> list[str]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    if not parts:
        return list(DEFAULT_SOURCES)
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
        "datasets>=4.0,<5.0",
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
    .add_local_file(
        str(PROJECT_ROOT / "benchmark" / "test_corpus_qlab" / "manifest.json"),
        remote_path="/app/benchmark/test_corpus_qlab/manifest.json",
    )
)

# Full EveryAyah/QUA ingest is many hours; 24h is Modal's typical max.
# 32 cores match fbank num_jobs. Resume without --force if a source times out.
FBANK_NUM_JOBS = 32
_FN_KW = dict(
    image=image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=32,
    memory=65536,
    timeout=24 * 3600,
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
        "reused_flac": 0,
    }


def patch_hf_list_feature() -> None:
    """Alias datasets-4 `List` so older `datasets` 3.x can read parquet metadata.

    Some QUA mushaf configs (first seen: `ahmed_amer_tvquran`) embed
    `{"_type": "List", ...}` in Arrow schema metadata. No-op on datasets 4+
    where `List` is already registered. Staging image now pins datasets 4.x
    (Audio still `decode=False` so torchcodec is not required).
    """
    from datasets.features import features as feat_mod

    types = feat_mod._FEATURE_TYPES
    if "List" in types:
        return
    seq = types.get("Sequence")
    if seq is None:
        raise RuntimeError("datasets has neither List nor Sequence feature type")
    types["List"] = seq


def _boot_remote() -> None:
    sys.path.insert(0, "/app")
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    Path("/vol/hf_cache").mkdir(parents=True, exist_ok=True)
    Path("/vol/audio").mkdir(parents=True, exist_ok=True)
    Path("/vol/manifests").mkdir(parents=True, exist_ok=True)
    Path("/vol/licenses").mkdir(parents=True, exist_ok=True)
    Path("/vol/fbank").mkdir(parents=True, exist_ok=True)
    patch_hf_list_feature()


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

    patch_hf_list_feature()

    # `split="all"` is a reserved keyword in datasets>=3 and cannot be passed
    # as the split= argument even when the config advertises a split named all.
    if split == "all":
        dd = (
            load_dataset(repo, name, streaming=True)
            if name
            else load_dataset(repo, streaming=True)
        )
        ds = dd["all"] if hasattr(dd, "keys") and "all" in dd else dd
    else:
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


def iqra_match_scores(db, sentence: str, tashkeel: str) -> dict:
    """Compare match_verse / search / hamza-stripped scores for one Iqra row."""
    from shared.normalizer import normalize_arabic

    out: dict[str, float] = {}
    for label, raw in (("sentence", sentence or ""), ("tashkeel", tashkeel or "")):
        raw = str(raw)
        if not raw.strip():
            out[f"{label}_match"] = 0.0
            out[f"{label}_search"] = 0.0
            out[f"{label}_hamza"] = 0.0
            continue
        hit = db.match_verse(raw)
        out[f"{label}_match"] = float(hit["score"]) if hit else 0.0
        hits = db.search(raw, top_k=1)
        out[f"{label}_search"] = float(hits[0]["score"]) if hits else 0.0
        hit_h = db.match_verse(normalize_arabic(raw, strip_hamza=True))
        out[f"{label}_hamza"] = float(hit_h["score"]) if hit_h else 0.0
    return out


def iqra_row_keep_flags(scores: dict, threshold: float = MATCH_MIN_SCORE) -> dict:
    """Keep-rate flags for one Iqra score dict from `iqra_match_scores`."""
    baseline = max(float(scores.get("sentence_match") or 0), float(scores.get("tashkeel_match") or 0))
    search = max(float(scores.get("sentence_search") or 0), float(scores.get("tashkeel_search") or 0))
    hamza = max(float(scores.get("sentence_hamza") or 0), float(scores.get("tashkeel_hamza") or 0))
    return {
        "baseline": baseline >= threshold,
        "search": search >= threshold,
        "hamza": hamza >= threshold,
        "baseline_score": baseline,
        "search_score": search,
        "hamza_score": hamza,
    }


def match_iqra_row(db, sentence: str, tashkeel: str):
    """Primary: match_verse on tashkeel then sentence. Keep ≥ 0.95 unless diag says otherwise."""
    return _match_ayah(db, tashkeel or "") or _match_ayah(db, sentence or "")


def _phoneme_text(corpus, tokenizer, OOVError, surah: int, ayah: int, ayah_end: int | None):
    end = ayah_end if ayah_end is not None else ayah
    phonemes = corpus.span_phonemes(surah, ayah, end)
    tokenizer.encode(phonemes)
    return phonemes


def _save_cuts(source: str, cut_dicts: list, manifest_root: Path | str | None = None) -> None:
    from lhotse import CutSet

    merged = finalize_cut_dicts(list(cut_dicts or []))
    cuts = CutSet.from_dicts(merged)
    out = final_cuts_path(source, manifest_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    cuts.to_file(str(out))
    remove_partial_cuts(source, manifest_root)
    print(f"[{source}] wrote {out} ({len(cuts)} cuts)")


def _write_stats(stats: dict) -> None:
    path = Path(f"/vol/manifests/{stats['source']}_stats.json")
    path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"[{stats['source']}] stats: {json.dumps(stats, default=str)}")


def _sync_stats(stats: dict, state: dict) -> None:
    stats["clips"] = int(state.get("clips_kept") or 0)
    stats["hours"] = float(state.get("hours_kept") or 0.0)


def _begin_or_skip(source: str, force: bool, stats: dict) -> tuple[dict | None, dict | None]:
    """Return (state, None) to ingest, or (None, skip_stats) if already complete.

    A leftover `*_cuts.partial.jsonl` means a crash mid-run: resume even if a
    stale final gzip exists. `--force` deletes partial + progress + final + audio
    + fbank dir + `*_cuts_fbank.jsonl.gz` (smoke leftovers must not skip fbank).
    """
    if force:
        state = restore_partial_state(source, force=True)
        audio_dir = Path("/vol/audio") / source
        if audio_dir.is_dir():
            shutil.rmtree(audio_dir)
        fbank_dir = Path("/vol/fbank") / source
        if fbank_dir.is_dir():
            shutil.rmtree(fbank_dir)
        vol.commit()
        print(
            f"[{source}] --force: cleared partial, progress, final cuts, "
            f"fbank gzip, {audio_dir}, and {fbank_dir}"
        )
        return state, None
    partial = partial_cuts_path(source)
    final = final_cuts_path(source)
    if final.is_file() and not partial.is_file():
        print(f"[{source}] {final} exists; skip (pass --force to redo)")
        existing = Path(f"/vol/manifests/{source}_stats.json")
        if existing.is_file():
            try:
                loaded = json.loads(existing.read_text(encoding="utf-8"))
                loaded["skipped_existing"] = True
                _write_stats(loaded)
                return None, loaded
            except json.JSONDecodeError:
                pass
        stats["skipped_existing"] = True
        _write_stats(stats)
        return None, stats
    state = restore_partial_state(source, force=False)
    n = len(state["cut_dicts"])
    if n:
        _sync_stats(stats, state)
        stats["reused_flac"] = n
        stats["prefix_restored"] = n
        print(
            f"[{source}] resume prefix_cuts={n} rows_seen={state['rows_seen']} "
            f"hours_kept={state['hours_kept']:.4f} clips_kept={state['clips_kept']}"
        )
    return state, None


def _commit(every_n: int, n: int) -> None:
    if n > 0 and n % every_n == 0:
        vol.commit()
        print(f"  volume commit at {n} clips")


def _maybe_crash_after(crash_after: int, limit: int, clips_kept: int) -> None:
    """Hidden test knob: raise after N kept clips when `--limit` is also set."""
    if crash_after > 0 and limit > 0 and int(clips_kept) >= crash_after:
        vol.commit()
        raise RuntimeError(f"crash-after {crash_after} (kept {clips_kept} clips)")


def _finish_row(
    source: str,
    state: dict,
    stats: dict,
    crash_after: int,
    limit: int,
    *,
    kept: bool,
    commit_every: int = 100,
    manifest_root: Path | str | None = None,
) -> None:
    persist_progress(source, state, manifest_root)
    _sync_stats(stats, state)
    if kept:
        _maybe_crash_after(crash_after, limit, int(state.get("clips_kept") or 0))
        _commit(commit_every, int(state.get("clips_kept") or 0))


def _finalize_source(source: str, state: dict, stats: dict) -> dict:
    _sync_stats(stats, state)
    n_prefix = int(stats.get("prefix_restored") or 0)
    n_reused = int(stats.get("reused_flac") or 0)
    if state.get("cut_dicts"):
        _save_cuts(source, state["cut_dicts"])
        print(
            f"[{source}] finalize cuts={stats['clips']} "
            f"prefix_restored={n_prefix} reused_flac={n_reused} "
            f"new_flac={max(int(stats['clips']) - n_reused, 0)}"
        )
    _write_stats(stats)
    vol.commit()
    return stats


def _append_cut(state, source, *, clip_id, flac_path, phonemes, speaker, custom, duration: float, manifest_root=None):
    """Write one MonoCut.to_dict() line immediately, then bump hours/clips."""
    from lhotse import MonoCut, Recording, SupervisionSegment

    rec = Recording.from_file(str(flac_path), recording_id=clip_id)
    cut = MonoCut(
        id=clip_id,
        start=0.0,
        duration=rec.duration,
        channel=0,
        recording=rec,
        supervisions=[
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
        ],
    )
    cut_dict = cut.to_dict()
    append_cut_dict(partial_cuts_path(source, manifest_root), cut_dict)
    state.setdefault("cut_dicts", []).append(cut_dict)
    state["clips_kept"] = int(state.get("clips_kept") or 0) + 1
    state["hours_kept"] = float(state.get("hours_kept") or 0.0) + float(duration) / 3600.0


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
    state: dict,
    extra_custom: dict | None = None,
    split: str | None = None,
    audio_root: Path | str | None = None,
    manifest_root: Path | str | None = None,
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
    clip_id = make_clip_id(source, idx, surah, ayah, split=split)
    flac_path = flac_clip_path(source, clip_id, audio_root=audio_root)
    if wav is not None:
        _write_flac(flac_path, wav)
    elif existing_flac_duration(flac_path) is None:
        _bump_skip(stats, "missing_flac")
        return False
    else:
        stats["reused_flac"] = int(stats.get("reused_flac") or 0) + 1
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
        state,
        source,
        clip_id=clip_id,
        flac_path=flac_path,
        phonemes=phonemes,
        speaker=str(speaker or "unknown"),
        custom=custom,
        duration=duration,
        manifest_root=manifest_root,
    )
    _sync_stats(stats, state)
    return True


def _audio_for_clip(
    *,
    source: str,
    idx: int,
    surah: int,
    ayah: int,
    audio_obj,
    force: bool,
    stats: dict,
    split: str | None = None,
    audio_root: Path | str | None = None,
):
    """Decode audio, or reuse an existing non-empty FLAC (crash resume)."""
    clip_id = make_clip_id(source, idx, surah, ayah, split=split)
    flac_path = flac_clip_path(source, clip_id, audio_root=audio_root)
    if not force:
        dur = existing_flac_duration(flac_path)
        if dur is not None:
            return None, dur
    try:
        return _audio_to_16k_mono(audio_obj)
    except Exception:
        _bump_skip(stats, "audio_error")
        return None, None


# ---------------------------------------------------------------------------
# Per-source prepare functions
# ---------------------------------------------------------------------------


def _prepare_everyayah(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("everyayah")
    print(
        f"[everyayah] splits={list(EVERYAYAH_SPLITS)} "
        "(never test: q-lab everyayah_heldout is curated test shards)"
    )
    state, skipped = _begin_or_skip("everyayah", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()

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
        stats["split"] = "+".join(EVERYAYAH_SPLITS)
        stats["features"] = str(feats)
        for split in EVERYAYAH_SPLITS:
            if limit and state["clips_kept"] >= limit:
                break
            ds = _stream_ds(HF_EVERYAYAH_CURATED, split)
            already = int((state.get("split_rows") or {}).get(split) or 0)
            ds = skip_hf_stream(ds, already, f"everyayah/{split}")
            for row in ds:
                if limit and state["clips_kept"] >= limit:
                    break
                idx = consume_row(state, split=split)
                sa = pick_surah_ayah(row)
                if sa is None:
                    _bump_skip(stats, "no_surah_ayah")
                    _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                    continue
                surah, ayah = sa
                wav, dur = _audio_for_clip(
                    source="everyayah",
                    idx=idx,
                    surah=surah,
                    ayah=ayah,
                    audio_obj=row.get("audio"),
                    force=force,
                    stats=stats,
                    split=split,
                )
                if dur is None:
                    _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                    continue
                kept = _ingest_clip(
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
                    state=state,
                    split=split,
                )
                _finish_row("everyayah", state, stats, crash_after, limit, kept=kept)
    else:
        builder = load_dataset_builder(HF_EVERYAYAH)
        _print_features("everyayah", HF_EVERYAYAH, builder.info.features, builder.info.splits)
        stats["hf_repo"] = HF_EVERYAYAH
        stats["split"] = "train"
        stats["features"] = str(builder.info.features)
        already = int(state.get("rows_seen") or 0)
        ds = skip_hf_stream(_stream_ds(HF_EVERYAYAH, "train"), already, "everyayah")
        for row in ds:
            if limit and state["clips_kept"] >= limit:
                break
            idx = consume_row(state)
            hit = _match_ayah(db, row.get("text") or "")
            if hit is None:
                _bump_skip(stats, "low_match")
                _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                continue
            wav, dur = _audio_for_clip(
                source="everyayah",
                idx=idx,
                surah=int(hit["surah"]),
                ayah=int(hit["ayah"]),
                audio_obj=row.get("audio"),
                force=force,
                stats=stats,
            )
            if dur is None:
                _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                continue
            kept = _ingest_clip(
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
                state=state,
            )
            _finish_row("everyayah", state, stats, crash_after, limit, kept=kept)

    return _finalize_source("everyayah", state, stats)


def _prepare_qua(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("qua")
    stats["hf_repo"] = HF_QUA
    print("[qua] dropping catalog rows whose slug/name contains nufais (q-lab qul_alnufais held-out)")
    state, skipped = _begin_or_skip("qua", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()

    catalog_builder = load_dataset_builder(HF_QUA, "mushafs")
    _print_features("qua/mushafs", HF_QUA, catalog_builder.info.features, catalog_builder.info.splits)
    catalog = _stream_ds(HF_QUA, "all", name="mushafs")
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
        if is_nufais_holdout(
            slug=slug,
            reciter=row.get("name_en") or "",
            reciter_id=row.get("reciter_id") or "",
            name_ar=row.get("name_ar") or "",
        ):
            _bump_skip(stats, "nufais_holdout")
            continue
        if not is_hafs_riwayah(row.get("riwayah")):
            _bump_skip(stats, "not_hafs")
            continue
        kept.append(row)
    nufais_n = int((stats.get("skipped") or {}).get("nufais_holdout") or 0)
    print(
        f"[qua] kept {len(kept)} hafs non-tarteel non-nufais mushafs "
        f"(excluded nufais={nufais_n} tarteel={int((stats.get('skipped') or {}).get('tarteel_dupe') or 0)} "
        f"not_hafs={int((stats.get('skipped') or {}).get('not_hafs') or 0)})"
    )
    if kept:
        sample = kept[0]
        print(
            f"[qua] first kept slug={sample.get('slug')} riwayah={sample.get('riwayah')} "
            f"recording_context={sample.get('recording_context')!r} reciter={sample.get('name_en')}"
        )

    printed_ayah_schema = False
    for mushaf in kept:
        slug = str(mushaf["slug"])
        if limit and state["clips_kept"] >= limit:
            break
        if not printed_ayah_schema:
            b = load_dataset_builder(HF_QUA, slug)
            _print_features(f"qua/{slug}", HF_QUA, b.info.features, b.info.splits)
            stats["features"] = str(b.info.features)
            printed_ayah_schema = True
        ctx = str(mushaf.get("recording_context") or "")
        condition = "studio" if "studio" in ctx.lower() else "crowd"
        speaker = str(mushaf.get("name_en") or slug)
        already = int((state.get("split_rows") or {}).get(slug) or 0)
        print(f"[qua] mushaf slug={slug} already={already}", flush=True)
        try:
            b = load_dataset_builder(HF_QUA, slug)
            splits = getattr(b.info, "splits", None) or {}
            n_ex = int(getattr(splits.get("train"), "num_examples", 0) or 0)
            if n_ex and already >= n_ex:
                print(f"[qua/{slug}] already complete ({already}/{n_ex}); skip load", flush=True)
                continue
            ds = skip_hf_stream(
                _stream_ds(HF_QUA, "train", name=slug), already, f"qua/{slug}"
            )
        except Exception as e:
            print(f"[qua/{slug}] load_dataset failed: {type(e).__name__}: {e}", flush=True)
            _bump_skip(stats, "mushaf_load_error")
            failed = stats.setdefault("failed_slugs", [])
            failed.append(slug)
            continue
        for row in ds:
            if limit and state["clips_kept"] >= limit:
                break
            idx = consume_row(state, split=slug)
            sa = pick_surah_ayah(row)
            if sa is None:
                _bump_skip(stats, "no_surah_ayah")
                _finish_row("qua", state, stats, crash_after, limit, kept=False, commit_every=50)
                continue
            surah, ayah = sa
            wav, dur = _audio_for_clip(
                source="qua",
                idx=idx,
                surah=surah,
                ayah=ayah,
                audio_obj=row.get("audio"),
                force=force,
                stats=stats,
                split=slug,
            )
            if dur is None:
                _finish_row("qua", state, stats, crash_after, limit, kept=False, commit_every=50)
                continue
            extra = {
                "recording_context": ctx,
                "slug": slug,
                "riwayah": mushaf.get("riwayah"),
            }
            kept_clip = _ingest_clip(
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
                state=state,
                extra_custom=extra,
                split=slug,
            )
            _finish_row("qua", state, stats, crash_after, limit, kept=kept_clip, commit_every=50)

    return _finalize_source("qua", state, stats)


def _prepare_qurantts(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder
    from huggingface_hub import hf_hub_download

    stats = _empty_stats("qurantts")
    stats["hf_repo"] = HF_QURANTTS
    state, skipped = _begin_or_skip("qurantts", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()

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
    already = int(state.get("rows_seen") or 0)
    ds = skip_hf_stream(ds, already, "qurantts")
    for row in ds:
        if limit and state["clips_kept"] >= limit:
            break
        idx = consume_row(state)
        fn = row.get("file_name") or row.get("path") or ""
        sa = pick_surah_ayah(row) or parse_surah_ayah_filename(fn)
        if sa is None:
            _bump_skip(stats, "no_surah_ayah")
            _finish_row("qurantts", state, stats, crash_after, limit, kept=False)
            continue
        surah, ayah = sa
        listed = float(row.get("duration_s") or 0)
        if listed > MAX_DURATION_S:
            _bump_skip(stats, "skip_long")
            _finish_row("qurantts", state, stats, crash_after, limit, kept=False)
            continue
        if not fn:
            _bump_skip(stats, "no_file_name")
            _finish_row("qurantts", state, stats, crash_after, limit, kept=False)
            continue
        clip_id = make_clip_id("qurantts", idx, surah, ayah)
        flac_path = flac_clip_path("qurantts", clip_id)
        wav = None
        dur = None if force else existing_flac_duration(flac_path)
        if dur is None:
            try:
                local = hf_hub_download(HF_QURANTTS, fn, repo_type="dataset")
                wav, dur = _wav_from_file(local)
            except Exception as e:
                print(f"[qurantts] audio fail {fn}: {e}")
                _bump_skip(stats, "audio_error")
                _finish_row("qurantts", state, stats, crash_after, limit, kept=False)
                continue
        kept = _ingest_clip(
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
            state=state,
            extra_custom={"riwaya": row.get("riwaya"), "file_name": fn},
        )
        _finish_row("qurantts", state, stats, crash_after, limit, kept=kept)

    return _finalize_source("qurantts", state, stats)


def _prepare_iqra(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("iqra")
    stats["hf_repo"] = HF_IQRA
    print(
        "[iqra] match_verse ≥ 0.95 on tashkeel_sentence then sentence "
        "(Iqra_train is MSA + Quran mix; non-Quran is the drop, not a norm bug); "
        "first 200 rows also score search(top_k=1) and hamza-stripped normalize"
    )
    state, skipped = _begin_or_skip("iqra", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()
    builder = load_dataset_builder(HF_IQRA)
    _print_features("iqra", HF_IQRA, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "train"
    already = int(state.get("rows_seen") or 0)
    ds = skip_hf_stream(_stream_ds(HF_IQRA, "train"), already, "iqra")
    diag_n = 0
    diag_keep = {"baseline": 0, "search": 0, "hamza": 0, "multi_ayah": 0}
    diag_miss_printed = 0
    for row in ds:
        if limit and state["clips_kept"] >= limit:
            break
        idx = consume_row(state)
        sentence = str(row.get("sentence") or "")
        tashkeel = str(row.get("tashkeel_sentence") or "")
        if diag_n < 200:
            scores = iqra_match_scores(db, sentence, tashkeel)
            flags = iqra_row_keep_flags(scores)
            diag_n += 1
            for k in ("baseline", "search", "hamza"):
                if flags[k]:
                    diag_keep[k] += 1
            if not flags["baseline"] and diag_miss_printed < 8:
                words = len(sentence.split())
                print(
                    f"[iqra] miss#{diag_miss_printed} words={words} "
                    f"baseline={flags['baseline_score']:.3f} search={flags['search_score']:.3f} "
                    f"hamza={flags['hamza_score']:.3f} tashkeel_len={len(tashkeel)} "
                    f"sentence={sentence[:80]!r}"
                )
                diag_miss_printed += 1
            if diag_n == 200:
                def _rate(k: str) -> str:
                    return f"{diag_keep[k]}/{diag_n} ({100.0 * diag_keep[k] / diag_n:.1f}%)"

                print(
                    f"[iqra] diag n={diag_n} keep@0.95 "
                    f"baseline(match_verse)={_rate('baseline')} "
                    f"search(top_k=1)={_rate('search')} "
                    f"hamza-strip={_rate('hamza')}"
                )
        hit = match_iqra_row(db, sentence, tashkeel)
        if hit is None:
            _bump_skip(stats, "low_match")
            _finish_row("iqra", state, stats, crash_after, limit, kept=False)
            continue
        ayah_end = hit.get("ayah_end") or int(hit["ayah"])
        if ayah_end != int(hit["ayah"]):
            diag_keep["multi_ayah"] += 1
        wav, dur = _audio_for_clip(
            source="iqra",
            idx=idx,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            audio_obj=row.get("audio"),
            force=force,
            stats=stats,
        )
        if dur is None:
            _finish_row("iqra", state, stats, crash_after, limit, kept=False)
            continue
        kept = _ingest_clip(
            source="iqra",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            ayah_end=ayah_end,
            speaker=str(row.get("id") or "iqra"),
            condition="crowd",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            state=state,
        )
        _finish_row("iqra", state, stats, crash_after, limit, kept=kept)
    if diag_n and diag_n < 200:
        def _rate_partial(k: str) -> str:
            return f"{diag_keep[k]}/{diag_n} ({100.0 * diag_keep[k] / max(diag_n, 1):.1f}%)"

        print(
            f"[iqra] diag n={diag_n} keep@0.95 "
            f"baseline(match_verse)={_rate_partial('baseline')} "
            f"search(top_k=1)={_rate_partial('search')} "
            f"hamza-strip={_rate_partial('hamza')}"
        )
    stats["iqra_match_diag"] = {
        "rows": diag_n,
        "keep_baseline": diag_keep["baseline"],
        "keep_search": diag_keep["search"],
        "keep_hamza": diag_keep["hamza"],
        "multi_ayah_kept": diag_keep["multi_ayah"],
    }
    return _finalize_source("iqra", state, stats)


def _prepare_retasy(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("retasy")
    stats["hf_repo"] = HF_RETASY
    state, skipped = _begin_or_skip("retasy", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()
    builder = load_dataset_builder(HF_RETASY)
    _print_features("retasy", HF_RETASY, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "train"
    already = int(state.get("rows_seen") or 0)
    ds = skip_hf_stream(_stream_ds(HF_RETASY, "train"), already, "retasy")
    for row in ds:
        if limit and state["clips_kept"] >= limit:
            break
        idx = consume_row(state)
        label = row.get("final_label")
        if not retasy_keep(label):
            reason = "bad_label" if label in BAD_RETASY_LABELS else "not_correct"
            _bump_skip(stats, reason)
            _finish_row("retasy", state, stats, crash_after, limit, kept=False)
            continue
        hit = _match_ayah(db, row.get("Aya") or "")
        if hit is None:
            _bump_skip(stats, "low_match")
            _finish_row("retasy", state, stats, crash_after, limit, kept=False)
            continue
        wav, dur = _audio_for_clip(
            source="retasy",
            idx=idx,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            audio_obj=row.get("audio"),
            force=force,
            stats=stats,
        )
        if dur is None:
            _finish_row("retasy", state, stats, crash_after, limit, kept=False)
            continue
        kept = _ingest_clip(
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
            state=state,
            extra_custom={"final_label": label},
        )
        _finish_row("retasy", state, stats, crash_after, limit, kept=kept)
    return _finalize_source("retasy", state, stats)


def _prepare_tlog(limit: int, force: bool, tlog_max_hours: float, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("tlog")
    stats["hf_repo"] = HF_TLOG
    stats["tlog_max_hours"] = tlog_max_hours
    excl = load_qlab_exclusions()
    holdout = excl["tlog_ids"]
    sample = sorted(holdout)[:3]
    print(f"[tlog] q-lab tlog_holdout exclusion: {len(holdout)} ids sample={sample}")
    state, skipped = _begin_or_skip("tlog", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()
    builder = load_dataset_builder(HF_TLOG)
    _print_features("tlog", HF_TLOG, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "clean"
    already = int(state.get("rows_seen") or 0)
    ds = skip_hf_stream(_stream_ds(HF_TLOG, "clean"), already, "tlog")
    for row in ds:
        if limit and state["clips_kept"] >= limit:
            break
        if tlog_max_hours > 0 and float(state.get("hours_kept") or 0.0) >= tlog_max_hours:
            _bump_skip(stats, "hours_cap")
            break
        idx = consume_row(state)
        if not row.get("is_clean", True):
            _bump_skip(stats, "unclean")
            _finish_row("tlog", state, stats, crash_after, limit, kept=False, commit_every=50)
            continue
        audio = row.get("audio") or {}
        path = ""
        if isinstance(audio, dict):
            path = str(audio.get("path") or "")
        for cand in (path, row.get("file_name"), row.get("id"), row.get("label")):
            if cand and parse_surah_ayah_filename(str(cand)):
                path = str(cand)
                break
        parsed = parse_surah_ayah_filename(path)
        if parsed is None:
            if int((stats.get("skipped") or {}).get("unmapped") or 0) < 3:
                print(f"[tlog] unmapped path={path!r} audio_keys={list(audio) if isinstance(audio, dict) else type(audio)}")
            _bump_skip(stats, "unmapped")
            _finish_row("tlog", state, stats, crash_after, limit, kept=False, commit_every=50)
            continue
        if tlog_holdout_key(path) in holdout:
            _bump_skip(stats, "qlab_holdout")
            _finish_row("tlog", state, stats, crash_after, limit, kept=False, commit_every=50)
            continue
        surah, ayah = parsed
        wav, dur = _audio_for_clip(
            source="tlog",
            idx=idx,
            surah=surah,
            ayah=ayah,
            audio_obj=audio,
            force=force,
            stats=stats,
        )
        if dur is None:
            _finish_row("tlog", state, stats, crash_after, limit, kept=False, commit_every=50)
            continue
        kept = _ingest_clip(
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
            state=state,
        )
        _finish_row("tlog", state, stats, crash_after, limit, kept=kept, commit_every=50)
    return _finalize_source("tlog", state, stats)


@app.function(**_FN_KW)
def prepare_everyayah(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_everyayah(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_qua(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_qua(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_qurantts(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_qurantts(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_iqra(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_iqra(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_retasy(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_retasy(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_tlog(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_tlog(limit, force, tlog_max_hours, crash_after=crash_after)


@app.function(**_FN_KW)
def compute_fbank(source: str, no_speed_perturb: bool = False, force: bool = False):
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
    if out_path.is_file() and not force:
        print(f"[fbank/{source}] {out_path} exists; skip (pass --force to redo)")
        return {"source": source, "skipped_existing": True, "path": str(out_path)}
    print(f"[fbank/{source}] load {cuts_path}")
    cuts = CutSet.from_file(str(cuts_path))
    if not no_speed_perturb:
        print(f"[fbank/{source}] speed perturb 0.9/1.1")
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
    storage = Path(f"/vol/fbank/{source}")
    storage.mkdir(parents=True, exist_ok=True)
    extractor = Fbank(FbankConfig(**LHOTSE_FBANK_CONFIG))
    print(f"[fbank/{source}] extract → {storage} ({len(cuts)} cuts, num_jobs={FBANK_NUM_JOBS})")
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=str(storage),
        storage_type=LilcomChunkyWriter,
        num_jobs=FBANK_NUM_JOBS,
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
    print(f"{'source':<12} {'clips':>8} {'hours':>10} {'oov':>6} {'reused':>8} {'license_ok':>12}  skipped")
    print("-" * 88)
    for src in ALL_SOURCES:
        r = summary.get(src)
        if not r:
            print(f"{src:<12} {'—':>8}")
            continue
        skipped = r.get("skipped") or {}
        skip_s = ",".join(f"{k}={v}" for k, v in skipped.items()) or "—"
        print(
            f"{src:<12} {r.get('clips', 0):8d} {float(r.get('hours') or 0):10.3f} "
            f"{r.get('oov', 0):6d} {int(r.get('reused_flac') or 0):8d} "
            f"{str(r.get('license_ok', True)):>12}  {skip_s}"
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
    sources: str = "everyayah,qua,iqra,retasy,tlog",
    limit: int = 0,
    force: bool = False,
    skip_fbank: bool = False,
    tlog_max_hours: float = 100.0,
    summary_only: bool = False,
    no_speed_perturb: bool = False,
    crash_after: int = 0,
):
    if summary_only:
        summarize.remote()
        return
    selected = parse_sources(sources)
    print(
        f"staging sources={selected} limit={limit} force={force} "
        f"skip_fbank={skip_fbank} crash_after={crash_after}"
    )
    handles = {
        src: PREPARE_FNS[src].spawn(limit, force, tlog_max_hours, crash_after)
        for src in selected
    }
    for src, handle in handles.items():
        try:
            stats = handle.get()
            print(
                f"DONE {src}: clips={stats.get('clips')} hours={stats.get('hours')} "
                f"reused_flac={stats.get('reused_flac')}"
            )
        except Exception as e:
            print(f"FAILED {src}: {type(e).__name__}: {e}")
    if not skip_fbank:
        fbank_handles = {
            src: compute_fbank.spawn(src, no_speed_perturb, force) for src in selected
        }
        for src, handle in fbank_handles.items():
            try:
                print(f"FBANK {src}: {handle.get()}")
            except Exception as e:
                print(f"FBANK FAILED {src}: {type(e).__name__}: {e}")
    summarize.remote()
