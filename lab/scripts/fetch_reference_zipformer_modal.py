"""Fetch Quran-Lab/zipformer_p-arabic-v3 onto the zipformer-ctc-training volume.

Idempotent: skips files that already exist at the same size. Always prints
size + sha256 per file under /vol/reference/.

Usage:
  modal run --detach scripts/fetch_reference_zipformer_modal.py
  modal run scripts/fetch_reference_zipformer_modal.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import modal

REPO_ID = "Quran-Lab/zipformer_p-arabic-v3"
VOL_DIR = Path("/vol/reference")

app = modal.App("zipformer-ctc-reference-fetch")
vol = modal.Volume.from_name("zipformer-ctc-training", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]>=0.23", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory(root: Path) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        if rel.startswith(".cache/") or "/.cache/" in rel or rel.startswith(".huggingface"):
            continue
        size = path.stat().st_size
        rows.append({"file": rel, "size": size, "sha256": _sha256(path)})
        print(f"{rel}\t{size}\t{rows[-1]['sha256']}")
    return rows


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def fetch_reference() -> dict:
    from huggingface_hub import list_repo_files, snapshot_download

    VOL_DIR.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(REPO_ID)
    print(f"HF {REPO_ID} files ({len(files)}):")
    for name in files:
        print(f"  {name}")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(VOL_DIR),
    )
    vol.commit()
    print("\n/vol/reference inventory:")
    rows = _inventory(VOL_DIR)
    return {"repo": REPO_ID, "n_files": len(rows), "files": rows}


@app.local_entrypoint()
def main():
    result = fetch_reference.remote()
    print(f"fetched {result['n_files']} files from {result['repo']}")
