"""Resolve the checkout `data/` directory across worktrees and Modal images.

Order: `TILAWA_DATA_ROOT` (data dir or repo root) → `<repo>/data` if it
contains `quran.json` → the main local checkout's `data/`.
"""

from __future__ import annotations

import os
from pathlib import Path

_MAIN_DATA = Path("/Users/rock/ai/projects/offline-tarteel/data")


def data_root() -> Path:
    """Directory that contains `quran.json` (and usually `prompter/`)."""
    env = os.environ.get("TILAWA_DATA_ROOT")
    if env:
        p = Path(env).expanduser()
        if (p / "quran.json").is_file():
            return p.resolve()
        nested = p / "data"
        if (nested / "quran.json").is_file():
            return nested.resolve()
        return p.resolve()
    repo = Path(__file__).resolve().parent.parent
    local = repo / "data"
    if (local / "quran.json").is_file():
        return local
    return _MAIN_DATA
