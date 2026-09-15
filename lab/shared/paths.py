"""Resolve the checkout `data/` directory across worktrees and Modal images.

Order for each lookup: `TILAWA_DATA_ROOT` (data dir or repo root) →
`<repo>/data` → the main local checkout's `data/`. `data_root()` picks the
first of those that contains `quran.json`. Use `resolve_data_file(rel)` when
the file you need may be missing from an earlier root (worktrees track
`data/quran.json` but not `data/prompter/quran.json`).
"""

from __future__ import annotations

import os
from pathlib import Path

_MAIN_DATA = Path("/Users/rock/ai/projects/offline-tarteel/data")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_roots() -> list[Path]:
    """Candidate `data/` dirs, env first then repo then main checkout.

    `TILAWA_DATA_ROOT` may be a `data/` dir or a repo root containing `data/`.
    Missing roots stay in the list so `resolve_data_file` can fall through.
    """
    out: list[Path] = []
    env = os.environ.get("TILAWA_DATA_ROOT")
    if env:
        p = Path(env).expanduser()
        out.append(p)
        out.append(p / "data")
    out.append(_repo_root() / "data")
    out.append(_MAIN_DATA)
    seen: set[Path] = set()
    uniq: list[Path] = []
    for root in out:
        try:
            key = root.resolve()
        except OSError:
            key = root
        if key in seen:
            continue
        seen.add(key)
        uniq.append(root)
    return uniq


def resolve_data_file(rel: str | Path) -> Path:
    """First existing `data/<rel>` across `data_roots()`.

    Raises FileNotFoundError if the file is absent from every root.
    """
    rel_path = Path(rel)
    tried: list[str] = []
    for root in data_roots():
        cand = root / rel_path
        tried.append(str(cand))
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"{rel_path.as_posix()} not found in TILAWA_DATA_ROOT, <repo>/data, "
        f"or {_MAIN_DATA} (tried: {tried})"
    )


def data_root() -> Path:
    """Directory that contains `quran.json` (and usually `prompter/`)."""
    try:
        return resolve_data_file("quran.json").parent
    except FileNotFoundError:
        env = os.environ.get("TILAWA_DATA_ROOT")
        if env:
            p = Path(env).expanduser()
            nested = p / "data"
            if nested.is_dir():
                return nested.resolve()
            return p.resolve()
        return _MAIN_DATA
