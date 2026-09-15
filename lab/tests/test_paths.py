"""data_root() fallback chain (env → repo data/ → main checkout)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.paths import data_root  # noqa: E402


def test_data_root_env_data_dir(monkeypatch, tmp_path: Path):
    (tmp_path / "quran.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TILAWA_DATA_ROOT", str(tmp_path))
    assert data_root() == tmp_path.resolve()


def test_data_root_env_repo_root(monkeypatch, tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "quran.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TILAWA_DATA_ROOT", str(tmp_path))
    assert data_root() == data.resolve()


def test_data_root_fallback_without_env(monkeypatch):
    monkeypatch.delenv("TILAWA_DATA_ROOT", raising=False)
    root = data_root()
    assert root.name == "data"
    local = ROOT / "data" / "quran.json"
    if local.is_file():
        assert root == (ROOT / "data").resolve()
    else:
        assert root == Path("/Users/rock/ai/projects/offline-tarteel/data")
