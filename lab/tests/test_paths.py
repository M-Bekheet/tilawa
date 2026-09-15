"""data_root() / resolve_data_file() fallback chain."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.paths import data_root, resolve_data_file  # noqa: E402

MAIN_PROMPTER = Path("/Users/rock/ai/projects/offline-tarteel/data/prompter/quran.json")


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


def test_resolve_data_file_env_wins(monkeypatch, tmp_path: Path):
    dest = tmp_path / "prompter"
    dest.mkdir()
    f = dest / "quran.json"
    f.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TILAWA_DATA_ROOT", str(tmp_path))
    assert resolve_data_file("prompter/quran.json") == f.resolve()


def test_resolve_data_file_env_repo_root(monkeypatch, tmp_path: Path):
    dest = tmp_path / "data" / "prompter"
    dest.mkdir(parents=True)
    f = dest / "quran.json"
    f.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TILAWA_DATA_ROOT", str(tmp_path))
    assert resolve_data_file("prompter/quran.json") == f.resolve()


def test_resolve_data_file_falls_through_when_local_missing(monkeypatch):
    monkeypatch.delenv("TILAWA_DATA_ROOT", raising=False)
    local = ROOT / "data" / "prompter" / "quran.json"
    p = resolve_data_file("prompter/quran.json")
    assert p.is_file()
    if local.is_file():
        assert p == local.resolve()
    else:
        assert p == MAIN_PROMPTER.resolve()


def test_resolve_data_file_env_miss_falls_through(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TILAWA_DATA_ROOT", str(tmp_path))
    p = resolve_data_file("prompter/quran.json")
    assert p.is_file()


def test_resolve_data_file_missing_everywhere(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TILAWA_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="definitely-not-here"):
        resolve_data_file("prompter/definitely-not-here.json")
