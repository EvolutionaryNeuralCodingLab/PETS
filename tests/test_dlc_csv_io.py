"""Tests for DLC CSV discovery and default selection."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from eye_tracking_system_tools.preprocessing.dlc_csv_io import (
    default_dlc_csv,
    list_dlc_csvs,
    resolve_dlc_csv,
)


def _touch(path: Path, *, mtime_offset: float = 0.0) -> None:
    path.write_text("scorer,bodyparts\n", encoding="utf-8")
    if mtime_offset:
        ts = time.time() + mtime_offset
        path.touch()
        import os

        os.utime(path, (ts, ts))


def test_list_dlc_csvs_empty_folder(tmp_path: Path) -> None:
    assert list_dlc_csvs(tmp_path) == []


def test_default_dlc_csv_prefers_newest_filtered(tmp_path: Path) -> None:
    old_filtered = tmp_path / "trialDLC_model_filtered.csv"
    new_filtered = tmp_path / "trialDLC_model_v2_filtered.csv"
    newest_plain = tmp_path / "trialDLC_model_v3.csv"
    _touch(old_filtered, mtime_offset=-100)
    _touch(new_filtered, mtime_offset=-10)
    _touch(newest_plain, mtime_offset=100)

    chosen = default_dlc_csv(list_dlc_csvs(tmp_path))
    assert chosen == new_filtered


def test_default_dlc_csv_newest_when_no_filtered(tmp_path: Path) -> None:
    older = tmp_path / "trialDLC_old.csv"
    newer = tmp_path / "trialDLC_new.csv"
    _touch(older, mtime_offset=-50)
    _touch(newer, mtime_offset=50)

    chosen = default_dlc_csv(list_dlc_csvs(tmp_path))
    assert chosen == newer


def test_default_dlc_csv_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        default_dlc_csv([])


def test_resolve_dlc_csv_uses_default_when_none(tmp_path: Path) -> None:
    only = tmp_path / "trialDLC_only.csv"
    _touch(only)
    assert resolve_dlc_csv(tmp_path, None) == only


def test_resolve_dlc_csv_accepts_basename(tmp_path: Path) -> None:
    a = tmp_path / "trialDLC_a.csv"
    b = tmp_path / "trialDLC_b.csv"
    _touch(a, mtime_offset=-10)
    _touch(b, mtime_offset=10)
    assert resolve_dlc_csv(tmp_path, b.name) == b


def test_resolve_dlc_csv_rejects_unknown(tmp_path: Path) -> None:
    _touch(tmp_path / "trialDLC_a.csv")
    with pytest.raises(ValueError):
        resolve_dlc_csv(tmp_path, "missingDLC.csv")
