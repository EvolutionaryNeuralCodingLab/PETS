"""Tests for DLC CSV discovery and default selection."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from eye_tracking_system_tools.preprocessing.dlc_csv_io import (
    default_dlc_csv,
    likelihood_threshold_stats,
    list_dlc_csvs,
    load_dlc_likelihood_values,
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


def _write_minimal_dlc_csv(path: Path) -> None:
    """Two bodyparts (Pupil, edge0), three likelihood samples after header trim."""
    path.write_text(
        "scorer,scorer,scorer,scorer,scorer,scorer\n"
        "bodyparts,Pupil,Pupil,Pupil,edge0,edge0,edge0\n"
        "coords,x,y,likelihood,x,y,likelihood\n"
        "0,1,2,0.10,3,4,0.20\n"
        "1,1,2,0.40,3,4,0.60\n"
        "2,1,2,0.90,3,4,0.95\n"
        "3,1,2,0.99,3,4,1.00\n",
        encoding="utf-8",
    )


def test_load_dlc_likelihood_values(tmp_path: Path) -> None:
    path = tmp_path / "trialDLC.csv"
    _write_minimal_dlc_csv(path)
    values = load_dlc_likelihood_values(path)
    # header=1 + iloc[1:] drops the coords label row; all numeric frames remain
    assert values.size == 8
    np.testing.assert_allclose(
        np.sort(values),
        [0.10, 0.20, 0.40, 0.60, 0.90, 0.95, 0.99, 1.00],
    )


def test_likelihood_threshold_stats_matches_gt_rule() -> None:
    values = np.array([0.4, 0.6, 0.9, 0.95, 0.99, 1.0])
    stats = likelihood_threshold_stats(values, 0.9)
    assert stats["n_total"] == 6
    assert stats["n_kept"] == 3  # > 0.9
    assert stats["n_removed"] == 3
    assert stats["frac_kept"] == pytest.approx(0.5)
