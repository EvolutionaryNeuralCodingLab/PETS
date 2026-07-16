"""Tests for ellipse fitting robustness and fit-report side channel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync


def _make_dlc_df(n_frames: int = 12, *, collinear: bool = False) -> pd.DataFrame:
    """Minimal header=1-style DLC table for eye_tracking_analysis."""
    cols = [
        "bodyparts",
        "Pupil_1",
        "Pupil_1.1",
        "Pupil_1.2",
        "Pupil_2",
        "Pupil_2.1",
        "Pupil_2.2",
        "Pupil_3",
        "Pupil_3.1",
        "Pupil_3.2",
        "Pupil_4",
        "Pupil_4.1",
        "Pupil_4.2",
        "Pupil_5",
        "Pupil_5.1",
        "Pupil_5.2",
        "Pupil_6",
        "Pupil_6.1",
        "Pupil_6.2",
        "Caudal_edge",
        "Caudal_edge.1",
        "Caudal_edge.2",
        "Rostral_edge",
        "Rostral_edge.1",
        "Rostral_edge.2",
    ]
    rows: list[list[object]] = [
        ["coords", *sum([["x", "y", "likelihood"]] * 8, [])],
    ]
    for i in range(n_frames):
        if collinear:
            x_base = 100.0 + i
            pts = [
                x_base,
                200.0,
                0.99,
                x_base + 1,
                200.0,
                0.99,
                x_base + 2,
                200.0,
                0.99,
                x_base + 3,
                200.0,
                0.99,
                x_base + 4,
                200.0,
                0.99,
                x_base + 5,
                200.0,
                0.99,
            ]
        else:
            pts = [
                100.0 + i,
                200.0,
                0.99,
                110.0 + i,
                205.0,
                0.99,
                105.0 + i,
                195.0,
                0.99,
                115.0 + i,
                210.0,
                0.99,
                108.0 + i,
                198.0,
                0.99,
                112.0 + i,
                202.0,
                0.99,
            ]
        rows.append([i, *pts])
    return pd.DataFrame(rows, columns=cols)


def _expected_ellipse_rows(df: pd.DataFrame) -> int:
    """Rows fitted by eye_tracking_analysis for a header=1 DLC table."""
    return max(0, len(df) - 3)


def test_eye_tracking_analysis_returns_dataframe_only() -> None:
    df = _make_dlc_df()
    out = BlockSync.eye_tracking_analysis(df, 0.95)
    assert isinstance(out, pd.DataFrame)
    assert "center_x" in out.columns
    assert "ellipse_size" in out.columns


def test_eye_tracking_analysis_collinear_points_do_not_raise() -> None:
    df = _make_dlc_df(collinear=True)
    report: dict = {}
    out = BlockSync.eye_tracking_analysis(df, 0.95, fit_report=report)
    assert len(out) == _expected_ellipse_rows(df)
    assert report["n_fit_failed"] >= 1
    assert out["center_x"].isna().any()


def test_eye_tracking_analysis_populates_fit_report() -> None:
    df = _make_dlc_df()
    report: dict = {}
    BlockSync.eye_tracking_analysis(df, 0.95, fit_report=report)
    assert report["n_frames"] == _expected_ellipse_rows(df)
    assert "yield_pct" in report
    assert "mean_likelihood_all" in report
    assert "mean_likelihood_used" in report
    assert report["uncertainty_thr"] == 0.95


def test_read_dlc_data_sets_report_attribute(tmp_path: Path, monkeypatch) -> None:
    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync

    block = object.__new__(BlockSync)
    block.analysis_path = tmp_path
    block.l_e_path = tmp_path / "LE"
    block.r_e_path = tmp_path / "RE"
    block.l_e_path.mkdir()
    block.r_e_path.mkdir()
    block.sample_rate = 20000

    le_csv = block.l_e_path / "trialDLC_le.csv"
    re_csv = block.r_e_path / "trialDLC_re.csv"
    dlc = _make_dlc_df()
    dlc.to_csv(le_csv, index=False)
    dlc.to_csv(re_csv, index=False)

    block.final_sync_df = pd.DataFrame(
        {
            "Arena_frame": [0, 1],
            "L_eye_frame": [0, 1],
            "R_eye_frame": [0, 1],
            "Arena_TTL": [0, 1000],
        }
    )

    monkeypatch.setattr(
        BlockSync,
        "eye_tracking_analysis",
        staticmethod(lambda *_a, **_k: pd.DataFrame({"center_x": [1.0]})),
    )

    result = block.read_dlc_data(export=False, overwrite=True)
    assert result is None
    assert block.dlc_ellipse_fit_report is not None
    assert "left" in block.dlc_ellipse_fit_report
    assert "right" in block.dlc_ellipse_fit_report
