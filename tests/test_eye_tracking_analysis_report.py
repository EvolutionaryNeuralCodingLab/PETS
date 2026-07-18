"""Tests for ellipse fitting robustness and fit-report side channel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.ellipse_fit import LsqEllipse


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


def test_lsq_ellipse_rejects_complex_eigendecomposition() -> None:
    """Point sets whose eigendecomposition is complex must not yield complex params.

    For this cloud, ``np.linalg.eig`` returns one real + one complex-conjugate
    pair. The real eigenvector fails ``4ac - b^2 > 0``; the old code accepted a
    complex eigenvector and returned complex ellipse parameters.
    """
    X = np.array(
        [
            [442.02758307005246, 521.1634866983218],
            [465.4948413949485, 526.6130910266664],
            [470.48509853488355, 526.4895912859606],
            [476.30316188740596, 525.2027738217914],
            [469.3302916039463, 525.4063969809491],
            [437.81940697022685, 520.565617289417],
            [470.020123700742, 524.2983374845327],
            [473.56330992647406, 526.8620246125631],
            [411.4920636182465, 519.4167832115719],
            [464.90165194482864, 524.2896286837652],
            [469.5435777379704, 524.5836403346974],
            [416.0009110505036, 521.508699566414],
        ]
    )
    with pytest.raises(ValueError, match="ellipse constraint"):
        LsqEllipse().fit(X)

    # Well-conditioned ellipses still fit and stay real float64.
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    el = LsqEllipse().fit(np.c_[3 * np.cos(t) + 10, 2 * np.sin(t) + 20])
    center, width, height, phi = el.as_parameters()
    assert el.coefficients.dtype == np.float64
    assert not any(
        np.iscomplexobj(v) for v in (center[0], center[1], width, height, phi)
    )


def test_eye_tracking_analysis_keeps_float_dtypes_with_pathological_points() -> None:
    """One complex fit must not upcast the whole ellipse DataFrame."""
    df = _make_dlc_df(n_frames=30)
    # Overwrite one data row with a random non-elliptical pupil cloud.
    rng = np.random.default_rng(7)
    row_idx = 5  # first data row after the coords header row at index 0
    for k in range(6):
        df.iloc[row_idx, 1 + 3 * k] = float(rng.uniform(0, 640))
        df.iloc[row_idx, 2 + 3 * k] = float(rng.uniform(0, 480))
        df.iloc[row_idx, 3 + 3 * k] = 0.99

    out = BlockSync.eye_tracking_analysis(df, 0.95)
    for col in ("center_x", "center_y", "width", "height", "phi", "ellipse_size"):
        assert out[col].dtype == np.float64, col
        assert not np.iscomplexobj(out[col].to_numpy())


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
