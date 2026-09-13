"""Tests for ellipse QC frame processing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.analysis.dlc_validation.ellipse_qc import (
    characterize_filtering,
    process_video_h5,
)
from eye_tracking_system_tools.analysis.dlc_validation.summarize import summarize_hierarchy


def _make_long_df(n_frames: int = 5) -> pd.DataFrame:
    rows = []
    for fi in range(n_frames):
        for i, bp in enumerate(["Pupil_1", "Pupil_2", "Pupil_3", "Pupil_4", "Pupil_5", "Pupil_6"]):
            angle = 2 * np.pi * i / 6
            rows.append(
                {
                    "frame_idx": fi,
                    "frame_key": str(fi),
                    "bodypart": bp,
                    "x": 100 + 20 * np.cos(angle),
                    "y": 100 + 10 * np.sin(angle),
                    "likelihood": 0.95 if i < 5 else 0.3,
                    "scorer": "test",
                    "source": "test",
                }
            )
    return pd.DataFrame(rows)


def test_process_video_from_long(monkeypatch, tmp_path):
    long_df = _make_long_df(3)
    pupil_bps = ["Pupil_1", "Pupil_2", "Pupil_3", "Pupil_4", "Pupil_5", "Pupil_6"]

    def fake_load(path, pupil_bodyparts):
        return long_df

    import eye_tracking_system_tools.analysis.dlc_validation.ellipse_qc as eqc

    monkeypatch.setattr(eqc, "load_analyzed_video_long", fake_load)
    h5 = tmp_path / "fakeDLC_test.h5"
    h5.touch()
    df = process_video_h5(h5, pupil_bps, likelihood_p_cutoff=0.6, min_points=5, show_progress=False)
    assert len(df) == 3
    assert df["all_fit_valid"].all()
    assert df["filt_fit_valid"].all()


def test_characterize_filtering():
    df = pd.DataFrame(
        {
            "all_landmarks_above_p_cutoff": [True, False],
            "all_fit_valid": [True, True],
            "filt_fit_valid": [True, False],
            "filt_n_landmarks_used": [6, 3],
            "delta_residual_rmse": [0.1, np.nan],
        }
    )
    stats = characterize_filtering(df)
    assert stats["n_frames"] == 2
    assert stats["frac_all_landmarks_above_p_cutoff"] == 0.5


def test_summarize_hierarchy():
    df = pd.DataFrame(
        {
            "video": ["v1", "v1", "v2", "v2"],
            "eye": ["LE", "LE", "RE", "RE"],
            "animal": ["a1"] * 4,
            "species": ["lizard"] * 4,
            "filt_fit_valid": [True, True, True, False],
            "filt_residual_rmse": [1.0, 2.0, 3.0, np.nan],
            "filt_residual_rmse_norm": [0.01, 0.02, 0.03, np.nan],
            "filt_diameter": [100.0, 100.0, 90.0, np.nan],
            "filt_likelihood_mean": [0.9, 0.95, 0.8, 0.5],
        }
    )
    summaries = summarize_hierarchy(df)
    assert "video" in summaries
    assert len(summaries["video"]) == 2
    assert "species_rollup" in summaries
