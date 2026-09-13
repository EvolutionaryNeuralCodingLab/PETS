"""Tests for DLC project I/O and bodypart detection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eye_tracking_system_tools.analysis.dlc_validation.dlc_h5_io import wide_to_long
from eye_tracking_system_tools.analysis.dlc_validation.landmark_validation import compute_landmark_errors
from eye_tracking_system_tools.analysis.dlc_validation.project_io import (
    is_pupil_bodypart,
    load_dlc_project,
    sort_pupil_bodyparts,
)


def test_is_pupil_bodypart():
    assert is_pupil_bodypart("Pupil_1")
    assert is_pupil_bodypart("pupil_12")
    assert not is_pupil_bodypart("Caudal_edge")
    assert not is_pupil_bodypart("Rostral_edge")


def test_sort_pupil_bodyparts():
    names = ["Pupil_12", "Pupil_3", "Pupil_10", "Pupil_1"]
    assert sort_pupil_bodyparts(names) == ["Pupil_1", "Pupil_3", "Pupil_10", "Pupil_12"]


def test_wide_to_long_pupil_only():
    cols = pd.MultiIndex.from_product(
        [["scorer"], ["Pupil_1", "Caudal_edge"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    wide = pd.DataFrame([[1, 2, 0.9, 3, 4, 0.8]], index=[0], columns=cols)
    long = wide_to_long(wide, pupil_only=True)
    assert len(long) == 1
    assert long.iloc[0]["bodypart"] == "Pupil_1"


def test_compute_landmark_errors_synthetic():
    frame_keys = ["f0", "f1"]
    split_labels = ["train", "test"]
    gt = pd.DataFrame(
        {
            "frame_key": ["f0", "f0", "f1", "f1"],
            "bodypart": ["Pupil_1", "Pupil_2", "Pupil_1", "Pupil_2"],
            "x": [10.0, 20.0, 30.0, 40.0],
            "y": [10.0, 20.0, 30.0, 40.0],
            "likelihood": [1.0, 1.0, 1.0, 1.0],
            "source": ["gt"] * 4,
        }
    )
    pred = pd.DataFrame(
        {
            "frame_key": ["f0", "f0", "f1", "f1"],
            "bodypart": ["Pupil_1", "Pupil_2", "Pupil_1", "Pupil_2"],
            "x_pred": [11.0, 20.0, 30.0, 42.0],
            "y_pred": [10.0, 21.0, 30.0, 40.0],
            "likelihood_pred": [0.99, 0.5, 0.99, 0.99],
            "scorer": ["dlc"] * 4,
            "source": ["pred"] * 4,
        }
    )
    # merge manually for test since compute expects separate gt/pred long
    pred_long = pred.rename(columns={"x_pred": "x", "y_pred": "y", "likelihood_pred": "likelihood"})
    frame_df, summary = compute_landmark_errors(
        gt,
        pred_long,
        split_labels,
        frame_keys,
        likelihood_p_cutoff=0.6,
    )
    assert len(frame_df) == 4
    test_rmse = summary[(summary["split"] == "test") & (summary["filter"] == "all")]["rmse_px"].iloc[0]
    assert test_rmse == pytest.approx(2.0 ** 0.5)  # Pupil_2 off by 2 px; Pupil_1 exact


def test_load_dlc_project_minimal(tmp_path: Path):
    root = tmp_path / "dlc"
    root.mkdir()
    (root / "config.yaml").write_text(
        """
Task: TestTask
scorer: Tester
date: Jan1
bodyparts:
  - Pupil_1
  - Pupil_2
  - Caudal_edge
TrainingFraction: [0.95]
iteration: 0
pcutoff: 0.6
""",
        encoding="utf-8",
    )
    proj = load_dlc_project(root, species="lizard")
    assert proj.scorer == "Tester"
    assert proj.pupil_bodyparts == ["Pupil_1", "Pupil_2"]
    assert "Caudal_edge" not in proj.pupil_bodyparts
