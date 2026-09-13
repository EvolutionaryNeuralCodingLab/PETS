"""Unit tests for eye-on-sensor sizing helpers (no GUI / no video I/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.analysis.eye_size_on_sensor import (
    NOMINAL_FRAME_DIAGONAL,
    average_species_eye_sizes,
    default_species_blocks,
    format_measurements_report,
    measure_eye_roi,
    pick_large_pupil_frame_index,
)


def test_default_species_blocks_match_species_traces():
    blocks = default_species_blocks()
    assert [b.species for b in blocks] == ["lizard", "mouse", "turtle"]
    assert blocks[0].animal == "PV_126"
    assert blocks[0].block_path.name == "block_007"
    assert blocks[1].animal == "M_002"
    assert blocks[2].animal == "T_18"


def test_measure_eye_roi_area_and_diagonal_pct():
    m = measure_eye_roi(
        (10, 20, 100, 80),
        species="lizard",
        animal="PV_126",
        side="left",
        block_path="/tmp/block_007",
        video_path="/tmp/le.mp4",
        frame_shape_hw=(480, 640),
    )
    assert m.width_px == 100
    assert m.height_px == 80
    assert m.area_px == 8000
    assert m.diagonal_px == pytest.approx(float(np.hypot(100, 80)))
    assert m.diagonal_pct_of_nominal_frame == pytest.approx(
        100.0 * m.diagonal_px / NOMINAL_FRAME_DIAGONAL
    )
    assert NOMINAL_FRAME_DIAGONAL == pytest.approx(800.0)


def test_measure_eye_roi_rejects_empty():
    with pytest.raises(ValueError, match="positive"):
        measure_eye_roi(
            (0, 0, 0, 10),
            species="mouse",
            animal="M_002",
            side="right",
            block_path=".",
            video_path="x.mp4",
        )


def test_average_requires_both_eyes_and_uses_area_and_diagonal():
    left = measure_eye_roi(
        (0, 0, 100, 80),
        species="lizard",
        animal="PV_126",
        side="left",
        block_path="/data/PV_126/block_007",
        video_path="/data/le.mp4",
    )
    right = measure_eye_roi(
        (0, 0, 120, 90),
        species="lizard",
        animal="PV_126",
        side="right",
        block_path="/data/PV_126/block_007",
        video_path="/data/re.mp4",
    )
    assert average_species_eye_sizes([left], require_both_eyes=True) == []

    avg = average_species_eye_sizes([left, right], require_both_eyes=True)
    assert len(avg) == 1
    a = avg[0]
    assert a.species == "lizard"
    assert a.n_eyes == 2
    assert a.mean_area_px == pytest.approx((8000 + 10800) / 2)
    mean_diag = (left.diagonal_px + right.diagonal_px) / 2
    assert a.mean_diagonal_px == pytest.approx(mean_diag)
    assert a.mean_diagonal_pct_of_frame == pytest.approx(
        100.0 * mean_diag / NOMINAL_FRAME_DIAGONAL
    )


def test_format_report_one_row_per_animal_with_lens():
    left = measure_eye_roi(
        (0, 0, 160, 120),
        species="turtle",
        animal="T_18",
        side="left",
        block_path=Path("/data/T_18/block_001"),
        video_path="/data/le.mp4",
    )
    right = measure_eye_roi(
        (0, 0, 160, 120),
        species="turtle",
        animal="T_18",
        side="right",
        block_path=Path("/data/T_18/block_001"),
        video_path="/data/re.mp4",
    )
    text = format_measurements_report([left, right])
    assert "QILENS" in text
    assert "3.7 mm" in text
    assert "90°" in text
    assert "turtle" in text
    assert "T_18" in text
    assert "area:" in text
    assert "diagonal:" in text
    assert "% of frame diagonal" in text
    assert "left:" not in text.lower() or "L=" in text


def test_pick_large_pupil_frame_index_above_p75(tmp_path):
    csv = tmp_path / "left_eye_data_degrees_raw_verified.csv"
    n = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "eye_frame": np.arange(n),
        "major_ax": rng.uniform(5, 30, size=n),
    })
    df.to_csv(csv, index=False)

    threshold = float(np.percentile(df["major_ax"], 75))
    picked = pick_large_pupil_frame_index(csv, rng=np.random.default_rng(0))
    row = df.loc[df["eye_frame"] == picked]
    assert not row.empty
    assert float(row["major_ax"].iloc[0]) >= threshold


def test_pick_large_pupil_frame_index_deterministic_with_seed(tmp_path):
    csv = tmp_path / "eye.csv"
    df = pd.DataFrame({
        "eye_frame": np.arange(100),
        "major_ax": np.linspace(5, 30, 100),
    })
    df.to_csv(csv, index=False)

    a = pick_large_pupil_frame_index(csv, rng=np.random.default_rng(99))
    b = pick_large_pupil_frame_index(csv, rng=np.random.default_rng(99))
    assert a == b


def test_pick_large_pupil_frame_index_rejects_missing_cols(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="major_ax"):
        pick_large_pupil_frame_index(csv)
