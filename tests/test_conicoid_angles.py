"""3D gaze score: Kerr-chart inversion, yaw/pitch chart, angle between normals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.conicoid.angles import (
    angle_between_normals_deg,
    chart_yaw_pitch,
    gazes_to_kerr_angles,
    kerr_angles_to_normals,
    yaw_pitch_to_normals,
)

REPO = Path(__file__).resolve().parents[1]
SIM_CSV = (
    REPO
    / "development"
    / "kerr_relative_error"
    / "ellipse_angle_mapping_correct_diameter_08mm_distance_13mm.csv"
)


@pytest.fixture(scope="module")
def command_grid() -> tuple[np.ndarray, np.ndarray]:
    if not SIM_CSV.is_file():
        pytest.skip(f"synthetic table missing: {SIM_CSV}")
    df = pd.read_csv(SIM_CSV)
    return df["x_angle"].to_numpy(dtype=float), df["y_angle"].to_numpy(dtype=float)


def test_straight_ahead_is_minus_z():
    n = kerr_angles_to_normals(0.0, 0.0)
    np.testing.assert_allclose(n, [[0.0, 0.0, -1.0]], atol=1e-15)


def test_chart_returns_command_angles(command_grid):
    x, y = command_grid
    yaw, pitch = chart_yaw_pitch(yaw_pitch_to_normals(x, y))
    assert np.nanmax(np.abs(yaw - x)) < 1e-10
    assert np.nanmax(np.abs(pitch - y)) < 1e-10


def test_direction_against_itself_is_zero(command_grid):
    n = yaw_pitch_to_normals(*command_grid)
    assert np.nanmax(angle_between_normals_deg(n, n)) < 1e-5


def test_kerr_pair_roundtrip(command_grid):
    truth = yaw_pitch_to_normals(*command_grid)
    phi, theta = gazes_to_kerr_angles(truth)
    recovered = kerr_angles_to_normals(phi, theta)
    assert np.nanmax(angle_between_normals_deg(recovered, truth)) < 1e-5


def test_pair_subtraction_is_not_the_gaze_error(command_grid):
    """The old score subtracted Kerr's chart from command angles: ~4.94° median."""
    x, y = command_grid
    phi, theta = gazes_to_kerr_angles(yaw_pitch_to_normals(x, y))
    old = np.hypot(phi - x, theta - y)
    assert np.nanmedian(old) == pytest.approx(4.94, abs=0.01)


def test_angle_between_known_rotation():
    a = yaw_pitch_to_normals(0.0, 0.0)
    b = yaw_pitch_to_normals(10.0, 0.0)
    assert angle_between_normals_deg(a, b)[0] == pytest.approx(10.0, abs=1e-9)


def test_non_finite_rows_are_nan():
    a = np.array([[0.0, 0.0, -1.0], [np.nan, 0.0, -1.0], [0.0, 0.0, 0.0]])
    out = angle_between_normals_deg(a, a)
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(out[1]) and np.isnan(out[2])
    phi_nan = kerr_angles_to_normals([np.nan, 5.0], [1.0, 2.0])
    assert np.isnan(phi_nan[0]).all() and np.isfinite(phi_nan[1]).all()
