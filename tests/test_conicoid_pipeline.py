"""Pipeline smoke tests: ellipse tables → c_phi / c_theta, NaNs preserved."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.calculate_conicoid_angles import (
    append_conicoid_angle_data,
)
from eye_tracking_system_tools.preprocessing.conicoid import (
    default_intrinsics,
    fit_eye_and_gaze,
)
from eye_tracking_system_tools.preprocessing.conicoid.geometry import Circle3D
from eye_tracking_system_tools.preprocessing.conicoid.unproject import (
    angular_error_deg,
    project_circle,
)


def _synthetic_eye_table(camera, *, n_nan: int = 1) -> tuple[pd.DataFrame, list[np.ndarray]]:
    f = camera.focal_px
    eye_c = np.array([0.5, -0.2, 13.0])
    radius = 4.0
    rows = []
    true_n = []
    k = 0
    for ph in np.linspace(-0.3, 0.3, 5):
        for th in np.linspace(-0.2, 0.2, 4):
            nx = np.sin(ph)
            ny = np.sin(th) * np.cos(ph)
            nz = -np.sqrt(max(1e-9, 1.0 - nx * nx - ny * ny))
            n = np.array([nx, ny, nz])
            n = n / np.linalg.norm(n)
            circ = Circle3D(centre=eye_c + radius * n, normal=n, radius=1.0)
            el = project_circle(circ, f)
            rows.append(
                {
                    "OE_timestamp": k * 100,
                    "eye_frame": k,
                    "ms_axis": float(k),
                    "center_x": el.cx + camera.cx,
                    "center_y": el.cy + camera.cy,
                    "major_ax": el.major_radius,
                    "minor_ax": el.minor_radius,
                    "width": el.major_radius,
                    "height": el.minor_radius,
                    "phi": el.angle,
                }
            )
            true_n.append(n)
            k += 1
    for i in range(n_nan):
        rows.insert(
            2 + i,
            {
                "OE_timestamp": 10_000 + i,
                "eye_frame": 900 + i,
                "ms_axis": np.nan,
                "center_x": np.nan,
                "center_y": np.nan,
                "major_ax": np.nan,
                "minor_ax": np.nan,
                "width": np.nan,
                "height": np.nan,
                "phi": np.nan,
            },
        )
    return pd.DataFrame(rows), true_n


def test_fit_eye_and_gaze_preserves_nans_and_matches_truth():
    camera = default_intrinsics("left")
    df, true_n = _synthetic_eye_table(camera, n_nan=1)
    out, fit = fit_eye_and_gaze(df, camera, use_ransac=False)
    assert set(("c_phi", "c_theta", "c_nx", "c_ny", "c_nz")).issubset(out.columns)
    assert int(out["c_phi"].isna().sum()) == 1
    assert int(out["c_phi"].notna().sum()) == len(true_n)
    assert np.isfinite(out.loc[out["c_phi"].isna(), "c_nx"]).sum() == 0

    errs = []
    ti = 0
    for _, row in out.iterrows():
        if not np.isfinite(row["c_nx"]):
            continue
        errs.append(
            angular_error_deg([row["c_nx"], row["c_ny"], row["c_nz"]], true_n[ti])
        )
        ti += 1
    assert max(errs) == pytest.approx(0.0, abs=1e-5)
    assert fit.sphere.radius == pytest.approx(4.0, rel=1e-6)


def test_append_conicoid_angle_data_is_idempotent():
    eye = pd.DataFrame(
        {
            "OE_timestamp": [0, 100, 200],
            "center_x": [1.0, 2.0, 3.0],
        }
    )
    angles = pd.DataFrame(
        {
            "OE_timestamp": [0, 100, 200],
            "c_phi": [1.0, 2.0, 3.0],
            "c_theta": [-1.0, 0.0, 1.0],
            "c_nx": [0.1, 0.2, 0.3],
            "c_ny": [0.0, 0.0, 0.0],
            "c_nz": [-1.0, -1.0, -1.0],
        }
    )
    once = append_conicoid_angle_data(eye, angles)
    twice = append_conicoid_angle_data(once, angles)
    assert list(twice.columns).count("c_phi") == 1
    pd.testing.assert_series_equal(twice["c_phi"], angles["c_phi"], check_names=False)
