"""Spin maximizer: 1-D contrast search over ellipse phi."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.conicoid.refined_io import (
    attach_spin_columns,
    rotation_fixed_eye_csv_path,
    write_rotation_fixed_eye_table,
)
from eye_tracking_system_tools.preprocessing.conicoid.spin_max import (
    SpinMaxSettings,
    apply_jitter_to_spin_table,
    canonicalize_ellipse_table,
    crop_roi,
    raw_ellipse_table_from_dlc_df,
    spin_maximize_eye_table,
    spin_maximize_frame,
    spin_score,
    x_flip_ellipse_table,
)
from eye_tracking_system_tools.preprocessing.ellipse_fit import canonicalize_ellipse_phi


def _dark_blob(
    angle_deg: float,
    *,
    a: float = 40.0,
    b: float = 15.0,
    size: int = 128,
    cx: float = 64.0,
    cy: float = 64.0,
) -> np.ndarray:
    img = np.full((size, size), 200, dtype=np.uint8)
    phi = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:size, 0:size]
    dx = xx - cx
    dy = yy - cy
    c, s = np.cos(phi), np.sin(phi)
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] = 20
    return img


def test_crop_roi_nan_center_does_not_raise():
    img = np.zeros((16, 16), dtype=np.uint8)
    crop, x0, y0 = crop_roi(img, np.nan, 8.0, 4.0, 3.0, 1.2)
    assert crop.size >= 1
    assert x0 == 0 and y0 == 0


def test_canonicalize_swaps_axes_and_wraps_phi():
    w, h, phi = canonicalize_ellipse_phi(10.0, 20.0, 0.1)
    assert w == pytest.approx(20.0)
    assert h == pytest.approx(10.0)
    assert phi == pytest.approx(0.1 + 0.5 * np.pi)
    w2, h2, phi2 = canonicalize_ellipse_phi(20.0, 10.0, 0.2)
    assert w2 == pytest.approx(20.0)
    assert h2 == pytest.approx(10.0)
    assert 0.0 <= phi2 < np.pi
    w3, h3, phi3 = canonicalize_ellipse_phi(8.0, 8.0, -0.1)
    assert w3 == pytest.approx(8.0)
    assert phi3 == pytest.approx(np.mod(-0.1, np.pi))


def test_canonicalize_ellipse_table_orders_axes():
    df = pd.DataFrame({"width": [10.0], "height": [20.0], "phi": [0.1]})
    out = canonicalize_ellipse_table(df)
    assert out.loc[0, "width"] == pytest.approx(20.0)
    assert out.loc[0, "height"] == pytest.approx(10.0)
    assert out.loc[0, "phi"] == pytest.approx(0.1 + 0.5 * np.pi)
    assert df.loc[0, "width"] == pytest.approx(10.0)


def test_spin_score_prefers_aligned_over_plus_90():
    img = _dark_blob(35.0)
    true = np.deg2rad(35.0)
    aligned = spin_score(img, 64.0, 64.0, 40.0, 15.0, true)
    rotated = spin_score(img, 64.0, 64.0, 40.0, 15.0, true + 0.5 * np.pi)
    assert aligned > rotated


def test_spin_maximize_recovers_known_angle():
    img = _dark_blob(35.0)
    _w, _h, phi, score = spin_maximize_frame(
        img,
        64.0,
        64.0,
        40.0,
        15.0,
        0.05,
        settings=SpinMaxSettings(threshold=80, roi_mult=1.2),
    )
    assert np.isfinite(phi)
    assert np.isfinite(score)
    err = min(abs(phi - np.deg2rad(35.0)), abs(phi - np.deg2rad(35.0) + np.pi))
    assert err < np.deg2rad(8.0)


def test_near_circle_still_returns_finite_phi():
    img = _dark_blob(20.0, a=22.0, b=21.5)
    _w, _h, phi, _score = spin_maximize_frame(
        img,
        64.0,
        64.0,
        22.0,
        21.5,
        0.3,
        settings=SpinMaxSettings(),
    )
    assert np.isfinite(phi)
    assert 0.0 <= phi < np.pi


def test_raw_ellipse_table_from_dlc_df_uses_raw_xy():
    le = pd.DataFrame(
        {
            "L_eye_frame": [3],
            "center_x": [10.0],
            "center_y": [20.0],
            "center_x_corrected": [99.0],
            "center_y_corrected": [88.0],
            "width": [8.0],
            "height": [5.0],
            "phi": [0.2],
        }
    )
    out = raw_ellipse_table_from_dlc_df(le, "left")
    assert out.loc[0, "eye_frame"] == 3
    assert out.loc[0, "center_x"] == pytest.approx(10.0)
    assert out.loc[0, "center_y"] == pytest.approx(20.0)
    assert "center_x_corrected" not in out.columns


def test_spin_maximize_eye_table_writes_spin_columns():
    img = _dark_blob(30.0)
    rows = []
    for k in range(4):
        rows.append(
            {
                "eye_frame": k,
                "center_x": 64.0,
                "center_y": 64.0,
                "width": 40.0,
                "height": 15.0,
                "phi": 0.1,
            }
        )
    df = pd.DataFrame(rows)

    def grab(_idx: int):
        return img

    out = spin_maximize_eye_table(
        df, grab, settings=SpinMaxSettings(threshold=80)
    )
    assert "phi_spin" in out.columns
    assert np.isfinite(out.loc[0, "phi_spin"])
    assert out.loc[0, "center_x_spin"] == pytest.approx(64.0)
    assert out.loc[0, "width_spin"] >= out.loc[0, "height_spin"]


def test_rotation_fixed_sidecar_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "eye_frame": [0, 1],
            "center_x": [10.0, 11.0],
            "center_y": [20.0, 21.0],
            "width": [8.0, 8.0],
            "height": [5.0, 5.0],
            "phi": [0.2, 0.3],
            "center_x_spin": [10.0, 11.0],
            "center_y_spin": [20.0, 21.0],
            "width_spin": [8.0, 8.0],
            "height_spin": [5.0, 5.0],
            "phi_spin": [0.4, 0.5],
        }
    )
    path = rotation_fixed_eye_csv_path(tmp_path, "left")
    write_rotation_fixed_eye_table(df, path)
    assert path.name == "left_rotation_fixed_eye_data.csv"
    right = rotation_fixed_eye_csv_path(tmp_path, "right")
    assert right.name == "right_rotation_fixed_eye_data.csv"
    assert right != path
    eye = df[["eye_frame", "center_x", "center_y", "width", "height", "phi"]].copy()
    attached = attach_spin_columns(eye, df)
    assert attached.loc[0, "phi_spin"] == pytest.approx(0.4)
    assert attached.loc[0, "center_x"] == pytest.approx(10.0)


def test_attach_spin_uses_raw_xy_on_jitter_corrected_preview():
    raw = pd.DataFrame(
        {
            "eye_frame": [7],
            "center_x": [12.0],
            "center_y": [34.0],
            "center_x_spin": [12.0],
            "center_y_spin": [34.0],
            "width_spin": [9.0],
            "height_spin": [4.0],
            "phi_spin": [0.7],
        }
    )
    preview = pd.DataFrame(
        {
            "eye_frame": [7],
            "center_x": [99.0],
            "center_y": [88.0],
            "width": [9.0],
            "height": [4.0],
            "phi": [0.1],
        }
    )
    attached = attach_spin_columns(preview, raw)
    assert attached.loc[0, "center_x"] == pytest.approx(99.0)
    assert attached.loc[0, "center_x_spin"] == pytest.approx(12.0)
    assert attached.loc[0, "phi_spin"] == pytest.approx(0.7)


def test_x_flip_ellipse_table_mirrors_like_verify():
    df = pd.DataFrame(
        {
            "center_x": [100.0],
            "center_y": [40.0],
            "width": [12.0],
            "height": [5.0],
            "phi": [0.3],
        }
    )
    out = x_flip_ellipse_table(df, 640.0)
    assert out.loc[0, "center_x"] == pytest.approx(540.0)
    assert out.loc[0, "center_y"] == pytest.approx(40.0)
    assert out.loc[0, "phi"] == pytest.approx(np.pi - 0.3)
    assert df.loc[0, "center_x"] == pytest.approx(100.0)


def test_spin_then_jitter_xy_matches_eye_data():
    """After spin-max, applying jitter recovers Verify ``left_eye_data`` xy."""
    frames = np.arange(6)
    raw_x = np.full(frames.size, 80.0)
    raw_y = np.full(frames.size, 50.0)
    displacement_x = np.array([1.0, 2.0, -1.0, 4.0, 0.0, 3.0])
    displacement_y = np.array([0.5, -0.5, 1.5, 0.0, 2.0, -1.0])
    jitter = {"x_displacement": displacement_x, "y_displacement": displacement_y}
    img = _dark_blob(30.0, a=20.0, b=8.0, size=160, cx=80.0, cy=50.0)

    raw = pd.DataFrame(
        {
            "eye_frame": frames,
            "center_x": raw_x,
            "center_y": raw_y,
            "width": np.full(frames.size, 20.0),
            "height": np.full(frames.size, 8.0),
            "phi": np.full(frames.size, 0.1),
        }
    )

    def grab(_idx: int):
        return img

    spun = spin_maximize_eye_table(raw, grab, settings=SpinMaxSettings(threshold=80))
    jittered = apply_jitter_to_spin_table(spun, jitter)
    again = apply_jitter_to_spin_table(jittered, jitter)

    from eye_tracking_system_tools.preprocessing.conicoid.spin_max import (
        jitter_deltas_for_frames,
    )

    dx, dy = jitter_deltas_for_frames(jitter, frames)
    eye_x = raw_x + dx
    eye_y = raw_y + dy
    np.testing.assert_allclose(jittered["center_x_spin"].to_numpy(), eye_x)
    np.testing.assert_allclose(jittered["center_y_spin"].to_numpy(), eye_y)
    np.testing.assert_allclose(again["center_x_spin"].to_numpy(), eye_x)
    np.testing.assert_allclose(spun["center_x_spin"].to_numpy(), raw_x)
    assert np.isfinite(jittered["phi_spin"]).all()
    # Phi is rotation-fixed, not copied from the jittered xy.
    assert not np.allclose(jittered["phi_spin"].to_numpy(), 0.1)


def test_x_flip_spin_stores_mirrored_xy():
    img = np.full((40, 100), 180, dtype=np.uint8)
    df = pd.DataFrame(
        {
            "eye_frame": [0],
            "center_x": [20.0],
            "center_y": [10.0],
            "width": [8.0],
            "height": [5.0],
            "phi": [0.2],
        }
    )

    def grab(_idx: int):
        return img

    out = spin_maximize_eye_table(
        df,
        grab,
        settings=SpinMaxSettings(threshold=80, x_flip=True, frame_width=100.0),
    )
    assert out.loc[0, "center_x_spin"] == pytest.approx(80.0)
    assert out.loc[0, "center_y_spin"] == pytest.approx(10.0)
    assert out.loc[0, "center_x"] == pytest.approx(20.0)
