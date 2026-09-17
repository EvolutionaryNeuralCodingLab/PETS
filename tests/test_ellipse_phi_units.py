"""Ellipse ``phi`` is stored in radians and must be converted for OpenCV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
    draw_ellipse_overlay,
)
from eye_tracking_system_tools.preprocessing.data_verification_utils import (
    horizontal_flip_eye_data,
    rotate_phi_only,
)
from eye_tracking_system_tools.preprocessing.ellipse_fit import LsqEllipse


def _points_on_rotated_ellipse(deg: float, *, a: float = 40.0, b: float = 15.0, n: int = 60):
    phi = np.deg2rad(deg)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    y = a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    return x, y, phi


def test_lsq_ellipse_as_parameters_returns_radians_not_degrees():
    """A 35° ellipse must come back as ~0.611 rad, not 35.

    ``as_parameters`` uses ``0.5 * np.arctan2(...)`` (numpy radians).
    ``return_fit`` then applies ``cos(phi)/sin(phi)`` with no ``deg2rad``.
    BlockSync writes that value into ``left/right_eye_data.csv`` ``phi``.
    OpenCV ``cv2.ellipse`` documents ``angle`` as degrees, so overlays must
    convert. Passing the raw table value (~0.6) as if it were degrees leaves
    the overlay almost unrotated — the old Verify look.
    """
    x, y, true_rad = _points_on_rotated_ellipse(35.0)
    el = LsqEllipse().fit(np.column_stack([x, y]))
    _center, width, height, phi = el.as_parameters()
    phi = float(phi)
    assert phi == pytest.approx(true_rad, abs=1e-8)
    assert np.degrees(phi) == pytest.approx(35.0, abs=1e-6)
    assert abs(phi - 35.0) > 30.0
    assert 0.5 < phi < 0.8
    assert float(width) == pytest.approx(40.0, rel=1e-6)
    assert float(height) == pytest.approx(15.0, rel=1e-6)

    recon = el.return_fit(n_points=200)
    orig = np.column_stack([x, y])
    d2 = ((orig[:, None, :] - recon[None, :, :]) ** 2).sum(axis=2)
    err_rad = float(np.mean(np.sqrt(d2.min(axis=1))))
    phi_as_deg = np.deg2rad(phi)
    tt = np.linspace(0.0, 2.0 * np.pi, 200)
    cx, cy = float(_center[0]), float(_center[1])
    xw = (
        cx
        + float(width) * np.cos(tt) * np.cos(phi_as_deg)
        - float(height) * np.sin(tt) * np.sin(phi_as_deg)
    )
    yw = (
        cy
        + float(width) * np.cos(tt) * np.sin(phi_as_deg)
        + float(height) * np.sin(tt) * np.cos(phi_as_deg)
    )
    d2w = ((orig[:, None, :] - np.column_stack([xw, yw])[None, :, :]) ** 2).sum(axis=2)
    err_if_degrees = float(np.mean(np.sqrt(d2w.min(axis=1))))
    assert err_rad < 0.5
    assert err_if_degrees > 5.0


def test_draw_ellipse_overlay_converts_radians_to_opencv_degrees():
    phi = float(np.deg2rad(35.0))
    df = pd.DataFrame(
        {
            "eye_frame": [0],
            "center_x": [100.0],
            "center_y": [100.0],
            "width": [40.0],
            "height": [10.0],
            "phi": [phi],
        }
    )
    blank = np.zeros((201, 201, 3), dtype=np.uint8)
    painted = draw_ellipse_overlay(blank, df, "eye_frame", 0, phi_unit="radians")
    rad = np.deg2rad(35.0)
    px = int(round(100 + 40 * np.cos(rad)))
    py = int(round(100 + 40 * np.sin(rad)))
    assert painted[py, px].sum() > 0
    assert painted[100, 140].sum() == 0

    passthrough = draw_ellipse_overlay(blank, df, "eye_frame", 0, phi_unit="passthrough")
    assert passthrough[100, 140].sum() > 0
    assert passthrough[py, px].sum() == 0


def test_draw_ellipse_overlay_skips_missing_refined_columns():
    df = pd.DataFrame(
        {
            "eye_frame": [0],
            "center_x": [80.0],
            "center_y": [60.0],
            "width": [20.0],
            "height": [10.0],
            "phi": [0.4],
        }
    )
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_ellipse_overlay(
        blank, df, "eye_frame", 0, color=(255, 0, 255), column_suffix="refined"
    )
    assert np.array_equal(out, blank)

    df["center_x_refined"] = np.nan
    df["center_y_refined"] = np.nan
    df["width_refined"] = np.nan
    df["height_refined"] = np.nan
    df["phi_refined"] = np.nan
    out_nan = draw_ellipse_overlay(
        blank, df, "eye_frame", 0, color=(255, 0, 255), column_suffix="refined"
    )
    assert np.array_equal(out_nan, blank)


def test_verify_phi_buttons_operate_in_radians():
    df = pd.DataFrame({"phi": [0.2], "center_x": [10.0]})
    rotated = rotate_phi_only(df)
    assert float(rotated.loc[0, "phi"]) == pytest.approx(0.2 + 0.5 * np.pi)
    flipped = horizontal_flip_eye_data(df, frame_width=100)
    assert float(flipped.loc[0, "center_x"]) == pytest.approx(90.0)
    assert float(flipped.loc[0, "phi"]) == pytest.approx(np.pi - 0.2)
