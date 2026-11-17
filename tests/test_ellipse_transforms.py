"""Tests for ellipse orientation transforms and OpenCV unit conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.data_verification_utils import (
    horizontal_flip_eye_data,
    rotate_phi_only,
)
from eye_tracking_system_tools.preprocessing.ellipse_fit import ELLIPSE_GEOMETRY_VERSION
from eye_tracking_system_tools.preprocessing.ellipse_overlay import (
    draw_ellipse_overlay,
)


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
    painted = draw_ellipse_overlay(
        blank, df, "eye_frame", 0, ellipse_geometry_version=ELLIPSE_GEOMETRY_VERSION
    )
    rad = np.deg2rad(35.0)
    px = int(round(100 + 40 * np.cos(rad)))
    py = int(round(100 + 40 * np.sin(rad)))
    assert painted[py, px].sum() > 0
    # Axis-aligned major axis tip should not be painted for a 35° ellipse.
    assert painted[100, 140].sum() == 0


def test_draw_ellipse_overlay_legacy_passthrough():
    """Version 1 tables historically passed raw values straight to OpenCV."""
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
    legacy = draw_ellipse_overlay(
        blank, df, "eye_frame", 0, ellipse_geometry_version=1
    )
    # Tiny radian value treated as degrees ≈ axis-aligned.
    assert legacy[100, 140].sum() > 0


def test_horizontal_flip_uses_radian_reflection():
    df = pd.DataFrame(
        {
            "center_x": [10.0],
            "center_y": [50.0],
            "phi": [0.3],
            "width": [40.0],
            "height": [20.0],
        }
    )
    out = horizontal_flip_eye_data(df, frame_width=200)
    assert out["center_x"].iloc[0] == pytest.approx(190.0)
    assert out["phi"].iloc[0] == pytest.approx((np.pi - 0.3) % np.pi)


def test_rotate_phi_only_physical_quarter_turn():
    df = pd.DataFrame(
        {
            "center_x": [10.0],
            "phi": [0.2],
            "width": [40.0],
            "height": [20.0],
        }
    )
    out = rotate_phi_only(df)
    assert out["phi"].iloc[0] == pytest.approx((0.2 + 0.5 * np.pi) % np.pi)
    assert out["width"].iloc[0] == pytest.approx(40.0)
    assert out["height"].iloc[0] == pytest.approx(20.0)
