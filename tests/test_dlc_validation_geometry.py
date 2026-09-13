"""Tests for DLC validation geometry helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eye_tracking_system_tools.analysis.dlc_validation.geometry import (
    fit_ellipse,
    point_on_ellipse,
    point_to_ellipse_distance,
    representative_diameter,
    residual_stats,
)


def test_representative_diameter_geometric_mean():
    d = representative_diameter(10.0, 5.0, "geometric_mean")
    assert d == pytest.approx(2 * math.sqrt(50))


def test_point_on_ellipse_at_axis():
    x, y = point_on_ellipse(0.0, 10.0, 5.0)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(0.0)


def test_point_to_ellipse_distance_on_boundary():
    major, minor = 10.0, 5.0
    x, y = point_on_ellipse(math.pi / 4, major, minor)
    d = point_to_ellipse_distance(x, y, 0.0, 0.0, major, minor, 0.0)
    assert d == pytest.approx(0.0, abs=0.05)


def test_point_to_ellipse_distance_at_center_equals_minor_axis():
    major, minor = 10.0, 5.0
    d = point_to_ellipse_distance(0.0, 0.0, 0.0, 0.0, major, minor, 0.0)
    assert d == pytest.approx(minor, abs=0.1)


def test_point_to_ellipse_distance_rotated():
    major, minor = 12.0, 6.0
    phi = math.pi / 6
    x, y = point_on_ellipse(0.0, major, minor)
    xr = x * math.cos(phi) - y * math.sin(phi)
    yr = x * math.sin(phi) + y * math.cos(phi)
    d = point_to_ellipse_distance(xr, yr, 0.0, 0.0, major, minor, phi)
    assert d == pytest.approx(0.0, abs=0.1)


def test_fit_ellipse_circle_points():
  # points on a circle
    angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    xs = 8.0 * np.cos(angles)
    ys = 8.0 * np.sin(angles)
    ell = fit_ellipse(xs, ys, min_points=5)
    assert ell is not None
    assert ell["major_ax"] == pytest.approx(8.0, rel=0.05)
    assert ell["minor_ax"] == pytest.approx(8.0, rel=0.05)
    assert ell["center_x"] == pytest.approx(0.0, abs=0.5)
    assert ell["center_y"] == pytest.approx(0.0, abs=0.5)


def test_fit_ellipse_insufficient_points():
    assert fit_ellipse(np.array([1.0]), np.array([2.0]), min_points=5) is None


def test_residual_stats():
    stats = residual_stats(np.array([1.0, 2.0, 3.0]))
    assert stats["residual_mae"] == pytest.approx(2.0)
    assert stats["residual_rmse"] == pytest.approx(math.sqrt((1 + 4 + 9) / 3))
    assert stats["residual_max"] == pytest.approx(3.0)
