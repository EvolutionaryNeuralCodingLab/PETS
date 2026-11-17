"""Tests for canonical ellipse geometry extraction."""

from __future__ import annotations

import numpy as np
import pytest

from eye_tracking_system_tools.preprocessing.ellipse_fit import (
    LsqEllipse,
    canonicalize_ellipse_axes_phi,
    conic_coefficients_to_parameters,
    orientation_is_reliable,
    unwrap_ellipse_phi,
)


def _points_on_ellipse(
    deg: float,
    *,
    a: float = 40.0,
    b: float = 20.0,
    cx: float = 100.0,
    cy: float = 80.0,
    n: int = 80,
):
    phi = np.deg2rad(deg)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    y = cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    return np.column_stack([x, y]), phi


def _angular_diff_mod_pi(a: float, b: float) -> float:
    """Smallest absolute difference of two π-periodic angles."""
    d = (a - b + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return abs(float(d))


@pytest.mark.parametrize("deg", [0.0, 30.0, 44.0, 45.0, 46.0, 60.0, 90.0])
def test_as_parameters_major_axis_contract(deg):
    X, true_phi = _points_on_ellipse(deg)
    el = LsqEllipse().fit(X)
    center, width, height, phi = el.as_parameters()
    assert float(width) == pytest.approx(40.0, rel=1e-3)
    assert float(height) == pytest.approx(20.0, rel=1e-3)
    assert float(width) >= float(height)
    assert np.isfinite(phi)
    assert 0.0 <= phi < np.pi
    assert _angular_diff_mod_pi(phi, true_phi % np.pi) < np.deg2rad(1.0)
    assert center[0] == pytest.approx(100.0, abs=1e-2)
    assert center[1] == pytest.approx(80.0, abs=1e-2)


def test_coefficient_sign_invariance():
    X, true_phi = _points_on_ellipse(35.0)
    el = LsqEllipse().fit(X)
    c1 = el.as_parameters()
    coef = el.coefficients.copy()
    # Flip all conic coefficients; geometry must be unchanged.
    a, b2, c, d2, f2, g = coef
    # coefficients are [a, 2b, c, 2d, 2f, g]
    b = b2 / 2.0
    d = d2 / 2.0
    f = f2 / 2.0
    c2 = conic_coefficients_to_parameters(-a, -b, -c, -d, -f, -g)
    assert c2[1] == pytest.approx(c1[1], rel=1e-6)
    assert c2[2] == pytest.approx(c1[2], rel=1e-6)
    assert _angular_diff_mod_pi(c2[3], c1[3]) < 1e-6


def test_exact_45_degree_boundary():
    X, true_phi = _points_on_ellipse(45.0)
    _c, width, height, phi = LsqEllipse().fit(X).as_parameters()
    assert width > height
    assert _angular_diff_mod_pi(phi, true_phi % np.pi) < np.deg2rad(1.0)


def test_near_circle_marks_orientation_unreliable():
    X, _ = _points_on_ellipse(30.0, a=40.0, b=39.5)
    _c, width, height, phi = LsqEllipse().fit(X).as_parameters()
    assert width >= height
    assert np.isnan(phi)
    assert not orientation_is_reliable(width, height)


def test_exact_circle_orientation_nan():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    X = np.column_stack([100 + 30 * np.cos(t), 80 + 30 * np.sin(t)])
    _c, width, height, phi = LsqEllipse().fit(X).as_parameters()
    assert width == pytest.approx(height, rel=1e-3)
    assert np.isnan(phi)


def test_canonicalize_swap_preserves_geometry():
    w, h, phi = canonicalize_ellipse_axes_phi(20.0, 40.0, 0.1)
    assert w == pytest.approx(40.0)
    assert h == pytest.approx(20.0)
    assert phi == pytest.approx((0.1 + 0.5 * np.pi) % np.pi)


def test_unwrap_does_not_bridge_gaps():
    phi = np.array([0.1, 0.2, np.nan, 0.3, 0.4])
    out = unwrap_ellipse_phi(phi)
    assert np.isnan(out[2])
    assert np.isfinite(out[0]) and np.isfinite(out[1])
    assert np.isfinite(out[3]) and np.isfinite(out[4])


def test_return_fit_reconstructs_points():
    X, _ = _points_on_ellipse(60.0)
    el = LsqEllipse().fit(X)
    recon = el.return_fit(n_points=200)
    d2 = ((X[:, None, :] - recon[None, :, :]) ** 2).sum(axis=2)
    err = float(np.mean(np.sqrt(d2.min(axis=1))))
    assert err < 0.5
