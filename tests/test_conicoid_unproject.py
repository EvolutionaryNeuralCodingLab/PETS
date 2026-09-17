"""Round-trip tests for Safaee-Rad / Swirski circular unprojection."""

from __future__ import annotations

import numpy as np
import pytest

from eye_tracking_system_tools.preprocessing.conicoid.geometry import Circle3D, conic_from_ellipse
from eye_tracking_system_tools.preprocessing.conicoid.unproject import (
    angular_error_deg,
    project_circle,
    unproject,
)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_round_trip_circle_facing_camera():
    f, r = 250.0, 1.0
    true = Circle3D(centre=np.array([0.0, 0.0, 20.0]), normal=np.array([0.0, 0.0, -1.0]), radius=r)
    ellipse = project_circle(true, f)
    assert ellipse.major_radius == pytest.approx(ellipse.minor_radius, rel=1e-9)
    a, b = unproject(ellipse, r, f)
    err = min(angular_error_deg(a.normal, true.normal), angular_error_deg(b.normal, true.normal))
    assert err == pytest.approx(0.0, abs=1e-8)
    cen = min(np.linalg.norm(a.centre - true.centre), np.linalg.norm(b.centre - true.centre))
    assert cen == pytest.approx(0.0, abs=1e-8)


def test_round_trip_tilted_circles():
    """Plausible pupil tilts (within ~50° of the camera), not grazing views."""
    f, r = 250.0, 1.0
    rng = np.random.default_rng(1)
    n_ok = 0
    while n_ok < 12:
        n = _unit(rng.normal(size=3))
        if n[2] > 0:
            n = -n
        if n[2] > -0.6:
            continue
        centre = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(18, 28)])
        true = Circle3D(centre=centre, normal=n, radius=r)
        a, b = unproject(project_circle(true, f), r, f)
        nerr = min(angular_error_deg(a.normal, n), angular_error_deg(b.normal, n))
        cerr = min(np.linalg.norm(a.centre - centre), np.linalg.norm(b.centre - centre))
        assert nerr == pytest.approx(0.0, abs=1e-6)
        assert cerr == pytest.approx(0.0, abs=1e-6)
        n_ok += 1


def test_near_circular_ellipse_solutions_coincident():
    f, r = 300.0, 1.0
    true = Circle3D(
        centre=np.array([0.0, 0.0, 18.0]),
        normal=np.array([0.0, 0.0, -1.0]),
        radius=r,
    )
    ellipse = project_circle(true, f)
    assert ellipse.minor_radius / ellipse.major_radius > 0.999
    a, b = unproject(ellipse, r, f)
    assert angular_error_deg(a.normal, b.normal) == pytest.approx(0.0, abs=1e-4)
    assert angular_error_deg(a.normal, true.normal) == pytest.approx(0.0, abs=1e-4)


def test_projected_conic_vanishes_on_circle_samples():
    f, r = 250.0, 1.0
    n = _unit(np.array([0.3, -0.2, -1.0]))
    true = Circle3D(centre=np.array([1.5, 0.4, 21.0]), normal=n, radius=r)
    ellipse = project_circle(true, f)
    conic = conic_from_ellipse(ellipse)
    t = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    # orthonormal basis of the circle plane
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(n, tmp)
    x = x / np.linalg.norm(x)
    y = np.cross(n, x)
    vals = []
    for ti in t:
        p = true.centre + r * (np.cos(ti) * x + np.sin(ti) * y)
        u = f * p[0] / p[2]
        v = f * p[1] / p[2]
        vals.append(conic.evaluate(u, v))
    assert np.max(np.abs(vals)) == pytest.approx(0.0, abs=1e-9)
