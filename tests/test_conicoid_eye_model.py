"""Swirski sphere initialisation from projected 3D pupils."""

from __future__ import annotations

import numpy as np
import pytest

from eye_tracking_system_tools.preprocessing.conicoid.angles import gaze_to_kerr_angles
from eye_tracking_system_tools.preprocessing.conicoid.eye_model import fit_sphere_from_ellipses
from eye_tracking_system_tools.preprocessing.conicoid.geometry import Circle3D
from eye_tracking_system_tools.preprocessing.conicoid.unproject import (
    angular_error_deg,
    project_circle,
)


def _grid_pupils(eye_c: np.ndarray, radius: float, focal: float):
    ellipses = []
    true_n = []
    for ph in np.linspace(-0.4, 0.4, 6):
        for th in np.linspace(-0.3, 0.3, 5):
            nx = np.sin(ph)
            ny = np.sin(th) * np.cos(ph)
            nz = -np.sqrt(max(1e-9, 1.0 - nx * nx - ny * ny))
            n = np.array([nx, ny, nz])
            n = n / np.linalg.norm(n)
            circ = Circle3D(centre=eye_c + radius * n, normal=n, radius=1.0)
            ellipses.append(project_circle(circ, focal))
            true_n.append(n)
    return ellipses, true_n


def test_sphere_from_known_gazes():
    f = 250.0
    eye_c = np.array([0.0, 1.0, 13.0])
    radius = 4.0
    ellipses, true_n = _grid_pupils(eye_c, radius, f)
    fit = fit_sphere_from_ellipses(
        ellipses, focal_length=f, pupil_radius=1.0, eye_z=13.0, use_ransac=False
    )
    assert np.linalg.norm(fit.sphere.centre - eye_c) == pytest.approx(0.0, abs=1e-8)
    assert fit.sphere.radius == pytest.approx(radius, rel=1e-8)
    errs = [
        angular_error_deg(obs.circle.normal, n)
        for obs, n in zip(fit.pupils, true_n)
        if obs.circle is not None
    ]
    assert errs
    assert max(errs) == pytest.approx(0.0, abs=1e-6)


def test_sphere_fit_with_ransac_on_clean_data():
    f = 250.0
    eye_c = np.array([0.0, 1.0, 13.0])
    ellipses, true_n = _grid_pupils(eye_c, 4.0, f)
    rng = np.random.default_rng(0)
    fit = fit_sphere_from_ellipses(
        ellipses, focal_length=f, eye_z=13.0, use_ransac=True, rng=rng
    )
    assert np.linalg.norm(fit.sphere.centre - eye_c) < 1e-6
    assert fit.n_inliers >= 2
    mid = len(true_n) // 2
    assert angular_error_deg(fit.pupils[mid].circle.normal, true_n[mid]) < 1e-5


def test_disambiguation_picks_matching_branch():
    f = 250.0
    eye_c = np.array([0.4, -0.3, 13.0])
    ellipses, true_n = _grid_pupils(eye_c, 4.0, f)
    fit = fit_sphere_from_ellipses(
        ellipses, focal_length=f, eye_z=13.0, use_ransac=False
    )
    for obs, n in zip(fit.pupils, true_n):
        assert obs.circle is not None
        assert angular_error_deg(obs.circle.normal, n) < 1e-6
        # The other unprojection branch is far from the true normal.
        other = obs.pair[0] if angular_error_deg(obs.pair[0].normal, n) > 1e-3 else obs.pair[1]
        assert angular_error_deg(other.normal, n) > 5.0


def test_kerr_angles_zero_when_looking_at_camera():
    phi, theta = gaze_to_kerr_angles(np.array([0.0, 0.0, -1.0]))
    assert phi == pytest.approx(0.0, abs=1e-12)
    assert theta == pytest.approx(0.0, abs=1e-12)


def test_kerr_angles_positive_x_gives_positive_phi():
    n = np.array([0.3, 0.0, -np.sqrt(1.0 - 0.3 ** 2)])
    phi, theta = gaze_to_kerr_angles(n)
    assert phi > 0
    assert theta == pytest.approx(0.0, abs=1e-8)
