"""Ellipse geometry helpers including true point-to-ellipse distance."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from eye_tracking_system_tools.preprocessing.ellipse_fit import LsqEllipse


def representative_diameter(major_ax: float, minor_ax: float, method: str = "geometric_mean") -> float:
    """
    Compute a scalar pupil diameter from semi-axes.

    ``geometric_mean``: 2 * sqrt(major * minor)  (default, matches plan)
    ``major``: 2 * major
    ``mean``: major + minor
    """
    if not (np.isfinite(major_ax) and np.isfinite(minor_ax) and major_ax > 0 and minor_ax > 0):
        return float("nan")
    if method == "major":
        return 2.0 * major_ax
    if method == "mean":
        return major_ax + minor_ax
    return 2.0 * math.sqrt(major_ax * minor_ax)


def fit_ellipse(
    xs: np.ndarray,
    ys: np.ndarray,
    min_points: int = 6,
) -> dict[str, Any] | None:
    """
    Fit an ellipse to 2D points using :class:`LsqEllipse`.

    Returns dict with center, axes, phi, area, diameter or None if fit fails.
    """
    X = np.column_stack([xs, ys]).astype(float)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] < min_points:
        return None
    try:
        el = LsqEllipse().fit(X)
        center, width, height, phi = el.as_parameters()
        vals = [center[0], center[1], width, height, phi]
        if any(np.iscomplexobj(v) for v in vals):
            return None
        cx = float(np.real(center[0]))
        cy = float(np.real(center[1]))
        major = float(np.real(max(width, height)))
        minor = float(np.real(min(width, height)))
        phi_f = float(np.real(phi))
        if not all(np.isfinite(v) for v in (cx, cy, major, minor, phi_f)):
            return None
        if major <= 0 or minor <= 0:
            return None
        area = math.pi * major * minor
        diameter = representative_diameter(major, minor)
        return {
            "center_x": cx,
            "center_y": cy,
            "major_ax": major,
            "minor_ax": minor,
            "phi": phi_f,
            "area": area,
            "diameter": diameter,
            "width": float(np.real(width)),
            "height": float(np.real(height)),
        }
    except (ValueError, IndexError, np.linalg.LinAlgError):
        return None


def _rotate_to_ellipse_frame(
    px: float,
    py: float,
    cx: float,
    cy: float,
    phi: float,
) -> tuple[float, float]:
    """Translate to center and rotate by ``-phi`` so major axis aligns with x."""
    dx = px - cx
    dy = py - cy
    c = math.cos(-phi)
    s = math.sin(-phi)
    xr = c * dx - s * dy
    yr = s * dx + c * dy
    return xr, yr


def point_on_ellipse(
    theta: float,
    major: float,
    minor: float,
) -> tuple[float, float]:
    return major * math.cos(theta), minor * math.sin(theta)


def point_to_ellipse_distance(
    px: float,
    py: float,
    cx: float,
    cy: float,
    major: float,
    minor: float,
    phi: float = 0.0,
    *,
    n_angles: int = 180,
) -> float:
    """
    Shortest Euclidean distance from point ``(px, py)`` to the ellipse.

    Uses dense angular sampling in the ellipse-aligned frame (fast, stable for QC).
    """
    if not all(np.isfinite(v) for v in (px, py, cx, cy, major, minor, phi)):
        return float("nan")
    if major <= 0 or minor <= 0:
        return float("nan")

    xr, yr = _rotate_to_ellipse_frame(px, py, cx, cy, phi)
    thetas = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    ex = major * np.cos(thetas)
    ey = minor * np.sin(thetas)
    d2 = (xr - ex) ** 2 + (yr - ey) ** 2
    t = float(thetas[int(np.argmin(d2))])
    best_d2 = float(np.min(d2))
    # Local refinement around the grid minimum
    step = math.pi / n_angles
    for _ in range(10):
        improved = False
        for tc in (t - step, t + step):
            ex_p = major * math.cos(tc)
            ey_p = minor * math.sin(tc)
            dd = (xr - ex_p) ** 2 + (yr - ey_p) ** 2
            if dd < best_d2:
                best_d2 = dd
                t = tc
                improved = True
        if not improved:
            step *= 0.5
        if step < 1e-6:
            break
    return math.sqrt(max(0.0, best_d2))


def point_to_ellipse_distances(
    xs: np.ndarray,
    ys: np.ndarray,
    ellipse: dict[str, Any],
) -> np.ndarray:
    """Vectorized point-to-ellipse distances for aligned landmark arrays."""
    cx = ellipse["center_x"]
    cy = ellipse["center_y"]
    major = ellipse["major_ax"]
    minor = ellipse["minor_ax"]
    phi = ellipse["phi"]
    out = np.empty(len(xs), dtype=float)
    for i, (x, y) in enumerate(zip(xs, ys)):
        out[i] = point_to_ellipse_distance(x, y, cx, cy, major, minor, phi)
    return out


def residual_stats(distances: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, max from finite point-to-ellipse distances."""
    d = distances[np.isfinite(distances)]
    if d.size == 0:
        return {"residual_mae": np.nan, "residual_rmse": np.nan, "residual_max": np.nan}
    return {
        "residual_mae": float(np.mean(np.abs(d))),
        "residual_rmse": float(np.sqrt(np.mean(d ** 2))),
        "residual_max": float(np.max(d)),
    }
