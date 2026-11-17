"""Map a 3D pupil/gaze normal to Kerr-style ``phi`` / ``theta`` (degrees).

Kerr (``BlockSync.kerr``) uses nested arcsin on image offsets from a
roundest-pupil reference:

    phi   = arcsin((x - aEC) / f_z)
    theta = arcsin((y - bEC) / (cos(phi) * f_z))

After Swirski initialisation the gaze is the unit radial from the sphere
centre through the pupil, flipped to point **toward the camera**. For a
pinhole camera at the origin looking along +z that vector is typically
``n_z < 0``. Identifying ``n_x`` with Kerr's ``(x-aEC)/f_z`` and ``n_y``
with ``(y-bEC)/f_z`` gives the same nested layout:

    phi   = arcsin(n_x)
    theta = arcsin(n_y / cos(phi))

Looking straight at the camera (``n = (0, 0, -1)``) yields ``phi = theta = 0``.
A pupil displaced to +x in the image yields ``n_x > 0`` and positive ``phi``.

That nested pair is a chart, not two independent rotations. Degrees of gaze
error are always the angle between two unit normals
(:func:`angle_between_normals_deg`); invert a Kerr pair with
:func:`kerr_angles_to_normals` first. Never subtract pairs from different
charts, and never report ``hypot(dphi, dtheta)`` as a gaze angle. For maps
and arrows use :func:`chart_yaw_pitch`, the inverse of a horizontal rotation
followed by a vertical one (:func:`yaw_pitch_to_normals`).
"""

from __future__ import annotations

import numpy as np


def gaze_to_kerr_angles(normal: np.ndarray) -> tuple[float, float]:
    """Return ``(phi_deg, theta_deg)`` for a unit gaze toward the camera."""
    n = np.asarray(normal, dtype=float).reshape(3)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm == 0.0:
        return float("nan"), float("nan")
    n = n / norm
    if n[2] > 0:
        n = -n
    phi = float(np.arcsin(np.clip(n[0], -1.0, 1.0)))
    cphi = float(np.cos(phi))
    if abs(cphi) < 1e-12:
        theta = float("nan")
    else:
        theta = float(np.arcsin(np.clip(n[1] / cphi, -1.0, 1.0)))
    return float(np.degrees(phi)), float(np.degrees(theta))


def gazes_to_kerr_angles(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised wrapper; ``normals`` is ``(N, 3)``."""
    arr = np.asarray(normals, dtype=float)
    if arr.ndim == 1:
        phi, theta = gaze_to_kerr_angles(arr)
        return np.array([phi]), np.array([theta])
    phi = np.full(arr.shape[0], np.nan)
    theta = np.full(arr.shape[0], np.nan)
    for i, row in enumerate(arr):
        phi[i], theta[i] = gaze_to_kerr_angles(row)
    return phi, theta


def _as_float_array(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))


def kerr_angles_to_normals(phi_deg, theta_deg) -> np.ndarray:
    """Invert Kerr's nested ``(phi, theta)`` to the toward-camera unit normal.

    ``phi = arcsin(n_x)`` and ``theta = arcsin(n_y / cos phi)`` come back as
    ``(sin phi, sin theta cos phi, -cos phi cos theta)``. The same inverse
    recovers a conicoid normal from its stored ``c_phi`` / ``c_theta``.
    Returns ``(N, 3)``; rows with a non-finite angle are NaN.
    """
    phi = np.deg2rad(_as_float_array(phi_deg))
    theta = np.deg2rad(_as_float_array(theta_deg))
    out = np.full((phi.shape[0], 3), np.nan)
    valid = np.isfinite(phi) & np.isfinite(theta)
    if np.any(valid):
        p = phi[valid]
        t = theta[valid]
        out[valid, 0] = np.sin(p)
        out[valid, 1] = np.sin(t) * np.cos(p)
        out[valid, 2] = -np.cos(p) * np.cos(t)
    return out


def yaw_pitch_to_normals(yaw_deg, pitch_deg) -> np.ndarray:
    """Toward-camera unit normal after a horizontal then a vertical rotation.

    ``(sin yaw cos pitch, sin pitch, -cos yaw cos pitch)``; inverse of
    :func:`chart_yaw_pitch`. Returns ``(N, 3)``.
    """
    x = np.deg2rad(_as_float_array(yaw_deg))
    y = np.deg2rad(_as_float_array(pitch_deg))
    return np.column_stack((np.sin(x) * np.cos(y), np.sin(y), -np.cos(x) * np.cos(y)))


def chart_yaw_pitch(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yaw and pitch (degrees) of toward-camera normals.

    ``yaw = atan2(n_x, -n_z)`` and ``pitch = arcsin(n_y)``. Use it to draw
    directions; it is not an error metric.
    """
    arr = np.asarray(normals, dtype=float).reshape(-1, 3)
    yaw = np.full(arr.shape[0], np.nan)
    pitch = np.full(arr.shape[0], np.nan)
    valid = np.isfinite(arr).all(axis=1)
    if np.any(valid):
        n = arr[valid] / np.linalg.norm(arr[valid], axis=1, keepdims=True)
        yaw[valid] = np.rad2deg(np.arctan2(n[:, 0], -n[:, 2]))
        pitch[valid] = np.rad2deg(np.arcsin(np.clip(n[:, 1], -1.0, 1.0)))
    return yaw, pitch


def angle_between_normals_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between paired gaze directions ``a`` and ``b`` (``(N, 3)``).

    Inputs are normalised per row. Rows where either side is non-finite or
    zero-length are NaN.
    """
    a = np.asarray(a, dtype=float).reshape(-1, 3)
    b = np.asarray(b, dtype=float).reshape(-1, 3)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    out = np.full(a.shape[0], np.nan)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (
        np.isfinite(a).all(axis=1)
        & np.isfinite(b).all(axis=1)
        & (na > 0)
        & (nb > 0)
    )
    if np.any(valid):
        dots = np.sum(a[valid] * b[valid], axis=1) / (na[valid] * nb[valid])
        out[valid] = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    return out
