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
