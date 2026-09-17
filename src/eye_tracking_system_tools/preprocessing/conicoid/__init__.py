"""Conicoid (Safaee-Rad / Swirski / Dierkes) eye-vector estimation from 2D ellipses.

This package is a Python port of the **non-iterative initialisation** in
Świrski & Dodgson 2013 (C++: https://github.com/LeszekSwirski/singleeyefitter),
which itself implements Safaee-Rad et al. 1992 circular unprojection.

Pipeline seam: after ``BlockSync.read_dlc_data`` / ``create_eye_data`` have
produced fitted ellipses (``center_x``, ``center_y``, ``major_ax``,
``minor_ax``, ``phi``). Kerr remains the production angle method
(``k_phi`` / ``k_theta``); this package writes parallel ``c_phi`` / ``c_theta``.

Image refine lives in ``refine.py`` (sidecar ``*_refined_*.csv``). Sphere
centre can use Swirski 2D projected-line intersection or Dierkes 2019 3D
line intersection. Refraction maps are optional (``refraction.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .angles import gaze_to_kerr_angles, gazes_to_kerr_angles
from .camera import CameraIntrinsics, default_intrinsics, resolve_intrinsics
from .eye_model import EyeModelFit, fit_sphere_from_ellipses
from .geometry import Ellipse2D, ellipse_from_pets
from .unproject import angular_error_deg, project_circle, unproject

CONICOID_ANGLE_COLS = ("c_phi", "c_theta")
CONICOID_VECTOR_COLS = ("c_nx", "c_ny", "c_nz")
CONICOID_SPHERE_COLS = ("c_ex", "c_ey", "c_ez", "c_eradius")

_ELLIPSE_COLS = ("center_x", "center_y", "major_ax", "minor_ax", "phi")


def _ensure_axes(df: pd.DataFrame) -> pd.DataFrame:
    work = df
    if "major_ax" not in work.columns or "minor_ax" not in work.columns:
        if not {"width", "height"}.issubset(work.columns):
            raise ValueError(
                "ellipse table needs major_ax/minor_ax or width/height"
            )
        work = work.copy()
        work["major_ax"] = np.nanmax(work[["width", "height"]].to_numpy(), axis=1)
        work["minor_ax"] = np.nanmin(work[["width", "height"]].to_numpy(), axis=1)
    return work


def fit_eye_and_gaze(
    ellipse_df: pd.DataFrame,
    camera: CameraIntrinsics,
    *,
    pupil_radius: float = 1.0,
    eye_z: float = 13.0,
    use_ransac: bool = True,
    rng: np.random.Generator | None = None,
    sphere_method: str = "dierkes_3d",
    refraction_maps=None,
    n_ref: float | None = None,
) -> tuple[pd.DataFrame, EyeModelFit]:
    """Fit a Swirski/Dierkes sphere to a PETS ellipse table and emit gaze columns.

    Returns ``(out_df, fit)``. ``out_df`` is indexed like ``ellipse_df`` and
    contains ``c_phi``, ``c_theta``, ``c_nx/y/z``, and repeated sphere
    centre/radius. Invalid rows stay NaN. ``refraction_maps`` is a
    :class:`RefractionMaps` instance, an npz path, or ``None`` (off).
    """
    if ellipse_df is None or ellipse_df.empty:
        raise ValueError("ellipse_df is empty")

    work = _ensure_axes(ellipse_df)
    n = len(work)
    c_phi = np.full(n, np.nan)
    c_theta = np.full(n, np.nan)
    nx = np.full(n, np.nan)
    ny = np.full(n, np.nan)
    nz = np.full(n, np.nan)

    valid = (
        np.isfinite(work["center_x"].to_numpy(dtype=float))
        & np.isfinite(work["center_y"].to_numpy(dtype=float))
        & np.isfinite(work["major_ax"].to_numpy(dtype=float))
        & np.isfinite(work["minor_ax"].to_numpy(dtype=float))
        & np.isfinite(work["phi"].to_numpy(dtype=float))
        & (work["major_ax"].to_numpy(dtype=float) > 0)
        & (work["minor_ax"].to_numpy(dtype=float) > 0)
    )
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 2:
        raise ValueError("Need at least two valid ellipses for a sphere fit")

    ellipses: list[Ellipse2D] = []
    for i in valid_idx:
        row = work.iloc[int(i)]
        width = float(row["width"]) if "width" in work.columns else None
        height = float(row["height"]) if "height" in work.columns else None
        ellipses.append(
            ellipse_from_pets(
                float(row["center_x"]),
                float(row["center_y"]),
                float(row["major_ax"]),
                float(row["minor_ax"]),
                float(row["phi"]),
                cx=camera.cx,
                cy=camera.cy,
                width=width,
                height=height,
            )
        )

    fit = fit_sphere_from_ellipses(
        ellipses,
        focal_length=camera.focal_px,
        pupil_radius=pupil_radius,
        eye_z=eye_z,
        use_ransac=use_ransac,
        rng=rng,
        sphere_method=sphere_method,
    )
    if refraction_maps is not None:
        from .refraction import apply_refraction_to_fit, load_refraction_maps

        maps = (
            load_refraction_maps(refraction_maps)
            if isinstance(refraction_maps, (str, Path))
            else refraction_maps
        )
        fit = apply_refraction_to_fit(fit, maps, n_ref=n_ref)

    for local, obs in enumerate(fit.pupils):
        if obs.circle is None:
            continue
        gi = int(valid_idx[local])
        nvec = np.asarray(obs.circle.normal, dtype=float)
        nx[gi], ny[gi], nz[gi] = nvec
        c_phi[gi], c_theta[gi] = gaze_to_kerr_angles(nvec)

    out = _assemble_output(work, c_phi, c_theta, nx, ny, nz, sphere=fit.sphere)
    return out, fit


def _assemble_output(
    work: pd.DataFrame,
    c_phi: np.ndarray,
    c_theta: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    sphere,
) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {
        "c_phi": c_phi,
        "c_theta": c_theta,
        "c_nx": nx,
        "c_ny": ny,
        "c_nz": nz,
    }
    if sphere is None:
        for name in CONICOID_SPHERE_COLS:
            cols[name] = np.full(len(work), np.nan)
    else:
        c = np.asarray(sphere.centre, dtype=float)
        cols["c_ex"] = np.full(len(work), float(c[0]))
        cols["c_ey"] = np.full(len(work), float(c[1]))
        cols["c_ez"] = np.full(len(work), float(c[2]))
        cols["c_eradius"] = np.full(len(work), float(sphere.radius))

    out = pd.DataFrame(cols, index=work.index)
    keep = [c for c in ("OE_timestamp", "eye_frame", "ms_axis") if c in work.columns]
    if keep:
        out = pd.concat([work[keep], out], axis=1)
    return out


__all__ = [
    "CONICOID_ANGLE_COLS",
    "CameraIntrinsics",
    "EyeModelFit",
    "angular_error_deg",
    "default_intrinsics",
    "fit_eye_and_gaze",
    "gaze_to_kerr_angles",
    "gazes_to_kerr_angles",
    "project_circle",
    "resolve_intrinsics",
    "unproject",
]
