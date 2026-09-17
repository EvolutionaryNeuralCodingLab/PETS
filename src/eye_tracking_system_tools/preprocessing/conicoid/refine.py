"""Swirski image-based per-frame pupil refine (frozen sphere).

After :func:`fit_sphere_from_ellipses`, each frame's 3D pupil
``(θ, ψ, r)`` is optimized with the eyeball frozen. Two residuals:

* **Region contrast** (Swirski 2013): thin bands around the reprojected
  ellipse; maximize ``mean(outside) - mean(inside)`` for a dark pupil.
* **Edge distance** (optional): squared approximate distance from DLC
  ``Pupil_*`` inliers to the reprojected ellipse.

Joint ``3 + 3N`` over a whole recording is too large for GUI/batch; this
module refines frames independently (optionally a subsample to update the
sphere once).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .camera import CameraIntrinsics
from .eye_model import (
    EyeModelFit,
    PupilObservation,
    Sphere,
    circle_from_params,
    dierkes_eyeball_centre,
    fit_sphere_from_ellipses,
    params_from_circle,
    snap_pupils_to_sphere,
)
from .geometry import Circle3D, Ellipse2D, ellipse_from_pets
from .refraction import RefractionMaps, apply_refraction_to_fit
from .unproject import project_circle, unproject

REFINED_COLS = (
    "center_x_refined",
    "center_y_refined",
    "width_refined",
    "height_refined",
    "phi_refined",
)


@dataclass
class RefineSettings:
    """User-tunable Swirski refine knobs (GUI slider defaults in parentheses)."""

    metric: str = "contrast"  # contrast | edge | both
    band_width: float = 5.0
    epsilon: float = 0.5
    downsample: float = 1.0
    dlc_likelihood: float = 0.95
    sphere_method: str = "dierkes_3d"
    lock_sphere: bool = True
    eye_z: float = 13.0
    use_ransac: bool = True
    pupil_radius: float = 1.0
    apply_refraction: bool = False
    refraction_maps: RefractionMaps | None = None
    edge_weight: float = 1.0
    sphere_update_stride: int = 10


@dataclass(frozen=True)
class PixelEllipse:
    cx: float
    cy: float
    major_radius: float
    minor_radius: float
    angle: float


def smootherstep_heaviside(t: np.ndarray | float, eps: float) -> np.ndarray:
    """C¹ smooth step ``H_ε``: 0 for ``t ≤ -ε``, 1 for ``t ≥ ε``."""
    arr = np.asarray(t, dtype=float)
    e = max(float(eps), 1e-12)
    x = np.clip((arr + e) / (2.0 * e), 0.0, 1.0)
    return (6.0 * x ** 5 - 15.0 * x ** 4 + 10.0 * x ** 3).astype(float)


def signed_ellipse_distance(
    xs: np.ndarray,
    ys: np.ndarray,
    ellipse: PixelEllipse,
) -> np.ndarray:
    """Approximate signed distance in pixels; **positive inside**.

    Map to the unit circle in the ellipse's principal frame, then scale by
    the major radius (Swirski's cheap distance).
    """
    dx = np.asarray(xs, dtype=float) - float(ellipse.cx)
    dy = np.asarray(ys, dtype=float) - float(ellipse.cy)
    c = float(np.cos(ellipse.angle))
    s = float(np.sin(ellipse.angle))
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    a = max(float(ellipse.major_radius), 1e-9)
    b = max(float(ellipse.minor_radius), 1e-9)
    rho = np.hypot(xr / a, yr / b)
    return (1.0 - rho) * a


def _as_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] == 3:
        r = arr[..., 0].astype(np.float64)
        g = arr[..., 1].astype(np.float64)
        b = arr[..., 2].astype(np.float64)
        return (0.299 * r + 0.587 * g + 0.114 * b)
    if arr.shape[-1] == 4:
        return _as_gray(arr[..., :3])
    raise ValueError("Expected gray or RGB(A) image")


def region_contrast_cost(
    image: np.ndarray,
    ellipse: PixelEllipse,
    *,
    band_width: float = 5.0,
    epsilon: float = 0.5,
) -> float:
    """``-(mean(R+) - mean(R-))`` so a dark pupil is a minimization problem."""
    gray = _as_gray(image)
    w = float(band_width)
    pad = float(ellipse.major_radius) + w + abs(float(epsilon)) + 2.0
    h, ww = gray.shape[:2]
    y0 = max(0, int(np.floor(ellipse.cy - pad)))
    y1 = min(h, int(np.ceil(ellipse.cy + pad)) + 1)
    x0 = max(0, int(np.floor(ellipse.cx - pad)))
    x1 = min(ww, int(np.ceil(ellipse.cx + pad)) + 1)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    crop = gray[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = signed_ellipse_distance(xx, yy, ellipse)
    w_out = smootherstep_heaviside(-d, epsilon) * smootherstep_heaviside(d + w, epsilon)
    w_in = smootherstep_heaviside(d, epsilon) * smootherstep_heaviside(w - d, epsilon)
    s_out = float(np.sum(w_out))
    s_in = float(np.sum(w_in))
    if s_out < 1e-9 or s_in < 1e-9:
        return 0.0
    mean_out = float(np.sum(crop * w_out) / s_out)
    mean_in = float(np.sum(crop * w_in) / s_in)
    return -(mean_out - mean_in)


def edge_distance_cost(points: np.ndarray, ellipse: PixelEllipse) -> float:
    """Mean squared approximate distance of inlier points to the ellipse contour."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    d = signed_ellipse_distance(pts[:, 0], pts[:, 1], ellipse)
    return float(np.mean(d * d))


def pixel_ellipse_from_circle(circle: Circle3D, camera: CameraIntrinsics) -> PixelEllipse:
    el = project_circle(circle, camera.focal_px)
    return PixelEllipse(
        cx=float(el.cx) + float(camera.cx),
        cy=float(el.cy) + float(camera.cy),
        major_radius=float(el.major_radius),
        minor_radius=float(el.minor_radius),
        angle=float(el.angle),
    )


def refined_row_from_circle(circle: Circle3D, camera: CameraIntrinsics) -> dict[str, float]:
    pel = pixel_ellipse_from_circle(circle, camera)
    return {
        "center_x_refined": pel.cx,
        "center_y_refined": pel.cy,
        "width_refined": pel.major_radius,
        "height_refined": pel.minor_radius,
        "phi_refined": pel.angle,
    }


def ellipse_from_eye_row(row, camera: CameraIntrinsics) -> Ellipse2D | None:
    cx = row["center_x"] if "center_x" in row.index else np.nan
    cy = row["center_y"] if "center_y" in row.index else np.nan
    if not np.isfinite(cx) or not np.isfinite(cy):
        return None
    major = row["major_ax"] if "major_ax" in row.index else np.nan
    minor = row["minor_ax"] if "minor_ax" in row.index else np.nan
    width = float(row["width"]) if "width" in row.index else None
    height = float(row["height"]) if "height" in row.index else None
    phi = row["phi"] if "phi" in row.index else np.nan
    if not np.isfinite(phi):
        return None
    if (not np.isfinite(major) or not np.isfinite(minor)) and (
        width is None or height is None
    ):
        return None
    if width is not None and (not np.isfinite(width) or width <= 0):
        width = None
    if height is not None and (not np.isfinite(height) or height <= 0):
        height = None
    return ellipse_from_pets(
        float(cx),
        float(cy),
        float(major) if np.isfinite(major) else float(width or 1.0),
        float(minor) if np.isfinite(minor) else float(height or 1.0),
        float(phi),
        cx=camera.cx,
        cy=camera.cy,
        width=width,
        height=height,
    )


def extract_pupil_inliers(
    dlc_csv: Path | str,
    *,
    likelihood: float = 0.95,
) -> dict[int, np.ndarray]:
    """``Pupil_*`` (x, y) with likelihood strictly above ``threshold``, keyed by row.

    Loading matches ``eye_tracking_analysis``: ``header=1`` then ``iloc[1:]``.
    The remaining row index is treated as the video / ``eye_frame`` index.
    """
    data = pd.read_csv(dlc_csv, header=1, low_memory=False)
    data = data.iloc[1:].apply(pd.to_numeric, errors="coerce")
    pupil_cols = [c for c in data.columns if "Pupil" in str(c)]
    out: dict[int, np.ndarray] = {}
    if len(pupil_cols) < 3:
        return out
    n_parts = len(pupil_cols) // 3
    values = data[pupil_cols].to_numpy(dtype=float)
    thresh = float(likelihood)
    for i in range(values.shape[0]):
        pts = []
        for k in range(n_parts):
            x, y, lik = values[i, 3 * k : 3 * k + 3]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(lik) and lik > thresh:
                pts.append((float(x), float(y)))
        if pts:
            out[i] = np.asarray(pts, dtype=float)
    return out


def _downsample(
    image: np.ndarray, ellipse: PixelEllipse, scale: float
) -> tuple[np.ndarray, PixelEllipse]:
    s = float(scale)
    if s >= 0.999:
        return image, ellipse
    s = max(s, 0.05)
    try:
        import cv2
    except ImportError:
        return image, ellipse
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled = PixelEllipse(
        cx=ellipse.cx * s,
        cy=ellipse.cy * s,
        major_radius=ellipse.major_radius * s,
        minor_radius=ellipse.minor_radius * s,
        angle=ellipse.angle,
    )
    return resized, scaled


def circle_image_cost(
    image: np.ndarray,
    circle: Circle3D,
    camera: CameraIntrinsics,
    *,
    settings: RefineSettings | None = None,
    inliers: np.ndarray | None = None,
) -> float:
    """2D residual of a 3D circle reprojected onto ``image``."""
    settings = settings or RefineSettings()
    metric = str(settings.metric).strip().lower()
    pel = pixel_ellipse_from_circle(circle, camera)
    img, pel_s = _downsample(image, pel, settings.downsample)
    total = 0.0
    if metric in ("contrast", "both"):
        total += region_contrast_cost(
            img,
            pel_s,
            band_width=settings.band_width * float(settings.downsample),
            epsilon=settings.epsilon * float(settings.downsample),
        )
    if metric in ("edge", "both") and inliers is not None and len(inliers):
        pts = np.asarray(inliers, dtype=float)
        if settings.downsample < 0.999:
            pts = pts * float(settings.downsample)
        total += float(settings.edge_weight) * edge_distance_cost(pts, pel_s)
    return float(total)


def _minimize(fun, x0: np.ndarray, bounds) -> tuple[np.ndarray, float]:
    x0 = np.asarray(x0, dtype=float)
    try:
        from scipy.optimize import minimize

        res = minimize(fun, x0, method="L-BFGS-B", bounds=bounds)
        return np.asarray(res.x, dtype=float), float(res.fun)
    except Exception:
        return _coordinate_descent(fun, x0, bounds)


def _coordinate_descent(fun, x0: np.ndarray, bounds) -> tuple[np.ndarray, float]:
    x = np.asarray(x0, dtype=float).copy()
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=float)
    hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=float)
    best = float(fun(x))
    steps = np.array([0.05, 0.02, 0.005, 0.002])
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(x.size):
                for delta in (-step, step):
                    trial = x.copy()
                    trial[i] = float(np.clip(trial[i] + delta, lo[i], hi[i]))
                    val = float(fun(trial))
                    if val < best - 1e-12:
                        x, best = trial, val
                        improved = True
    return x, best


def _initial_circle_on_sphere(
    ellipse: Ellipse2D,
    sphere: Sphere,
    camera: CameraIntrinsics,
    pupil_radius: float,
) -> Circle3D:
    pair = unproject(ellipse, float(pupil_radius), camera.focal_px)
    scored = []
    for circ in pair:
        dist = abs(float(np.linalg.norm(circ.centre - sphere.centre)) - float(sphere.radius))
        scored.append((dist, circ))
    scored.sort(key=lambda t: t[0])
    chosen = scored[0][1]
    dummy = EyeModelFit(
        sphere=sphere,
        pupils=[PupilObservation(ellipse=ellipse, pair=pair, circle=chosen, inlier=True)],
    )
    dummy = snap_pupils_to_sphere(dummy)
    if dummy.pupils[0].circle is None:
        return chosen
    return dummy.pupils[0].circle


def refine_circle(
    image: np.ndarray,
    circle: Circle3D,
    sphere: Sphere,
    camera: CameraIntrinsics,
    *,
    settings: RefineSettings | None = None,
    inliers: np.ndarray | None = None,
) -> Circle3D:
    """L-BFGS-B on ``(θ, ψ, r)`` with the sphere frozen."""
    settings = settings or RefineSettings()
    theta0, psi0, r0 = params_from_circle(sphere, circle)
    r_max = max(0.95 * float(sphere.radius), r0 * 2.0, 0.2)

    def cost(params: np.ndarray) -> float:
        circ = circle_from_params(
            sphere, float(params[0]), float(params[1]), float(params[2])
        )
        return circle_image_cost(
            image, circ, camera, settings=settings, inliers=inliers
        )

    bounds = [
        (1e-3, np.pi - 1e-3),
        (None, None),
        (1e-3, r_max),
    ]
    x_hat, _ = _minimize(cost, np.array([theta0, psi0, r0]), bounds)
    return circle_from_params(sphere, float(x_hat[0]), float(x_hat[1]), float(x_hat[2]))


def _ensure_axes(df: pd.DataFrame) -> pd.DataFrame:
    work = df
    if "major_ax" not in work.columns or "minor_ax" not in work.columns:
        if {"width", "height"}.issubset(work.columns):
            work = work.copy()
            work["major_ax"] = np.nanmax(work[["width", "height"]].to_numpy(), axis=1)
            work["minor_ax"] = np.nanmin(work[["width", "height"]].to_numpy(), axis=1)
    return work


def _ellipses_from_table(
    work: pd.DataFrame, camera: CameraIntrinsics
) -> tuple[list[Ellipse2D], np.ndarray]:
    valid = (
        np.isfinite(work["center_x"].to_numpy(dtype=float))
        & np.isfinite(work["center_y"].to_numpy(dtype=float))
        & np.isfinite(work["phi"].to_numpy(dtype=float))
    )
    if "major_ax" in work.columns:
        valid = (
            valid
            & np.isfinite(work["major_ax"].to_numpy(dtype=float))
            & (work["major_ax"].to_numpy(dtype=float) > 0)
        )
    idx = np.flatnonzero(valid)
    ellipses: list[Ellipse2D] = []
    kept: list[int] = []
    for i in idx:
        el = ellipse_from_eye_row(work.iloc[int(i)], camera)
        if el is None:
            continue
        ellipses.append(el)
        kept.append(int(i))
    return ellipses, np.asarray(kept, dtype=int)


def _fit_sphere(
    ellipses: list[Ellipse2D],
    camera: CameraIntrinsics,
    settings: RefineSettings,
) -> EyeModelFit:
    fit = fit_sphere_from_ellipses(
        ellipses,
        focal_length=camera.focal_px,
        pupil_radius=settings.pupil_radius,
        eye_z=settings.eye_z,
        use_ransac=settings.use_ransac,
        sphere_method=settings.sphere_method,
    )
    if settings.apply_refraction and settings.refraction_maps is not None:
        fit = apply_refraction_to_fit(fit, settings.refraction_maps)
    return fit


def _inliers_for_row(
    row,
    i: int,
    inliers_by_frame: Mapping[int, np.ndarray] | None,
) -> np.ndarray | None:
    if not inliers_by_frame:
        return None
    if "eye_frame" in row.index and np.isfinite(row["eye_frame"]):
        key = int(row["eye_frame"])
        if key in inliers_by_frame:
            return inliers_by_frame[key]
    if i in inliers_by_frame:
        return inliers_by_frame[i]
    return None


def refine_eye_table(
    ellipse_df: pd.DataFrame,
    camera: CameraIntrinsics,
    *,
    frame_image: Callable[[int], np.ndarray | None] | None = None,
    inliers_by_frame: Mapping[int, np.ndarray] | None = None,
    settings: RefineSettings | None = None,
    frame_indices: list[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, EyeModelFit]:
    """Fit a frozen sphere, then refine each requested row.

    ``frame_image(eye_frame)`` returns a raw OpenCV/RGB frame (or ``None``).
    Contrast metric is skipped when the image is missing; edge metric still
    runs if inliers are present.
    """
    settings = settings or RefineSettings()
    if ellipse_df is None or ellipse_df.empty:
        raise ValueError("ellipse_df is empty")
    work = _ensure_axes(ellipse_df)
    ellipses, valid_idx = _ellipses_from_table(work, camera)
    if len(ellipses) < 2:
        raise ValueError("Need at least two valid ellipses for a sphere fit")
    fit = _fit_sphere(ellipses, camera, settings)

    n = len(work)
    out_cols = {name: np.full(n, np.nan) for name in REFINED_COLS}
    local_of = {int(gi): local for local, gi in enumerate(valid_idx)}

    targets = frame_indices if frame_indices is not None else list(range(n))
    # Optional one-shot sphere update from a subsample of refined circles.
    if not settings.lock_sphere:
        stride = max(int(settings.sphere_update_stride), 1)
        subsample = [i for i in targets if i in local_of][::stride]
        circles: list[Circle3D] = []
        for count, i in enumerate(subsample):
            circ = _refine_row(
                work, i, local_of, fit, camera, settings, frame_image, inliers_by_frame
            )
            if circ is not None:
                circles.append(circ)
            if progress is not None:
                progress(count + 1, len(subsample) + len(targets))
        if len(circles) >= 2:
            centre = dierkes_eyeball_centre(circles, fit.sphere.radius)
            fit.sphere = Sphere(centre=centre, radius=fit.sphere.radius)
            fit = snap_pupils_to_sphere(fit)

    for count, i in enumerate(targets):
        circ = _refine_row(
            work, i, local_of, fit, camera, settings, frame_image, inliers_by_frame
        )
        if circ is not None:
            row = refined_row_from_circle(circ, camera)
            for name, value in row.items():
                out_cols[name][i] = value
        if progress is not None:
            extra = 0 if settings.lock_sphere else len(targets)
            progress(extra + count + 1, extra + len(targets))

    result = work.copy()
    for name, values in out_cols.items():
        result[name] = values
    return result, fit


def _refine_row(
    work: pd.DataFrame,
    i: int,
    local_of: dict[int, int],
    fit: EyeModelFit,
    camera: CameraIntrinsics,
    settings: RefineSettings,
    frame_image: Callable[[int], np.ndarray | None] | None,
    inliers_by_frame: Mapping[int, np.ndarray] | None,
) -> Circle3D | None:
    if i not in local_of:
        return None
    local = local_of[i]
    obs = fit.pupils[local]
    row = work.iloc[i]
    image = None
    if frame_image is not None:
        frame_key = int(row["eye_frame"]) if "eye_frame" in row.index and np.isfinite(row["eye_frame"]) else i
        image = frame_image(frame_key)
    inliers = _inliers_for_row(row, i, inliers_by_frame)
    metric = str(settings.metric).strip().lower()
    start = obs.circle
    if start is None:
        el = ellipse_from_eye_row(row, camera)
        if el is None:
            return None
        start = _initial_circle_on_sphere(
            el, fit.sphere, camera, settings.pupil_radius
        )
    # No 2D metric → do not write a snap-to-sphere reproject as "refined".
    if image is None and (inliers is None or metric == "contrast"):
        return None
    img = image if image is not None else np.zeros((1, 1), dtype=np.uint8)
    use = RefineSettings(
        metric="edge" if image is None else settings.metric,
        band_width=settings.band_width,
        epsilon=settings.epsilon,
        downsample=settings.downsample,
        dlc_likelihood=settings.dlc_likelihood,
        sphere_method=settings.sphere_method,
        lock_sphere=settings.lock_sphere,
        eye_z=settings.eye_z,
        use_ransac=settings.use_ransac,
        pupil_radius=settings.pupil_radius,
        edge_weight=settings.edge_weight,
    )
    refined = refine_circle(
        img, start, fit.sphere, camera, settings=use, inliers=inliers
    )
    before = circle_image_cost(img, start, camera, settings=use, inliers=inliers)
    after = circle_image_cost(img, refined, camera, settings=use, inliers=inliers)
    if not np.isfinite(after) or not np.isfinite(before) or after >= before:
        return None
    return refined


def refine_single_frame(
    image: np.ndarray,
    row: pd.Series,
    sphere: Sphere,
    camera: CameraIntrinsics,
    *,
    settings: RefineSettings | None = None,
    inliers: np.ndarray | None = None,
) -> dict[str, float]:
    """Preview helper: refine one table row given an already-fitted sphere."""
    settings = settings or RefineSettings()
    el = ellipse_from_eye_row(row, camera)
    if el is None:
        raise ValueError("Row has no valid ellipse")
    start = _initial_circle_on_sphere(el, sphere, camera, settings.pupil_radius)
    circ = refine_circle(
        image, start, sphere, camera, settings=settings, inliers=inliers
    )
    return refined_row_from_circle(circ, camera)
