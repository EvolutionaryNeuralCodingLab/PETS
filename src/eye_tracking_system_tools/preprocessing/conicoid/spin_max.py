"""1-D ellipse orientation search (spin maximizer).

Uses **raw** DLC ellipses (``le_df``/``re_df`` ``center_x``/``center_y``),
not jitter-corrected ``left/right_eye_data``. Center and axes stay frozen;
``phi`` is swept in ``[0, π)`` to maximize dark-inside minus dark-in-outer-band.
Round pupils are not skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import sys

from eye_tracking_system_tools.preprocessing.ellipse_fit import canonicalize_ellipse_phi

SPIN_GEOMETRY_COLS = (
    "center_x_spin",
    "center_y_spin",
    "width_spin",
    "height_spin",
    "phi_spin",
)


def canonicalize_ellipse_table(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with ``width >= height`` and ``phi`` wrapped to ``[0, π)``."""
    out = df.copy()
    if not {"width", "height", "phi"}.issubset(out.columns):
        return out
    w, h, ang = canonicalize_ellipse_phi(
        out["width"].to_numpy(dtype=float),
        out["height"].to_numpy(dtype=float),
        out["phi"].to_numpy(dtype=float),
    )
    out["width"] = w
    out["height"] = h
    out["phi"] = ang
    if "major_ax" in out.columns:
        out["major_ax"] = np.fmax(w, h)
    if "minor_ax" in out.columns:
        out["minor_ax"] = np.fmin(w, h)
    return out


def raw_ellipse_table_from_dlc_df(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Build a spin-max table from ``read_dlc_data`` ``le_df`` / ``re_df``.

    Uses raw ``center_x`` / ``center_y`` (on-pupil in the video), never
    ``center_*_corrected``. Frame index is ``L_eye_frame`` / ``R_eye_frame``.
    """
    if df is None or df.empty:
        raise ValueError(f"{side} DLC ellipse table is empty")
    side_lc = str(side).strip().lower()
    if side_lc in ("left", "l", "le"):
        frame_src = "L_eye_frame" if "L_eye_frame" in df.columns else "eye_frame"
    elif side_lc in ("right", "r", "re"):
        frame_src = "R_eye_frame" if "R_eye_frame" in df.columns else "eye_frame"
    else:
        raise ValueError(f"Unknown eye side {side!r}")
    needed = [frame_src, "center_x", "center_y", "width", "height", "phi"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"DLC {side} table missing {missing}; have {list(df.columns)}. "
            "Run Read DLC so analysis/le_df.csv and re_df.csv exist."
        )
    out = pd.DataFrame(
        {
            "eye_frame": pd.to_numeric(df[frame_src], errors="coerce"),
            "center_x": pd.to_numeric(df["center_x"], errors="coerce"),
            "center_y": pd.to_numeric(df["center_y"], errors="coerce"),
            "width": pd.to_numeric(df["width"], errors="coerce"),
            "height": pd.to_numeric(df["height"], errors="coerce"),
            "phi": pd.to_numeric(df["phi"], errors="coerce"),
        }
    )
    if "OE_timestamp" in df.columns:
        out["OE_timestamp"] = df["OE_timestamp"].to_numpy()
    elif "Arena_TTL" in df.columns:
        out["OE_timestamp"] = df["Arena_TTL"].to_numpy()
    return canonicalize_ellipse_table(out)


def x_flip_ellipse_table(df: pd.DataFrame, frame_width: float) -> pd.DataFrame:
    """Mirror ellipse xy/phi about the vertical axis (Verify tab X-flip).

    ``center_x → width - center_x``, ``phi → π - phi``. Axes stay the same.
    Operates in raw camera pixels; the video itself is not flipped.
    """
    if df is None or df.empty:
        raise ValueError("ellipse table is empty")
    if "center_x" not in df.columns:
        raise KeyError("ellipse table needs center_x for an X-flip")
    out = df.copy()
    width = float(frame_width)
    out["center_x"] = width - pd.to_numeric(out["center_x"], errors="coerce")
    if "phi" in out.columns:
        out["phi"] = np.pi - pd.to_numeric(out["phi"], errors="coerce")
    return canonicalize_ellipse_table(out)


def jitter_deltas_for_frames(
    jitter_eye: Mapping[str, Any] | None,
    frame_indices: np.ndarray,
    *,
    median_k: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-filtered ``(dx, dy)`` aligned to ``frame_indices``.

    Same sign as BlockSync ``correct_jitter``:
    ``center_*_corrected = original + medfilt(displacement, 13)``.
    Missing / out-of-range frames get 0.
    """
    frames = np.asarray(frame_indices, dtype=float)
    n = frames.size
    zeros = np.zeros(n, dtype=float)
    if not jitter_eye or "x_displacement" not in jitter_eye:
        return zeros, zeros.copy()
    x = np.asarray(jitter_eye["x_displacement"], dtype=float)
    y = np.asarray(jitter_eye.get("y_displacement", np.zeros_like(x)), dtype=float)
    xm = _medfilt_1d(x, median_k)
    ym = _medfilt_1d(y, median_k)
    dx = np.zeros(n, dtype=float)
    dy = np.zeros(n, dtype=float)
    valid = np.isfinite(frames)
    idx = np.zeros(n, dtype=int)
    idx[valid] = frames[valid].astype(int)
    in_range = valid & (idx >= 0) & (idx < xm.size)
    dx[in_range] = np.where(np.isfinite(xm[idx[in_range]]), xm[idx[in_range]], 0.0)
    in_y = valid & (idx >= 0) & (idx < ym.size)
    dy[in_y] = np.where(np.isfinite(ym[idx[in_y]]), ym[idx[in_y]], 0.0)
    return dx, dy


def apply_jitter_to_spin_table(
    df: pd.DataFrame,
    jitter_eye: Mapping[str, Any] | None,
    *,
    median_k: int = 13,
) -> pd.DataFrame:
    """Add jitter to ``center_*_spin`` so xy matches Verify ``left/right_eye_data``.

    Call this **after** spin-max. ``phi_spin`` is left unchanged.
    """
    if df is None or df.empty:
        raise ValueError("spin table is empty")
    if "center_x_spin" not in df.columns or "center_y_spin" not in df.columns:
        raise KeyError("spin table needs center_x_spin / center_y_spin")
    if not jitter_eye or "x_displacement" not in jitter_eye:
        raise ValueError(
            "No jitter report for this eye. Run jitter on the Sync tab first."
        )
    out = df.copy()
    if "center_x_spin_raw" not in out.columns:
        out["center_x_spin_raw"] = pd.to_numeric(out["center_x_spin"], errors="coerce")
    if "center_y_spin_raw" not in out.columns:
        out["center_y_spin_raw"] = pd.to_numeric(out["center_y_spin"], errors="coerce")
    frame_col = "eye_frame" if "eye_frame" in out.columns else "frame"
    if frame_col not in out.columns:
        raise KeyError("spin table needs eye_frame to apply jitter")
    frames = pd.to_numeric(out[frame_col], errors="coerce").to_numpy(dtype=float)
    dx, dy = jitter_deltas_for_frames(jitter_eye, frames, median_k=median_k)
    out["center_x_spin"] = pd.to_numeric(out["center_x_spin_raw"], errors="coerce") + dx
    out["center_y_spin"] = pd.to_numeric(out["center_y_spin_raw"], errors="coerce") + dy
    return out


@dataclass
class SpinMaxSettings:
    """User knobs from the Refine spin-maximizer dialog."""

    threshold: int | None = None  # None → Otsu
    roi_mult: float = 1.2
    band_width: float = 5.0
    median_k: int = 5
    x_flip: bool = False
    frame_width: float | None = None


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr
    elif arr.shape[-1] == 3:
        r = arr[..., 0].astype(np.float64)
        g = arr[..., 1].astype(np.float64)
        b = arr[..., 2].astype(np.float64)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    elif arr.shape[-1] == 4:
        return _as_gray_u8(arr[..., :3])
    else:
        raise ValueError("Expected gray or RGB(A) image")
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def binarize_gray(gray: np.ndarray, threshold: int | None) -> tuple[np.ndarray, int]:
    """Threshold so bright is 255, dark is 0. ``threshold is None`` uses Otsu."""
    import cv2

    src = _as_gray_u8(gray)
    if threshold is None:
        used, binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary, int(round(float(used)))
    used = int(np.clip(int(threshold), 0, 255))
    _, binary = cv2.threshold(src, used, 255, cv2.THRESH_BINARY)
    return binary, used


def roi_half_side(width: float, height: float, roi_mult: float) -> float:
    """Half the square ROI side: ``multiplier * max(semi-axes)``.

    Full side is ``2 * roi_mult * max(width, height)``, so ``1.0`` just
    contains the ellipse.
    """
    return max(float(roi_mult), 0.05) * max(float(width), float(height), 1.0)


def crop_roi(
    image: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
    roi_mult: float,
) -> tuple[np.ndarray, int, int]:
    """Return ``(crop, x0, y0)`` of the square ROI in image coordinates."""
    if not all(np.isfinite(float(v)) for v in (cx, cy, width, height, roi_mult)):
        return np.zeros((1, 1), dtype=image.dtype), 0, 0
    half = roi_half_side(width, height, roi_mult)
    h, w = image.shape[:2]
    x0 = int(np.floor(cx - half))
    y0 = int(np.floor(cy - half))
    x1 = int(np.ceil(cx + half))
    y1 = int(np.ceil(cy + half))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return np.zeros((1, 1), dtype=image.dtype), x0c, y0c
    return image[y0c:y1c, x0c:x1c].copy(), x0c, y0c


def _dark_mask(binary_roi: np.ndarray) -> np.ndarray:
    dark = np.asarray(binary_roi) < 128
    if dark.ndim > 2:
        dark = dark[..., 0]
    return dark.astype(np.float64, copy=False)


def _spin_scores(
    dark: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
    phis: np.ndarray,
    band_width: float,
) -> np.ndarray:
    """Vectorized dark-inside minus dark-outer-band over many ``phi`` values."""
    hh, ww = dark.shape[:2]
    n_phi = int(np.asarray(phis).size)
    if hh < 2 or ww < 2 or n_phi == 0:
        return np.full(max(n_phi, 1), -np.inf, dtype=float)
    a = max(float(width), 1e-6)
    b = max(float(height), 1e-6)
    band = max(float(band_width), 1.0)
    phis = np.asarray(phis, dtype=float).reshape(-1)
    dx = xx - float(cx)
    dy = yy - float(cy)
    c = np.cos(phis)[:, None, None]
    s = np.sin(phis)[:, None, None]
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    rho = np.hypot(xr / a, yr / b)
    dist = (1.0 - rho) * a
    inside = dist >= 0.0
    outer = (dist < 0.0) & (dist >= -band)
    n_in = np.count_nonzero(inside, axis=(1, 2)).astype(np.float64)
    n_out = np.count_nonzero(outer, axis=(1, 2)).astype(np.float64)
    dark3 = dark[None, ...]
    dark_in = np.sum(dark3 * inside, axis=(1, 2)) / np.maximum(n_in, 1.0)
    dark_out = np.sum(dark3 * outer, axis=(1, 2)) / np.maximum(n_out, 1.0)
    scores = dark_in - dark_out
    scores[(n_in < 4.0) | (n_out < 4.0)] = -np.inf
    return scores


def spin_score(
    binary_roi: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
    phi: float,
    *,
    band_width: float = 5.0,
) -> float:
    """Dark-inside minus dark-in-outer-band. Higher is better."""
    dark = _dark_mask(binary_roi)
    hh, ww = dark.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]
    return float(
        _spin_scores(
            dark, xx, yy, cx, cy, width, height, np.array([phi]), band_width
        )[0]
    )


def score_curve(
    binary_roi: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
    *,
    band_width: float = 5.0,
    step_deg: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``phi`` samples in ``[0, π)`` and the corresponding spin scores."""
    step = max(float(step_deg), 0.5) * np.pi / 180.0
    phis = np.arange(0.0, np.pi, step)
    dark = _dark_mask(binary_roi)
    hh, ww = dark.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]
    scores = _spin_scores(
        dark, xx, yy, cx, cy, width, height, phis, band_width
    )
    return phis, scores


def spin_maximize_frame(
    image: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
    phi: float,
    *,
    settings: SpinMaxSettings | None = None,
) -> tuple[float, float, float, float]:
    """Return ``(width, height, phi, score)``. Always returns a finite ``phi``."""
    settings = settings or SpinMaxSettings()
    w, h, ang0 = canonicalize_ellipse_phi(width, height, phi)
    fallback = float(ang0) if np.isfinite(ang0) else 0.0
    if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
        return w, h, fallback, float("nan")
    crop, x0, y0 = crop_roi(image, cx, cy, w, h, settings.roi_mult)
    binary, _used = binarize_gray(crop, settings.threshold)
    lx = float(cx) - float(x0)
    ly = float(cy) - float(y0)
    dark = _dark_mask(binary)
    hh, ww = dark.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww]
    band = settings.band_width
    phis = np.arange(0.0, np.pi, 5.0 * np.pi / 180.0)
    scores = _spin_scores(dark, xx, yy, lx, ly, w, h, phis, band)
    finite = np.isfinite(scores) & (scores > float("-inf"))
    if not np.any(finite):
        return w, h, fallback, float("nan")
    best = float(phis[int(np.nanargmax(scores))])
    best_s = float(np.nanmax(scores))
    for step_deg in (1.0, 0.25):
        candidates = np.mod(
            best
            + np.linspace(-10.0, 10.0, int(round(20.0 / step_deg)) + 1)
            * (np.pi / 180.0),
            np.pi,
        )
        cand_s = _spin_scores(dark, xx, yy, lx, ly, w, h, candidates, band)
        j = int(np.nanargmax(cand_s))
        if cand_s[j] > best_s:
            best_s = float(cand_s[j])
            best = float(candidates[j])
    best = float(np.mod(best, np.pi))
    best_s = float(
        _spin_scores(dark, xx, yy, lx, ly, w, h, np.array([best]), band)[0]
    )
    if not np.isfinite(best_s) or best_s == float("-inf"):
        return w, h, fallback, float("nan")
    return w, h, best, float(best_s)


def _medfilt_1d(values: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    k = max(int(k), 1)
    if k % 2 == 0:
        k += 1
    if k == 1 or x.size == 0:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        window = xp[i : i + k]
        finite = window[np.isfinite(window)]
        out[i] = float(np.median(finite)) if finite.size else np.nan
    return out


def _unwrap_ellipse_phi(phi: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(phi, dtype=float).shape, np.nan, dtype=float)
    finite = np.isfinite(phi)
    if not np.any(finite):
        return out
    idx = np.flatnonzero(finite)
    folded = np.mod(np.asarray(phi, dtype=float)[idx], np.pi)
    doubled = np.unwrap(2.0 * folded) / 2.0
    out[idx] = doubled
    return out


def spin_maximize_eye_table(
    df: pd.DataFrame,
    frame_image: Callable[[int], np.ndarray | None],
    *,
    settings: SpinMaxSettings | None = None,
    frame_indices: list[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
    progress_desc: str = "ellipse rotation",
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """Add ``*_spin`` columns. ``df`` must already be in raw DLC pixel space."""
    settings = settings or SpinMaxSettings()
    if df is None or df.empty:
        raise ValueError("ellipse_df is empty")
    work = df.copy()
    n = len(work)
    cols = {name: np.full(n, np.nan) for name in SPIN_GEOMETRY_COLS}
    frame_col = "eye_frame" if "eye_frame" in work.columns else "frame"
    targets = frame_indices if frame_indices is not None else list(range(n))
    raw_phi = np.full(n, np.nan)
    total = len(targets)
    iterator: Any = targets
    if show_tqdm:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                targets,
                desc=progress_desc,
                unit="frame",
                file=sys.stderr,
                mininterval=0.3,
                miniters=1,
                disable=False,
                dynamic_ncols=True,
                ascii=True,
            )
        except Exception:
            iterator = targets

    for count, i in enumerate(iterator):
        row = work.iloc[int(i)]
        cx = float(row["center_x"]) if "center_x" in row.index else np.nan
        cy = float(row["center_y"]) if "center_y" in row.index else np.nan
        width = float(row["width"]) if "width" in row.index else np.nan
        height = float(row["height"]) if "height" in row.index else np.nan
        phi = float(row["phi"]) if "phi" in row.index else np.nan
        w0, h0, ang0 = canonicalize_ellipse_phi(width, height, phi)
        if not np.isfinite(cx) or not np.isfinite(cy):
            if progress is not None:
                progress(count + 1, total)
            continue
        frame_key = (
            int(row[frame_col])
            if frame_col in row.index and np.isfinite(row[frame_col])
            else int(i)
        )
        image = frame_image(frame_key)
        frame_w = settings.frame_width
        if frame_w is None and image is not None:
            frame_w = float(image.shape[1])
        if settings.x_flip:
            if frame_w is None:
                if progress is not None:
                    progress(count + 1, total)
                continue
            cx = float(frame_w) - cx
            phi = np.pi - phi
            w0, h0, ang0 = canonicalize_ellipse_phi(width, height, phi)
        cols["center_x_spin"][i] = cx
        cols["center_y_spin"][i] = cy
        cols["width_spin"][i] = w0
        cols["height_spin"][i] = h0
        if image is None:
            raw_phi[i] = ang0 if np.isfinite(ang0) else np.nan
            if progress is not None:
                progress(count + 1, total)
            continue
        if not np.isfinite(w0) or not np.isfinite(h0):
            raw_phi[i] = ang0 if np.isfinite(ang0) else np.nan
            if progress is not None:
                progress(count + 1, total)
            continue
        _w, _h, ang, _score = spin_maximize_frame(
            image, cx, cy, w0, h0, ang0, settings=settings
        )
        raw_phi[i] = ang if np.isfinite(ang) else ang0
        if progress is not None:
            progress(count + 1, total)

    unwrapped = _unwrap_ellipse_phi(raw_phi)
    smoothed = _medfilt_1d(unwrapped, settings.median_k)
    cols["phi_spin"] = np.mod(smoothed, np.pi)
    missing = ~np.isfinite(raw_phi)
    cols["phi_spin"][missing] = np.nan
    for name, values in cols.items():
        work[name] = values
    return work
