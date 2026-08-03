"""Pupil physiological perimeter (rect/circle) — Verify draw + Sync DLC pre-fit mask."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PERIMETERS_FILENAME = "pupil_perimeters.yaml"

_GEOMETRY_COLS = (
    "center_x",
    "center_y",
    "width",
    "height",
    "phi",
    "major_ax",
    "minor_ax",
    "ratio",
    "ellipse_size",
)


def perimeters_path(block_path: Path | str) -> Path:
    return Path(block_path) / "analysis" / PERIMETERS_FILENAME


def normalize_perimeter(d: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate and normalize a perimeter dict; return None if empty/invalid."""
    if not d or not isinstance(d, dict):
        return None
    shape = str(d.get("shape", "")).lower().strip()
    if shape == "rect":
        try:
            x, y, w, h = (float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"]))
        except (KeyError, TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        return {"shape": "rect", "x": x, "y": y, "w": w, "h": h}
    if shape == "circle":
        try:
            cx, cy, r = (float(d["cx"]), float(d["cy"]), float(d["r"]))
        except (KeyError, TypeError, ValueError):
            return None
        if r <= 0:
            return None
        return {"shape": "circle", "cx": cx, "cy": cy, "r": r}
    return None


def perimeter_summary(d: dict[str, Any] | None) -> str:
    peri = normalize_perimeter(d)
    if peri is None:
        return "(none)"
    if peri["shape"] == "rect":
        return (
            f"rect x={peri['x']:.0f} y={peri['y']:.0f} "
            f"w={peri['w']:.0f} h={peri['h']:.0f}"
        )
    return f"circle cx={peri['cx']:.0f} cy={peri['cy']:.0f} r={peri['r']:.0f}"


def point_in_perimeter(
    x: float | np.ndarray,
    y: float | np.ndarray,
    perimeter: dict[str, Any] | None,
) -> bool | np.ndarray:
    """Return True where (x, y) lies inside the perimeter (raw video coords)."""
    peri = normalize_perimeter(perimeter)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if peri is None:
        # No perimeter → everything kept
        if x_arr.shape == ():
            return True
        return np.ones(x_arr.shape, dtype=bool)

    if peri["shape"] == "rect":
        x0, y0, w, h = peri["x"], peri["y"], peri["w"], peri["h"]
        inside = (x_arr >= x0) & (x_arr <= x0 + w) & (y_arr >= y0) & (y_arr <= y0 + h)
    else:
        cx, cy, r = peri["cx"], peri["cy"], peri["r"]
        inside = (x_arr - cx) ** 2 + (y_arr - cy) ** 2 <= r ** 2

    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    inside = inside & finite
    if x_arr.shape == ():
        return bool(inside)
    return inside


def apply_perimeter_to_eye_df(
    df: pd.DataFrame,
    perimeter: dict[str, Any] | None,
) -> tuple[pd.DataFrame, int]:
    """
    NaN ellipse geometry columns for rows whose center falls outside ``perimeter``.

    Prefer Verify → Commit bad datapoints (noise-epoch catalog) for the GUI
    workflow; this helper remains for notebooks / opt-in masking.

    Returns ``(filtered_df, n_rows_hit)``.
    """
    peri = normalize_perimeter(perimeter)
    if peri is None or df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame(), 0
    if "center_x" not in df.columns or "center_y" not in df.columns:
        return df.copy(), 0

    out = df.copy()
    cx = out["center_x"].to_numpy(dtype=float)
    cy = out["center_y"].to_numpy(dtype=float)
    # Only consider rows that currently have a finite center
    has_center = np.isfinite(cx) & np.isfinite(cy)
    inside = point_in_perimeter(cx, cy, peri)
    if isinstance(inside, bool):
        inside = np.array([inside] * len(out), dtype=bool)
    hit = has_center & ~inside
    n_hit = int(hit.sum())
    if n_hit == 0:
        return out, 0
    cols = [c for c in _GEOMETRY_COLS if c in out.columns]
    for col in cols:
        out.loc[hit, col] = np.nan
    return out, n_hit


def mask_pupil_keypoints_outside_perimeter(
    pupil_xs: pd.DataFrame,
    pupil_ys: pd.DataFrame,
    perimeter: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Set Pupil keypoint cells outside ``perimeter`` to NaN (pre-ellipse-fit).

    Returns ``(xs, ys, n_points_masked)``.
    """
    peri = normalize_perimeter(perimeter)
    if peri is None:
        return pupil_xs, pupil_ys, 0
    xs = pupil_xs.copy()
    ys = pupil_ys.copy()
    x_vals = xs.to_numpy(dtype=float)
    y_vals = ys.to_numpy(dtype=float)
    inside = point_in_perimeter(x_vals, y_vals, peri)
    # Finite points that are outside
    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    mask_out = finite & ~inside
    n_masked = int(mask_out.sum())
    if n_masked:
        x_vals = x_vals.copy()
        y_vals = y_vals.copy()
        x_vals[mask_out] = np.nan
        y_vals[mask_out] = np.nan
        xs = pd.DataFrame(x_vals, index=xs.index, columns=xs.columns)
        ys = pd.DataFrame(y_vals, index=ys.index, columns=ys.columns)
    return xs, ys, n_masked


def read_pupil_perimeters(block_path: Path | str) -> dict[str, dict[str, Any] | None]:
    """Load ``analysis/pupil_perimeters.yaml`` → ``{"left": …, "right": …}``."""
    path = perimeters_path(block_path)
    empty: dict[str, dict[str, Any] | None] = {"left": None, "right": None}
    if not path.is_file():
        return empty
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return empty
    return {
        "left": normalize_perimeter(data.get("left")),
        "right": normalize_perimeter(data.get("right")),
    }


def write_pupil_perimeters(
    block_path: Path | str,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> Path:
    """Write ``analysis/pupil_perimeters.yaml`` (omits missing eyes)."""
    block_path = Path(block_path)
    out_dir = block_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PERIMETERS_FILENAME
    payload: dict[str, Any] = {}
    left_n = normalize_perimeter(left)
    right_n = normalize_perimeter(right)
    if left_n is not None:
        payload["left"] = left_n
    if right_n is not None:
        payload["right"] = right_n
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def display_to_raw_xy(
    x_view: float,
    y_view: float,
    frame_h: int,
) -> tuple[int, int]:
    """Map vertically flipped display coords → raw video coords (Kerr convention)."""
    y_raw = int(frame_h) - 1 - int(round(y_view))
    x_raw = int(round(x_view))
    return x_raw, y_raw


def raw_rect_from_display_drag(
    x0_view: float,
    y0_view: float,
    x1_view: float,
    y1_view: float,
    frame_h: int,
) -> dict[str, Any]:
    """Build a raw-coord rect perimeter from two display corners."""
    xa, ya = display_to_raw_xy(x0_view, y0_view, frame_h)
    xb, yb = display_to_raw_xy(x1_view, y1_view, frame_h)
    x = float(min(xa, xb))
    y = float(min(ya, yb))
    w = float(abs(xb - xa))
    h = float(abs(yb - ya))
    return {"shape": "rect", "x": x, "y": y, "w": max(w, 1.0), "h": max(h, 1.0)}


def raw_circle_from_display_drag(
    cx_view: float,
    cy_view: float,
    rim_x_view: float,
    rim_y_view: float,
    frame_h: int,
) -> dict[str, Any]:
    """Build a raw-coord circle from display center + rim point."""
    cx, cy = display_to_raw_xy(cx_view, cy_view, frame_h)
    rx, ry = display_to_raw_xy(rim_x_view, rim_y_view, frame_h)
    r = float(np.hypot(rx - cx, ry - cy))
    return {"shape": "circle", "cx": float(cx), "cy": float(cy), "r": max(r, 1.0)}
