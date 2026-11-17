"""Draw ellipse overlays without GUI toolkit imports."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def phi_to_opencv_degrees(phi: float, *, ellipse_geometry_version: int = 2) -> float:
    """Convert stored ellipse ``phi`` to OpenCV ``cv2.ellipse`` degrees."""
    if phi is None or (isinstance(phi, float) and not np.isfinite(phi)):
        return 0.0
    phi_val = float(phi)
    if int(ellipse_geometry_version) >= 2:
        return float(np.degrees(phi_val))
    return phi_val


def draw_ellipse_overlay(
    frame: np.ndarray,
    df: pd.DataFrame | None,
    frame_col: str,
    frame_idx: int,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    ellipse_geometry_version: int = 2,
) -> np.ndarray:
    """Draw one ellipse from an eye table onto ``frame``.

    Version 2 tables store ``phi`` in radians; OpenCV wants degrees.
    Version 1 (legacy) passes the stored value through unchanged.
    """
    if df is None or frame_col not in df.columns:
        return frame
    mask = df[frame_col] == frame_idx
    if not mask.any():
        return frame
    row = df.loc[mask].iloc[0]
    cx, cy = row.get("center_x"), row.get("center_y")
    if pd.isna(cx) or pd.isna(cy):
        return frame
    out = frame.copy()
    x = int(round(float(cx)))
    y = int(round(float(cy)))
    w = max(int(row.get("width", 1) or 1), 1)
    h = max(int(row.get("height", 1) or 1), 1)
    phi_raw = row.get("phi", 0.0)
    if pd.isna(phi_raw):
        phi_deg = 0.0
    else:
        phi_deg = phi_to_opencv_degrees(
            float(phi_raw), ellipse_geometry_version=ellipse_geometry_version
        )
    cv2.ellipse(out, (x, y), (w, h), phi_deg, 0, 360, color, thickness)
    return out
