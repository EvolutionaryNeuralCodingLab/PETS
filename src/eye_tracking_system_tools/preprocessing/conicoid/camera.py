"""Pinhole camera intrinsics for conicoid unprojection.

Image coordinates are OpenCV-style (origin at the top-left, x right, y down),
matching DLC / ``LsqEllipse`` ellipses. The principal point is subtracted
before Safaee-Rad unprojection so the cone vertex sits at ``(0, 0, -f)``.

Focal length in pixels is resolved, in order:

1. ``analysis/LR_focal_length.csv`` (``L_focal_length`` / ``R_focal_length``)
2. ``f_mm / pix_size_mm`` from ``analysis/LR_pix_size.csv``
3. A documented fallback from the QILENS 3.7 mm lens on an OV5647 sensor
   scaled to 640×480 (logged, never silent).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NOMINAL_WIDTH = 640
NOMINAL_HEIGHT = 480
LENS_FOCAL_MM = 3.7
# Omnivision OV5647: 1.4 µm pitch at native 2592×1944.
OV5647_PIXEL_PITCH_MM = 0.0014
OV5647_NATIVE_WIDTH = 2592
DEFAULT_FOCAL_PX = LENS_FOCAL_MM / OV5647_PIXEL_PITCH_MM * (
    NOMINAL_WIDTH / OV5647_NATIVE_WIDTH
)

FOCAL_LENGTH_CSV = "LR_focal_length.csv"
PIX_SIZE_CSV = "LR_pix_size.csv"

_SIDE_TO_FOCAL_COL = {"left": "L_focal_length", "right": "R_focal_length", "l": "L_focal_length", "r": "R_focal_length"}
_SIDE_TO_PIX_COL = {"left": "L_pix_size", "right": "R_pix_size", "l": "L_pix_size", "r": "R_pix_size"}


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera used by Safaee-Rad / Swirski unprojection."""

    width: int = NOMINAL_WIDTH
    height: int = NOMINAL_HEIGHT
    cx: float = NOMINAL_WIDTH / 2.0
    cy: float = NOMINAL_HEIGHT / 2.0
    focal_px: float = DEFAULT_FOCAL_PX
    source: str = "default_qilens_ov5647"
    side: str = ""
    lens_focal_mm: float = LENS_FOCAL_MM

    def to_camera_xy(self, x: float, y: float) -> tuple[float, float]:
        return float(x) - self.cx, float(y) - self.cy


def _normalize_side(side: str) -> str:
    key = str(side).strip().lower()
    if key in ("left", "l", "le", "left_eye"):
        return "left"
    if key in ("right", "r", "re", "right_eye"):
        return "right"
    raise ValueError(f"Unknown eye side {side!r}; expected left/right")


def _analysis_dir(block_path: Path | str) -> Path:
    path = Path(block_path)
    if path.name == "analysis":
        return path
    return path / "analysis"


def default_intrinsics(
    side: str = "",
    *,
    frame_width: int = NOMINAL_WIDTH,
    frame_height: int = NOMINAL_HEIGHT,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(frame_width),
        height=int(frame_height),
        cx=float(frame_width) / 2.0,
        cy=float(frame_height) / 2.0,
        focal_px=float(DEFAULT_FOCAL_PX),
        source=(
            "default_qilens_ov5647 "
            f"(f={LENS_FOCAL_MM} mm, pitch={OV5647_PIXEL_PITCH_MM} mm, "
            f"scaled {OV5647_NATIVE_WIDTH}→{frame_width})"
        ),
        side=side,
    )


def _read_csv_value(path: Path, column: str) -> float | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if column not in df.columns or df.empty:
        return None
    value = float(df.at[0, column])
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def resolve_intrinsics(
    block_path: Path | str | None,
    side: str,
    *,
    frame_width: int = NOMINAL_WIDTH,
    frame_height: int = NOMINAL_HEIGHT,
    lens_focal_mm: float = LENS_FOCAL_MM,
) -> CameraIntrinsics:
    """Resolve per-eye pinhole intrinsics from a block's analysis folder."""
    side_n = _normalize_side(side)
    cx = float(frame_width) / 2.0
    cy = float(frame_height) / 2.0
    fallback = default_intrinsics(side_n, frame_width=frame_width, frame_height=frame_height)

    if block_path is None:
        logger.warning(
            "No block path for %s eye; using fallback focal_px=%.3f (%s)",
            side_n,
            fallback.focal_px,
            fallback.source,
        )
        return fallback

    analysis = _analysis_dir(block_path)
    focal_col = _SIDE_TO_FOCAL_COL[side_n]
    pix_col = _SIDE_TO_PIX_COL[side_n]

    focal = _read_csv_value(analysis / FOCAL_LENGTH_CSV, focal_col)
    if focal is not None:
        source = str(analysis / FOCAL_LENGTH_CSV)
        logger.info("%s eye focal_px=%.3f from %s", side_n, focal, source)
        return CameraIntrinsics(
            width=int(frame_width),
            height=int(frame_height),
            cx=cx,
            cy=cy,
            focal_px=focal,
            source=source,
            side=side_n,
            lens_focal_mm=lens_focal_mm,
        )

    pix = _read_csv_value(analysis / PIX_SIZE_CSV, pix_col)
    if pix is not None:
        focal = float(lens_focal_mm) / pix
        source = f"{analysis / PIX_SIZE_CSV} ({lens_focal_mm} mm / {pix} mm/px)"
        logger.info("%s eye focal_px=%.3f from %s", side_n, focal, source)
        return CameraIntrinsics(
            width=int(frame_width),
            height=int(frame_height),
            cx=cx,
            cy=cy,
            focal_px=focal,
            source=source,
            side=side_n,
            lens_focal_mm=lens_focal_mm,
        )

    logger.warning(
        "%s eye: no %s or %s under %s; using fallback focal_px=%.3f (%s)",
        side_n,
        FOCAL_LENGTH_CSV,
        PIX_SIZE_CSV,
        analysis,
        fallback.focal_px,
        fallback.source,
    )
    return fallback
