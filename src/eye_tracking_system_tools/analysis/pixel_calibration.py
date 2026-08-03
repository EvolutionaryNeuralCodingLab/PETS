"""
Per-block pixel-size calibration (``analysis/LR_pix_size.csv``).

Standalone port of ``BlockSync.calibrate_pixel_size``: the user drags an ROI whose
**diagonal** spans a known real-world distance on a frame of each eye video, and

    pix_size = known_dist_mm / sqrt(w**2 + h**2)      # mm per pixel

The CSV keeps the exact ``L_pix_size`` / ``R_pix_size`` columns BlockSync reads, so
files written here are picked up by the preprocessing pipeline and vice versa.
No BlockSync object (and no sync / DLC data) is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PIX_SIZE_CSV = "LR_pix_size.csv"
DEFAULT_KNOWN_DIST_MM = 10.0
MM_TO_UM = 1000.0

_EYE_ALIASES = {
    "left_eye": "L",
    "left": "L",
    "l": "L",
    "le": "L",
    "right_eye": "R",
    "right": "R",
    "r": "R",
    "re": "R",
}


@dataclass(frozen=True)
class PixelSize:
    """Millimetres per pixel for each eye camera."""

    l_mm_per_px: float
    r_mm_per_px: float
    source: str = PIX_SIZE_CSV

    @property
    def l_um_per_px(self) -> float:
        return self.l_mm_per_px * MM_TO_UM

    @property
    def r_um_per_px(self) -> float:
        return self.r_mm_per_px * MM_TO_UM

    def um_per_px(self, eye: str) -> float:
        """Scale factor for an eye name (``left_eye`` / ``R`` / ``right`` …)."""
        side = _EYE_ALIASES.get(str(eye).strip().lower())
        if side is None:
            raise KeyError(f"Unknown eye {eye!r} (expected left/right variants)")
        return self.l_um_per_px if side == "L" else self.r_um_per_px


def analysis_dir(block_path: Path | str) -> Path:
    return Path(block_path) / "analysis"


def pix_size_csv_path(block_path: Path | str) -> Path:
    return analysis_dir(block_path) / PIX_SIZE_CSV


def has_pixel_calibration(block_path: Path | str) -> bool:
    return pix_size_csv_path(block_path).is_file()


def read_pixel_size(block_path: Path | str) -> PixelSize | None:
    """Return the block's calibration, or ``None`` when the CSV is absent."""
    path = pix_size_csv_path(block_path)
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    for col in ("L_pix_size", "R_pix_size"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing column {col}")
    left = float(df.at[0, "L_pix_size"])
    right = float(df.at[0, "R_pix_size"])
    if not (np.isfinite(left) and np.isfinite(right)) or left <= 0 or right <= 0:
        raise ValueError(f"{path}: non-positive pixel sizes ({left}, {right})")
    return PixelSize(l_mm_per_px=left, r_mm_per_px=right, source=str(path))


def write_pixel_size(
    block_path: Path | str,
    l_mm_per_px: float,
    r_mm_per_px: float,
    *,
    known_dist_mm: float | None = None,
    method: str = "roi_diagonal",
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write the BlockSync-compatible CSV plus a provenance sidecar."""
    out_dir = analysis_dir(block_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / PIX_SIZE_CSV
    pd.DataFrame([{"L_pix_size": float(l_mm_per_px), "R_pix_size": float(r_mm_per_px)}]).to_csv(
        csv_path, index=False
    )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "block_path": str(Path(block_path)),
        "units": "mm_per_pixel",
        "L_pix_size": float(l_mm_per_px),
        "R_pix_size": float(r_mm_per_px),
        "L_um_per_px": float(l_mm_per_px) * MM_TO_UM,
        "R_um_per_px": float(r_mm_per_px) * MM_TO_UM,
        "known_dist_mm": known_dist_mm,
        "method": method,
        "entrypoint": "eye_tracking_system_tools.analysis.pixel_calibration",
        **(extra_meta or {}),
    }
    with open(csv_path.with_suffix(".meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return csv_path


def find_eye_videos(block_path: Path | str) -> dict[str, Path | None]:
    """Locate the raw LE/RE mp4s (same rule as ``BlockSync.handle_eye_videos``)."""
    block_path = Path(block_path)
    out: dict[str, Path | None] = {"left": None, "right": None}
    for side, folder in (("left", "LE"), ("right", "RE")):
        d = block_path / "eye_videos" / folder
        if not d.is_dir():
            continue
        vids = sorted(p for p in d.rglob("*.mp4") if "DLC" not in str(p))
        if vids:
            out[side] = vids[0]
    return out


def pix_size_from_roi(known_dist_mm: float, roi) -> float:
    """``roi`` is OpenCV's ``(x, y, w, h)``; the diagonal spans ``known_dist_mm``."""
    _, _, w, h = (float(v) for v in roi)
    diag = float(np.sqrt(w**2 + h**2))
    if diag <= 0:
        raise ValueError("Empty ROI — drag a box across the known distance")
    return float(known_dist_mm) / diag


def grab_frame(video_path: Path | str, frame_index: int = 1):
    """Read one frame (BlockSync uses index 1, the second frame)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        cap.set(1, int(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
        return frame
    finally:
        cap.release()


def calibrate_block(
    block_path: Path | str,
    *,
    known_dist_mm: float = DEFAULT_KNOWN_DIST_MM,
    overwrite: bool = False,
    frame_index: int = 1,
) -> PixelSize:
    """
    Open an OpenCV ROI window per eye and write ``analysis/LR_pix_size.csv``.

    Drag a box whose **diagonal** spans ``known_dist_mm`` on the landmark, then
    press Enter/Space (Esc cancels). Requires a GUI-capable OpenCV build and a
    kernel running on the machine with the display.
    """
    import cv2

    block_path = Path(block_path)
    if not overwrite:
        existing = read_pixel_size(block_path)
        if existing is not None:
            print(f"calibration already exists for {block_path.name}: {pix_size_csv_path(block_path)}")
            return existing

    if not hasattr(cv2, "selectROI"):
        raise RuntimeError(
            "This OpenCV build has no GUI (opencv-python-headless). "
            "Install opencv-python, or use manual_calibration()."
        )

    videos = find_eye_videos(block_path)
    missing = [side for side, p in videos.items() if p is None]
    if missing:
        raise FileNotFoundError(
            f"{block_path}: no raw mp4 under eye_videos/{'/'.join(s.upper()[0] + 'E' for s in missing)} "
            "— cannot calibrate from video (use manual_calibration())."
        )

    title = f"Drag the ROI diagonal across {known_dist_mm} mm — {block_path.name}"
    sizes: dict[str, float] = {}
    try:
        for side in ("right", "left"):
            frame = grab_frame(videos[side], frame_index)
            roi = cv2.selectROI(f"{title} [{side} eye]", frame, showCrosshair=True)
            cv2.destroyWindow(f"{title} [{side} eye]")
            cv2.waitKey(1)
            sizes[side] = pix_size_from_roi(known_dist_mm, roi)
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    write_pixel_size(
        block_path,
        sizes["left"],
        sizes["right"],
        known_dist_mm=known_dist_mm,
        method="roi_diagonal_cv2",
        extra_meta={
            "left_video": str(videos["left"]),
            "right_video": str(videos["right"]),
            "frame_index": int(frame_index),
        },
    )
    result = PixelSize(l_mm_per_px=sizes["left"], r_mm_per_px=sizes["right"])
    print(
        f"{block_path.name}: L={result.l_um_per_px:.2f} µm/px, "
        f"R={result.r_um_per_px:.2f} µm/px → {pix_size_csv_path(block_path)}"
    )
    return result


def manual_calibration(
    block_path: Path | str,
    *,
    left_px: float,
    right_px: float,
    known_dist_mm: float = DEFAULT_KNOWN_DIST_MM,
) -> PixelSize:
    """
    Fallback without video: give the measured landmark length in pixels per eye.

    Useful when the raw mp4s are offline but the landmark was measured elsewhere.
    """
    if left_px <= 0 or right_px <= 0:
        raise ValueError("Pixel distances must be positive")
    left = float(known_dist_mm) / float(left_px)
    right = float(known_dist_mm) / float(right_px)
    write_pixel_size(
        block_path,
        left,
        right,
        known_dist_mm=known_dist_mm,
        method="manual_pixel_distance",
        extra_meta={"left_px": float(left_px), "right_px": float(right_px)},
    )
    return PixelSize(l_mm_per_px=left, r_mm_per_px=right)


def require_pixel_sizes(block_paths) -> dict[str, PixelSize]:
    """
    Map ``str(block_path) → PixelSize``, raising with the full list of blocks that
    still need calibration.
    """
    out: dict[str, PixelSize] = {}
    missing: list[str] = []
    for block_path in block_paths:
        ps = read_pixel_size(block_path)
        if ps is None:
            missing.append(str(block_path))
        else:
            out[str(block_path)] = ps
    if missing:
        listed = "\n  ".join(missing)
        raise ValueError(
            "Missing analysis/LR_pix_size.csv for:\n  "
            f"{listed}\n"
            "Calibrate them (development/jitter_mount_pipeline.ipynb, section 3) "
            "or pass units='px' to stay in pixel units."
        )
    return out
