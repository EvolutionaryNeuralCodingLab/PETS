"""
Eye size on the sensor for methods-text / review replies.

Measure a rectangular ROI around each eye on one representative block per
species (same defaults as :mod:`species_traces`), average left and right, and
report:

* **pixels** — mean ROI area (px²)
* **percentage** — mean ROI diagonal as a fraction of the 640×480 frame diagonal

Camera: Arducam B0066 (OV5647) with QILENS M7 3.7 mm pinhole lens — see
``CAMERA_SPECS``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.species_traces import DEFAULTS as SPECIES_TRACE_DEFAULTS

# Nominal recording format used throughout the paper pipeline.
NOMINAL_FRAME_W = 640
NOMINAL_FRAME_H = 480
NOMINAL_FRAME_AREA = NOMINAL_FRAME_W * NOMINAL_FRAME_H
NOMINAL_FRAME_DIAGONAL = float(np.hypot(NOMINAL_FRAME_W, NOMINAL_FRAME_H))  # 800
NOMINAL_FPS_HZ = 60.0

# Body: https://www.arducam.com/arducam-raspberry-pi-5mp-spy-camera-b0066.html
# Replacement lens (stock lens removed): AliExpress item 32944505429
#   QILENS CM7F3.7H7.8A0 — M7×0.35 pinhole, FL 3.7 mm, f/2.55, 90° on 1/3"
LENS_URL = (
    "https://www.aliexpress.com/item/32944505429.html"
)
CAMERA_SPECS: dict[str, Any] = {
    "product": "Arducam Raspberry Pi 5MP spy camera (B0066)",
    "sensor": "Omnivision OV5647",
    "sensor_optical_format": "1/4\"",
    "native_still_resolution": "2592×1944 (5 MP)",
    "shutter": "rolling",
    "ir_filter": "integral IR-cut (visible light)",
    "stock_lens_replaced": True,
    "lens": {
        "brand": "QILENS",
        "model": "CM7F3.7H7.8A0",
        "type": "pinhole / mini CCTV",
        "mount": "M7×0.35",
        "focal_length_mm": 3.7,
        "aperture": "f/2.55",
        "manufacturer_image_format": "1/3\"",
        "manufacturer_fov_deg": 90,
        "weight_g": 2,
        "url": LENS_URL,
    },
    "manufacturer_video_modes": (
        "1080p30",
        "720p60",
        "480p90",
        "640×480 at 60/90 fps (common OV5647/RPi mode)",
    ),
    "this_study_recording": {
        "resolution_px": (NOMINAL_FRAME_W, NOMINAL_FRAME_H),
        "framerate_hz": NOMINAL_FPS_HZ,
        "interface": "Raspberry Pi CSI",
    },
    "body_url": "https://www.arducam.com/arducam-raspberry-pi-5mp-spy-camera-b0066.html",
}

CAMERA_SPECS_TEXT = """\
Camera (eye views)
  Body:      Arducam B0066 Raspberry Pi 5MP spy camera
  Sensor:    Omnivision OV5647, 1/4\", rolling shutter, integral IR-cut
  Native:    2592×1944 still; manufacturer video modes include 1080p30 / 720p60 / 480p90
  Lens:      QILENS CM7F3.7H7.8A0 pinhole (stock lens replaced)
             FL 3.7 mm · f/2.55 · M7×0.35 mount · ~2 g
             Manufacturer FOV ≈ 90° (quoted for 1/3\" format; OV5647 is 1/4\")
  This study: 640×480 at 60 Hz via Raspberry Pi CSI
  Body ref:  https://www.arducam.com/arducam-raspberry-pi-5mp-spy-camera-b0066.html
  Lens ref:  https://www.aliexpress.com/item/32944505429.html
""".rstrip()


@dataclass(frozen=True)
class SpeciesBlock:
    """One representative block used for on-sensor eye sizing."""

    species: str
    animal: str
    block_path: Path
    block_num: str


@dataclass(frozen=True)
class EyeRoiMeasurement:
    """Pixel footprint of a user-drawn eye ROI on one eye video frame."""

    species: str
    animal: str
    side: str
    block_path: str
    video_path: str
    x: int
    y: int
    width_px: int
    height_px: int
    area_px: int
    frame_width_px: int
    frame_height_px: int
    nominal_frame_width_px: int = NOMINAL_FRAME_W
    nominal_frame_height_px: int = NOMINAL_FRAME_H

    @property
    def diagonal_px(self) -> float:
        return float(np.hypot(self.width_px, self.height_px))

    @property
    def diagonal_pct_of_nominal_frame(self) -> float:
        return 100.0 * self.diagonal_px / NOMINAL_FRAME_DIAGONAL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["diagonal_px"] = self.diagonal_px
        d["diagonal_pct_of_nominal_frame"] = self.diagonal_pct_of_nominal_frame
        return d


@dataclass(frozen=True)
class SpeciesEyeSize:
    """Left/right-averaged eye size for one animal (one number set)."""

    species: str
    animal: str
    block_path: str
    n_eyes: int
    mean_area_px: float
    mean_diagonal_px: float
    mean_diagonal_pct_of_frame: float
    left_area_px: float | None = None
    right_area_px: float | None = None
    left_diagonal_px: float | None = None
    right_diagonal_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_species_blocks(
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[SpeciesBlock]:
    """Lizard / mouse / turtle blocks from :data:`species_traces.DEFAULTS`."""
    cfg = {k: dict(v) for k, v in SPECIES_TRACE_DEFAULTS.items()}
    if overrides:
        for species, body in overrides.items():
            cfg.setdefault(species, {}).update(body)
    out: list[SpeciesBlock] = []
    for species in ("lizard", "mouse", "turtle"):
        body = cfg[species]
        out.append(
            SpeciesBlock(
                species=species,
                animal=str(body["animal"]),
                block_path=Path(body["block_path"]),
                block_num=str(body["block_num"]),
            )
        )
    return out


def measure_eye_roi(
    roi: tuple[int, int, int, int] | list[int],
    *,
    species: str,
    animal: str,
    side: str,
    block_path: Path | str,
    video_path: Path | str,
    frame_shape_hw: tuple[int, int] | None = None,
) -> EyeRoiMeasurement:
    """Convert an ``(x, y, w, h)`` ROI into pixel metrics."""
    x, y, w, h = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    if w < 1 or h < 1:
        raise ValueError(f"ROI must have positive size; got w={w}, h={h}")
    if frame_shape_hw is None:
        fh, fw = NOMINAL_FRAME_H, NOMINAL_FRAME_W
    else:
        fh, fw = int(frame_shape_hw[0]), int(frame_shape_hw[1])
    return EyeRoiMeasurement(
        species=str(species),
        animal=str(animal),
        side=str(side),
        block_path=str(Path(block_path)),
        video_path=str(Path(video_path)),
        x=x,
        y=y,
        width_px=w,
        height_px=h,
        area_px=w * h,
        frame_width_px=fw,
        frame_height_px=fh,
    )


def average_species_eye_sizes(
    measurements: Sequence[EyeRoiMeasurement],
    *,
    require_both_eyes: bool = True,
) -> list[SpeciesEyeSize]:
    """
    One averaged number set per species/animal.

    Averages left and right **area** (px²) and **diagonal** (px). Percentage is
    mean diagonal / frame diagonal (640×480 → 800 px).
    """
    by_species: dict[str, list[EyeRoiMeasurement]] = {}
    for m in measurements:
        by_species.setdefault(m.species, []).append(m)

    out: list[SpeciesEyeSize] = []
    for species in ("lizard", "mouse", "turtle", *sorted(set(by_species) - {"lizard", "mouse", "turtle"})):
        group = by_species.get(species)
        if not group:
            continue
        by_side = {m.side: m for m in group}
        left = by_side.get("left")
        right = by_side.get("right")
        if require_both_eyes and (left is None or right is None):
            continue
        used = [m for m in (left, right) if m is not None]
        if not used:
            continue
        mean_area = float(np.mean([m.area_px for m in used]))
        mean_diag = float(np.mean([m.diagonal_px for m in used]))
        out.append(
            SpeciesEyeSize(
                species=species,
                animal=used[0].animal,
                block_path=used[0].block_path,
                n_eyes=len(used),
                mean_area_px=mean_area,
                mean_diagonal_px=mean_diag,
                mean_diagonal_pct_of_frame=100.0 * mean_diag / NOMINAL_FRAME_DIAGONAL,
                left_area_px=None if left is None else float(left.area_px),
                right_area_px=None if right is None else float(right.area_px),
                left_diagonal_px=None if left is None else float(left.diagonal_px),
                right_diagonal_px=None if right is None else float(right.diagonal_px),
            )
        )
    return out


def pick_large_pupil_frame_index(
    eye_csv_path: Path | str,
    *,
    percentile: float = 75.0,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Randomly pick a frame index where the pupil diameter is above the
    ``percentile``-th value of ``major_ax`` in the eye CSV.

    Returns an ``eye_frame`` value suitable for ``grab_frame(video, idx)``.
    """
    df = pd.read_csv(eye_csv_path)
    if "major_ax" not in df.columns or "eye_frame" not in df.columns:
        raise ValueError(
            f"Eye CSV must contain 'major_ax' and 'eye_frame' columns: "
            f"{eye_csv_path}"
        )
    col = df["major_ax"].dropna()
    if col.empty:
        raise ValueError(f"No finite major_ax values in {eye_csv_path}")

    threshold = float(np.percentile(col, percentile))
    above = df.loc[df["major_ax"] >= threshold, "eye_frame"].dropna().astype(int)
    if above.empty:
        raise ValueError(
            f"No frames with major_ax ≥ {threshold:.1f} (p{percentile}) "
            f"in {eye_csv_path}"
        )
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(above.values))


def format_measurements_report(
    measurements: Iterable[EyeRoiMeasurement],
    *,
    include_camera_specs: bool = True,
    require_both_eyes: bool = True,
) -> str:
    """Human-readable summary: one averaged row per animal."""
    rows = list(measurements)
    averages = average_species_eye_sizes(rows, require_both_eyes=require_both_eyes)
    lines: list[str] = []
    if include_camera_specs:
        lines.append(CAMERA_SPECS_TEXT)
        lines.append("")
    lines.append(
        f"Eye size on sensor (L/R average; frame {NOMINAL_FRAME_W}×{NOMINAL_FRAME_H}, "
        f"diagonal {NOMINAL_FRAME_DIAGONAL:.0f} px)"
    )
    lines.append(
        "  Report: mean area (px²); mean ROI diagonal as % of frame diagonal."
    )
    if not averages:
        incomplete = []
        by_species: dict[str, set[str]] = {}
        for m in rows:
            by_species.setdefault(m.species, set()).add(m.side)
        for sp, sides in by_species.items():
            if sides != {"left", "right"}:
                incomplete.append(f"{sp} ({', '.join(sorted(sides)) or 'none'})")
        if incomplete:
            lines.append(
                "  (need both left and right ROIs for: " + "; ".join(incomplete) + ")"
            )
        else:
            lines.append("  (no ROIs yet)")
        return "\n".join(lines)

    for a in averages:
        block = Path(a.block_path).name
        lines.append(
            f"\n{a.species} ({a.animal}, {block})  [n={a.n_eyes} eyes]"
        )
        lines.append(
            f"  area:     {a.mean_area_px:.0f} px²"
            + (
                f"  (L={a.left_area_px:.0f}, R={a.right_area_px:.0f})"
                if a.left_area_px is not None and a.right_area_px is not None
                else ""
            )
        )
        lines.append(
            f"  diagonal: {a.mean_diagonal_px:.1f} px "
            f"= {a.mean_diagonal_pct_of_frame:.2f}% of frame diagonal"
            + (
                f"  (L={a.left_diagonal_px:.1f}, R={a.right_diagonal_px:.1f})"
                if a.left_diagonal_px is not None and a.right_diagonal_px is not None
                else ""
            )
        )
    return "\n".join(lines)
