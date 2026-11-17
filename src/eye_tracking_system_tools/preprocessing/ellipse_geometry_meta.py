"""Ellipse geometry schema / provenance helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eye_tracking_system_tools.preprocessing.ellipse_fit import (
    ELLIPSE_GEOMETRY_VERSION,
    NEAR_CIRCLE_AXIS_RATIO,
)

ELLIPSE_GEOMETRY_META_FILENAME = "ellipse_geometry_meta.json"


def ellipse_geometry_meta(
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance dict written at eye-table storage boundaries."""
    meta: dict[str, Any] = {
        "ellipse_geometry_version": int(ELLIPSE_GEOMETRY_VERSION),
        "phi_unit": "radians",
        "phi_range": "[0, pi)",
        "width": "major_semiaxis",
        "height": "minor_semiaxis",
        "coordinate_space": "raw_image_xy_right_down",
        "near_circle_axis_ratio": float(NEAR_CIRCLE_AXIS_RATIO),
        "notes": (
            "phi is major-axis orientation; convert to degrees only at "
            "OpenCV/cv2.ellipse call sites."
        ),
    }
    if extra:
        meta.update(extra)
    return meta


def write_ellipse_geometry_meta(
    analysis_path: Path | str,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``ellipse_geometry_meta.json`` under an analysis directory."""
    path = Path(analysis_path) / ELLIPSE_GEOMETRY_META_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = ellipse_geometry_meta(extra=extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def read_ellipse_geometry_version(
    analysis_path: Path | str | None = None,
    *,
    default: int = 1,
) -> int:
    """Read geometry version from sidecar; default 1 for legacy tables."""
    if analysis_path is None:
        return int(default)
    path = Path(analysis_path) / ELLIPSE_GEOMETRY_META_FILENAME
    if not path.is_file():
        return int(default)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("ellipse_geometry_version", default))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return int(default)
