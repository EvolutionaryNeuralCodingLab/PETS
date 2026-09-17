"""Per-block ``rotation_correction_params.yaml`` for ellipse rotation correction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from eye_tracking_system_tools.preprocessing.conicoid.refined_io import _analysis_dir
from eye_tracking_system_tools.preprocessing.conicoid.spin_max import SpinMaxSettings

PARAMS_FILENAME = "rotation_correction_params.yaml"
PARAMS_VERSION = 1


def rotation_correction_params_path(block_path: Path | str) -> Path:
    return _analysis_dir(block_path) / PARAMS_FILENAME


def settings_to_mapping(settings: SpinMaxSettings) -> dict[str, Any]:
    return {
        "threshold": None if settings.threshold is None else int(settings.threshold),
        "roi_mult": float(settings.roi_mult),
        "x_flip": bool(settings.x_flip),
        "frame_width": (
            None if settings.frame_width is None else float(settings.frame_width)
        ),
        "band_width": float(settings.band_width),
        "median_k": int(settings.median_k),
    }


def settings_from_mapping(payload: Mapping[str, Any] | None) -> SpinMaxSettings:
    data = dict(payload or {})
    threshold = data.get("threshold")
    if threshold is not None:
        threshold = int(threshold)
    frame_width = data.get("frame_width")
    if frame_width is not None:
        frame_width = float(frame_width)
    return SpinMaxSettings(
        threshold=threshold,
        roi_mult=float(data.get("roi_mult", 1.2)),
        band_width=float(data.get("band_width", 5.0)),
        median_k=int(data.get("median_k", 5)),
        x_flip=bool(data.get("x_flip", False)),
        frame_width=frame_width,
    )


@dataclass
class RotationCorrectionParams:
    """Portable per-block knobs (no video / DLC paths)."""

    left: SpinMaxSettings
    right: SpinMaxSettings
    apply_jitter: bool = True
    version: int = PARAMS_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "apply_jitter": bool(self.apply_jitter),
            "left": settings_to_mapping(self.left),
            "right": settings_to_mapping(self.right),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RotationCorrectionParams:
        if not isinstance(payload, Mapping):
            raise ValueError("rotation correction params must be a mapping")
        version = int(payload.get("version", PARAMS_VERSION))
        return cls(
            version=version,
            apply_jitter=bool(payload.get("apply_jitter", True)),
            left=settings_from_mapping(payload.get("left")),
            right=settings_from_mapping(payload.get("right")),
        )

    def eye_settings(self) -> dict[str, SpinMaxSettings]:
        return {"left": self.left, "right": self.right}

    def with_frame_widths(
        self, left_width: float | None, right_width: float | None
    ) -> RotationCorrectionParams:
        left = self.left
        right = self.right
        if left.frame_width is None and left_width is not None:
            left = replace(left, frame_width=float(left_width))
        if right.frame_width is None and right_width is not None:
            right = replace(right, frame_width=float(right_width))
        return replace(self, left=left, right=right)


def read_rotation_params(block_path: Path | str) -> RotationCorrectionParams:
    path = rotation_correction_params_path(block_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return RotationCorrectionParams.from_mapping(payload)


def try_read_rotation_params(
    block_path: Path | str,
) -> RotationCorrectionParams | None:
    path = rotation_correction_params_path(block_path)
    if not path.is_file():
        return None
    return read_rotation_params(block_path)


def write_rotation_params(
    block_path: Path | str,
    params: RotationCorrectionParams,
) -> Path:
    path = rotation_correction_params_path(block_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        params.to_mapping(),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    path.write_text(body, encoding="utf-8")
    return path
