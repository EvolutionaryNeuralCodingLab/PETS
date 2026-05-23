"""YAML config load/save for the Block Annotator."""

from __future__ import annotations

from pathlib import Path

import yaml

from eye_tracking_system_tools.annotation.block_annotator.models import AnnotatorConfig

DEFAULT_CONFIG_NAME = "annotator_config.yaml"


def default_config_path(output_folder: Path) -> Path:
    return Path(output_folder) / DEFAULT_CONFIG_NAME


def config_template_dict() -> dict:
    return {
        "event_types": ["saccade", "blink", "noise", "pupil event"],
        "default_range_half_width_ms": 100.0,
        "playback_fps": 60.0,
        "step_rows": 1,
    }


def ensure_config_template(output_folder: Path) -> Path:
    """Create annotator_config.yaml in output_folder if missing."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    path = default_config_path(output_folder)
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_template_dict(), f, sort_keys=False)
    return path


def load_config(path: Path | None, output_folder: Path) -> AnnotatorConfig:
    if path is None:
        path = ensure_config_template(output_folder)
    else:
        path = Path(path)
        if not path.exists():
            path = ensure_config_template(output_folder)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AnnotatorConfig.from_dict(data)


def save_config(path: Path, config: AnnotatorConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
