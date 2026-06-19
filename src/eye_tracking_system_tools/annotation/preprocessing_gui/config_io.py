"""YAML config load/save for the Preprocessing GUI.

Config lives **per-output-folder** as ``preproc_gui_config.yaml`` (mirroring
the Block Annotator's ``annotator_config.yaml`` convention). The user picks
an output folder once; the GUI seeds a default template there and persists
the last-used paths, default parameters, and tab-state on save.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = "preproc_gui_config.yaml"


@dataclass
class PreprocConfig:
    """All settings persisted between Preprocessing GUI sessions.

    Defaults reflect the values currently used in the source notebooks.
    """

    # ---- Block-selection defaults (last-used) ----
    last_experiment_path: str | None = None
    last_animal: str | None = None
    last_blocks: list[str] = field(default_factory=list)
    last_bad_blocks: list[str] = field(default_factory=list)
    last_channeldict: dict[str, str] | None = None  # keys stringified for YAML

    # ---- Stage 1 defaults ----
    arena_target_fps: float = 60.0
    arena_fps_tol_hz: float = 5.0
    final_sync_tol_frac: float = 0.9
    dlc_threshold_to_use: float = 0.95
    jitter_max_distance: int = 60
    jitter_diff_threshold: int = 5
    jitter_gap_to_bridge: int = 24

    # ---- Stage 3 default ----
    kerr_name_tag: str = "raw_verified"

    # ---- Stage 4 defaults ----
    behavior_window_size_ms: int = 10000
    behavior_step_size_ms: int = 1000
    behavior_threshold: float = 0.3

    # ---- Stage 5 (sync-free) defaults ----
    syncfree_artifact_tag: str = "v1"
    syncfree_uncertainty_thr: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PreprocConfig":
        data = data or {}
        fields = cls()
        for k, v in data.items():
            if hasattr(fields, k):
                setattr(fields, k, v)
        return fields


def default_config_path(output_folder: Path) -> Path:
    return Path(output_folder) / DEFAULT_CONFIG_NAME


def config_template_dict() -> dict[str, Any]:
    return PreprocConfig().to_dict()


def ensure_config_template(output_folder: Path) -> Path:
    """Create ``preproc_gui_config.yaml`` in *output_folder* if missing."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    path = default_config_path(output_folder)
    if not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_template_dict(), f, sort_keys=False)
    return path


def load_config(path: Path | None, output_folder: Path) -> PreprocConfig:
    if path is None:
        path = ensure_config_template(output_folder)
    else:
        path = Path(path)
        if not path.exists():
            path = ensure_config_template(output_folder)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return PreprocConfig.from_dict(data)


def save_config(path: Path, config: PreprocConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
