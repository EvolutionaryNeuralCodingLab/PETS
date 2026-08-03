"""Scan and load on-disk preprocessing artifacts into a shared BlockSync session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from eye_tracking_system_tools.annotation.preprocessing_gui.block_session import (
    BlockSyncSession,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    PreprocConfig,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker import (
    load_eye_brightness_lists,
)
from eye_tracking_system_tools.preprocessing.block_sync_core import (
    load_eye_tracking_df_csv,
    load_final_sync_df,
)
from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
    append_angle_data,
    load_eye_data,
    load_self_kerr_refs,
)
from eye_tracking_system_tools.preprocessing.data_verification_utils import (
    load_eye_data as load_eye_data_verify,
)
from eye_tracking_system_tools.preprocessing.sync_free_eye_io import (
    default_syncfree_paths,
    discover_eye_video_paths,
)


class ArtifactLoadState(Enum):
    NONE = "none"
    PARTIAL = "partial"
    READY = "ready"


@dataclass
class ArtifactSpec:
    id: str
    label: str
    paths_fn: Callable[[BlockHandle, PreprocConfig], list[Path]]
    load_fn: Callable[
        [BlockSyncSession, Any, BlockHandle, GuiState, PreprocConfig], None
    ]


@dataclass(frozen=True)
class TabArtifactProfile:
    tab_id: str
    artifacts: tuple[ArtifactSpec, ...]


@dataclass
class ArtifactScanResult:
    state: ArtifactLoadState
    found: list[ArtifactSpec] = field(default_factory=list)
    missing: list[ArtifactSpec] = field(default_factory=list)
    total: int = 0

    @property
    def found_count(self) -> int:
        return len(self.found)

    @property
    def summary(self) -> str:
        return f"{self.found_count}/{self.total} artifacts on disk"


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def message(self) -> str:
        lines = []
        if self.loaded:
            lines.append("Loaded: " + ", ".join(self.loaded))
        if self.skipped:
            lines.append("Skipped (missing): " + ", ".join(self.skipped))
        return "\n".join(lines) if lines else "Nothing to load."


def resolve_parsed_events_path(block: BlockHandle) -> Path | None:
    """Resolve ``parsed_events.csv`` without constructing BlockSync."""
    oe_root = block.block_path / "oe_files"
    if not oe_root.is_dir():
        return None
    for exp_dir in sorted(oe_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        direct = exp_dir / "parsed_events.csv"
        if direct.is_file():
            return direct
        for sub in sorted(exp_dir.iterdir()):
            if sub.is_dir() and (sub / "parsed_events.csv").is_file():
                return sub / "parsed_events.csv"
    return None


def _paths_exist(paths: list[Path]) -> bool:
    return bool(paths) and all(p.is_file() for p in paths)


def scan_tab_artifacts(
    profile: TabArtifactProfile,
    block: BlockHandle,
    config: PreprocConfig,
) -> ArtifactScanResult:
    found: list[ArtifactSpec] = []
    missing: list[ArtifactSpec] = []
    for spec in profile.artifacts:
        paths = spec.paths_fn(block, config)
        if _paths_exist(paths):
            found.append(spec)
        else:
            missing.append(spec)
    total = len(profile.artifacts)
    if not found:
        state = ArtifactLoadState.NONE
    elif len(found) == total:
        state = ArtifactLoadState.READY
    else:
        state = ArtifactLoadState.PARTIAL
    return ArtifactScanResult(
        state=state, found=found, missing=missing, total=total
    )


def artifact_checklist_text(result: ArtifactScanResult) -> str:
    lines = []
    found_ids = {s.id for s in result.found}
    for spec in result.found + result.missing:
        mark = "x" if spec.id in found_ids else " "
        lines.append(f"[{mark}] {spec.label}")
    return "\n".join(lines)


def load_tab_artifacts(
    profile: TabArtifactProfile,
    session: BlockSyncSession,
    block: BlockHandle,
    state: GuiState,
    config: PreprocConfig,
) -> LoadReport:
    scan = scan_tab_artifacts(profile, block, config)
    report = LoadReport()
    if scan.state is ArtifactLoadState.NONE:
        return report
    blocksync = session.get(block)
    session.ensure_eye_videos(blocksync)
    for spec in scan.found:
        try:
            spec.load_fn(session, blocksync, block, state, config)
            report.loaded.append(spec.label)
        except Exception as exc:
            report.skipped.append(f"{spec.label} ({exc})")
    for spec in scan.missing:
        report.skipped.append(spec.label)
    return report


# --- Sync tab loaders -------------------------------------------------------


def _load_parsed_events(session, bs, block, state, config) -> None:
    bs.parse_open_ephys_events(overwrite=False, interactive_on_fail=False)


def _load_brightness_pkl(session, bs, block, state, config) -> None:
    left, right = load_eye_brightness_lists(bs)
    bs.le_frame_val_list = left
    bs.re_frame_val_list = right


def _load_arena_brightness(session, bs, block, state, config) -> None:
    bs.arena_brightness_df = pd.read_csv(block.analysis_path / "arena_brightness.csv")


def _load_simple_sync_left(session, bs, block, state, config) -> None:
    state.df_left_simple_sync = pd.read_csv(
        block.analysis_path / "eye_left_simple_sync.csv"
    )


def _load_simple_sync_right(session, bs, block, state, config) -> None:
    state.df_right_simple_sync = pd.read_csv(
        block.analysis_path / "eye_right_simple_sync.csv"
    )


def _load_blocksync_df(session, bs, block, state, config) -> None:
    bs.blocksync_df = pd.read_csv(
        block.analysis_path / "blocksync_df.csv", engine="python"
    )
    session.sanitize(bs)


def _load_final_sync(session, bs, block, state, config) -> None:
    load_final_sync_df(bs, verbose=False)
    state.final_sync_df = bs.final_sync_df


def _load_le_df(session, bs, block, state, config) -> None:
    bs.le_df = load_eye_tracking_df_csv(block.analysis_path / "le_df.csv")


def _load_re_df(session, bs, block, state, config) -> None:
    bs.re_df = load_eye_tracking_df_csv(block.analysis_path / "re_df.csv")


def _load_jitter_report(session, bs, block, state, config) -> None:
    bs.get_jitter_reports(
        export=False,
        overwrite=False,
        remove_led_blinks=False,
        sort_on_loading=True,
    )


def _load_left_eye_data(session, bs, block, state, config) -> None:
    load_eye_data_verify(bs)


SYNC_ARTIFACT_PROFILE = TabArtifactProfile(
    tab_id="sync",
    artifacts=(
        ArtifactSpec(
            "parsed_events",
            "parsed_events.csv",
            lambda b, c: [p for p in [resolve_parsed_events_path(b)] if p],
            _load_parsed_events,
        ),
        ArtifactSpec(
            "eye_brightness",
            "eye_brightness_values_dict.pkl",
            lambda b, c: [b.analysis_path / "eye_brightness_values_dict.pkl"],
            _load_brightness_pkl,
        ),
        ArtifactSpec(
            "arena_brightness",
            "arena_brightness.csv",
            lambda b, c: [b.analysis_path / "arena_brightness.csv"],
            _load_arena_brightness,
        ),
        ArtifactSpec(
            "simple_sync_left",
            "eye_left_simple_sync.csv",
            lambda b, c: [b.analysis_path / "eye_left_simple_sync.csv"],
            _load_simple_sync_left,
        ),
        ArtifactSpec(
            "simple_sync_right",
            "eye_right_simple_sync.csv",
            lambda b, c: [b.analysis_path / "eye_right_simple_sync.csv"],
            _load_simple_sync_right,
        ),
        ArtifactSpec(
            "blocksync_df",
            "blocksync_df.csv",
            lambda b, c: [b.analysis_path / "blocksync_df.csv"],
            _load_blocksync_df,
        ),
        ArtifactSpec(
            "final_sync_df",
            "final_sync_df.csv",
            lambda b, c: [b.analysis_path / "final_sync_df.csv"],
            _load_final_sync,
        ),
        ArtifactSpec(
            "le_df",
            "le_df.csv",
            lambda b, c: [b.analysis_path / "le_df.csv"],
            _load_le_df,
        ),
        ArtifactSpec(
            "re_df",
            "re_df.csv",
            lambda b, c: [b.analysis_path / "re_df.csv"],
            _load_re_df,
        ),
        ArtifactSpec(
            "jitter_report",
            "jitter_report_dict.pkl",
            lambda b, c: [b.analysis_path / "jitter_report_dict.pkl"],
            _load_jitter_report,
        ),
        ArtifactSpec(
            "left_eye_data",
            "left_eye_data.csv",
            lambda b, c: [b.analysis_path / "left_eye_data.csv"],
            _load_left_eye_data,
        ),
        ArtifactSpec(
            "right_eye_data",
            "right_eye_data.csv",
            lambda b, c: [b.analysis_path / "right_eye_data.csv"],
            lambda s, bs, b, st, c: load_eye_data_verify(bs),
        ),
        ArtifactSpec(
            "noise_epochs_left",
            "noise_epochs_left.csv",
            lambda b, c: [b.analysis_path / "noise_epochs_left.csv"],
            lambda s, bs, b, st, c: None,
        ),
        ArtifactSpec(
            "noise_epochs_right",
            "noise_epochs_right.csv",
            lambda b, c: [b.analysis_path / "noise_epochs_right.csv"],
            lambda s, bs, b, st, c: None,
        ),
    ),
)


VERIFY_ARTIFACT_PROFILE = TabArtifactProfile(
    tab_id="verify",
    artifacts=(
        ArtifactSpec(
            "left_eye_data",
            "left_eye_data.csv",
            lambda b, c: [b.analysis_path / "left_eye_data.csv"],
            _load_left_eye_data,
        ),
        ArtifactSpec(
            "right_eye_data",
            "right_eye_data.csv",
            lambda b, c: [b.analysis_path / "right_eye_data.csv"],
            lambda s, bs, b, st, c: load_eye_data_verify(bs),
        ),
        ArtifactSpec(
            "kerr_refs",
            "self_kerr_refs.csv",
            lambda b, c: [b.analysis_path / "self_kerr_refs.csv"],
            lambda s, bs, b, st, c: load_self_kerr_refs(bs),
        ),
        ArtifactSpec(
            "pupil_perimeters",
            "pupil_perimeters.yaml",
            lambda b, c: [b.analysis_path / "pupil_perimeters.yaml"],
            lambda s, bs, b, st, c: None,
        ),
        ArtifactSpec(
            "noise_epochs_left",
            "noise_epochs_left.csv",
            lambda b, c: [b.analysis_path / "noise_epochs_left.csv"],
            lambda s, bs, b, st, c: None,
        ),
        ArtifactSpec(
            "noise_epochs_right",
            "noise_epochs_right.csv",
            lambda b, c: [b.analysis_path / "noise_epochs_right.csv"],
            lambda s, bs, b, st, c: None,
        ),
    ),
)


def kerr_artifact_profile(name_tag: str) -> TabArtifactProfile:
    tag = name_tag or "raw_verified"

    def _angle_paths(block: BlockHandle, _config: PreprocConfig) -> list[Path]:
        ap = block.analysis_path
        return [
            ap / f"left_kerr_angle_{tag}.csv",
            ap / f"right_kerr_angle_{tag}.csv",
        ]

    def _load_kerr_angles(session, bs, block, state, config) -> None:
        load_eye_data(bs)
        load_self_kerr_refs(bs)
        left_angle = block.analysis_path / f"left_kerr_angle_{tag}.csv"
        right_angle = block.analysis_path / f"right_kerr_angle_{tag}.csv"
        left_angles = pd.read_csv(left_angle)
        right_angles = pd.read_csv(right_angle)
        bs.left_eye_data = append_angle_data(bs.left_eye_data, left_angles)
        bs.right_eye_data = append_angle_data(bs.right_eye_data, right_angles)

    return TabArtifactProfile(
        tab_id="kerr",
        artifacts=(
            ArtifactSpec(
                "left_eye_data",
                "left_eye_data.csv",
                lambda b, c: [b.analysis_path / "left_eye_data.csv"],
                lambda s, bs, b, st, c: load_eye_data(bs),
            ),
            ArtifactSpec(
                "right_eye_data",
                "right_eye_data.csv",
                lambda b, c: [b.analysis_path / "right_eye_data.csv"],
                lambda s, bs, b, st, c: None,
            ),
            ArtifactSpec(
                "kerr_refs",
                "self_kerr_refs.csv",
                lambda b, c: [b.analysis_path / "self_kerr_refs.csv"],
                lambda s, bs, b, st, c: load_self_kerr_refs(bs),
            ),
            ArtifactSpec(
                "kerr_angles",
                f"Kerr angles ({tag})",
                _angle_paths,
                _load_kerr_angles,
            ),
        ),
    )


BEHAVIOR_ARTIFACT_PROFILE = TabArtifactProfile(
    tab_id="behavior",
    artifacts=(
        ArtifactSpec(
            "behavior_state",
            "behavior_state.csv",
            lambda b, c: [b.analysis_path / f"block_{b.block_num}_behavior_state.csv"],
            lambda s, bs, b, st, c: None,
        ),
    ),
)


EXPLORE_ARTIFACT_PROFILE = TabArtifactProfile(
    tab_id="explore",
    artifacts=(
        ArtifactSpec(
            "final_sync_df",
            "final_sync_df.csv",
            lambda b, c: [b.analysis_path / "final_sync_df.csv"],
            _load_final_sync,
        ),
        ArtifactSpec(
            "left_eye_data",
            "left_eye_data.csv",
            lambda b, c: [b.analysis_path / "left_eye_data.csv"],
            _load_left_eye_data,
        ),
        ArtifactSpec(
            "right_eye_data",
            "right_eye_data.csv",
            lambda b, c: [b.analysis_path / "right_eye_data.csv"],
            lambda s, bs, b, st, c: load_eye_data_verify(bs),
        ),
    ),
)


def _saccades_dir(block: BlockHandle) -> Path:
    return block.analysis_path / "saccades"


CALIBRATION_ARTIFACT_PROFILE = TabArtifactProfile(
    tab_id="calibration",
    artifacts=(
        ArtifactSpec(
            "pix_size",
            "LR_pix_size.csv",
            lambda b, c: [b.analysis_path / "LR_pix_size.csv"],
            lambda s, bs, b, st, c: None,
        ),
    ),
)


SACCADES_ARTIFACT_PROFILE = TabArtifactProfile(
    tab_id="saccades",
    artifacts=(
        ArtifactSpec(
            "left_eye_data",
            "left_eye_data*.csv (with Kerr angles)",
            lambda b, c: list(b.analysis_path.glob("left_eye_data*.csv")),
            lambda s, bs, b, st, c: None,
        ),
        ArtifactSpec(
            "right_eye_data",
            "right_eye_data*.csv (with Kerr angles)",
            lambda b, c: list(b.analysis_path.glob("right_eye_data*.csv")),
            lambda s, bs, b, st, c: None,
        ),
        ArtifactSpec(
            "saccade_events",
            "saccades/saccade_events.csv",
            lambda b, c: [_saccades_dir(b) / "saccade_events.csv"],
            lambda s, bs, b, st, c: None,
        ),
        ArtifactSpec(
            "saccade_params",
            "saccades/detection_params.yaml",
            lambda b, c: [_saccades_dir(b) / "detection_params.yaml"],
            lambda s, bs, b, st, c: None,
        ),
    ),
)


def syncfree_artifact_profile(artifact_tag: str) -> TabArtifactProfile:
    tag = artifact_tag or "v1"

    def _eye_data_paths(block: BlockHandle, _config: PreprocConfig) -> list[Path]:
        videos = discover_eye_video_paths(block)
        paths = []
        if "left" in videos:
            paths.append(
                default_syncfree_paths(videos["left"], "left", tag)["eye_data"]
            )
        if "right" in videos:
            paths.append(
                default_syncfree_paths(videos["right"], "right", tag)["eye_data"]
            )
        return paths

    return TabArtifactProfile(
        tab_id="syncfree",
        artifacts=(
            ArtifactSpec(
                "syncfree_eye_data",
                f"syncfree eye_data ({tag})",
                _eye_data_paths,
                lambda s, bs, b, st, c: None,
            ),
        ),
    )


def syncfree_status_signature_paths(
    block: BlockHandle, config: PreprocConfig
) -> list[Path]:
    """Path-only status signature for Sync-free tab (no BlockSync)."""
    profile = syncfree_artifact_profile(config.syncfree_artifact_tag)
    paths: list[Path] = []
    for spec in profile.artifacts:
        paths.extend(spec.paths_fn(block, config))
    return paths
