"""Tests for analysis artifact scan/load helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    ArtifactLoadState,
    SYNC_ARTIFACT_PROFILE,
    discover_eye_video_paths,
    scan_tab_artifacts,
    syncfree_status_signature_paths,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    PreprocConfig,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle


def _handle(block_path: Path) -> BlockHandle:
    return BlockHandle(
        animal_call="PV",
        experiment_date="2026_06_15",
        block_num="011",
        block_path=block_path,
        path_to_animal_folder=block_path.parent.parent.parent,
    )


def test_sync_manifest_has_twelve_artifacts():
    assert len(SYNC_ARTIFACT_PROFILE.artifacts) == 12


def test_scan_tab_artifacts_partial_and_full(tmp_path):
    block_path = tmp_path / "block_011"
    analysis = block_path / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "final_sync_df.csv").write_text("x")
    (analysis / "left_eye_data.csv").write_text("x")
    handle = _handle(block_path)
    config = PreprocConfig()
    result = scan_tab_artifacts(SYNC_ARTIFACT_PROFILE, handle, config)
    assert result.state is ArtifactLoadState.PARTIAL
    assert result.found_count == 2
    assert result.total == 12

    for spec in SYNC_ARTIFACT_PROFILE.artifacts:
        for p in spec.paths_fn(handle, config):
            if p.parent == analysis:
                p.parent.mkdir(parents=True, exist_ok=True)
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_text("stub")
    # parsed_events lives under oe_files
    oe = block_path / "oe_files" / "exp1" / "parsed_events.csv"
    oe.parent.mkdir(parents=True)
    oe.write_text("events")

    full = scan_tab_artifacts(SYNC_ARTIFACT_PROFILE, handle, config)
    assert full.state is ArtifactLoadState.READY
    assert full.found_count == 12


def test_discover_eye_video_paths(tmp_path):
    block_path = tmp_path / "block_011"
    le = block_path / "eye_videos" / "LE" / "sub"
    re = block_path / "eye_videos" / "RE" / "sub"
    le.mkdir(parents=True)
    re.mkdir(parents=True)
    (le / "left.mp4").write_bytes(b"\x00")
    (re / "right.mp4").write_bytes(b"\x00")
    paths = discover_eye_video_paths(_handle(block_path))
    assert "left" in paths and "right" in paths


def test_syncfree_status_signature_paths_without_blocksync(tmp_path):
    block_path = tmp_path / "block_011"
    le = block_path / "eye_videos" / "LE" / "sub"
    re = block_path / "eye_videos" / "RE" / "sub"
    le.mkdir(parents=True)
    re.mkdir(parents=True)
    (le / "left.mp4").write_bytes(b"\x00")
    (re / "right.mp4").write_bytes(b"\x00")
    config = PreprocConfig(syncfree_artifact_tag="v1")
    paths = syncfree_status_signature_paths(_handle(block_path), config)
    assert len(paths) == 2
    assert all(p.name.endswith("_eye_data.csv") for p in paths)
