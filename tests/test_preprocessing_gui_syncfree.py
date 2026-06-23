"""Phase 7 tests — Sync-free tab (unified artifacts)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    ensure_config_template,
    load_config,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.syncfree_tab import (
    SyncFreeTab,
)
from eye_tracking_system_tools.preprocessing.sync_free_eye_io import (
    analysis_syncfree_timeline_path,
    default_syncfree_paths,
    finalize_syncfree_eye,
    merge_self_kerr_refs,
    maybe_load_kerr_refs,
    resolve_syncfree_working_csv,
    run_syncfree_ellipses_for_eye,
    syncfree_mapped_is_stale,
    video_path_for_eye,
    write_syncfree_draft,
)

TEST_TAG = "_pytest_syncfree"


@pytest.fixture
def syncfree_test_tag() -> str:
    return TEST_TAG


def _cleanup_syncfree_artifacts(blocksync, tag: str) -> None:
    for eye in ("left", "right"):
        video = video_path_for_eye(blocksync, eye)
        paths = default_syncfree_paths(video, eye, tag)
        for path in paths.values():
            if path.is_file():
                path.unlink()
        legacy_names = (
            f"{eye}_syncfree_{tag}_ellipses.csv",
            f"{eye}_syncfree_{tag}_verified.csv",
            f"{eye}_syncfree_{tag}_kerr_angles.csv",
            f"{eye}_syncfree_{tag}_degrees.csv",
        )
        for name in legacy_names:
            p = video.parent / name
            if p.is_file():
                p.unlink()
    ap = Path(blocksync.analysis_path)
    for eye in ("left", "right"):
        for path in (
            analysis_syncfree_timeline_path(blocksync, eye, tag),
            ap / f"{eye}_eye_degrees_from_syncfree_{tag}.csv",
        ):
            if path.is_file():
                path.unlink()


def test_merge_self_kerr_refs_preserves_other_eye(tmp_path: Path):
    block = SimpleNamespace(
        analysis_path=tmp_path,
        kerr_ref_l_x=None,
        kerr_ref_l_y=None,
        kerr_ref_r_x=None,
        kerr_ref_r_y=None,
    )
    merge_self_kerr_refs(block, "left", 10, 20)
    merge_self_kerr_refs(block, "right", 30, 40)
    row = pd.read_csv(tmp_path / "self_kerr_refs.csv").iloc[0]
    assert int(row["kerr_ref_l_x"]) == 10
    assert int(row["kerr_ref_l_y"]) == 20
    assert int(row["kerr_ref_r_x"]) == 30
    assert int(row["kerr_ref_r_y"]) == 40


def test_maybe_load_kerr_refs_from_csv(tmp_path: Path):
    block = SimpleNamespace(
        analysis_path=tmp_path,
        kerr_ref_l_x=None,
        kerr_ref_l_y=None,
        kerr_ref_r_x=None,
        kerr_ref_r_y=None,
    )
    merge_self_kerr_refs(block, "left", 11, 22)
    kx, ky = maybe_load_kerr_refs(block, "left")
    assert (kx, ky) == (11, 22)


def test_syncfree_finalize_writes_unified_artifacts(sample_block, syncfree_test_tag: str):
    """Finalize produces eye_data + kerr_refs + meta (+ optional timeline)."""
    from eye_tracking_system_tools.preprocessing.block_sync_core import (
        load_final_sync_df,
    )

    block = sample_block
    load_final_sync_df(block, verbose=False)
    block.handle_eye_videos()
    tag = syncfree_test_tag
    thr = 0.95
    _cleanup_syncfree_artifacts(block, tag)

    try:
        for eye in ("left", "right"):
            video = video_path_for_eye(block, eye)
            df_ell, meta = run_syncfree_ellipses_for_eye(block, eye, thr)
            write_syncfree_draft(df_ell, meta, video, eye, tag)
            kx, ky = maybe_load_kerr_refs(block, eye)
            written = finalize_syncfree_eye(
                block,
                eye,
                df_ell,
                (kx, ky),
                tag=tag,
                map_to_timeline=True,
            )
            paths = default_syncfree_paths(video, eye, tag)
            assert paths["eye_data"].is_file()
            assert paths["kerr_refs"].is_file()
            assert paths["meta"].is_file()
            assert not paths["draft"].is_file()
            assert written["timeline"].is_file()

            eye_data = pd.read_csv(paths["eye_data"])
            assert "eye_frame" in eye_data.columns
            assert "k_r" in eye_data.columns
            assert "center_x" in eye_data.columns
            timeline = pd.read_csv(written["timeline"])
            assert "OE_timestamp" in timeline.columns
            assert "ms_axis" in timeline.columns
    finally:
        _cleanup_syncfree_artifacts(block, tag)


def test_resolve_syncfree_working_csv_prefers_draft(sample_block, syncfree_test_tag: str):
    block = sample_block
    block.handle_eye_videos()
    tag = syncfree_test_tag
    _cleanup_syncfree_artifacts(block, tag)
    try:
        video = video_path_for_eye(block, "left")
        df_ell, meta = run_syncfree_ellipses_for_eye(block, "left", 0.95)
        write_syncfree_draft(df_ell, meta, video, "left", tag)
        src = resolve_syncfree_working_csv(video, "left", tag)
        assert src is not None
        assert src.name.endswith("_draft.csv")
    finally:
        _cleanup_syncfree_artifacts(block, tag)


def test_syncfree_tab_loads_verifiers_from_draft(
    sample_block_path: Path, qapp_session, tmp_path: Path
):
    ensure_config_template(tmp_path)
    config = load_config(None, tmp_path)
    config.syncfree_artifact_tag = TEST_TAG
    state = GuiState(output_folder=tmp_path)
    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=sample_block_path,
        path_to_animal_folder=sample_block_path.parents[2],
    )
    tab = SyncFreeTab(state, config)
    tab._artifact_tag.setText(TEST_TAG)
    tab.set_block(block)

    blocksync = tab._require_blocksync()
    tag = TEST_TAG
    for eye in ("left", "right"):
        video = video_path_for_eye(blocksync, eye)
        df_ell, meta = run_syncfree_ellipses_for_eye(blocksync, eye, 0.95)
        write_syncfree_draft(df_ell, meta, video, eye, tag)
    tab._try_load_verifiers()
    assert tab._left_verifier is not None
    assert tab._right_verifier is not None
    _cleanup_syncfree_artifacts(blocksync, TEST_TAG)


def test_syncfree_mapped_is_stale_false_when_missing(sample_block_path: Path):
    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=sample_block_path,
        path_to_animal_folder=sample_block_path.parents[2],
    )
    assert not syncfree_mapped_is_stale(block, "left", "missing_tag")
