"""Phase 1 tests for the Preprocessing GUI Sync tab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    StageStatus,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.status_bus import StatusBus
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.sync_tab import SyncTab
from eye_tracking_system_tools.preprocessing.notebook_helpers import (
    build_arena_grid_df,
    build_final_sync_df_merge_nearest,
    shift_eye_df_by_index,
    simple_sync_build,
    verify_final_df_against_sources,
)


def test_promoted_helpers_match_notebook(sample_block):
    sample_block.handle_eye_videos()
    sample_block.parse_open_ephys_events()
    sample_block.handle_arena_files()
    sample_block.get_eye_brightness_vectors(use_auto_roi=True, create_if_missing=False)

    df_l, df_r = simple_sync_build(sample_block, export=False)
    assert {"frame_idx", "brightness", "oe_time_s"}.issubset(df_l.columns)
    assert {"frame_idx", "brightness", "oe_time_s"}.issubset(df_r.columns)

    arena_grid_df, _ = build_arena_grid_df(sample_block, target_fps=60.0, arena_fps_tol_hz=5.0)
    assert {"grid_time_s", "Arena_frame_60", "Arena_frame_src", "Arena_ttl_src"}.issubset(
        arena_grid_df.columns
    )

    final_df = build_final_sync_df_merge_nearest(
        sample_block,
        df_l,
        df_r,
        target_fps=60.0,
        tol_frac=0.9,
        export_csv=False,
    )
    expected_final = pd.read_csv(sample_block.analysis_path / "final_sync_df.csv")
    assert set(final_df.columns).issubset(set(expected_final.columns))
    assert len(final_df) == len(expected_final)

    stats = verify_final_df_against_sources(sample_block, final_df, df_l, df_r, target_fps=60.0, tol_frac=0.9)
    expected_keys = {
        "tick_ms",
        "tol_samples",
        "left_frame_match",
        "left_values_match",
        "right_frame_match",
        "right_values_match",
    }
    assert set(stats) == expected_keys
    assert all(0.0 <= float(stats[k]) <= 1.0 for k in stats if k.endswith("_match"))


def test_shift_eye_df_by_index_inverse():
    idx = np.arange(1000, 1100)
    df = pd.DataFrame(
        {
            "frame_idx": np.arange(100),
            "brightness": np.linspace(0.0, 1.0, 100),
            "oe_time_s": np.linspace(0.0, 1.65, 100),
        },
        index=idx,
    )
    shifted = shift_eye_df_by_index(df, 7)
    restored = shift_eye_df_by_index(shifted, -7)

    for col in ("frame_idx", "brightness", "oe_time_s"):
        a = df[col].to_numpy(dtype=float)
        b = restored[col].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[mask], b[mask], rtol=0.0, atol=1e-12)


def test_status_signature_sync(qapp_session, tmp_path):
    block_path = tmp_path / "PV_106" / "2025_09_04" / "block_015"
    analysis = block_path / "analysis"
    analysis.mkdir(parents=True)

    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=block_path,
        path_to_animal_folder=tmp_path,
    )

    class _DummyState:
        pass

    class _DummyCfg:
        arena_target_fps = 60.0
        arena_fps_tol_hz = 5.0
        final_sync_tol_frac = 0.9

    tab = SyncTab(_DummyState(), _DummyCfg())
    bus = StatusBus()
    bus.register_tab("sync", tab.status_signature, upstream_tabs=[])
    bus.set_block(block)
    assert bus.status_for("sync") == StageStatus.NOT_STARTED

    (analysis / "final_sync_df.csv").write_text("x\n1\n", encoding="utf-8")
    (analysis / "left_eye_data.csv").write_text("x\n1\n", encoding="utf-8")
    (analysis / "right_eye_data.csv").write_text("x\n1\n", encoding="utf-8")
    bus.refresh_all()
    assert bus.status_for("sync") == StageStatus.COMPLETE


def test_sync_tab_phase1_widgets_construct(qapp_session):
    class _State:
        df_left_simple_sync = None
        df_right_simple_sync = None
        arena_grid_df = None
        final_sync_df = None

    class _Cfg:
        arena_target_fps = 60.0
        arena_fps_tol_hz = 5.0
        final_sync_tol_frac = 0.9

    tab = SyncTab(_State(), _Cfg())
    assert tab._steps.count() == 5
    assert tab._stack.count() == 5
    assert tab._btn_prepare.text()
    assert tab._btn_parse_oe.text()
    assert tab._btn_extract_brightness.text()
    assert tab._btn_build_arena_grid.text()
    assert tab._btn_build_simple_sync.text()
    assert tab._btn_open_shift.text()
    assert tab._btn_apply_shifts.text()
    assert tab._btn_apply_insertions.text()
    assert tab._btn_build_final.text()
    assert tab._btn_verify_final.text()
    assert tab._btn_export_final.text()


def test_sync_tab_refresh_shift_label_uses_float_ticks(qapp_session):
    class _State:
        arena_grid_df = None
        final_sync_df = None
        df_left_simple_sync = pd.DataFrame(
            {"oe_time_s": [0.0, 0.02, 0.04], "frame_idx": [1, 2, 3], "brightness": [1.0, 2.0, 3.0]},
            index=[10, 20, 30],
        )
        df_right_simple_sync = pd.DataFrame(
            {"oe_time_s": [0.0, 0.02, 0.04], "frame_idx": [1, 2, 3], "brightness": [1.0, 2.0, 3.0]},
            index=[11, 21, 31],
        )

    class _Cfg:
        arena_target_fps = 60.0
        arena_fps_tol_hz = 5.0
        final_sync_tol_frac = 0.9

    tab = SyncTab(_State(), _Cfg())
    tab._left_shift.setValue(2)
    tab._right_shift.setValue(-1)
    tab._refresh_shift_ms_labels()
    assert "~40.000 ms" in tab._left_shift_ms.text()
    assert "~-20.000 ms" in tab._right_shift_ms.text()


def test_load_eye_tracking_df_csv_drops_level_0(tmp_path):
    from eye_tracking_system_tools.preprocessing.block_sync_core import (
        load_eye_tracking_df_csv,
    )

    path = tmp_path / "le_df.csv"
    path.write_text(
        "level_0,Arena_TTL,L_eye_frame,center_x,center_y\n"
        "0,100,1,10.0,20.0\n"
        "1,101,2,11.0,21.0\n",
        encoding="utf-8",
    )
    df = load_eye_tracking_df_csv(path)
    assert "level_0" not in df.columns
    assert "Arena_TTL" in df.columns
    assert len(df) == 2


def test_status_signature_eye_data_csv(qapp_session, tmp_path):
    """COMPLETE when left/right_eye_data.csv exist (Phase 2 signature)."""
    block_path = tmp_path / "PV_106" / "2025_09_04" / "block_015"
    analysis = block_path / "analysis"
    analysis.mkdir(parents=True)

    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=block_path,
        path_to_animal_folder=tmp_path,
    )

    class _DummyState:
        pass

    class _DummyCfg:
        arena_target_fps = 60.0
        arena_fps_tol_hz = 5.0
        final_sync_tol_frac = 0.9
        dlc_threshold_to_use = 0.95
        jitter_max_distance = 60
        jitter_diff_threshold = 5
        jitter_gap_to_bridge = 24

    tab = SyncTab(_DummyState(), _DummyCfg())
    bus = StatusBus()
    bus.register_tab("sync", tab.status_signature, upstream_tabs=[])
    bus.set_block(block)
    assert bus.status_for("sync") == StageStatus.NOT_STARTED

    (analysis / "final_sync_df.csv").write_text("x\n", encoding="utf-8")
    (analysis / "left_eye_data.csv").write_text("x\n", encoding="utf-8")
    (analysis / "right_eye_data.csv").write_text("x\n", encoding="utf-8")
    bus.refresh_all()
    assert bus.status_for("sync") == StageStatus.COMPLETE


def test_sync_tab_phase2_widgets_construct(qapp_session):
    class _State:
        df_left_simple_sync = None
        df_right_simple_sync = None
        arena_grid_df = None
        final_sync_df = None

    class _Cfg:
        arena_target_fps = 60.0
        arena_fps_tol_hz = 5.0
        final_sync_tol_frac = 0.9
        dlc_threshold_to_use = 0.95
        jitter_max_distance = 60
        jitter_diff_threshold = 5
        jitter_gap_to_bridge = 24

    tab = SyncTab(_State(), _Cfg())
    assert tab._btn_read_dlc.text()
    assert tab._dlc_le_combo is not None
    assert tab._dlc_re_combo is not None
    assert tab._dlc_overwrite is not None
    assert tab._dlc_overwrite.isChecked() is False
    assert tab._btn_likelihood_hist.text()
    assert tab._btn_jitter_report.text()
    assert tab._jitter_overwrite is not None
    assert tab._jitter_overwrite.isChecked() is False
    assert tab._btn_correct_jitter.text()
    assert tab._btn_preview_jitter.text()
    assert tab._btn_finalize_eye.text()
    assert tab._jitter_plot_left is not None
    assert tab._jitter_plot_right is not None
    assert tab._batch_op.count() == 4
    assert tab._btn_batch_run.text()
