#!/usr/bin/env python
"""Headless smoke-load of the Preprocessing GUI.

Mirrors ``scripts/validate_block_annotator_load.py``: instantiates every tab
on the sample block (offscreen QPA), confirms the tabs declare a sensible
``status_signature`` and that the StatusBus updates the per-tab status icon.
Exits 0 on success, non-zero on any structural failure.

Usage::

    python scripts/validate_preprocessing_gui_load.py \\
        --experiment-path D:\\sample_data_for_eye_repo \\
        --animal PV_106 --block 015
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--animal", type=str, required=True)
    parser.add_argument(
        "--block",
        action="append",
        required=True,
        help="Block number to load. Pass multiple times for batch mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder (config + logs). Defaults to a temp dir.",
    )
    args = parser.parse_args()

    # Offscreen QPA so this works on CI / headless boxes.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    output_folder = args.output or Path(tempfile.mkdtemp(prefix="pets_preproc_gui_"))

    from PyQt6 import QtWidgets

    from eye_tracking_system_tools.annotation.preprocessing_gui.app import (
        PreprocessingGuiWindow,
        _TAB_CLASSES,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.block_picker import (
        discover_blocks,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
        ensure_config_template,
        load_config,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
        GuiState,
        StageStatus,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    blocks = discover_blocks(args.experiment_path, args.animal, args.block)
    if not blocks:
        print(
            f"[FAIL] No blocks discovered under "
            f"{args.experiment_path / args.animal} for ids {args.block}"
        )
        return 1
    print(f"[OK] Discovered {len(blocks)} block(s):")
    for b in blocks:
        print(f"  {b.display_label} -> {b.block_path}")

    ensure_config_template(output_folder)
    config = load_config(None, output_folder)
    config_path = output_folder / "preproc_gui_config.yaml"

    state = GuiState(
        experiment_path=args.experiment_path,
        blocks=blocks,
        current_index=0,
        output_folder=output_folder,
    )
    state.ensure_session()

    win = PreprocessingGuiWindow(state, config, config_path)
    expected_ids = [cls.tab_id for cls in _TAB_CLASSES]
    if list(win._tabs.keys()) != expected_ids:
        print(
            f"[FAIL] Tab order mismatch: got {list(win._tabs.keys())}, "
            f"expected {expected_ids}"
        )
        return 2

    ok = True
    for tab_id, tab in win._tabs.items():
        sig = tab.status_signature(blocks[0])
        if not isinstance(sig, list):
            print(f"[FAIL] {tab_id}.status_signature did not return a list: {sig!r}")
            ok = False
            continue
        for p in sig:
            if not isinstance(p, Path):
                print(f"[FAIL] {tab_id}.status_signature contained non-Path {p!r}")
                ok = False
        status = win._status_bus.status_for(tab_id)
        if not isinstance(status, StageStatus):
            print(f"[FAIL] {tab_id} StatusBus returned non-StageStatus {status!r}")
            ok = False
        print(f"[OK] {tab_id}: signature={[p.name for p in sig]} status={status.value}")

    for tab_id, tab in win._tabs.items():
        if not hasattr(tab, "_btn_load_prev"):
            print(f"[FAIL] {tab_id} missing Load prev analysis button.")
            ok = False
        else:
            print(f"[OK] {tab_id}: Load prev analysis button present.")

    picker = win._block_picker
    for attr in ("_btn_add", "_btn_release"):
        if not hasattr(picker, attr):
            print(f"[FAIL] BlockPicker missing {attr}.")
            ok = False
    if hasattr(picker, "_btn_add") and hasattr(picker, "_btn_release"):
        print("[OK] BlockPicker add/release controls present.")

    # Phase 1: Sync tab should expose deterministic-stage widgets.
    sync_tab = win._tabs.get("sync")
    if sync_tab is None:
        print("[FAIL] Sync tab missing from tab registry.")
        ok = False
    else:
        required_attrs = [
            "_steps",
            "_stack",
            "_btn_prepare",
            "_btn_parse_oe",
            "_btn_manual_ttl",
            "_btn_extract_brightness",
            "_btn_preview_brightness",
            "_btn_manual_roi",
            "_btn_build_arena_grid",
            "_btn_build_simple_sync",
            "_btn_open_shift",
            "_btn_apply_shifts",
            "_btn_apply_insertions",
            "_btn_build_final",
            "_btn_verify_final",
            "_btn_export_final",
            "_sanity_plot",
            "_verify_text",
        ]
        missing = [name for name in required_attrs if not hasattr(sync_tab, name)]
        if missing:
            print(f"[FAIL] Sync tab missing expected Phase 1 widgets: {missing}")
            ok = False
        else:
            print("[OK] Sync tab Phase 1 widgets instantiated.")

        phase2_attrs = [
            "_btn_read_dlc",
            "_btn_jitter_report",
            "_btn_correct_jitter",
            "_btn_preview_jitter",
            "_btn_apply_jitter",
            "_btn_finalize_eye",
            "_btn_load_prev",
            "_jitter_plot_left",
            "_jitter_plot_right",
        ]
        missing_p2 = [name for name in phase2_attrs if not hasattr(sync_tab, name)]
        if missing_p2:
            print(f"[FAIL] Sync tab missing expected Phase 2 widgets: {missing_p2}")
            ok = False
        else:
            print("[OK] Sync tab Phase 2 widgets instantiated.")
            batch_attrs = ["_batch_op", "_btn_batch_run", "_btn_batch_cancel", "_batch_log"]
            missing_batch = [n for n in batch_attrs if not hasattr(sync_tab, n)]
            if missing_batch:
                print(f"[FAIL] Sync tab missing batch widgets: {missing_batch}")
                ok = False
            else:
                print("[OK] Sync tab batch panel instantiated.")

        phase3_attrs = ["_btn_manual_ttl", "_btn_preview_brightness", "_btn_manual_roi"]
        missing_p3 = [name for name in phase3_attrs if not hasattr(sync_tab, name)]
        if missing_p3:
            print(f"[FAIL] Sync tab missing expected Phase 3 widgets: {missing_p3}")
            ok = False
        else:
            print("[OK] Sync tab Phase 3 manual-fallback widgets instantiated.")
            try:
                from eye_tracking_system_tools.annotation.preprocessing_gui.manual_ttl_dialog import (
                    ManualTtlDialog,
                )
                from eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker import (
                    QtRoiPickerDialog,
                )
            except ImportError as e:
                print(f"[FAIL] Phase 3 modules import error: {e}")
                ok = False
            else:
                print("[OK] Phase 3 fallback modules import cleanly.")

    verify_tab = win._tabs.get("verify")
    if verify_tab is None:
        print("[FAIL] Verify tab missing from tab registry.")
        ok = False
    else:
        verify_attrs = ["_stale_banner", "_btn_save", "_verifier_host"]
        missing_v = [n for n in verify_attrs if not hasattr(verify_tab, n)]
        if missing_v:
            print(f"[FAIL] Verify tab missing expected Phase 4 attrs: {missing_v}")
            ok = False
        else:
            print("[OK] Verify tab Phase 4 shell instantiated.")
            try:
                from eye_tracking_system_tools.annotation.preprocessing_gui.ellipse_verifier import (
                    EllipseVerifierWidget,
                )
            except ImportError as e:
                print(f"[FAIL] ellipse_verifier import error: {e}")
                ok = False
            else:
                print("[OK] EllipseVerifierWidget imports cleanly.")
        if blocks:
            try:
                verify_tab.set_block(blocks[0])
                verify_tab._on_load_prev_analysis()
                if verify_tab._left_verifier is None or verify_tab._right_verifier is None:
                    print(
                        "[WARN] Verify tab verifiers not loaded (missing eye CSVs/videos on sample block)."
                    )
                else:
                    print("[OK] Verify tab loaded dual EllipseVerifierWidget instances.")
                    for verifier in (
                        verify_tab._left_verifier,
                        verify_tab._right_verifier,
                    ):
                        for attr in ("_btn_save", "_btn_quit"):
                            if hasattr(verifier, attr):
                                print(
                                    f"[FAIL] EllipseVerifierWidget should not expose {attr} "
                                    "(single tab-level Save only)."
                                )
                                ok = False
            except Exception as e:
                print(f"[WARN] Verify tab set_block: {e}")

    kerr_tab = win._tabs.get("kerr")
    if kerr_tab is None:
        print("[FAIL] Kerr tab missing from tab registry.")
        ok = False
    else:
        kerr_attrs = [
            "_name_tag",
            "_btn_calculate",
            "_btn_export",
            "_btn_batch",
            "_refs_banner",
        ]
        missing_k = [n for n in kerr_attrs if not hasattr(kerr_tab, n)]
        if missing_k:
            print(f"[FAIL] Kerr tab missing expected Phase 5 widgets: {missing_k}")
            ok = False
        else:
            print("[OK] Kerr tab Phase 5 widgets instantiated.")
        if blocks:
            try:
                kerr_tab.set_block(blocks[0])
                print("[OK] Kerr tab set_block completed.")
            except Exception as e:
                print(f"[WARN] Kerr tab set_block: {e}")

    behavior_tab = win._tabs.get("behavior")
    if behavior_tab is None:
        print("[FAIL] Behavior tab missing from tab registry.")
        ok = False
    else:
        behavior_attrs = [
            "_movement_plot",
            "_btn_load",
            "_btn_compute",
            "_btn_rolling",
            "_btn_export",
            "_threshold",
            "_lizmov_banner",
            "_calib_path_edit",
            "_headstage_combo",
        ]
        missing_b = [n for n in behavior_attrs if not hasattr(behavior_tab, n)]
        if missing_b:
            print(f"[FAIL] Behavior tab missing expected Phase 6 widgets: {missing_b}")
            ok = False
        else:
            print("[OK] Behavior tab Phase 6 widgets instantiated.")
        if blocks:
            try:
                behavior_tab.set_block(blocks[0])
                print("[OK] Behavior tab set_block completed (PV_106 — expect disabled workflow).")
            except Exception as e:
                print(f"[WARN] Behavior tab set_block: {e}")

    # Optional: smoke Behavior tab with lizMov-bearing block when present.
    behavior_blocks = discover_blocks(args.experiment_path, "PV_126", ["006"])
    if behavior_blocks:
        try:
            behavior_tab.set_block(behavior_blocks[0])
            if behavior_tab._btn_load.isEnabled():
                n_items = len(
                    behavior_tab._movement_plot.plot_widget.listDataItems()
                )
                if behavior_tab._rolling_df is not None and n_items >= 1:
                    print(
                        "[OK] Behavior tab enabled for PV_126 / block_006 "
                        f"(lizMov present, {len(behavior_tab._rolling_df)} windows plotted)."
                    )
                else:
                    print(
                        "[FAIL] PV_126 block_006 loaded but movement trace not plotted."
                    )
                    ok = False
            else:
                print("[WARN] PV_126 block_006 found but lizMov workflow disabled.")
        except Exception as e:
            print(f"[WARN] Behavior tab PV_126 set_block: {e}")

    syncfree_tab = win._tabs.get("syncfree")
    if syncfree_tab is None:
        print("[FAIL] Sync-free tab missing from tab registry.")
        ok = False
    else:
        syncfree_attrs = [
            "_artifact_tag",
            "_btn_ellipses",
            "_btn_finalize",
            "_btn_save_draft",
            "_stale_banner",
        ]
        missing_s = [n for n in syncfree_attrs if not hasattr(syncfree_tab, n)]
        if missing_s:
            print(f"[FAIL] Sync-free tab missing expected Phase 7 widgets: {missing_s}")
            ok = False
        else:
            print("[OK] Sync-free tab Phase 7 widgets instantiated.")
        if blocks:
            try:
                syncfree_tab.set_block(blocks[0])
                print("[OK] Sync-free tab set_block completed.")
            except Exception as e:
                print(f"[WARN] Sync-free tab set_block: {e}")

    explore_tab = win._tabs.get("explore")
    if explore_tab is None:
        print("[FAIL] Data Exploration tab missing from tab registry.")
        ok = False
    else:
        explore_attrs = [
            "_plot",
            "_video",
            "_time_label",
            "_stale_banner",
            "_btn_load_prev",
        ]
        missing_e = [n for n in explore_attrs if not hasattr(explore_tab, n)]
        if missing_e:
            print(f"[FAIL] Explore tab missing expected widgets: {missing_e}")
            ok = False
        else:
            print("[OK] Data Exploration tab Phase 0/1 widgets instantiated.")
            plot = explore_tab._plot
            if not hasattr(plot, "_box_zoom_btn"):
                print("[FAIL] Explore plot missing box-zoom toggle.")
                ok = False
            else:
                print("[OK] Explore plot box-zoom control present.")
            video = explore_tab._video
            video_attrs = [
                "_left_panel",
                "_arena_panel",
                "_right_panel",
                "_playback",
                "_slider",
            ]
            missing_v = [n for n in video_attrs if not hasattr(video, n)]
            if missing_v:
                print(f"[FAIL] Explore video panel missing: {missing_v}")
                ok = False
            else:
                print("[OK] Explore video LE/Arena/RE panels present.")
        if blocks:
            try:
                explore_tab.set_block(blocks[0])
                print("[OK] Explore tab set_block completed.")
            except Exception as e:
                print(f"[WARN] Explore tab set_block: {e}")

    # Touch the close path so persistence runs without crashing.
    try:
        win._persist_defaults()
        print(f"[OK] Persisted defaults to {config_path}")
    except Exception as e:
        print(f"[FAIL] _persist_defaults raised: {e}")
        ok = False

    win.close()
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
