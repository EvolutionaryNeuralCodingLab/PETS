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
            "_btn_extract_brightness",
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
