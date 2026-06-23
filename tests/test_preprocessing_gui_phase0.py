"""Phase 0 smoke tests for the Preprocessing GUI.

Verifies:

* The new ``preprocessing/notebook_helpers`` module exposes every helper
  the plan promotes out of the notebooks.
* The notebooks have been edited so they no longer carry an inline copy
  of those helpers (the refactor stuck).
* The GUI's package imports cleanly and the MainWindow instantiates
  headlessly with all five tabs declaring a sensible status signature.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROC_DIR = REPO_ROOT / "src" / "eye_tracking_system_tools" / "preprocessing"


# --- Helper-module shape ------------------------------------------------


PROMOTED_NAMES = [
    "_normalize_to_seconds",
    "_read_eye_internal_seconds",
    "_locate_eye_timestamps_csv",
    "_delta_analysis",
    "_first_ttl_sample",
    "_get_fs",
    "_assert_strictly_increasing",
    "_nearest_with_tol",
    "_shift_eye_df_by_index",
    "build_eye_df_simple",
    "simple_sync_build",
    "describe_eye_tick",
    "shift_eye_df_by_index",
    "build_arena_grid_df",
    "ArenaGridInfo",
    "_infer_ttl_fps",
    "_build_arena_grid",
    "build_final_sync_df_merge_nearest",
    "verify_final_df_against_sources",
    "export_final_sync_df",
    "load_final_sync_df",
    "plot_simple_sync_bokeh",
    "hover_inspect_eyes_bokeh",
    "sanity_plot_final_df",
    "bokeh_plotter",
    "_normalize_insert_positions",
    "insert_duplicate_frames_slide",
    "insert_dup_by_pos",
    "insert_dup_by_oe_sample",
    "find_jittery_frames",
    "add_intermediate_elements",
    "export_eye_data_2d",
    "rolling_window_analysis",
    "create_behavior_df",
]


def test_notebook_helpers_exports_every_promoted_name():
    from eye_tracking_system_tools.preprocessing import notebook_helpers as nh

    missing = [n for n in PROMOTED_NAMES if not hasattr(nh, n)]
    assert not missing, f"notebook_helpers is missing: {missing}"


def test_promoted_helpers_are_callable():
    from eye_tracking_system_tools.preprocessing import notebook_helpers as nh

    callable_names = [
        "simple_sync_build",
        "describe_eye_tick",
        "shift_eye_df_by_index",
        "build_arena_grid_df",
        "build_final_sync_df_merge_nearest",
        "verify_final_df_against_sources",
        "export_final_sync_df",
        "load_final_sync_df",
        "plot_simple_sync_bokeh",
        "hover_inspect_eyes_bokeh",
        "sanity_plot_final_df",
        "bokeh_plotter",
        "insert_dup_by_pos",
        "insert_dup_by_oe_sample",
        "rolling_window_analysis",
        "create_behavior_df",
    ]
    not_callable = [n for n in callable_names if not callable(getattr(nh, n))]
    assert not not_callable, f"Not callable: {not_callable}"


# --- Notebook refactor stuck --------------------------------------------


def _read_nb_source(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source") or []
        if isinstance(src, str):
            out.append(src)
        else:
            out.append("".join(src))
    return "\n\n".join(out)


def test_block_sync_notebook_imports_from_notebook_helpers():
    src = _read_nb_source(PREPROC_DIR / "block_synchronization.ipynb")
    assert (
        "from eye_tracking_system_tools.preprocessing.notebook_helpers import"
        in src
    ), "block_synchronization.ipynb should import from notebook_helpers"


def test_block_sync_notebook_no_longer_defines_promoted_helpers():
    src = _read_nb_source(PREPROC_DIR / "block_synchronization.ipynb")
    sentinel_defs = [
        "def simple_sync_build(",
        "def build_arena_grid_df(",
        "def build_final_sync_df_merge_nearest(",
        "def plot_simple_sync_bokeh(",
        "def insert_duplicate_frames_slide(",
    ]
    leftover = [d for d in sentinel_defs if d in src]
    assert not leftover, (
        f"These helper defs should have been removed from "
        f"block_synchronization.ipynb: {leftover}"
    )


def test_accelerometer_notebook_imports_from_notebook_helpers():
    src = _read_nb_source(
        PREPROC_DIR / "add_accelerometer_state_annotations.ipynb"
    )
    assert (
        "from eye_tracking_system_tools.preprocessing.notebook_helpers import"
        in src
    )
    sentinel_defs = ["def rolling_window_analysis(", "def create_behavior_df("]
    leftover = [d for d in sentinel_defs if d in src]
    assert not leftover, (
        f"These helper defs should have been removed from accelerometer "
        f"notebook: {leftover}"
    )


# --- Notebooks still compile-check OK -----------------------------------


def _compile_notebook_source(path: Path) -> None:
    src = _read_nb_source(path)
    # strip Jupyter magics + shell escapes so we can compile() the result
    src = re.sub(r"(?m)^\s*%[a-zA-Z].*$", "", src)
    src = re.sub(r"(?m)^\s*!.*$", "", src)
    compile(src, str(path), "exec")


def test_block_sync_notebook_compiles():
    _compile_notebook_source(PREPROC_DIR / "block_synchronization.ipynb")


def test_accelerometer_notebook_compiles():
    _compile_notebook_source(
        PREPROC_DIR / "add_accelerometer_state_annotations.ipynb"
    )


# --- GUI package smoke load ---------------------------------------------


def test_preprocessing_gui_package_imports():
    """Importing the GUI package + tabs must not raise at module level."""
    import eye_tracking_system_tools.annotation.preprocessing_gui  # noqa: F401
    from eye_tracking_system_tools.annotation.preprocessing_gui import app  # noqa: F401
    from eye_tracking_system_tools.annotation.preprocessing_gui import (
        block_picker,  # noqa: F401
        config_io,  # noqa: F401
        models,  # noqa: F401
        status_bus,  # noqa: F401
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs import (
        BehaviorTab,
        KerrTab,
        SyncFreeTab,
        SyncTab,
        VerifyTab,
    )
    assert SyncTab.tab_id == "sync"
    assert VerifyTab.tab_id == "verify"
    assert KerrTab.tab_id == "kerr"
    assert BehaviorTab.tab_id == "behavior"
    assert SyncFreeTab.tab_id == "syncfree"


def test_status_signature_lists_paths(sample_block_path, qapp_session, tmp_path):
    """Each tab declares a status_signature returning a list of pathlib.Path."""
    from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
        ensure_config_template,
        load_config,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
        BlockHandle,
        GuiState,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs import (
        BehaviorTab,
        KerrTab,
        SyncFreeTab,
        SyncTab,
        VerifyTab,
    )

    ensure_config_template(tmp_path)
    config = load_config(None, tmp_path)
    state = GuiState(output_folder=tmp_path)

    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=sample_block_path,
        path_to_animal_folder=sample_block_path.parents[2],
    )

    for cls in (SyncTab, VerifyTab, KerrTab, BehaviorTab, SyncFreeTab):
        tab = cls(state, config)
        sig = tab.status_signature(block)
        assert isinstance(sig, list)
        for p in sig:
            assert isinstance(p, Path)


def test_main_window_instantiates_headlessly(
    sample_block_path,
    qapp_session,
    tmp_path,
):
    """PreprocessingGuiWindow constructs and shows 5 tabs without a display."""
    from eye_tracking_system_tools.annotation.preprocessing_gui.app import (
        PreprocessingGuiWindow,
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
    )

    blocks = discover_blocks(sample_block_path.parents[2], "PV_106", ["015"])
    assert blocks, "Discovery failed for sample block."

    ensure_config_template(tmp_path)
    config = load_config(None, tmp_path)
    state = GuiState(
        experiment_path=sample_block_path.parents[2],
        blocks=blocks,
        output_folder=tmp_path,
    )

    win = PreprocessingGuiWindow(state, config, tmp_path / "preproc_gui_config.yaml")
    try:
        assert win._tab_widget.count() == 5
        labels = [
            win._tab_widget.tabText(i) for i in range(win._tab_widget.count())
        ]
        assert labels == ["Sync", "Verify", "Kerr", "Behavior", "Sync-free"]
    finally:
        win.close()


def test_behavior_tab_disables_when_lizmov_missing(
    sample_block_path,
    qapp_session,
    tmp_path,
):
    """Per open-item #5, no lizMov.mat -> tab disabled with tooltip."""
    from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
        load_config,
        ensure_config_template,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
        BlockHandle,
        GuiState,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.behavior_tab import (
        BehaviorTab,
        has_liz_mov,
    )

    ensure_config_template(tmp_path)
    cfg = load_config(None, tmp_path)
    state = GuiState(output_folder=tmp_path)

    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=sample_block_path,
        path_to_animal_folder=sample_block_path.parents[2],
    )
    # Sample block has no lizMov.mat per the spec.
    assert not has_liz_mov(block)

    tab = BehaviorTab(state, cfg)
    tab.set_block(block)
    assert not tab._btn_load.isEnabled()
    assert "lizMov" in tab.toolTip()
    assert "lizMov" in tab._lizmov_banner.text()


def test_simple_sync_build_matches_notebook(sample_block):
    """Equivalence check: helpers from notebook_helpers reproduce the
    schema/row-count of the sample block's saved simple-sync CSVs."""
    import pandas as pd

    from eye_tracking_system_tools.preprocessing.notebook_helpers import (
        simple_sync_build,
    )

    sample_block.handle_eye_videos()
    sample_block.parse_open_ephys_events()
    sample_block.handle_arena_files()
    sample_block.get_eye_brightness_vectors(use_auto_roi=True, create_if_missing=False)

    dfL, dfR = simple_sync_build(sample_block, export=False)
    expected_cols = {"frame_idx", "brightness", "oe_time_s"}
    assert expected_cols.issubset(dfL.columns)
    assert expected_cols.issubset(dfR.columns)

    saved_l = pd.read_csv(sample_block.analysis_path / "eye_left_simple_sync.csv")
    saved_r = pd.read_csv(sample_block.analysis_path / "eye_right_simple_sync.csv")
    assert len(dfL) == len(saved_l)
    assert len(dfR) == len(saved_r)
