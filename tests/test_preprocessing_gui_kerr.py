"""Phase 5 tests — Kerr tab."""

from __future__ import annotations

from pathlib import Path

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
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.kerr_tab import (
    KerrTab,
)


def test_kerr_tab_produces_csvs(sample_block_path, qapp_session, tmp_path):
    analysis = sample_block_path / "analysis"
    if not (analysis / "left_eye_data.csv").is_file():
        pytest.skip("Sample eye data not present.")
    if not (analysis / "self_kerr_refs.csv").is_file():
        pytest.skip("self_kerr_refs.csv not present — run Verify tab first.")

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

    tab = KerrTab(state, config)
    tab.set_block(block)
    tab._name_tag.setText("gui_test")

    tab._run_calculate()
    if tab._worker is not None:
        tab._worker.wait()

    tag = "gui_test"
    left_angle = analysis / f"left_kerr_angle_{tag}.csv"
    right_angle = analysis / f"right_kerr_angle_{tag}.csv"
    assert left_angle.is_file() and right_angle.is_file()

    tab._run_export_merged()
    left_merged = analysis / f"left_eye_data_{tag}.csv"
    right_merged = analysis / f"right_eye_data_{tag}.csv"
    assert left_merged.is_file() and right_merged.is_file()

    ldf = pd.read_csv(left_merged, index_col=0)
    rdf = pd.read_csv(right_merged, index_col=0)
    assert {"k_phi", "k_theta"}.issubset(ldf.columns)
    assert {"k_phi", "k_theta"}.issubset(rdf.columns)
