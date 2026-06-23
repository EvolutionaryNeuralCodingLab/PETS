"""Phase 6 tests — Behavior tab."""

from __future__ import annotations

import numpy as np
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
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.behavior_tab import (
    BehaviorTab,
    has_liz_mov,
)
from eye_tracking_system_tools.preprocessing.notebook_helpers import (
    create_behavior_df,
    rolling_window_analysis,
)


@pytest.fixture
def synthetic_liz_mov_df() -> pd.DataFrame:
    t = np.arange(0, 30_000, 4, dtype=float)
    mov = np.where(t < 10_000, 0.1, 0.4)
    return pd.DataFrame({"t_mov_ms": t, "movAll": mov})


def test_rolling_window_analysis_helpers(synthetic_liz_mov_df):
    rolling = rolling_window_analysis(
        synthetic_liz_mov_df,
        window_size=10_000,
        step_size=1_000,
    )
    assert list(rolling.columns) == ["window_start", "average_movAll"]
    assert len(rolling) > 0

    rolling["behavior"] = rolling["average_movAll"].apply(
        lambda x: "active" if x > 0.3 else "quiet"
    )
    behavior_df = create_behavior_df(rolling)
    assert list(behavior_df.columns) == ["start_time", "end_time", "annotation"]
    assert not behavior_df.empty
    assert set(behavior_df["annotation"].unique()).issubset({"active", "quiet"})


def test_behavior_tab_writes_csv(behavior_sample_block_path, qapp_session, tmp_path):
    ensure_config_template(tmp_path)
    config = load_config(None, tmp_path)
    state = GuiState(output_folder=tmp_path)
    block = BlockHandle(
        animal_call="PV_126",
        experiment_date="2024_07_18",
        block_num="006",
        block_path=behavior_sample_block_path,
        path_to_animal_folder=behavior_sample_block_path.parents[2],
    )
    assert has_liz_mov(block)

    tab = BehaviorTab(state, config)
    tab.set_block(block)
    assert tab.isEnabled()
    assert tab._rolling_df is not None
    assert len(tab._movement_plot.plot_widget.listDataItems()) == 1

    tab._run_export()
    out_path = block.analysis_path / "block_006_behavior_state.csv"
    assert out_path.is_file()
    exported = pd.read_csv(out_path)
    assert list(exported.columns) == ["start_time", "end_time", "annotation"]
