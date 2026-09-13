"""Tests for saccade viewer event subset selectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
    Figure2fPoints,
    _resolve_contra_peak,
    display_data_from_collected,
    select_figure_2f_roi,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.saccade_viewer.selectors.common import (
    enrich_events_for_viewer,
    numeric_event_columns,
    unique_events_by_identity,
)
from eye_tracking_system_tools.analysis.saccade_viewer.selectors.threshold_selector import (
    ThresholdRule,
    apply_threshold_rules,
)


class _Spec:
    def __init__(self, animal: str, block: str, block_path: Path) -> None:
        self.animal = animal
        self.block_num = block
        self.block_key = f"{animal}_block_{block}"
        self.block_path = block_path


class _Bundle:
    def __init__(self, animal: str, block: str, tmp_path: Path) -> None:
        bp = tmp_path / f"block_{block}"
        bp.mkdir(parents=True, exist_ok=True)
        self.spec = _Spec(animal, block, bp)
        self.all_saccades = pd.DataFrame()
        self.left = pd.DataFrame()
        self.right = pd.DataFrame()
        self.l_saccades = pd.DataFrame()
        self.r_saccades = pd.DataFrame()


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["M_002", "M_002", "M_002"],
            "block": ["012", "012", "013"],
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [1000.0, 2000.0, 1500.0],
            "net_angular_disp": [5.0, 12.0, 3.0],
            "head_movement": [False, True, False],
            "speed_profile_angular": [[1.0, 2.0], [3.0], [0.5]],
        }
    )


def test_numeric_event_columns_skips_profiles():
    cols = numeric_event_columns(_sample_events())
    assert "net_angular_disp" in cols
    assert "speed_profile_angular" not in cols
    assert "saccade_on_ms" in cols


def test_apply_threshold_rules_min_max():
    df = _sample_events()
    rules = [
        ThresholdRule("net_angular_disp", min_val=4.0),
        ThresholdRule("net_angular_disp", max_val=10.0),
    ]
    out = apply_threshold_rules(df, rules)
    assert len(out) == 1
    assert out.iloc[0]["net_angular_disp"] == 5.0


def test_enrich_events_for_viewer(tmp_path: Path):
    df = _sample_events()
    bundle = _Bundle("M_002", "012", tmp_path)
    tables = EventTables(
        blocks=[bundle],
        all_saccades=df,
        synced=pd.DataFrame(),
        non_synced=pd.DataFrame(),
        csv_meta=[],
        params={},
    )
    subset = df.iloc[[0]].copy()
    subset["right_peak_v"] = 0.1
    out = enrich_events_for_viewer(tables, subset)
    assert "block_path" in out.columns
    assert out.iloc[0]["block_path"] == str(bundle.spec.block_path)
    assert "right_peak_v" not in out.columns


def test_unique_events_by_identity():
    df = pd.concat([_sample_events(), _sample_events().iloc[[0]]], ignore_index=True)
    out = unique_events_by_identity(df)
    assert len(out) == 3


def test_select_figure_2f_roi():
    pts = pd.DataFrame(
        {
            "animal": ["A", "A", "B"],
            "block": ["1", "1", "2"],
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [1.0, 2.0, 3.0],
            "right_peak_v": [0.05, 0.2, 0.08],
            "left_peak_v": [0.04, 0.15, 0.09],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    collected = Figure2fPoints(
        points=pts,
        macro_range=(0.0, 0.5),
        micro_range=(0.0, 0.1),
        bins=10,
        cfg={},
    )
    out = select_figure_2f_roi(collected, 0.0, 0.1, 0.0, 0.1)
    assert len(out) == 2
    assert set(out["saccade_on_ms"].tolist()) == {1.0, 3.0}


def test_resolve_contra_peak_modes():
    eye_df = pd.DataFrame(
        {
            "ms_axis": [0.0, 10.0, 20.0, 30.0],
            "angular_speed_r": [1.0, 5.0, 2.0, 0.5],
        }
    )
    row = pd.Series({"saccade_on_ms": 10.0, "saccade_off_ms": 20.0})
    window_peak = _resolve_contra_peak(
        eye_df, row, sample_mode="contra_window", contra_sample_ms=11.0, frame_ms=10.0
    )
    span_peak = _resolve_contra_peak(
        eye_df, row, sample_mode="event_span", contra_sample_ms=11.0, frame_ms=10.0
    )
    assert window_peak == pytest.approx(0.5)
    assert span_peak == pytest.approx(0.5)

    narrow = pd.DataFrame(
        {"ms_axis": [10.0, 20.0], "angular_speed_r": [2.0, 8.0]}
    )
    window_n = _resolve_contra_peak(
        narrow,
        row,
        sample_mode="contra_window",
        contra_sample_ms=5.0,
        frame_ms=10.0,
    )
    span_n = _resolve_contra_peak(
        narrow,
        row,
        sample_mode="event_span",
        contra_sample_ms=5.0,
        frame_ms=10.0,
    )
    assert window_n == pytest.approx(0.2)
    assert span_n == pytest.approx(0.8)


def test_display_data_from_collected():
    pts = pd.DataFrame(
        {
            "animal": ["A"],
            "block": ["1"],
            "eye": ["L"],
            "saccade_on_ms": [1.0],
            "right_peak_v": [0.05],
            "left_peak_v": [0.04],
            "weight": [1.0],
        }
    )
    collected = Figure2fPoints(
        points=pts,
        macro_range=(0.0, 0.5),
        micro_range=(0.0, 0.1),
        bins=5,
        cfg={"macro_tick_list": [0, 0.25], "micro_tick_list": [0, 0.05]},
    )
    display = display_data_from_collected(collected)
    assert display.n_points == 1
    assert display.macro["norm_counts"].shape == (4, 4)
