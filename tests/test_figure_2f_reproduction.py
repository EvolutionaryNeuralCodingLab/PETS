"""Tests for Fig 2f legacy reproduction helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
    collect_figure_2f_points,
    compare_figure_2f_to_reference,
    summarize_figure_2f_event_table,
)
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables


def _eye_trace(times: np.ndarray, spikes: dict[float, float]) -> pd.DataFrame:
    speed = np.full(times.size, 0.01, dtype=float)
    for t_ms, val in spikes.items():
        speed[int(np.argmin(np.abs(times - t_ms)))] = val
    return pd.DataFrame({"ms_axis": times, "angular_speed_r": speed})


def _tables(tmp_path: Path) -> EventTables:
    frame_ms = 17.0
    times = np.arange(0.0, 2100.0, frame_ms)
    left = _eye_trace(times, {500.0: 0.5, 2000.0: 0.34})
    right = _eye_trace(times, {100.0: 99.0, 510.0: 0.2, 2000.0: 0.85})
    block_num = "007"
    events = pd.DataFrame(
        {
            "animal": ["PV_106", "PV_106", "PV_106", "PV_106"],
            "block": [block_num] * 4,
            "eye": ["L", "R", "L", "R"],
            "saccade_on_ms": [100.0, 110.0, 1500.0, 2000.0],
            "saccade_off_ms": [134.0, 144.0, 1534.0, 2034.0],
            "head_movement": [False, False, False, False],
            "speed_profile_angular": [[3.4], [5.1], [2.0], [2.2]],
        }
    )
    block_path = tmp_path / "block_007"
    (block_path / "analysis").mkdir(parents=True, exist_ok=True)
    spec = BlockSpec(animal="PV_106", block_path=block_path, block_num=block_num)
    bundle = BlockBundle(
        spec=spec,
        left=left,
        right=right,
        left_csv_meta={},
        right_csv_meta={},
        l_saccades=events.loc[events["eye"] == "L"].reset_index(drop=True),
        r_saccades=events.loc[events["eye"] == "R"].reset_index(drop=True),
        all_saccades=events.copy(),
    )
    params = {
        "binocular": {"sync_diff_ms": 34.0},
        "figure_2f": {
            "event_mode": "monocular",
            "pairing_mode": "legacy_contra_table",
            "contra_event_window_ms": 100.0,
            "require_head_stationary": True,
            "exclude_animals": [],
            "macro_range": [0.0, 0.5],
            "micro_range": [0.0, 0.1],
        },
    }
    return EventTables(
        blocks=[bundle],
        all_saccades=events.copy(),
        synced=pd.DataFrame(),
        non_synced=events.copy(),
        params=params,
    )


def test_legacy_monocular_skips_near_contra(tmp_path: Path):
    tables = _tables(tmp_path)
    collected = collect_figure_2f_points(tables)
    # L@100/R@110 are a binocular pair; L@1500 and R@2000 are monocular.
    assert len(collected.points) == 2
    assert set(collected.points["2f_source"]) == {"legacy_monocular"}


def test_head_filter_moving(tmp_path: Path):
    tables = _tables(tmp_path)
    events = tables.all_saccades.copy()
    events["head_movement"] = [True, True, False, False]
    tables = EventTables(
        blocks=tables.blocks,
        all_saccades=events,
        synced=tables.synced,
        non_synced=events,
        params=tables.params,
    )
    collected = collect_figure_2f_points(
        tables,
        cfg={"head_filter": "moving", "require_head_stationary": False, "event_mode": "all"},
    )
    assert len(collected.points) >= 1
    assert (collected.points["head_movement"] == True).all()  # noqa: E712


def test_summarize_figure_2f_event_table(tmp_path: Path):
    tables = _tables(tmp_path)
    summary = summarize_figure_2f_event_table(
        tables, {"require_head_stationary": True, "exclude_animals": []}
    )
    assert summary["n_events_total"] == 4
    assert summary["n_events_after_filters"] == 4


def test_compare_figure_2f_to_reference(tmp_path: Path):
    tables = _tables(tmp_path)
    collected = collect_figure_2f_points(tables)
    ref_path = tmp_path / "ref.pickle"
    import pickle

    with open(ref_path, "wb") as handle:
        pickle.dump(
            {
                "right_eye_speeds": collected.right_peak_v,
                "left_eye_speeds": collected.left_peak_v,
                "macro": {"norm_counts": np.zeros((59, 59))},
                "vmax_all": 1.0,
            },
            handle,
        )
    report = compare_figure_2f_to_reference(collected, ref_path)
    assert report["n_delta"] == 0
