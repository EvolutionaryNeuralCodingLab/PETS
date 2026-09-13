"""Reverse follow-up after large saccades vs lizMov head labels."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.movement_associated_nystagmus import (
    PLOT_ID,
    collect_reverse_followups,
    export_movement_associated_nystagmus,
    span_overlaps_movement,
    summarize_reverse_followups,
)
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables
from eye_tracking_system_tools.analysis.plot_bundle import is_plot_bundle


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "animal",
            "block",
            "eye",
            "saccade_on_ms",
            "saccade_off_ms",
            "net_angular_disp",
            "overall_angle_deg",
        ]
    )


def _write_lizmov(path: Path, t_ms: np.ndarray, mov: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("t_mov_ms", data=np.asarray(t_ms, dtype=float).reshape(-1, 1))
        f.create_dataset("movAll", data=np.asarray(mov, dtype=float).reshape(-1, 1))


def _tables(tmp_path: Path, events: pd.DataFrame, t_ms: np.ndarray, mov: np.ndarray) -> EventTables:
    animal, block_num = "PV_126", "007"
    block_path = tmp_path / f"block_{block_num}"
    (block_path / "analysis").mkdir(parents=True, exist_ok=True)
    _write_lizmov(
        block_path / "oe_files" / "rec" / "analysis" / f"Animal={animal}" / "lizMov.mat",
        t_ms,
        mov,
    )
    spec = BlockSpec(animal=animal, block_path=block_path, block_num=block_num)
    l_ev = events.loc[events["eye"] == "L"].reset_index(drop=True) if not events.empty else _empty()
    r_ev = events.loc[events["eye"] == "R"].reset_index(drop=True) if not events.empty else _empty()
    bundle = BlockBundle(
        spec=spec,
        left=pd.DataFrame(),
        right=pd.DataFrame(),
        left_csv_meta={},
        right_csv_meta={},
        l_saccades=l_ev,
        r_saccades=r_ev,
        all_saccades=events.copy(),
    )
    return EventTables(
        blocks=[bundle],
        all_saccades=events.copy(),
        synced=_empty(),
        non_synced=_empty(),
        params={"binocular": {"sync_diff_ms": 34.0}},
    )


def test_span_overlaps_movement_inclusive():
    times = np.array([80.0, 84.0, 88.0])
    assert span_overlaps_movement(80.0, 90.0, times) is True
    assert span_overlaps_movement(100.0, 120.0, times) is False
    assert span_overlaps_movement(88.0, 88.0, times) is True


def test_collect_counts_reverse_within_100ms_not_150ms(tmp_path: Path):
    t_ms = np.arange(80.0, 140.0, 4.0)
    mov = np.full_like(t_ms, 0.01)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 4,
            "block": ["007"] * 4,
            "eye": ["L", "L", "L", "L"],
            "saccade_on_ms": [100.0, 160.0, 300.0, 450.0],
            "saccade_off_ms": [120.0, 175.0, 320.0, 470.0],
            "net_angular_disp": [10.0, 3.0, 10.0, 3.0],
            "overall_angle_deg": [0.0, 180.0, 0.0, 180.0],
        }
    )
    tables = _tables(tmp_path, events, t_ms, mov)
    payload = collect_reverse_followups(tables, post_window_ms=100.0, amp_threshold_deg=5.0)
    large = payload["large"]
    assert set(large["saccade_on_ms"]) == {100.0, 300.0}
    moving = large.loc[large["saccade_on_ms"] == 100.0].iloc[0]
    still = large.loc[large["saccade_on_ms"] == 300.0].iloc[0]
    assert bool(moving["head_movement"]) is True
    assert bool(moving["has_reverse_followup"]) is True
    assert moving["reverse_latency_ms"] == 60.0
    assert bool(still["head_movement"]) is False
    assert bool(still["has_reverse_followup"]) is False
    summary = summarize_reverse_followups(payload)
    assert summary["overall"]["n"] == 2
    assert summary["overall"]["n_reverse"] == 1
    assert summary["head_moving"]["n_reverse"] == 1
    assert summary["head_still"]["n_reverse"] == 0


def test_binocular_partner_is_not_a_reverse_followup(tmp_path: Path):
    t_ms = np.arange(400.0, 460.0, 4.0)
    mov = np.full_like(t_ms, 0.002)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 2,
            "block": ["007"] * 2,
            "eye": ["L", "R"],
            "saccade_on_ms": [100.0, 110.0],
            "saccade_off_ms": [120.0, 130.0],
            "net_angular_disp": [12.0, 11.0],
            "overall_angle_deg": [0.0, 180.0],
        }
    )
    tables = _tables(tmp_path, events, t_ms, mov)
    payload = collect_reverse_followups(tables, amp_threshold_deg=5.0)
    q = payload["large"]
    assert len(q) == 1
    assert bool(q.iloc[0]["has_reverse_followup"]) is False


def test_export_writes_bundle(tmp_path: Path):
    t_ms = np.arange(80.0, 140.0, 4.0)
    mov = np.full_like(t_ms, 0.01)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 2,
            "block": ["007"] * 2,
            "eye": ["L", "L"],
            "saccade_on_ms": [100.0, 160.0],
            "saccade_off_ms": [120.0, 175.0],
            "net_angular_disp": [10.0, 3.0],
            "overall_angle_deg": [0.0, 170.0],
        }
    )
    tables = _tables(tmp_path, events, t_ms, mov)
    out = tmp_path / PLOT_ID
    written = export_movement_associated_nystagmus(tables, out, amp_threshold_deg=5.0, show=False)
    assert is_plot_bundle(out)
    assert written["reverse_followup_by_head.pdf"].is_file()
    assert written["captions.md"].is_file()
    assert (out / "metadata" / "summary.yaml").is_file()
