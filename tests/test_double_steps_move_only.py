"""Back-and-forth sequences restricted to head-moving unique events."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.double_steps_move_only import (
    PLOT_ID,
    collect_double_steps_move_only,
    export_double_steps_move_only,
    reviewer_response_text,
    summarize_double_steps_move_only,
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


def test_move_only_drops_still_primaries(tmp_path: Path):
    t_ms = np.arange(80.0, 140.0, 4.0)
    mov = np.full_like(t_ms, 0.01)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 5,
            "block": ["007"] * 5,
            "eye": ["L"] * 5,
            "saccade_on_ms": [100.0, 140.0, 180.0, 300.0, 360.0],
            "saccade_off_ms": [120.0, 155.0, 195.0, 320.0, 375.0],
            "net_angular_disp": [10.0, 3.0, 2.0, 10.0, 3.0],
            "overall_angle_deg": [0.0, 180.0, 10.0, 0.0, 180.0],
        }
    )
    tables = _tables(tmp_path, events, t_ms, mov)
    payload = collect_double_steps_move_only(tables, post_window_ms=150.0, amp_threshold_deg=5.0)
    unique = payload["all_unique"]
    large = payload["large"]
    assert payload["n_unique_dropped_still"] >= 1
    assert 300.0 not in set(pd.to_numeric(unique["saccade_on_ms"]))
    assert set(large["saccade_on_ms"]) == {100.0}
    assert bool(large.iloc[0]["has_back_and_forth"]) is True
    assert bool(large.iloc[0]["head_movement"]) is True
    summary = summarize_double_steps_move_only(payload)
    assert "head_still" not in summary
    assert "fisher_p_two_sided" not in summary
    assert summary["overall"]["n"] == 1
    assert summary["overall"]["n_hit"] == 1


def test_export_move_only_bundle_and_response(tmp_path: Path):
    t_ms = np.arange(80.0, 140.0, 4.0)
    mov = np.full_like(t_ms, 0.01)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 3,
            "block": ["007"] * 3,
            "eye": ["L"] * 3,
            "saccade_on_ms": [100.0, 140.0, 180.0],
            "saccade_off_ms": [120.0, 155.0, 195.0],
            "net_angular_disp": [10.0, 3.0, 2.0],
            "overall_angle_deg": [0.0, 180.0, 10.0],
        }
    )
    tables = _tables(tmp_path, events, t_ms, mov)
    out = tmp_path / PLOT_ID
    written = export_double_steps_move_only(tables, out, amp_threshold_deg=5.0, show=False)
    assert is_plot_bundle(out)
    assert written["back_and_forth_overall.pdf"].is_file()
    assert "back_and_forth_by_head.pdf" not in written
    text = written["reviewer_response.md"].read_text().lower()
    assert "during head movement" in text
    assert "nystagmus" in text
    assert "fisher" not in text
    assert "head-stationary" not in text
    assert "head-still" not in text
    summary = summarize_double_steps_move_only(
        collect_double_steps_move_only(tables, amp_threshold_deg=5.0)
    )
    assert "head-stationary" not in reviewer_response_text(summary).lower()
