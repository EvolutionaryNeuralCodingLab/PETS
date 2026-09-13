"""Back-and-forth sequences after large saccades."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.double_steps_rev_verbatim import (
    PLOT_ID,
    collect_double_steps,
    export_double_steps,
    is_back_and_forth_sequence,
    summarize_double_steps,
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


def test_sequence_requires_two_small_extras_and_a_flip():
    primary_amp, primary_ang = 10.0, 0.0
    reverse = {"amp_deg": 3.0, "overall_angle_deg": 180.0, "latency_ms": 40.0}
    forth = {"amp_deg": 2.0, "overall_angle_deg": 10.0, "latency_ms": 80.0}
    same = {"amp_deg": 2.0, "overall_angle_deg": 15.0, "latency_ms": 80.0}
    big = {"amp_deg": 12.0, "overall_angle_deg": 180.0, "latency_ms": 40.0}

    hit, n_small, n_flip = is_back_and_forth_sequence(
        [reverse, forth], primary_amp=primary_amp, primary_angle=primary_ang
    )
    assert hit is True
    assert n_small == 2
    assert n_flip == 1

    hit, n_small, _ = is_back_and_forth_sequence(
        [reverse], primary_amp=primary_amp, primary_angle=primary_ang
    )
    assert hit is False
    assert n_small == 1

    hit, _, _ = is_back_and_forth_sequence(
        [reverse, {"amp_deg": 2.5, "overall_angle_deg": 170.0, "latency_ms": 80.0}],
        primary_amp=primary_amp,
        primary_angle=primary_ang,
    )
    assert hit is False  # two backs, no forth

    hit, n_small, _ = is_back_and_forth_sequence(
        [big, forth], primary_amp=primary_amp, primary_angle=primary_ang
    )
    assert hit is False
    assert n_small == 1  # large extra dropped

    hit, _, _ = is_back_and_forth_sequence(
        [same, {"amp_deg": 2.0, "overall_angle_deg": 20.0, "latency_ms": 90.0}],
        primary_amp=primary_amp,
        primary_angle=primary_ang,
    )
    assert hit is False  # no reverse vs primary


def test_collect_double_step_vs_single_reverse(tmp_path: Path):
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
    payload = collect_double_steps(tables, post_window_ms=150.0, amp_threshold_deg=5.0)
    large = payload["large"]
    assert set(large["saccade_on_ms"]) == {100.0, 300.0}
    seq = large.loc[large["saccade_on_ms"] == 100.0].iloc[0]
    one = large.loc[large["saccade_on_ms"] == 300.0].iloc[0]
    assert bool(seq["has_back_and_forth"]) is True
    assert bool(seq["head_movement"]) is True
    assert int(seq["n_small_followups"]) == 2
    assert bool(one["has_back_and_forth"]) is False
    assert bool(one["head_movement"]) is False
    summary = summarize_double_steps(payload)
    assert summary["overall"]["n_hit"] == 1
    assert summary["head_moving"]["n_hit"] == 1
    assert summary["head_still"]["n_hit"] == 0


def test_export_double_steps_bundle(tmp_path: Path):
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
    written = export_double_steps(tables, out, amp_threshold_deg=5.0, show=False)
    assert is_plot_bundle(out)
    assert written["back_and_forth_by_head.pdf"].is_file()
    assert written["reviewer_response.md"].is_file()
    assert "back-and-forth" in written["reviewer_response.md"].read_text().lower()
