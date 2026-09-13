"""Head-bout onset vs unique saccade timing."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables
from eye_tracking_system_tools.analysis.head_labels import bout_onsets_from_times
from eye_tracking_system_tools.analysis.saccade_head_timing import (
    PLOT_ID,
    collect_saccade_head_timing,
    export_saccade_head_timing,
    nearest_onset,
    summarize_timing,
    unique_saccade_onsets,
)


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["animal", "block", "eye", "saccade_on_ms", "saccade_off_ms"]
    )


def _write_lizmov(path: Path, t_ms: np.ndarray, mov: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("t_mov_ms", data=np.asarray(t_ms, dtype=float).reshape(-1, 1))
        f.create_dataset("movAll", data=np.asarray(mov, dtype=float).reshape(-1, 1))
    return path


def _tables(
    tmp_path: Path,
    *,
    events: pd.DataFrame,
    with_lizmov: bool = True,
    t_ms: np.ndarray | None = None,
    mov: np.ndarray | None = None,
    animal: str = "PV_126",
    block_num: str = "007",
) -> EventTables:
    block_path = tmp_path / f"block_{block_num}"
    (block_path / "analysis").mkdir(parents=True, exist_ok=True)
    if with_lizmov:
        if t_ms is None:
            t_ms = np.arange(0.0, 200.0, 10.0)
            mov = np.zeros_like(t_ms)
            mov[(t_ms >= 80.0) & (t_ms < 120.0)] = 1.0
        _write_lizmov(
            block_path / "oe_files" / "rec" / "analysis" / f"Animal={animal}" / "lizMov.mat",
            t_ms,
            mov if mov is not None else np.zeros_like(t_ms),
        )
    spec = BlockSpec(animal=animal, block_path=block_path, block_num=block_num)
    l_ev = events.loc[events["eye"] == "L"].reset_index(drop=True) if not events.empty else _empty_events()
    r_ev = events.loc[events["eye"] == "R"].reset_index(drop=True) if not events.empty else _empty_events()
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
        synced=_empty_events(),
        non_synced=_empty_events(),
        params={"binocular": {"sync_diff_ms": 34.0}},
    )


def test_bout_onsets_gap_cluster_sparse_always_positive():
    times = np.concatenate([np.arange(0.0, 80.0, 4.0), np.arange(200.0, 260.0, 4.0)])
    onsets = bout_onsets_from_times(times, gap_ms=40.0)
    np.testing.assert_array_equal(onsets, [0.0, 200.0])


def test_nearest_onset_sign_convention():
    onsets = np.array([80.0, 200.0])
    nearest, dt = nearest_onset(100.0, onsets)
    assert nearest == 80.0
    assert dt == -20.0  # head started 20 ms before the saccade
    nearest, dt = nearest_onset(190.0, onsets)
    assert nearest == 200.0
    assert dt == 10.0  # saccade first


def test_unique_saccade_onsets_merges_binocular_pair():
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 3,
            "block": ["007"] * 3,
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [100.0, 110.0, 400.0],
        }
    )
    unique = unique_saccade_onsets(events, sync_diff_ms=34.0)
    assert len(unique) == 2
    conc = set(unique["concurrency"])
    assert conc == {"binocular", "monocular"}
    paired = unique.loc[unique["concurrency"] == "binocular", "saccade_on_ms"].iloc[0]
    assert paired == 100.0


def test_collect_skips_blocks_without_lizmov(tmp_path: Path):
    events = pd.DataFrame(
        {
            "animal": ["PV_126"],
            "block": ["007"],
            "eye": ["L"],
            "saccade_on_ms": [100.0],
            "saccade_off_ms": [120.0],
        }
    )
    tables = _tables(tmp_path, events=events, with_lizmov=False)
    ev, diag, peth = collect_saccade_head_timing(tables, window_ms=500.0)
    assert ev.empty
    assert diag.iloc[0]["note"] == "no lizMov.mat"
    assert int(peth["n_saccades"]) == 0


def test_collect_head_leads_and_saccade_leads(tmp_path: Path):
    # Head bout starts at 80 ms (rising edge). Second bout at 300 ms.
    t_ms = np.arange(0.0, 400.0, 10.0)
    mov = np.zeros_like(t_ms)
    mov[(t_ms >= 80.0) & (t_ms < 140.0)] = 1.0
    mov[(t_ms >= 300.0) & (t_ms < 340.0)] = 1.0
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 4,
            "block": ["007"] * 4,
            "eye": ["L", "R", "L", "R"],
            "saccade_on_ms": [100.0, 108.0, 280.0, 500.0],
            "saccade_off_ms": [120.0, 128.0, 300.0, 520.0],
        }
    )
    tables = _tables(tmp_path, events=events, t_ms=t_ms, mov=mov)
    ev, diag = collect_saccade_head_timing(tables, window_ms=50.0)[:2]
    assert int(diag.iloc[0]["n_head_onsets"]) == 2
    # L/R at 100/108 collapse to one event at 100 → dt = 80-100 = -20 (head leads)
    # L at 280 → nearest 300 → dt = +20 (saccade leads), in 50 ms window
    # R at 500 → nearest 300 → dt = -200, outside window
    assert int(diag.iloc[0]["n_unique_events"]) == 3
    assert int(diag.iloc[0]["n_in_window"]) == 2
    in_win = ev.loc[ev["in_window"]].sort_values("saccade_on_ms")
    assert list(in_win["dt_ms"]) == [-20.0, 20.0]
    summary = summarize_timing(ev, diag, window_ms=50.0, simultaneous_ms=17.0)
    assert summary["n_head_leads"] == 1
    assert summary["n_saccade_leads"] == 1
    assert summary["median_dt_ms"] == 0.0


def test_simultaneous_is_one_video_frame():
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 3,
            "block": ["007"] * 3,
            "dt_ms": [-17.0, 0.0, 17.0],
            "in_window": True,
        }
    )
    diag = pd.DataFrame({"lizmov_path": ["x"]})
    summary = summarize_timing(events, diag, simultaneous_ms=17.0)
    assert summary["n_simultaneous"] == 3
    assert summary["n_head_leads"] == 0
    assert summary["n_saccade_leads"] == 0
    just_outside = events.copy()
    just_outside["dt_ms"] = [-17.1, 17.1, 20.0]
    summary = summarize_timing(just_outside, diag, simultaneous_ms=17.0)
    assert summary["n_simultaneous"] == 0
    assert summary["n_head_leads"] == 1
    assert summary["n_saccade_leads"] == 2


def test_collect_sparse_always_positive_lizmov(tmp_path: Path):
    """Real lizMov.mat stores only supra-threshold samples (movAll always > 0)."""
    t_ms = np.concatenate([np.arange(80.0, 140.0, 4.0), np.arange(300.0, 340.0, 4.0)])
    mov = np.full_like(t_ms, 0.002)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"],
            "block": ["007"],
            "eye": ["L"],
            "saccade_on_ms": [100.0],
            "saccade_off_ms": [120.0],
        }
    )
    tables = _tables(tmp_path, events=events, t_ms=t_ms, mov=mov)
    ev, diag, _peth = collect_saccade_head_timing(tables, window_ms=50.0, bout_gap_ms=40.0)
    assert int(diag.iloc[0]["n_head_onsets"]) == 2
    assert bool(ev.iloc[0]["in_window"])
    assert ev.iloc[0]["dt_ms"] == -20.0


def test_export_writes_plots_and_metadata(tmp_path: Path):
    t_ms = np.arange(0.0, 200.0, 10.0)
    mov = np.zeros_like(t_ms)
    mov[(t_ms >= 80.0) & (t_ms < 120.0)] = 1.0
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 2,
            "block": ["007"] * 2,
            "eye": ["L", "R"],
            "saccade_on_ms": [90.0, 200.0],
            "saccade_off_ms": [110.0, 220.0],
        }
    )
    tables = _tables(tmp_path, events=events, t_ms=t_ms, mov=mov)
    out = tmp_path / PLOT_ID
    written = export_saccade_head_timing(tables, out, window_ms=150.0, show=False)
    assert (out / "plots" / "head_onset_relative_to_saccade.pdf").is_file()
    assert (out / "plots" / "head_onset_relative_to_saccade_by_animal.pdf").is_file()
    assert (out / "plots" / "lead_lag_summary.pdf").is_file()
    assert (out / "metadata" / "saccade_head_offsets.csv").is_file()
    assert (out / "metadata" / "block_diagnostics.csv").is_file()
    assert (out / "metadata" / "summary.yaml").is_file()
    assert (out / "metadata" / "LOGIC.md").is_file()
    assert (out / "replot.py").is_file()
    assert written["head_onset_relative_to_saccade.pdf"].is_file()
    assert (out / "plots" / "head_motion_peth.pdf").is_file()
