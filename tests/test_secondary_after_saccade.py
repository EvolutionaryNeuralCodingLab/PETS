"""Secondary detected eye movements after large still-head saccades."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables
from eye_tracking_system_tools.analysis.plot_bundle import is_plot_bundle
from eye_tracking_system_tools.analysis.secondary_after_saccade import (
    PLOT_ID,
    circular_diff_deg,
    collect_secondary_after_saccade,
    direction_class,
    export_secondary_after_saccade,
    head_stationary_at_onset,
    propose_amplitude_threshold,
    summarize_secondary,
    unique_gaze_events,
)


def _empty_events() -> pd.DataFrame:
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


def _eye_trace(*, t0: float = 0.0, n: int = 80, dt: float = 10.0, phi0: float = 0.0) -> pd.DataFrame:
    t = t0 + np.arange(n) * dt
    phi = np.full(n, phi0, dtype=float)
    th = np.zeros(n, dtype=float)
    return pd.DataFrame({"ms_axis": t, "k_phi": phi, "k_theta": th})


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
    left: pd.DataFrame | None = None,
    right: pd.DataFrame | None = None,
    t_ms: np.ndarray | None = None,
    mov: np.ndarray | None = None,
    with_lizmov: bool = True,
    animal: str = "PV_126",
    block_num: str = "007",
) -> EventTables:
    block_path = tmp_path / f"block_{block_num}"
    (block_path / "analysis").mkdir(parents=True, exist_ok=True)
    if with_lizmov:
        if t_ms is None:
            t_ms = np.arange(400.0, 480.0, 4.0)
            mov = np.full_like(t_ms, 0.002)
        _write_lizmov(
            block_path / "oe_files" / "rec" / "analysis" / f"Animal={animal}" / "lizMov.mat",
            t_ms,
            mov if mov is not None else np.zeros_like(t_ms),
        )
    spec = BlockSpec(animal=animal, block_path=block_path, block_num=block_num)
    if left is None:
        left = _eye_trace()
    if right is None:
        right = _eye_trace()
    l_ev = events.loc[events["eye"] == "L"].reset_index(drop=True) if not events.empty else _empty_events()
    r_ev = events.loc[events["eye"] == "R"].reset_index(drop=True) if not events.empty else _empty_events()
    bundle = BlockBundle(
        spec=spec,
        left=left,
        right=right,
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


def test_circular_diff_and_direction_class():
    assert circular_diff_deg(10.0, 0.0) == 10.0
    assert abs(circular_diff_deg(350.0, 10.0) - (-20.0)) < 1e-9
    assert direction_class(0.0, 20.0) == "same"
    assert direction_class(0.0, 180.0) == "reverse"
    assert direction_class(10.0, 200.0) == "reverse"
    assert direction_class(np.nan, 20.0) == "unknown"


def test_head_stationary_at_onset_lookback_exclusive():
    times = np.array([80.0, 84.0, 88.0, 200.0])
    assert head_stationary_at_onset(100.0, times, lookback_ms=50.0) is False
    assert head_stationary_at_onset(100.0, times, lookback_ms=10.0) is True
    assert head_stationary_at_onset(80.0, times, lookback_ms=50.0) is True  # sample at t_on excluded
    assert head_stationary_at_onset(250.0, np.array([]), lookback_ms=50.0) is True


def test_propose_amplitude_threshold_p75_and_floor():
    amps = np.arange(1.0, 11.0)  # P75 = 7.75 → 7.5
    thr, stats = propose_amplitude_threshold(amps, percentile=75.0, floor_deg=5.0)
    assert thr == 7.5
    assert stats["p75"] == 7.75
    tiny = np.linspace(0.6, 2.0, 20)
    thr2, _ = propose_amplitude_threshold(tiny, percentile=75.0, floor_deg=5.0)
    assert thr2 == 5.0


def test_unique_gaze_events_merges_binocular_and_keeps_amp():
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 3,
            "block": ["007"] * 3,
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [100.0, 110.0, 400.0],
            "saccade_off_ms": [120.0, 128.0, 418.0],
            "net_angular_disp": [12.0, 8.0, 3.0],
            "overall_angle_deg": [10.0, 15.0, 200.0],
        }
    )
    unique = unique_gaze_events(events, sync_diff_ms=34.0)
    assert len(unique) == 2
    pair = unique.loc[unique["concurrency"] == "binocular"].iloc[0]
    assert pair["saccade_on_ms"] == 100.0
    assert pair["saccade_off_ms"] == 128.0
    assert pair["amp_deg"] == 12.0
    assert pair["primary_eye"] == "L"
    assert pair["eyes"] == 2
    mono = unique.loc[unique["concurrency"] == "monocular"].iloc[0]
    assert mono["amp_deg"] == 3.0


def test_collect_still_head_large_reverse_and_later_head(tmp_path: Path):
    # Primary at 100 ms while still; reverse follow-up at 160 ms; head bout at 200 ms.
    t_ms = np.arange(200.0, 280.0, 4.0)
    mov = np.full_like(t_ms, 0.002)
    left = _eye_trace(t0=0.0, n=40, dt=10.0, phi0=0.0)
    left.loc[left["ms_axis"] >= 100.0, "k_phi"] = 10.0
    left.loc[left["ms_axis"] >= 160.0, "k_phi"] = 2.0
    right = _eye_trace(t0=0.0, n=40, dt=10.0, phi0=0.0)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 3,
            "block": ["007"] * 3,
            "eye": ["L", "L", "R"],
            "saccade_on_ms": [100.0, 160.0, 500.0],
            "saccade_off_ms": [120.0, 175.0, 520.0],
            "net_angular_disp": [10.0, 3.0, 12.0],
            "overall_angle_deg": [0.0, 180.0, 0.0],
        }
    )
    tables = _tables(tmp_path, events=events, left=left, right=right, t_ms=t_ms, mov=mov)
    payload = collect_secondary_after_saccade(
        tables,
        lookback_ms=50.0,
        post_window_ms=250.0,
        amp_threshold_deg=5.0,
        reload_missing_traces=False,
    )
    stat = payload["stationary"]
    assert set(stat["saccade_on_ms"]) == {100.0, 160.0, 500.0}
    q = payload["qualifying"]
    assert set(q["saccade_on_ms"]) == {100.0, 500.0}
    primary = q.loc[q["saccade_on_ms"] == 100.0].iloc[0]
    assert bool(primary["has_secondary"])
    assert primary["secondary_class"] == "reverse"
    assert primary["first_secondary_latency_ms"] == 60.0
    assert bool(primary["has_subsequent_head"])
    assert primary["head_latency_ms"] == 100.0
    late = q.loc[q["saccade_on_ms"] == 500.0].iloc[0]
    assert bool(late["has_secondary"]) is False
    assert bool(late["has_subsequent_head"]) is False
    assert np.isfinite(primary["remaining_frac"])
    assert primary["remaining_frac"] < 0.5

    summary = summarize_secondary(payload)
    assert summary["n_qualifying"] == 2
    assert summary["n_with_secondary"] == 1
    assert summary["n_first_reverse"] == 1


def test_ongoing_head_at_onset_is_excluded(tmp_path: Path):
    t_ms = np.arange(80.0, 140.0, 4.0)
    mov = np.full_like(t_ms, 0.01)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"],
            "block": ["007"],
            "eye": ["L"],
            "saccade_on_ms": [100.0],
            "saccade_off_ms": [120.0],
            "net_angular_disp": [12.0],
            "overall_angle_deg": [0.0],
        }
    )
    tables = _tables(tmp_path, events=events, t_ms=t_ms, mov=mov)
    payload = collect_secondary_after_saccade(
        tables, amp_threshold_deg=5.0, reload_missing_traces=False
    )
    assert payload["stationary"].empty
    assert payload["qualifying"].empty
    assert payload["diag"].iloc[0]["n_unique_events"] == 1
    assert payload["diag"].iloc[0]["n_stationary_onset"] == 0


def test_binocular_constituent_is_not_counted_as_followup(tmp_path: Path):
    t_ms = np.arange(400.0, 480.0, 4.0)
    mov = np.full_like(t_ms, 0.002)
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 2,
            "block": ["007"] * 2,
            "eye": ["L", "R"],
            "saccade_on_ms": [100.0, 110.0],
            "saccade_off_ms": [120.0, 130.0],
            "net_angular_disp": [12.0, 11.0],
            "overall_angle_deg": [0.0, 5.0],
        }
    )
    tables = _tables(tmp_path, events=events, t_ms=t_ms, mov=mov)
    payload = collect_secondary_after_saccade(
        tables, amp_threshold_deg=5.0, reload_missing_traces=False
    )
    q = payload["qualifying"]
    assert len(q) == 1
    assert int(q.iloc[0]["n_secondary"]) == 0
    assert q.iloc[0]["concurrency"] == "binocular"


def test_export_writes_plots_metadata_captions(tmp_path: Path):
    t_ms = np.arange(200.0, 260.0, 4.0)
    mov = np.full_like(t_ms, 0.002)
    left = _eye_trace()
    left.loc[left["ms_axis"] >= 100.0, "k_phi"] = 8.0
    events = pd.DataFrame(
        {
            "animal": ["PV_126"] * 2,
            "block": ["007"] * 2,
            "eye": ["L", "L"],
            "saccade_on_ms": [100.0, 180.0],
            "saccade_off_ms": [120.0, 195.0],
            "net_angular_disp": [8.0, 2.0],
            "overall_angle_deg": [0.0, 170.0],
        }
    )
    tables = _tables(tmp_path, events=events, left=left, t_ms=t_ms, mov=mov)
    out = tmp_path / PLOT_ID
    written = export_secondary_after_saccade(
        tables, out, amp_threshold_deg=5.0, show=False, n_browser_per_cell=1
    )
    assert is_plot_bundle(out)
    assert written["amplitude_distribution.pdf"].is_file()
    assert written["event_browser.pdf"].is_file()
    assert written["secondary_fractions.pdf"].is_file()
    assert written["captions.md"].is_file()
    assert (out / "metadata" / "qualifying_events.csv").is_file()
    assert (out / "metadata" / "summary.yaml").is_file()
    assert (out / "metadata" / "LOGIC.md").is_file()
    assert (out / "replot.py").is_file()
    text = written["captions.md"].read_text()
    assert "not applied" in text.lower()
    assert "threshold" in text.lower()
