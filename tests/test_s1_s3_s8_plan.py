"""Tests for S1 Kerr bars, S3 paper settings, isolation mask, S8j axis scaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.analysis.figures_2c_2e import (
    keep_samples_outside_neighbors,
    neighbor_on_off,
)
from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
    event_square_percentile,
    export_figure_s3,
    half_step_ticks,
    histogram2d_xy,
    percentile_speed_max,
    publication_square_limits,
    rebin_s8j_payload,
    zoom_square_limits,
)
from eye_tracking_system_tools.analysis.kerr_component_error_export import (
    animal_level_from_blocks,
    export_s1_kerr_bars,
    figure_s1c_overall,
)
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables
from eye_tracking_system_tools.analysis.block_registry import BlockSpec


def test_publication_square_limits_and_half_step_ticks():
    hi, ticks = publication_square_limits(0.95, snap=True)
    assert hi == pytest.approx(1.0)
    assert ticks == [0.0, 0.5, 1.0]
    hi_m, ticks_m = publication_square_limits(1.25, snap=False)
    assert hi_m == pytest.approx(1.25)
    assert ticks_m == [0.0, 0.5, 1.0]
    assert half_step_ticks(1.5) == [0.0, 0.5, 1.0, 1.5]


def test_s8j_hist_grid_stays_59():
    hist = histogram2d_xy(np.array([0.1, 0.2]), np.array([0.1, 0.2]), None, (0.0, 1.25), 60)
    assert hist["norm_counts"].shape == (59, 59)


def test_percentile_speed_max_matches_nanpercentile():
    right = np.array([0.1, 0.2, 1.0])
    left = np.array([0.1, 0.3, 0.9])
    got = percentile_speed_max(right, left, pct=99.5)
    expect = float(np.nanpercentile(np.concatenate([right, left]), 99.5))
    assert got == pytest.approx(expect)


def test_keep_samples_outside_neighbors():
    t = np.linspace(0, 200, 21)
    ons = np.array([0.0, 80.0])
    offs = np.array([20.0, 100.0])
    keep = keep_samples_outside_neighbors(t, 0.0, 20.0, ons, offs)
    assert keep[t == 10].all()
    assert not keep[(t >= 80) & (t <= 100)].any()


def test_neighbor_on_off_groups_by_eye():
    events = pd.DataFrame(
        {
            "animal": ["M_002", "M_002"],
            "block": ["012", "012"],
            "eye": ["L", "L"],
            "saccade_on_ms": [0.0, 50.0],
            "saccade_off_ms": [10.0, 60.0],
            "net_angular_disp": [2.0, 2.0],
        }
    )
    mapping = neighbor_on_off(events)
    ons, offs = mapping[("M_002", "012", "L")]
    assert ons.size == 2
    assert offs[0] == pytest.approx(10.0)


def test_s1c_overall_from_csv(tmp_path: Path):
    kerr = Path("/Users/nimi/Projects/PETS/development/kerr_relative_error/outputs")
    written = export_s1_kerr_bars(tmp_path, show=False)
    assert written["cohort_component_error_overall_phi_theta.pdf"].is_file()
    assert written["cohort_component_error_across_animals.pdf"].is_file()
    import pickle

    with open(written["kerr_component_error.pkl"], "rb") as f:
        payload = pickle.load(f)
    assert float(payload["phi_mean"]) == pytest.approx(0.84, abs=0.01)
    assert float(payload["theta_mean"]) == pytest.approx(1.19, abs=0.01)
    across = pd.read_csv(kerr / "cohort_component_error_across_animals.csv")
    fig = figure_s1c_overall(across)
    assert fig.axes[0].get_ylabel()
    import matplotlib.pyplot as plt

    plt.close(fig)
    both = across[(across["eye"] == "both") & (across["axis"] == "phi")].iloc[0]
    assert float(both["mean_of_mean_abs_deg"]) == pytest.approx(0.84, abs=0.01)


def test_animal_level_from_blocks():
    per_block = pd.DataFrame(
        {
            "animal": ["A", "A", "A"],
            "block_key": ["b1", "b2", "b1"],
            "eye": ["L", "L", "R"],
            "axis": ["phi", "phi", "phi"],
            "mean_abs_deg": [1.0, 3.0, 2.0],
        }
    )
    out = animal_level_from_blocks(per_block)
    left = out[(out["eye"] == "L") & (out["axis"] == "phi")].iloc[0]
    assert left["mean_abs_deg"] == pytest.approx(2.0)
    assert left["n_blocks"] == 2


def _eye_trace(times: np.ndarray, spikes: dict[float, float]) -> pd.DataFrame:
    speed = np.full(times.size, 0.01, dtype=float)
    for t_ms, val in spikes.items():
        speed[int(np.argmin(np.abs(times - t_ms)))] = val
    phi = np.cumsum(speed)
    return pd.DataFrame(
        {
            "ms_axis": times,
            "angular_speed_r": speed,
            "k_phi": phi,
            "k_theta": np.zeros_like(times),
        }
    )


def test_export_figure_s3_paper_settings(tmp_path: Path):
    frame_ms = 17.0
    times = np.arange(0.0, 2100.0, frame_ms)
    left = _eye_trace(times, {500.0: 0.5, 2000.0: 0.34})
    right = _eye_trace(times, {510.0: 0.2, 2000.0: 0.85})
    events = pd.DataFrame(
        {
            "animal": ["PV_106", "PV_106", "PV_62", "PV_62"],
            "block": ["007"] * 4,
            "eye": ["L", "R", "L", "R"],
            "saccade_on_ms": [500.0, 510.0, 500.0, 510.0],
            "saccade_off_ms": [534.0, 544.0, 534.0, 544.0],
            "head_movement": [False, False, True, True],
            "speed_profile_angular": [[3.4], [5.1], [2.0], [2.2]],
        }
    )
    block_path = tmp_path / "block_007"
    (block_path / "analysis").mkdir(parents=True)
    spec = BlockSpec(animal="PV_106", block_path=block_path, block_num="007")
    spec62 = BlockSpec(animal="PV_62", block_path=block_path, block_num="007")
    bundle = BlockBundle(
        spec=spec,
        left=left,
        right=right,
        left_csv_meta={},
        right_csv_meta={},
        l_saccades=events.loc[(events["eye"] == "L") & (events["animal"] == "PV_106")].reset_index(drop=True),
        r_saccades=events.loc[(events["eye"] == "R") & (events["animal"] == "PV_106")].reset_index(drop=True),
        all_saccades=events.loc[events["animal"] == "PV_106"].copy(),
    )
    bundle62 = BlockBundle(
        spec=spec62,
        left=left,
        right=right,
        left_csv_meta={},
        right_csv_meta={},
        l_saccades=events.loc[(events["eye"] == "L") & (events["animal"] == "PV_62")].reset_index(drop=True),
        r_saccades=events.loc[(events["eye"] == "R") & (events["animal"] == "PV_62")].reset_index(drop=True),
        all_saccades=events.loc[events["animal"] == "PV_62"].copy(),
    )
    tables = EventTables(
        blocks=[bundle, bundle62],
        all_saccades=events.copy(),
        synced=pd.DataFrame(),
        non_synced=events.copy(),
        params={"binocular": {"sync_diff_ms": 34.0}, "figure_2f": {}},
    )
    written = export_figure_s3(tables, tmp_path, show=False)
    import pickle

    with open(written["figure_S3.pickle"], "rb") as f:
        payload = pickle.load(f)
    assert tuple(payload["macro_range"]) == (0.0, 0.2)
    assert int(payload["bins"]) == 100
    still = payload["still"]
    assert still["norm_counts"].shape == (99, 99)
    assert payload.get("pairing_mode") == "legacy_contra_table"
    assert "PV_62" in payload.get("exclude_animals", [])


def test_rebin_s8j_payload_keeps_mouse_xmax():
    rng = np.random.default_rng(0)
    liz_r = rng.uniform(0, 0.9, 200)
    liz_l = rng.uniform(0, 0.9, 200)
    mou_r = rng.uniform(0, 1.2, 200)
    mou_l = rng.uniform(0, 1.2, 200)
    data = {
        "lizard": {
            "right": liz_r,
            "left": liz_l,
            "weights": np.ones(liz_r.size),
            "range": (0.0, 0.3),
        },
        "mouse": {
            "right": mou_r,
            "left": mou_l,
            "weights": np.ones(mou_r.size),
            "range": (0.0, 1.25),
        },
        "bins": 60,
    }
    out = rebin_s8j_payload(data, pct=99.5)
    assert out["mouse"]["range"][1] == pytest.approx(1.25)
    assert out["mouse"]["ticks"] == [0.0, 0.5, 1.0]
    assert out["lizard"]["range"][1] == pytest.approx(1.0)
    assert out["lizard"]["ticks"] == [0.0, 0.5, 1.0]
    assert out["lizard"]["hist"]["norm_counts"].shape == (59, 59)
    assert out["mouse"]["hist"]["norm_counts"].shape == (59, 59)
    zoom = out["zoom"]
    assert zoom["lizard"]["hist"]["norm_counts"].shape == (59, 59)
    assert zoom["mouse"]["hist"]["norm_counts"].shape == (59, 59)
    assert zoom["lizard"]["range"][1] < out["lizard"]["range"][1]
    assert zoom["mouse"]["range"][1] < out["mouse"]["range"][1]


def test_zoom_square_limits_round_even():
    hi, ticks = zoom_square_limits(0.195)
    assert hi == pytest.approx(0.2)
    assert ticks == [0.0, 0.1, 0.2]
    hi, ticks = zoom_square_limits(0.365)
    assert hi == pytest.approx(0.4)
    assert ticks == [0.0, 0.2, 0.4]


def test_event_square_percentile_keeps_half():
    right = np.array([0.1, 0.2, 0.8, 0.9])
    left = np.array([0.1, 0.2, 0.8, 0.9])
    x = event_square_percentile(right, left, pct=50.0)
    inside = np.mean((right <= x) & (left <= x))
    assert inside == pytest.approx(0.5)
