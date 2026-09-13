"""Monocular rate / asymmetry helpers (lizard vs mouse exploratory)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.binocular import find_synced_saccades_ms
from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.monocular_species import (
    annotate_monocular_asymmetry,
    asymmetry_index,
    concurrency_count_row,
    export_monocular_species_compare,
    hierarchical_bootstrap_difference,
    monocular_counts_by_group,
    rank_against,
)
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "animal",
            "block",
            "eye",
            "saccade_on_ms",
            "saccade_off_ms",
            "peak_velocity",
        ]
    )


def _tables(
    tmp_path: Path,
    *,
    events: pd.DataFrame,
    animal: str,
    block_num: str,
    params: dict | None = None,
) -> EventTables:
    block_path = tmp_path / animal / f"block_{block_num}"
    (block_path / "analysis").mkdir(parents=True, exist_ok=True)
    spec = BlockSpec(animal=animal, block_path=block_path, block_num=block_num)
    empty = _empty_events()
    l_ev = events.loc[events["eye"] == "L"].reset_index(drop=True) if not events.empty else empty.copy()
    r_ev = events.loc[events["eye"] == "R"].reset_index(drop=True) if not events.empty else empty.copy()
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
    synced, nons = (
        find_synced_saccades_ms(events, sync_diff_ms=34.0)
        if not events.empty
        else (empty.copy(), empty.copy())
    )
    return EventTables(
        blocks=[bundle],
        all_saccades=events.copy(),
        synced=synced,
        non_synced=nons,
        params=params or {"binocular": {"sync_diff_ms": 34.0}, "saccade": {"speed_threshold_deg_per_frame": 0.8}},
    )


def _merge(tables: list[EventTables]) -> EventTables:
    return EventTables(
        blocks=[b for t in tables for b in t.blocks],
        all_saccades=pd.concat([t.all_saccades for t in tables], ignore_index=True),
        synced=pd.concat([t.synced for t in tables], ignore_index=True),
        non_synced=pd.concat([t.non_synced for t in tables], ignore_index=True),
        params=tables[0].params,
    )


def test_concurrency_counts_unique_gaze_numbers():
    events = pd.DataFrame(
        {
            "animal": ["PV_1"] * 5,
            "block": ["001"] * 5,
            "eye": ["L", "R", "L", "R", "L"],
            "saccade_on_ms": [0.0, 10.0, 200.0, 205.0, 400.0],
        }
    )
    row = concurrency_count_row(events, sync_diff_ms=34.0)
    assert row["n_binocular"] == 2
    assert row["n_monocular"] == 1
    assert row["n_eye_events"] == 5
    assert row["n_unique_gaze"] == 3
    assert abs(row["pct_monocular_unique_gaze"] - 100.0 / 3.0) < 1e-9
    assert abs(row["pct_monocular_per_eye"] - 20.0) < 1e-9


def test_asymmetry_index_bounds():
    ai = asymmetry_index([10.0, 10.0, 0.0], [0.0, 10.0, 0.0])
    assert abs(ai[0] - 1.0) < 1e-12
    assert abs(ai[1] - 0.0) < 1e-12
    assert np.isnan(ai[2])


def test_annotate_monocular_uses_eye_for_ipsi():
    points = pd.DataFrame(
        {
            "animal": ["PV_1", "PV_1"],
            "block": ["001", "001"],
            "eye": ["L", "R"],
            "2f_source": ["monocular_window", "monocular_window"],
            "left_peak_v": [0.4, 0.05],
            "right_peak_v": [0.02, 0.5],
        }
    )
    out = annotate_monocular_asymmetry(points, detector_deg_per_ms=0.05)
    assert list(out["v_ipsi"]) == [0.4, 0.5]
    assert list(out["v_contra"]) == [0.02, 0.05]
    assert out["strict_silent"].tolist() == [True, False]
    assert out["ai"].iloc[0] > out["ai"].iloc[1]


def test_rank_outside_and_bootstrap_sign(tmp_path: Path):
    rank = rank_against(80.0, np.array([40.0, 45.0, 50.0, 55.0, 60.0]))
    assert rank["outside_range"] is True
    assert rank["n_below"] == 5
    boot = hierarchical_bootstrap_difference(
        np.array([40.0, 45.0, 50.0, 55.0, 60.0]),
        np.array([78.0, 80.0, 82.0, 84.0]),
        n_boot=500,
        rng=np.random.default_rng(0),
    )
    assert boot["observed_diff"] > 0
    assert boot["ci_lo"] > 0


def test_export_without_2f_writes_bundle(tmp_path: Path):
    lizards = []
    for i, animal in enumerate(["PV_106", "PV_143", "PV_126"]):
        t_l = np.array([0.0, 200.0, 400.0, 800.0]) + 10 * i
        t_r = t_l + (8.0 if i < 2 else 80.0)
        events = pd.DataFrame(
            {
                "animal": [animal] * 8,
                "block": ["001"] * 8,
                "eye": ["L"] * 4 + ["R"] * 4,
                "saccade_on_ms": np.concatenate([t_l, t_r]),
                "peak_velocity": 1.2,
            }
        )
        lizards.append(
            _tables(
                tmp_path,
                events=events,
                animal=animal,
                block_num="001",
                params={
                    "binocular": {"sync_diff_ms": 34.0},
                    "saccade": {"speed_threshold_deg_per_frame": 0.8},
                },
            )
        )
    lizard = _merge(lizards)

    mouse_parts = []
    for b in ("012", "013"):
        t_l = np.array([0.0, 300.0, 600.0])
        t_r = t_l + 90.0
        events = pd.DataFrame(
            {
                "animal": ["M_002"] * 6,
                "block": [b] * 6,
                "eye": ["L"] * 3 + ["R"] * 3,
                "saccade_on_ms": np.concatenate([t_l, t_r]),
                "peak_velocity": 4.0,
            }
        )
        mouse_parts.append(
            _tables(
                tmp_path,
                events=events,
                animal="M_002",
                block_num=b,
                params={
                    "binocular": {"sync_diff_ms": 34.0},
                    "saccade": {"speed_threshold_deg_per_frame": 3.23},
                },
            )
        )
    mouse = _merge(mouse_parts)
    written = export_monocular_species_compare(
        lizard,
        mouse,
        tmp_path / "run",
        n_shuffle=20,
        n_boot=50,
        collect_2f=False,
        rng=np.random.default_rng(0),
        show=False,
    )
    assert "monocular_pct_by_animal.pdf" in written
    assert "monocular_shuffle.pdf" in written
    bundle = tmp_path / "run" / "monocular_species_compare"
    assert (bundle / "metadata" / "LOGIC.md").is_file()
    assert (bundle / "metadata" / "species_summary.csv").is_file()
    assert (bundle / "metadata" / "monocular_counts.csv").is_file()
    counts = pd.read_csv(bundle / "metadata" / "monocular_counts.csv")
    animal_rows = counts.loc[counts["level"] == "animal"]
    assert set(animal_rows["species"]) == {"lizard", "mouse"}
    liz_animal = monocular_counts_by_group(lizard, level="animal")
    assert len(liz_animal) == 3
