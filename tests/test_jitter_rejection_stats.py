"""Interframe jitter-diff rejection counts."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.jitter_rejection_stats import (
    CSV_NAME,
    build_rejection_table,
    collect_block_eye_stats,
    count_diff_rejections,
    load_dataset_specs,
    rebase_block_path,
    run_jitter_rejection_stats,
)


def test_count_diff_rejections_signed_threshold():
    dist = np.array([0.0, 0.0, 6.0, 6.0, 0.0, 1.0])
    n_frames, n_removed = count_diff_rejections(dist, threshold=5.0)
    assert n_frames == 6
    # diffs: 0, 6, 0, -6, 1 → only the +6 exceeds 5
    assert n_removed == 1


def test_count_diff_rejections_integer_below_six():
    dist = np.array([0.0, 5.0, 10.0, 16.0])
    _n, n_removed = count_diff_rejections(dist, threshold=5.0)
    # diffs 5, 5, 6 → only 6 > 5
    assert n_removed == 1


def _write_report(block_path: Path, left: np.ndarray, right: np.ndarray) -> None:
    analysis = block_path / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    payload = {
        "left_eye": {"top_correlation_dist": np.asarray(left, dtype=float)},
        "right_eye": {"top_correlation_dist": np.asarray(right, dtype=float)},
    }
    with open(analysis / "jitter_report_dict.pkl", "wb") as f:
        pickle.dump(payload, f)


def test_collect_and_aggregate_per_animal(tmp_path: Path):
    a1 = tmp_path / "PV_A" / "2025_01_01" / "block_001"
    a2 = tmp_path / "PV_A" / "2025_01_01" / "block_002"
    b1 = tmp_path / "PV_B" / "2025_01_02" / "block_001"
    _write_report(a1, np.array([0.0, 0.0, 8.0]), np.array([0.0, 0.0, 0.0]))
    _write_report(a2, np.array([0.0, 1.0]), np.array([0.0, 9.0]))
    _write_report(b1, np.array([0.0, 0.0]), np.array([0.0, 0.0, 7.0]))

    specs = [
        BlockSpec("PV_A", a1, "001"),
        BlockSpec("PV_A", a2, "002"),
        BlockSpec("PV_B", b1, "001"),
    ]
    rows = []
    for spec in specs:
        rows.extend(collect_block_eye_stats(spec, threshold=5.0))
    df = build_rejection_table(rows, threshold=5.0)

    animal = df[df["level"] == "animal"].set_index("animal")
    # PV_A: left a1 +1, right a2 +1; frames 3+3 + 2+2 = 10, removed 2
    assert int(animal.loc["PV_A", "n_frames"]) == 10
    assert int(animal.loc["PV_A", "n_frames_removed"]) == 2
    assert animal.loc["PV_A", "pct_frames_removed"] == 20.0
    # PV_B: left 2 frames 0 removed; right 3 frames 1 removed
    assert int(animal.loc["PV_B", "n_frames"]) == 5
    assert int(animal.loc["PV_B", "n_frames_removed"]) == 1

    overall = df[df["level"] == "all"].iloc[0]
    assert int(overall["n_frames"]) == 15
    assert int(overall["n_frames_removed"]) == 3
    assert overall["pct_frames_removed"] == 20.0


def test_run_writes_csv_from_paper_registry(tmp_path: Path):
    block = tmp_path / "PV_106" / "2025_08_06" / "block_008"
    _write_report(block, np.array([0.0, 6.1]), np.array([0.0, 0.0]))
    registry = tmp_path / "paper_blocks.yaml"
    registry.write_text(
        yaml.safe_dump({"animals": {"PV_106": [str(block)]}}),
        encoding="utf-8",
    )
    out = tmp_path / "outputs" / "jitter_rejection_overall" / CSV_NAME
    df, path = run_jitter_rejection_stats(registry=registry, out_csv=out, threshold=5.0)
    assert path == out
    assert path.is_file()
    loaded = pd.read_csv(path)
    all_row = loaded[loaded["level"] == "all"].iloc[0]
    assert int(all_row["n_frames"]) == 4
    assert int(all_row["n_frames_removed"]) == 1
    assert list(df["level"].unique()) == ["block", "animal", "all"]


def test_load_dataset_specs_jitter_registry(tmp_path: Path):
    block = tmp_path / "PV_24" / "block_012"
    block.mkdir(parents=True)
    registry = tmp_path / "jitter.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "blocks": [
                    {
                        "animal": "PV_24",
                        "block_path": str(block),
                        "mount_type": "modular",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = load_dataset_specs(registry)
    assert len(specs) == 1
    assert specs[0].animal == "PV_24"
    assert specs[0].block_path == block


def test_rebase_block_path_from_experiments_prefix(tmp_path: Path):
    local = tmp_path / "PV_106" / "2025_08_06" / "block_008"
    local.mkdir(parents=True)
    registered = Path("/Volumes/Data-1/Nimrod/experiments/PV_106/2025_08_06/block_008")
    assert rebase_block_path(registered, tmp_path) == local
