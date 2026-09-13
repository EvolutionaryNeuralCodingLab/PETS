"""Tests for the data yield report (DLC + ellipse completeness)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from eye_tracking_system_tools.analysis.data_yield import (
    apply_yield_removals_in_memory,
    collect_eye_yield,
    collect_eye_yield_from_dataset,
    collect_likelihood_pools_from_dataset,
    fitted_mask,
    load_yield_dataset,
    mask_eye_df_by_manual_ms,
    missing_epoch_lengths,
    pool_yield_summary,
    read_manual_outlier_intervals,
    resolve_yield_eye_csv,
    retag_dataset_block,
    run_yield_report,
    seed_yield_registry_from_jitter,
    summarize_missing_epochs,
    write_yield_registry,
    yield_registry_table_from_dataset,
)
from eye_tracking_system_tools.analysis.data_yield_gui import yield_registry_table
from eye_tracking_system_tools.analysis.jitter_epochs import (
    JitterBlockSpec,
    guess_mount_type,
    pool_block_samples,
    read_registry_blocks,
)
from eye_tracking_system_tools.preprocessing.noise_epochs import (
    CATEGORY_LED_BLINK,
    append_epochs,
)


def test_guess_mount_type_turtle_and_mouse() -> None:
    assert guess_mount_type("Turtle_01") == "turtle"
    assert guess_mount_type("turtle_02") == "turtle"
    assert guess_mount_type("M_002") == "mouse"
    assert guess_mount_type("PV_143") == "modular"


def test_registry_roundtrip_accepts_turtle(tmp_path: Path) -> None:
    specs = [
        JitterBlockSpec("PV_1", tmp_path / "block_a", "modular"),
        JitterBlockSpec("Turtle_1", tmp_path / "block_b", "turtle"),
    ]
    path = tmp_path / "yield.yaml"
    write_yield_registry(path, specs)
    loaded = read_registry_blocks(path)
    assert [s.mount_type for s in loaded] == ["modular", "turtle"]
    assert "Data yield report registry" in path.read_text(encoding="utf-8")


def test_jitter_pool_includes_turtle() -> None:
    from eye_tracking_system_tools.analysis.jitter_epochs import BlockSamples

    samples = [
        BlockSamples(
            spec=JitterBlockSpec("PV_1", Path("/x/block_a"), "modular"),
            samples={"left_eye": np.array([1.0, 2.0])},
            units="px",
        ),
        BlockSamples(
            spec=JitterBlockSpec("Turtle_1", Path("/x/block_b"), "turtle"),
            samples={"left_eye": np.array([9.0, 9.0])},
            units="px",
        ),
    ]
    pools = pool_block_samples(samples)
    assert pools["turtle"].tolist() == [9.0, 9.0]
    assert pools["modular"].tolist() == [1.0, 2.0]
    assert pools["mouse"].size == 0
    assert pools["rigid"].size == 0


def test_missing_epoch_lengths_and_summary() -> None:
    # F F T T T F T T T T T T F  → lengths 3 and 6
    missing = np.array(
        [False, False, True, True, True, False, True, True, True, True, True, True, False]
    )
    lengths = missing_epoch_lengths(missing)
    assert lengths.tolist() == [3, 6]
    stats = summarize_missing_epochs(missing, short_max=5)
    assert stats.n_missing_frames == 9
    assert stats.n_short_epochs == 1
    assert stats.n_long_epochs == 1
    assert stats.n_short_frames == 3
    assert stats.n_long_frames == 6
    assert stats.pct_missing_frames_short == pytest.approx(100.0 * 3 / 9)
    assert stats.pct_missing_frames_long == pytest.approx(100.0 * 6 / 9)
    assert stats.median_epoch_frames == pytest.approx(4.5)


def test_summarize_missing_empty() -> None:
    stats = summarize_missing_epochs(np.zeros(10, dtype=bool))
    assert stats.n_missing_frames == 0
    assert stats.n_epochs == 0
    assert np.isnan(stats.median_epoch_frames)


def _make_eye_df(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eye_frame": np.arange(n),
            "ms_axis": np.arange(n) * 10.0,
            "center_x": np.linspace(1.0, 2.0, n),
            "center_y": np.linspace(1.0, 2.0, n),
            "width": np.ones(n),
            "height": np.ones(n),
            "phi": np.zeros(n),
            "k_phi": np.zeros(n),
            "k_theta": np.zeros(n),
        }
    )


def test_noise_and_manual_mask_in_memory(tmp_path: Path) -> None:
    block = tmp_path / "block_001"
    analysis = block / "analysis"
    analysis.mkdir(parents=True)
    df = _make_eye_df(20)
    append_epochs(
        block,
        "left",
        category=CATEGORY_LED_BLINK,
        frames=[2, 3, 4],
        replace_category=True,
    )
    # Manual: frames 10–12 via ms_axis (100–120)
    man = pd.DataFrame(
        {
            "animal_call": ["PV_1"] * 2,
            "block": ["block_001"] * 2,
            "eye": ["left", "right"],
            "start_ms": [100.0, 0.0],
            "end_ms": [120.0, 10.0],
            "manual_outlier_detected": [True, True],
        }
    )
    man.to_csv(analysis / "manual_event_annotations.csv", index=False)

    masked, hits = apply_yield_removals_in_memory(df, block, "left")
    assert hits["n_noise_hit"] == 3
    assert hits["n_manual_hit"] == 3
    assert masked.loc[[2, 3, 4, 10, 11, 12], "center_x"].isna().all()
    # Original untouched
    assert df["center_x"].notna().all()
    # Right-eye intervals should not affect left
    assert masked.loc[0, "center_x"] == pytest.approx(df.loc[0, "center_x"])


def test_mask_eye_df_by_manual_ms_counts_only_finite() -> None:
    df = _make_eye_df(5)
    df.loc[1, "center_x"] = np.nan
    out, n_hit = mask_eye_df_by_manual_ms(df, [(10.0, 20.0)])  # frames 1–2
    assert n_hit == 1  # frame 2 only (frame 1 already NaN)
    assert out.loc[2, "center_x"] != out.loc[2, "center_x"]  # isnan


def test_resolve_yield_eye_csv_prefers_raw_verified(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    plain = analysis / "left_eye_data.csv"
    raw = analysis / "left_eye_data_raw_verified.csv"
    plain.write_text("center_x\n1\n", encoding="utf-8")
    raw.write_text("center_x\n2\n", encoding="utf-8")
    # Make plain newer — still prefer raw_verified
    import os
    import time

    now = time.time()
    os.utime(raw, (now - 100, now - 100))
    os.utime(plain, (now, now))
    choice = resolve_yield_eye_csv(analysis, "left")
    assert choice.rule == "raw_verified"
    assert choice.path.name == raw.name


def test_collect_eye_yield_end_to_end(tmp_path: Path) -> None:
    block = tmp_path / "PV_1" / "2025_01_01" / "block_001"
    analysis = block / "analysis"
    analysis.mkdir(parents=True)
    df = _make_eye_df(10)
    df.loc[5:6, ["center_x", "center_y"]] = np.nan
    df.to_csv(analysis / "left_eye_data_raw_verified.csv", index=False)
    df.to_csv(analysis / "right_eye_data_raw_verified.csv", index=False)

    spec = JitterBlockSpec("PV_1", block, "rigid")
    recs = collect_eye_yield(spec)
    assert len(recs) == 2
    left = next(r for r in recs if r.eye == "left")
    assert left.n_frames == 10
    assert left.n_fitted == 8
    assert left.yield_pct == pytest.approx(80.0)
    assert left.missing.n_missing_frames == 2
    assert left.missing.n_short_epochs == 1

    summary = pool_yield_summary(recs)
    assert summary["groups"]["rigid"]["yield_pct_weighted"] == pytest.approx(80.0)


def test_run_yield_report_writes_artifacts(tmp_path: Path) -> None:
    block = tmp_path / "PV_1" / "2025_01_01" / "block_001"
    analysis = block / "analysis"
    analysis.mkdir(parents=True)
    df = _make_eye_df(12)
    df.to_csv(analysis / "left_eye_data_raw_verified.csv", index=False)
    df.to_csv(analysis / "right_eye_data_raw_verified.csv", index=False)

    # Minimal DLC-like CSV so likelihood path is exercised (or skipped cleanly)
    le = block / "eye_videos" / "LE"
    le.mkdir(parents=True)
    # No real DLC — report should still succeed for ellipse figures

    specs = [JitterBlockSpec("PV_1", block, "modular")]
    figures = tmp_path / "figures"
    metadata = tmp_path / "metadata"
    written = run_yield_report(specs, figures, metadata, show=False, verbose=False)
    assert (metadata / "per_block_yield.csv").is_file()
    assert (metadata / "yield_pool_summary.yaml").is_file()
    assert (metadata / "data_yield_report.pickle").is_file()
    assert "ellipse_modular_vs_rigid" in written
    assert written["ellipse_modular_vs_rigid"].is_file()
    ellipse = written["ellipse_modular_vs_rigid"]
    assert ellipse.parent.name == "plots"
    assert ellipse.parent.parent.name == "yield_ellipse_modular_vs_rigid"
    assert (ellipse.parent.parent / "replot.py").is_file()
    cohort = yaml.safe_load(
        (ellipse.parent.parent / "metadata" / "cohort.yaml").read_text()
    )
    assert cohort["mount_type"] == "modular_vs_rigid"


def test_seed_yield_registry_from_jitter(tmp_path: Path) -> None:
    jitter = tmp_path / "jitter.yaml"
    yield_reg = tmp_path / "yield.yaml"
    jitter.write_text(
        yaml.safe_dump(
            {
                "blocks": [
                    {
                        "animal": "PV_1",
                        "block_path": str(tmp_path / "b1"),
                        "mount_type": "rigid",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    seed_yield_registry_from_jitter(yield_reg, jitter, overwrite=True)
    specs = read_registry_blocks(yield_reg)
    assert len(specs) == 1
    assert specs[0].mount_type == "rigid"
    # No overwrite when already populated
    jitter.write_text(
        yaml.safe_dump(
            {
                "blocks": [
                    {
                        "animal": "M_1",
                        "block_path": str(tmp_path / "b2"),
                        "mount_type": "mouse",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    seed_yield_registry_from_jitter(yield_reg, jitter, overwrite=False)
    specs2 = read_registry_blocks(yield_reg)
    assert specs2[0].animal == "PV_1"


def test_fitted_mask() -> None:
    df = _make_eye_df(4)
    df.loc[1, "center_x"] = np.nan
    assert fitted_mask(df).tolist() == [True, False, True, True]


def test_read_manual_outlier_intervals_filters(tmp_path: Path) -> None:
    block = tmp_path / "block"
    (block / "analysis").mkdir(parents=True)
    pd.DataFrame(
        {
            "eye": ["left", "left", "right"],
            "start_ms": [0.0, 10.0, 0.0],
            "end_ms": [5.0, 15.0, 5.0],
            "manual_outlier_detected": [True, False, True],
        }
    ).to_csv(block / "analysis" / "manual_event_annotations.csv", index=False)
    left = read_manual_outlier_intervals(block, "left")
    assert left == [(0.0, 5.0)]
    assert read_manual_outlier_intervals(block, "right") == [(0.0, 5.0)]


def _write_minimal_block(block: Path, *, n: int = 10, nan_slice: slice | None = None) -> None:
    analysis = block / "analysis"
    analysis.mkdir(parents=True)
    df = _make_eye_df(n)
    if nan_slice is not None:
        df.loc[nan_slice, ["center_x", "center_y"]] = np.nan
    df.to_csv(analysis / "left_eye_data_raw_verified.csv", index=False)
    df.to_csv(analysis / "right_eye_data_raw_verified.csv", index=False)
    append_epochs(
        block,
        "left",
        category=CATEGORY_LED_BLINK,
        frames=[1, 2],
        replace_category=True,
    )


def test_load_yield_dataset_and_compute_from_cache(tmp_path: Path) -> None:
    block = tmp_path / "PV_1" / "2025_01_01" / "block_001"
    _write_minimal_block(block, n=10, nan_slice=slice(5, 7))
    spec = JitterBlockSpec("PV_1", block, "modular")
    dataset = load_yield_dataset([spec], verbose=False)
    assert len(dataset) == 1
    assert dataset.blocks[spec.block_key].left.eye_df is not None
    assert dataset.blocks[spec.block_key].left.has_noise_file

    recs = collect_eye_yield_from_dataset(dataset)
    assert len(recs) == 2
    left = next(r for r in recs if r.eye == "left")
    # Noise epochs [1, 2] plus two pre-existing NaNs at frames 5–6.
    assert left.n_noise_hit == 2
    assert left.n_frames == 10
    assert left.n_fitted <= 8  # at least the pre-existing NaNs
    assert left.n_fitted == 10 - left.missing.n_missing_frames
    assert left.yield_pct == pytest.approx(100.0 * left.n_fitted / 10)

    disk_recs = collect_eye_yield(spec)
    disk_left = next(r for r in disk_recs if r.eye == "left")
    assert left.n_fitted == disk_left.n_fitted
    assert left.yield_pct == pytest.approx(disk_left.yield_pct)

    qc = yield_registry_table_from_dataset(dataset)
    assert float(qc.loc[0, "yield_left_pct"]) == pytest.approx(round(left.yield_pct, 2))

    # GUI helper accepts dataset
    qc2 = yield_registry_table([spec], dataset=dataset)
    assert float(qc2.loc[0, "yield_left_pct"]) == pytest.approx(round(left.yield_pct, 2))

    written = run_yield_report(
        [spec],
        tmp_path / "figures",
        tmp_path / "metadata",
        dataset=dataset,
        show=False,
        verbose=False,
    )
    assert "ellipse_modular_vs_rigid" in written
    pools = collect_likelihood_pools_from_dataset(dataset)
    assert pools["modular"].size == 0  # no DLC in this fixture


def test_retag_dataset_block(tmp_path: Path) -> None:
    block = tmp_path / "PV_1" / "2025_01_01" / "block_001"
    _write_minimal_block(block)
    spec = JitterBlockSpec("PV_1", block, "modular")
    dataset = load_yield_dataset([spec], verbose=False)
    updated = retag_dataset_block(dataset, spec.block_key, "rigid")
    assert updated.mount_type == "rigid"
    assert dataset.blocks[spec.block_key].spec.mount_type == "rigid"
    assert dataset.specs[0].mount_type == "rigid"


def test_yield_registry_table_without_dataset_skips_yield(tmp_path: Path) -> None:
    block = tmp_path / "PV_1" / "2025_01_01" / "block_001"
    _write_minimal_block(block)
    spec = JitterBlockSpec("PV_1", block, "rigid")
    qc = yield_registry_table([spec])
    assert qc.loc[0, "yield_left_pct"] is None or pd.isna(qc.loc[0, "yield_left_pct"])
    assert "load DATA" in str(qc.loc[0, "yield_note"])


def test_ellipse_figures_render_with_labels() -> None:
    from eye_tracking_system_tools.analysis.data_yield import (
        figure_ellipse_yield_by_condition,
        figure_ellipse_yield_modular_vs_rigid,
        figure_ellipse_yield_single,
    )
    import matplotlib.pyplot as plt

    summary = {
        "groups": {
            "modular": {
                "yield_pct_weighted": 91.2,
                "pct_missing_frames_short": 40.5,
                "pct_missing_frames_long": 59.5,
            },
            "rigid": {
                "yield_pct_weighted": 88.1,
                "pct_missing_frames_short": 33.3,
                "pct_missing_frames_long": 66.7,
            },
            "mouse": {
                "yield_pct_weighted": 95.0,
                "pct_missing_frames_short": 10.0,
                "pct_missing_frames_long": 90.0,
            },
        }
    }
    figs = [
        figure_ellipse_yield_modular_vs_rigid(summary),
        figure_ellipse_yield_single(summary, "mouse", color="gray"),
        figure_ellipse_yield_by_condition(summary),
    ]
    assert all(f is not None for f in figs)
    for f in figs:
        plt.close(f)
