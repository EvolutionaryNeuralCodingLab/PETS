"""Tests for Fig 2f / S3 colormap trial exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.figures_2f_colormap_export import (
    COLORMAP_VARIANTS,
    colormap_trials_run_dir,
    export_2f_colormap_bundle,
    resolve_figure_2f_colormap,
    verify_colormap_trials_run,
)
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables


def _eye_trace(times: np.ndarray, spikes: dict[float, float]) -> pd.DataFrame:
    speed = np.full(times.size, 0.01, dtype=float)
    for t_ms, val in spikes.items():
        speed[int(np.argmin(np.abs(times - t_ms)))] = val
    return pd.DataFrame({"ms_axis": times, "angular_speed_r": speed})


def _tables(
    tmp_path: Path,
    *,
    head_flags: list[bool],
    animal: str = "PV_126",
) -> EventTables:
    from eye_tracking_system_tools.analysis.binocular import find_synced_saccades_ms

    frame_ms = 17.0
    times = np.arange(0.0, 2100.0, frame_ms)
    left = _eye_trace(times, {2000.0: 0.34, 500.0: 0.5})
    right = _eye_trace(times, {100.0: 99.0, 1000.0: 0.85, 600.0: 0.4})
    block_num = "007"
    events = pd.DataFrame(
        {
            "animal": [animal] * len(head_flags),
            "block": [block_num] * len(head_flags),
            "eye": ["L", "R", "L", "R", "L", "R"][: len(head_flags)],
            "saccade_on_ms": [100.0, 110.0, 1000.0, 2000.0, 600.0, 610.0][: len(head_flags)],
            "saccade_off_ms": [134.0, 144.0, 1034.0, 2034.0, 634.0, 644.0][: len(head_flags)],
            "head_movement": head_flags,
            "speed_profile_angular": [[3.4], [5.1], [1.7], [2.55], [2.0], [2.2]][: len(head_flags)],
        }
    )
    block_path = tmp_path / f"block_{block_num}"
    (block_path / "analysis").mkdir(parents=True, exist_ok=True)
    spec = BlockSpec(animal=animal, block_path=block_path, block_num=block_num)
    l_ev = events.loc[events["eye"] == "L"].reset_index(drop=True)
    r_ev = events.loc[events["eye"] == "R"].reset_index(drop=True)
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
    params = {
        "binocular": {"sync_diff_ms": 34.0},
        "figure_2f": {
            "event_mode": "all",
            "sample_mode": "contra_window",
            "contra_sample_ms": 51.0,
            "require_head_stationary": False,
            "exclude_animals": [],
            "macro_range": [0.0, 0.5],
            "micro_range": [0.0, 0.1],
        },
    }
    synced, nons = find_synced_saccades_ms(events, sync_diff_ms=34.0)
    return EventTables(
        blocks=[bundle],
        all_saccades=events.copy(),
        synced=synced,
        non_synced=nons,
        params=params,
    )


def test_resolve_figure_2f_colormap_variants():
    paper = resolve_figure_2f_colormap("paper_white0")
    rgba = paper(0.0)
    assert rgba[0] == pytest.approx(1.0)
    assert rgba[1] == pytest.approx(1.0)
    assert rgba[2] == pytest.approx(1.0)
    turbo = resolve_figure_2f_colormap("turbo")
    assert turbo(0.0)[0] != pytest.approx(1.0)
    hot = resolve_figure_2f_colormap("hot")
    # Standard hot: zero is black, not white.
    assert hot(0.0)[0] == pytest.approx(0.0, abs=0.05)
    hot_r = resolve_figure_2f_colormap("hot_r")
    # Reversed hot: low is white, high is black.
    lo = hot_r(0.0)
    hi = hot_r(1.0)
    assert lo[0] == pytest.approx(1.0, abs=0.05)
    assert lo[1] == pytest.approx(1.0, abs=0.05)
    assert lo[2] == pytest.approx(1.0, abs=0.05)
    assert hi[0] == pytest.approx(0.0, abs=0.05)
    assert hi[1] == pytest.approx(0.0, abs=0.05)
    assert hi[2] == pytest.approx(0.0, abs=0.05)


def test_export_2f_colormap_bundle_2f_and_s3(tmp_path: Path):
    run = colormap_trials_run_dir(tmp_path, "")
    tables = _tables(tmp_path, head_flags=[False, False, False, False, True, True])

    written_2f = export_2f_colormap_bundle(
        tables, run, plot_id="figure_2f", figure_kind="2f", show=False
    )
    assert len(written_2f) == len(COLORMAP_VARIANTS) * 2 + 1  # pdfs + pickle
    plots = run / "figure_2f" / "plots"
    for variant in COLORMAP_VARIANTS:
        assert (plots / f"figure_2f_{variant}.pdf").is_file()
        assert (plots / f"figure_2f_colorbar_{variant}.pdf").is_file()
    assert (run / "figure_2f" / "replot.py").is_file()
    assert (run / "figure_2f" / "metadata" / "LOGIC.md").is_file()

    written_s3 = export_2f_colormap_bundle(
        tables, run, plot_id="figure_S3", figure_kind="s3", show=False
    )
    s3_plots = run / "figure_S3" / "plots"
    for variant in COLORMAP_VARIANTS:
        assert (s3_plots / f"figure_S3_head_still_{variant}.pdf").is_file()
        assert (s3_plots / f"figure_S3_head_moving_{variant}.pdf").is_file()
        assert (s3_plots / f"figure_S3_colorbar_{variant}.pdf").is_file()
    assert len(written_s3) >= len(COLORMAP_VARIANTS) * 3


def test_export_mouse_colormap_bundle(tmp_path: Path):
    run = colormap_trials_run_dir(tmp_path, "mouse")
    tables = _tables(tmp_path, head_flags=[False, False, False, False], animal="M_002")
    tables.params["figure_2f"]["auto_view_limits"] = True

    export_2f_colormap_bundle(
        tables,
        run,
        plot_id="mouse_figure_2f",
        figure_kind="2f",
        cohort_override="mouse",
        show=False,
    )
    plots = run / "mouse_figure_2f" / "plots"
    assert (plots / "mouse_figure_2f_paper_white0.pdf").is_file()


def test_verify_colormap_trials_run(tmp_path: Path):
    run = colormap_trials_run_dir(tmp_path, "")
    tables = _tables(tmp_path, head_flags=[False, False, False, False, True, True])
    export_2f_colormap_bundle(tables, run, plot_id="figure_2f", figure_kind="2f", show=False)
    export_2f_colormap_bundle(tables, run, plot_id="figure_S3", figure_kind="s3", show=False)
    mouse_tables = _tables(tmp_path, head_flags=[False, False, False, False], animal="M_002")
    export_2f_colormap_bundle(
        mouse_tables,
        run,
        plot_id="mouse_figure_2f",
        figure_kind="2f",
        cohort_override="mouse",
        show=False,
    )
    report = verify_colormap_trials_run(run)
    assert report["all_complete"]
    assert report["bundles"]["figure_2f"]["complete"]
    assert report["bundles"]["mouse_figure_2f"]["complete"]
    assert report["bundles"]["figure_S3"]["complete"]
