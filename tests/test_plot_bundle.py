"""Self-contained plot-bundle layout, cohort naming, and replot.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest
import yaml

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.figure_catalog import run_figure
from eye_tracking_system_tools.analysis.pipeline import BlockBundle, EventTables
from eye_tracking_system_tools.analysis.plot_bundle import (
    begin_plot_bundle,
    catalog_folder_id,
    finish_plot_bundle,
    infer_catalog_cohort,
)
from eye_tracking_system_tools.analysis.s1_occupancy import export_s1_occupancy


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["animal", "block", "eye", "saccade_on_ms", "saccade_off_ms"]
    )


def _tables(animal: str, tmp_path: Path) -> EventTables:
    block_path = tmp_path / "block_007"
    (block_path / "analysis").mkdir(parents=True, exist_ok=True)
    spec = BlockSpec(animal=animal, block_path=block_path, block_num="007")
    events = _empty_events()
    bundle = BlockBundle(
        spec=spec,
        left=pd.DataFrame(),
        right=pd.DataFrame(),
        left_csv_meta={},
        right_csv_meta={},
        l_saccades=events.copy(),
        r_saccades=events.copy(),
        all_saccades=events.copy(),
    )
    return EventTables(
        blocks=[bundle],
        all_saccades=events.copy(),
        synced=events.copy(),
        non_synced=events.copy(),
        params={},
    )


def test_infer_catalog_cohort_mouse_vs_lizard(tmp_path: Path):
    mouse = infer_catalog_cohort(_tables("M_002", tmp_path))
    assert mouse["cohort"] == "mouse"
    assert mouse["rule"] == "animal_ids"
    lizard = infer_catalog_cohort(_tables("PV_126", tmp_path))
    assert lizard["cohort"] == "lizard"
    assert catalog_folder_id("2f", "mouse") == "mouse_figure_2f"
    assert catalog_folder_id("2f", "lizard") == "figure_2f"
    assert catalog_folder_id("s3", "lizard") == "figure_S3"


def test_infer_catalog_cohort_mixed_raises(tmp_path: Path):
    a = _tables("M_002", tmp_path)
    b = _tables("PV_126", tmp_path)
    mixed = EventTables(
        blocks=a.blocks + b.blocks,
        all_saccades=pd.concat([a.all_saccades, b.all_saccades], ignore_index=True),
        synced=a.synced,
        non_synced=a.non_synced,
        params={},
    )
    with pytest.raises(ValueError, match="Mixed"):
        infer_catalog_cohort(mixed)
    forced = infer_catalog_cohort(mixed, override="mouse")
    assert forced["rule"] == "explicit"
    assert forced["cohort"] == "mouse"


def test_begin_finish_writes_replot_and_cohort(tmp_path: Path):
    tables = _tables("PV_126", tmp_path)
    bundle = begin_plot_bundle(
        tmp_path,
        "s1_occupancy",
        kind="s1_occupancy",
        tables=tables,
        logic_key="s1_occupancy",
    )
    (bundle.plots_dir / "s1_empirical_ratio_hist.pdf").write_bytes(b"%PDF")
    finish_plot_bundle(bundle)
    assert (bundle.bundle_dir / "replot.py").is_file()
    assert (bundle.metadata_dir / "LOGIC.md").is_file()
    cohort = yaml.safe_load((bundle.metadata_dir / "cohort.yaml").read_text())
    assert cohort["cohort"] == "lizard"
    assert "PV_126" in cohort["animals"]
    params = yaml.safe_load((bundle.metadata_dir / "params.yaml").read_text())
    assert params["kind"] == "s1_occupancy"


def test_export_s1_nests_under_plot_id(tmp_path: Path):
    n = 50
    left = pd.DataFrame(
        {
            "k_phi": list(range(n)),
            "k_theta": [0] * n,
            "ms_axis": [i * 16.67 for i in range(n)],
        }
    )
    tables = _tables("PV_126", tmp_path)
    tables.blocks[0].left = left
    tables.blocks[0].right = left.copy()
    written = export_s1_occupancy(tables, tmp_path, show=False)
    pdf = written["s1_empirical_ratio_hist.pdf"]
    assert pdf.parent.name == "plots"
    assert pdf.parent.parent.name == "s1_occupancy"
    assert (pdf.parent.parent / "replot.py").is_file()


def test_run_figure_mouse_prefix(tmp_path: Path):
    events = pd.DataFrame(
        {
            "animal": ["M_002"] * 4,
            "block": ["007"] * 4,
            "eye": ["L", "R", "L", "R"],
            "saccade_on_ms": [0.0, 10.0, 80.0, 90.0],
            "saccade_off_ms": [20.0, 30.0, 100.0, 110.0],
            "net_angular_disp": [2.0, 3.0, 4.0, 5.0],
            "peak_velocity": [1.0, 1.2, 1.1, 1.3],
            "head_movement": False,
            "overall_angle_deg": [0.0, 10.0, 20.0, 30.0],
        }
    )
    tables = _tables("M_002", tmp_path)
    tables = EventTables(
        blocks=tables.blocks,
        all_saccades=events,
        synced=events.iloc[0:0],
        non_synced=events,
        params={"main_sequence": {"min_events_per_bin": 1, "frame_rate_fps": 60.0}},
    )
    tables.blocks[0].all_saccades = events
    out = run_figure("2e", tables, tmp_path, show=False)
    assert (tmp_path / "mouse_figure_2e" / "plots" / "figure_2e.pdf").is_file()
    assert (tmp_path / "mouse_figure_2e" / "replot.py").is_file()
    assert any(Path(p).name == "figure_2e.pdf" for p in out.values())


def test_run_figure_mixed_animals_raises(tmp_path: Path):
    a = _tables("M_002", tmp_path)
    b = _tables("PV_126", tmp_path)
    mixed = EventTables(
        blocks=a.blocks + b.blocks,
        all_saccades=pd.concat([a.all_saccades, b.all_saccades], ignore_index=True),
        synced=a.synced,
        non_synced=a.non_synced,
        params={},
    )
    with pytest.raises(ValueError, match="Mixed"):
        run_figure("2e", mixed, tmp_path, show=False)


def test_pickle_meta_uses_relative_path(tmp_path: Path):
    from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta

    pkl = tmp_path / "figure_2f" / "metadata" / "figure_2f_nodowncast.pickle"
    write_pickle_with_meta({"ok": True}, pkl, meta={"n": 1}, entrypoint="test")
    sidecar = yaml.safe_load((Path(str(pkl) + ".meta.yaml")).read_text())
    assert sidecar["pickle"] == "metadata/figure_2f_nodowncast.pickle"
    assert not Path(sidecar["pickle"]).is_absolute()


def test_copy_plot_bundles_keeps_folder_tree(tmp_path: Path):
    from eye_tracking_system_tools.analysis.review_collect import copy_plot_bundles

    src = tmp_path / "src_run"
    bundle = begin_plot_bundle(
        src,
        "jitter_turtle",
        kind="jitter_histogram",
        cohort={
            "cohort": "turtle",
            "animals": ["T_18"],
            "block_keys": ["T_18_block_001"],
            "rule": "registry_mount_type",
            "mount_type": "turtle",
        },
        logic_key="jitter_histogram",
    )
    (bundle.plots_dir / "jitter_turtle.pdf").write_bytes(b"%PDF-1.4")
    finish_plot_bundle(bundle)
    dest = tmp_path / "review_answers_latest"
    written = copy_plot_bundles([src], dest)
    copied = dest / "jitter_turtle"
    assert written["jitter_turtle"] == copied
    assert (copied / "plots" / "jitter_turtle.pdf").is_file()
    assert (copied / "replot.py").is_file()
    assert not (dest / "figures").exists()


def test_run_plot_pooled_turtle_bundle(tmp_path: Path):
    import numpy as np

    from eye_tracking_system_tools.analysis.jitter_epochs import (
        BlockSamples,
        JitterBlockSpec,
        run_plot_pooled,
    )

    spec = JitterBlockSpec("T_18", tmp_path / "block_001", "turtle")
    samples = BlockSamples(
        spec=spec,
        samples={"left_eye": np.array([1.0, 2.0, 3.0, 4.0])},
        units="um",
    )
    figures = tmp_path / "figures"
    metadata = tmp_path / "metadata"
    figures.mkdir()
    metadata.mkdir()
    written = run_plot_pooled(
        [],
        figures,
        metadata,
        block_samples=[samples],
        show=False,
    )
    pdf = written["turtle"]
    assert pdf.parent.name == "plots"
    assert pdf.parent.parent.name == "jitter_turtle"
    cohort = yaml.safe_load((pdf.parent.parent / "metadata" / "cohort.yaml").read_text())
    assert cohort["mount_type"] == "turtle"
    assert cohort["block_keys"] == ["T_18_block_001"]
    assert (metadata / "jitter_pool_summary.yaml").is_file()
    assert not (pdf.parent.parent / "metadata" / "jitter_epochs").exists()
    assert (pdf.parent.parent / "replot.py").is_file()


def test_unified_jitter_shared_xmax(tmp_path: Path):
    import numpy as np

    from eye_tracking_system_tools.analysis.jitter_epochs import (
        BlockSamples,
        JitterBlockSpec,
        UNIFIED_JITTER_PDF,
        UNIFIED_JITTER_PLOT_ID,
        export_unified_jitter,
        figure_unified_jitter,
        shared_jitter_xmax,
    )

    rng = np.random.default_rng(0)
    samples = [
        BlockSamples(
            spec=JitterBlockSpec("PV_106", tmp_path / "block_001", "rigid"),
            samples={"left_eye": rng.normal(5.0, 1.0, 80)},
            units="um",
        ),
        BlockSamples(
            spec=JitterBlockSpec("PV_62", tmp_path / "block_002", "modular"),
            samples={"left_eye": rng.normal(40.0, 8.0, 80)},
            units="um",
        ),
        BlockSamples(
            spec=JitterBlockSpec("M_002", tmp_path / "block_003", "mouse"),
            samples={"left_eye": rng.normal(8.0, 1.5, 80)},
            units="um",
        ),
        BlockSamples(
            spec=JitterBlockSpec("T_18", tmp_path / "block_004", "turtle"),
            samples={"left_eye": rng.normal(12.0, 2.0, 80)},
            units="um",
        ),
    ]
    pools = {
        "rigid": samples[0].values,
        "modular": samples[1].values,
        "mouse": samples[2].values,
        "turtle": samples[3].values,
    }
    xmax, source = shared_jitter_xmax(pools)
    assert source == "modular"
    assert xmax == pytest.approx(float(np.nanpercentile(pools["modular"], 99.5)))

    figures = tmp_path / "figures"
    metadata = tmp_path / "metadata"
    figures.mkdir()
    metadata.mkdir()
    written = export_unified_jitter(
        [],
        figures,
        metadata,
        block_samples=samples,
        show=False,
    )
    pdf = written["unified"]
    assert pdf.name == UNIFIED_JITTER_PDF
    assert pdf.parent.parent.name == UNIFIED_JITTER_PLOT_ID
    params = yaml.safe_load((pdf.parent.parent / "metadata" / "params.yaml").read_text())
    assert params["xmax_source"] == "modular"
    assert params["xmax"] == pytest.approx(xmax)
    assert params["bin_widths"]
    assert (pdf.parent.parent / "replot.py").is_file()
    fig = figure_unified_jitter(pools, xmax=xmax, n_bins=15)
    xlims = [ax.get_xlim() for ax in fig.axes]
    assert all(lo == pytest.approx(0.0) and hi == pytest.approx(xmax) for lo, hi in xlims)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_unified_jitter_px_writes_separate_bundle(tmp_path: Path):
    import numpy as np

    from eye_tracking_system_tools.analysis.jitter_epochs import (
        BlockSamples,
        JitterBlockSpec,
        UNIFIED_JITTER_PDF,
        UNIFIED_JITTER_PLOT_ID,
        UNIFIED_JITTER_PLOT_ID_PX,
        export_unified_jitter,
    )

    rng = np.random.default_rng(1)
    samples = [
        BlockSamples(
            spec=JitterBlockSpec("PV_106", tmp_path / "block_001", "rigid"),
            samples={"left_eye": rng.choice([0.0, 1.0, np.sqrt(2), 2.0], 80)},
            units="px",
        ),
        BlockSamples(
            spec=JitterBlockSpec("PV_62", tmp_path / "block_002", "modular"),
            samples={"left_eye": rng.choice([0.0, 1.0, np.sqrt(2), 2.0, 3.0], 80)},
            units="px",
        ),
        BlockSamples(
            spec=JitterBlockSpec("M_002", tmp_path / "block_003", "mouse"),
            samples={"left_eye": rng.choice([0.0, 1.0, np.sqrt(2)], 80)},
            units="px",
        ),
        BlockSamples(
            spec=JitterBlockSpec("T_18", tmp_path / "block_004", "turtle"),
            samples={"left_eye": rng.choice([0.0, 1.0, 2.0], 80)},
            units="px",
        ),
    ]
    figures = tmp_path / "figures"
    metadata = tmp_path / "metadata"
    figures.mkdir()
    metadata.mkdir()
    written = export_unified_jitter(
        [],
        figures,
        metadata,
        units="px",
        block_samples=samples,
        show=False,
    )
    pdf = written["unified"]
    assert pdf.name == UNIFIED_JITTER_PDF
    assert pdf.parent.parent.name == UNIFIED_JITTER_PLOT_ID_PX
    assert pdf.parent.parent.name != UNIFIED_JITTER_PLOT_ID
    params = yaml.safe_load((pdf.parent.parent / "metadata" / "params.yaml").read_text())
    assert params["units"] == "px"


def test_unified_jitter_per_panel_bin_width_fills_lattice(tmp_path: Path):
    import numpy as np

    from eye_tracking_system_tools.analysis.jitter_epochs import (
        BlockSamples,
        JitterBlockSpec,
        export_unified_jitter,
        figure_unified_jitter,
        mount_bin_widths,
        unified_jitter_bin_edges,
    )

    px = np.array([0.0, 1.0, np.sqrt(2), 2.0, np.sqrt(5), np.sqrt(8), 3.0])
    turtle_scale = 50.0
    lizard_scale = 20.0
    turtle_vals = np.repeat(turtle_scale * px, 80)
    lizard_fine = np.repeat(lizard_scale * px, 80)
    lizard_coarse = np.repeat(40.0 * px, 80)
    samples = [
        BlockSamples(
            spec=JitterBlockSpec("PV_106", tmp_path / "block_001", "rigid"),
            samples={"left_eye": lizard_fine},
            units="um",
        ),
        BlockSamples(
            spec=JitterBlockSpec("PV_62", tmp_path / "block_002", "modular"),
            samples={"left_eye": lizard_coarse},
            units="um",
        ),
        BlockSamples(
            spec=JitterBlockSpec("M_002", tmp_path / "block_003", "mouse"),
            samples={"left_eye": lizard_fine},
            units="um",
        ),
        BlockSamples(
            spec=JitterBlockSpec("T_18", tmp_path / "block_004", "turtle"),
            samples={"left_eye": turtle_vals},
            units="um",
        ),
        BlockSamples(
            spec=JitterBlockSpec("T_18", tmp_path / "block_005", "turtle"),
            samples={"left_eye": np.repeat(30.0 * px, 40)},
            units="um",
        ),
    ]
    widths = mount_bin_widths(samples)
    assert widths["turtle"] == pytest.approx(50.0)
    assert widths["modular"] == pytest.approx(40.0)
    assert widths["rigid"] == pytest.approx(20.0)

    xmax = 300.0
    shared = np.linspace(0, xmax, 16)
    hist_shared, _ = np.histogram(turtle_vals, bins=shared)
    assert (hist_shared == 0).any()

    edges = unified_jitter_bin_edges(xmax, bin_width=widths["turtle"])
    hist_panel, _ = np.histogram(turtle_vals, bins=edges)
    occupied = hist_panel[edges[:-1] < turtle_vals.max()]
    assert occupied.size and np.all(occupied > 0)

    (tmp_path / "figures").mkdir()
    (tmp_path / "metadata").mkdir()
    written = export_unified_jitter(
        [],
        tmp_path / "figures",
        tmp_path / "metadata",
        block_samples=samples,
        show=False,
    )
    params = yaml.safe_load(
        (written["unified"].parent.parent / "metadata" / "params.yaml").read_text()
    )
    assert params["bin_widths"]["turtle"] == pytest.approx(50.0)
    assert params["n_bins"] is None

    k_map = {"turtle": 1.0, "lizard": 1.0, "mouse": 2}
    widths_k = mount_bin_widths(samples, k=k_map)
    assert widths_k["turtle"] == pytest.approx(50.0)
    assert widths_k["rigid"] == pytest.approx(20.0)
    assert widths_k["modular"] == pytest.approx(40.0)
    assert widths_k["mouse"] == pytest.approx(40.0)
    written_k = export_unified_jitter(
        [],
        tmp_path / "figures",
        tmp_path / "metadata",
        block_samples=samples,
        pixel_bin_k=k_map,
        show=False,
    )
    params_k = yaml.safe_load(
        (written_k["unified"].parent.parent / "metadata" / "params.yaml").read_text()
    )
    assert params_k["pixel_bin_k"] == {"turtle": 1.0, "lizard": 1.0, "mouse": 2.0}
    assert params_k["bin_widths"]["mouse"] == pytest.approx(40.0)

    pools = {
        "rigid": samples[0].values,
        "modular": samples[1].values,
        "mouse": samples[2].values,
        "turtle": samples[3].values,
    }
    fig = figure_unified_jitter(pools, xmax=xmax, bin_widths=widths)
    xlims = [ax.get_xlim() for ax in fig.axes]
    assert all(lo == pytest.approx(0.0) and hi == pytest.approx(xmax) for lo, hi in xlims)
    bar_widths = [ax.patches[0].get_width() for ax in fig.axes if ax.patches]
    assert len(set(round(w, 6) for w in bar_widths)) > 1
    import matplotlib.pyplot as plt

    plt.close(fig)
