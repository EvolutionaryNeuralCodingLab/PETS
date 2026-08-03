"""Figures 3e (quiet vs active pupil histogram) and 3f (per-animal z-scored diff).

Ports ``development/old_pipelines_for_ref/pupil_diameter_state_clean.ipynb``
(``plot_combined_eye_probability_histograms`` /
``compare_animals_difference_histogram_zscore_colored``) onto the
``EventTables`` / ``BlockBundle`` pipeline, following the pattern of
``figures_2g_2j.py``: :func:`resolve_figure_dirs`, :func:`write_pickle_with_meta`,
:func:`build_color_map`, :func:`show_and_close`.

**No seaborn.** Histograms are drawn with plain matplotlib bars; the KDE
overlay uses :func:`scipy.stats.gaussian_kde`.

Both figures require per-frame eye traces (``BlockBundle.left`` / ``.right``
with ``ms_axis`` + ``major_ax``) — build ``EventTables`` with
``keep_traces=True`` before calling into this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Collection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde

from eye_tracking_system_tools.analysis.behavior_state import (
    has_behavior_state,
    normalize_label,
    read_behavior_state,
)
from eye_tracking_system_tools.analysis.colors import build_color_map
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.pixel_calibration import read_pixel_size
from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

logger = logging.getLogger(__name__)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def _pupil_diameter_mm(df: pd.DataFrame, mm_per_px: float) -> np.ndarray:
    """Pupil diameter in mm for one eye's trace.

    ``major_ax`` is the raw pupil-diameter measurement in **pixels**;
    ``eye_trace_io`` aliases it to a ``pupil_diameter`` column that is *still
    pixels*. Some raw CSVs additionally carry a legacy ``pupil_diameter``
    column already pre-converted to mm by an earlier pipeline run, so the only
    safe rule is: always multiply ``major_ax`` by the calibration factor when
    ``major_ax`` is present; fall back to ``pupil_diameter`` (assumed px, the
    ``major_ax`` alias) only when ``major_ax`` itself is missing.
    """
    if "major_ax" in df.columns:
        px = df["major_ax"].to_numpy(dtype=float)
    elif "pupil_diameter" in df.columns:
        px = df["pupil_diameter"].to_numpy(dtype=float)
    else:
        raise KeyError("eye trace has neither 'major_ax' nor 'pupil_diameter' column")
    return px * float(mm_per_px)


def aggregate_pupil_mm_by_state(
    tables: EventTables,
    *,
    animals: Collection[str] | None = None,
    exclude_block_keys: Collection[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Pool pupil diameter (mm, both eyes) into ``{animal: {'quiet': [...], 'active': [...]}}``.

    For each block with a behavior-state timeline and pixel calibration, every
    annotated segment slices both eyes' ``ms_axis``:

    * left eye:  ``ms_axis > start`` and ``ms_axis < end``   (strict)
    * right eye: ``ms_axis >= start`` and ``ms_axis <= end`` (inclusive)

    matching the notebook's slicing convention exactly (including the
    left/right asymmetry). Blocks without behavior state, pixel calibration,
    or loaded eye traces are skipped with a log message.
    """
    exclude = {str(k) for k in exclude_block_keys} if exclude_block_keys else set()
    animal_filter = {str(a) for a in animals} if animals is not None else None

    per_animal: dict[str, dict[str, list[float]]] = {}
    for bundle in tables.blocks:
        spec = bundle.spec
        if animal_filter is not None and spec.animal not in animal_filter:
            continue
        if spec.block_key in exclude:
            continue
        if not has_behavior_state(spec):
            continue

        state_df = read_behavior_state(spec)
        if state_df is None or state_df.empty:
            continue

        pix = read_pixel_size(spec.block_path)
        if pix is None:
            logger.warning("[%s] skipped (no LR_pix_size.csv calibration)", spec.block_key)
            continue

        left_df, right_df = bundle.left, bundle.right
        if left_df is None or right_df is None or left_df.empty or right_df.empty:
            logger.warning(
                "[%s] skipped (no eye traces loaded; build EventTables with keep_traces=True)",
                spec.block_key,
            )
            continue
        if "ms_axis" not in left_df.columns or "ms_axis" not in right_df.columns:
            logger.warning("[%s] skipped (eye traces missing ms_axis column)", spec.block_key)
            continue

        left_ms = left_df["ms_axis"].to_numpy(dtype=float)
        right_ms = right_df["ms_axis"].to_numpy(dtype=float)
        left_mm = _pupil_diameter_mm(left_df, pix.l_mm_per_px)
        right_mm = _pupil_diameter_mm(right_df, pix.r_mm_per_px)

        bucket = per_animal.setdefault(spec.animal, {"quiet": [], "active": []})
        n_segments = 0
        for _, row in state_df.iterrows():
            label = normalize_label(row["annotation"])
            if label not in ("quiet", "active"):
                continue
            start, end = float(row["start_time"]), float(row["end_time"])

            lmask = (left_ms > start) & (left_ms < end)
            rmask = (right_ms >= start) & (right_ms <= end)
            lvals = left_mm[lmask]
            rvals = right_mm[rmask]
            lvals = lvals[np.isfinite(lvals)]
            rvals = rvals[np.isfinite(rvals)]

            bucket[label].extend(lvals.tolist())
            bucket[label].extend(rvals.tolist())
            n_segments += 1

        logger.info(
            "[%s] pooled %d behavior segments (quiet n=%d, active n=%d)",
            spec.block_key,
            n_segments,
            len(bucket["quiet"]),
            len(bucket["active"]),
        )

    return per_animal


def _compute_bin_edges(
    all_data: np.ndarray,
    num_bins: int,
    x_range: tuple[float, float],
    outlier_percentiles: tuple[float, float],
) -> np.ndarray:
    """Shared bin-edge logic (ported from the archived Fig 3e reproduction script)."""
    if all_data.size == 0:
        raise ValueError("No pupil data found; nothing to plot.")

    in_range = all_data[(all_data >= x_range[0]) & (all_data <= x_range[1])]

    if in_range.size == 0:
        raw_min, raw_max = float(np.nanmin(all_data)), float(np.nanmax(all_data))
        logger.warning(
            "No data within x_range %s; global span=[%.4g, %.4g]. Expanding.",
            x_range,
            raw_min,
            raw_max,
        )
        o_min, o_max = np.percentile(all_data, outlier_percentiles)
        start, stop = min(o_min, o_max), max(o_min, o_max)
        if not np.isfinite(start) or not np.isfinite(stop) or start == stop:
            start, stop = raw_min, raw_max
            if start == stop:
                stop = start + 1e-6
    else:
        o_min, o_max = np.percentile(in_range, outlier_percentiles)
        start = max(min(o_min, o_max), x_range[0])
        stop = min(max(o_min, o_max), x_range[1])
        if start >= stop:
            start, stop = (start, start + 1e-6) if start == stop else sorted([start, stop])

    bin_edges = np.linspace(start, stop, num_bins + 1)
    if not np.all(bin_edges[1:] > bin_edges[:-1]):
        eps = np.finfo(float).eps * max(1.0, abs(stop))
        bin_edges = np.linspace(start, stop + eps * (num_bins + 1), num_bins + 1)
    return bin_edges


def export_figure_3e(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    """Quiet vs active pupil-diameter probability histogram (+ KDE overlay).

    Paper call site: ``animals: [PV_62]`` (block 038 is excluded implicitly —
    it has no behavior_state.csv at all, so :func:`aggregate_pupil_mm_by_state`
    skips it automatically).
    """
    cfg = dict(tables.params.get("figure_3e", {}))
    num_bins = int(cfg.get("num_bins", 40))
    x_range = tuple(float(v) for v in cfg.get("x_range", [1.0, 2.35]))
    outlier_percentiles = tuple(float(v) for v in cfg.get("outlier_percentiles", [0.001, 99.999]))
    figsize = tuple(float(v) for v in cfg.get("figsize", [2.2, 1.5]))
    colors = {"quiet": cfg.get("color_quiet", "blue"), "active": cfg.get("color_active", "orange")}
    animals = cfg.get("animals")
    exclude_block_keys = cfg.get("exclude_block_keys") or cfg.get("exclude_blocks")

    # After per-figure block filtering, honour YAML ``animals`` only when those
    # animals are still present; otherwise fall back to the selected blocks.
    available = {b.spec.animal for b in tables.blocks}
    if animals:
        animals = [a for a in animals if str(a) in available]
        if not animals:
            animals = sorted(available)
            logger.info(
                "[3e] configured animals absent after block filter; using %s",
                animals,
            )

    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)

    per_animal = aggregate_pupil_mm_by_state(
        tables, animals=animals, exclude_block_keys=exclude_block_keys
    )

    combined: dict[str, list[float]] = {"quiet": [], "active": []}
    for bucket in per_animal.values():
        combined["quiet"].extend(bucket["quiet"])
        combined["active"].extend(bucket["active"])

    all_data = np.asarray(combined["quiet"] + combined["active"], dtype=float)
    all_data = all_data[np.isfinite(all_data)]
    bin_edges = _compute_bin_edges(all_data, num_bins, x_range, outlier_percentiles)
    bin_width = float(bin_edges[1] - bin_edges[0])

    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    plotted_any = False
    hist_data: dict[str, dict[str, np.ndarray]] = {}

    for label in ("quiet", "active"):
        data = np.asarray(combined[label], dtype=float)
        data = data[np.isfinite(data) & (data >= bin_edges[0]) & (data <= bin_edges[-1])]
        if data.size == 0:
            logger.info("[3e] '%s': no samples within bin range; skipping.", label)
            continue

        counts, _ = np.histogram(data, bins=bin_edges)
        probs = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
        ax.bar(
            bin_edges[:-1],
            probs,
            width=np.diff(bin_edges),
            align="edge",
            alpha=0.5,
            color=colors.get(label, "gray"),
            label=label,
        )

        if data.size >= 2 and np.nanstd(data) > 0:
            xs = np.linspace(bin_edges[0], bin_edges[-1], 200)
            kde = gaussian_kde(data)
            ys = kde(xs) * bin_width
            ax.plot(xs, ys, color=colors.get(label, "gray"), lw=1.2)

        hist_data[label] = {"n": data.size, "probability": probs, "raw": data}
        plotted_any = True
        logger.info("[3e] '%s': n=%d", label, data.size)

    ax.set_xlabel("Pupil diameter [mm]", fontsize=10)
    ax.set_ylabel("Likelihood", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    if plotted_any:
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(
            0.5, 0.5, "No data to plot in selected range",
            transform=ax.transAxes, ha="center", va="center", fontsize=8,
        )

    out_pdf = figures_dir / "figure_3e.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    show_and_close(fig, show)

    pkl = metadata_dir / "figure_3e_data.pickle"
    write_pickle_with_meta(
        {
            "figure_name": "figure_3e",
            "bin_edges": bin_edges,
            "combined_aggregated": {k: np.asarray(v, dtype=float) for k, v in combined.items()},
            "per_animal": {
                a: {k: np.asarray(v, dtype=float) for k, v in bucket.items()}
                for a, bucket in per_animal.items()
            },
            "hist": hist_data,
            "num_bins": num_bins,
            "x_range": x_range,
            "outlier_percentiles": outlier_percentiles,
            "colors": colors,
            "animals_filter": list(animals) if animals is not None else None,
        },
        pkl,
        meta={
            "csv_choices": tables.csv_meta,
            "params": cfg,
            "figure": "3e",
            "animals_used": sorted(per_animal.keys()),
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_3e_3f_pupil.export_figure_3e",
    )
    return pkl


def export_figure_3f(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    """Per-animal z-scored (active - quiet) pupil-diameter probability difference."""
    cfg = dict(tables.params.get("figure_3f", {}))
    num_bins = int(cfg.get("num_bins", 15))
    x_range = tuple(float(v) for v in cfg.get("x_range", [-3.0, 3.0]))
    outlier_percentiles = tuple(float(v) for v in cfg.get("outlier_percentiles", [0.1, 99.9]))
    figsize = tuple(float(v) for v in cfg.get("figsize", [2.2, 1.7]))
    exclude_animals = {str(a) for a in cfg.get("exclude_animals", ["PV_57"])}

    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)

    all_per_animal = aggregate_pupil_mm_by_state(tables, animals=None)
    per_animal = {a: b for a, b in all_per_animal.items() if a not in exclude_animals}

    overall = np.asarray(
        [v for bucket in per_animal.values() for v in (bucket["quiet"] + bucket["active"])],
        dtype=float,
    )
    if overall.size == 0:
        raise ValueError("No pupil diameter data found for figure 3f (after exclude_animals).")

    o_min, o_max = np.percentile(overall, outlier_percentiles)
    bin_edges = np.linspace(x_range[0], x_range[1], num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    animal_diffs: dict[str, np.ndarray] = {}
    for animal, bucket in per_animal.items():
        a = np.asarray(bucket["active"], dtype=float)
        q = np.asarray(bucket["quiet"], dtype=float)
        a = a[(a >= o_min) & (a <= o_max)]
        q = q[(q >= o_min) & (q <= o_max)]
        comb = np.concatenate([a, q])
        if comb.size == 0:
            continue
        mu, sd = float(np.nanmean(comb)), float(np.nanstd(comb))
        sd = sd if sd > 0 else 1.0
        az, qz = (a - mu) / sd, (q - mu) / sd

        ha, _ = np.histogram(az, bins=bin_edges)
        hq, _ = np.histogram(qz, bins=bin_edges)
        pa = ha / ha.sum() if ha.sum() > 0 else ha.astype(float)
        pq = hq / hq.sum() if hq.sum() > 0 else hq.astype(float)
        animal_diffs[animal] = pa - pq

    animals = sorted(animal_diffs.keys())
    if not animals:
        raise ValueError("No valid animals remained for figure 3f after preprocessing.")

    color_map = build_color_map(animals, template="okabeito", order=animals)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    legend_handles, legend_labels = [], []
    for animal in animals:
        h, = ax.plot(bin_centers, animal_diffs[animal], lw=1.5, color=color_map[animal], label=animal)
        legend_handles.append(h)
        legend_labels.append(animal)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlim(x_range)
    ax.set_xlabel("Z-scored Pupil Diameter", fontsize=10)
    ax.set_ylabel("Probability Difference", fontsize=10)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=True)
    ax.grid(False)
    fig.tight_layout()

    out_pdf = figures_dir / "figure_3f.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    show_and_close(fig, show)

    legend_pdf = figures_dir / "figure_3f_legend.pdf"
    if legend_handles:
        fig_leg = plt.figure(figsize=(2.0, 0.28 * max(1, len(legend_labels)) + 0.4), dpi=300)
        fig_leg.legend(legend_handles, legend_labels, loc="center", frameon=False, ncol=1, prop={"size": 8})
        fig_leg.savefig(legend_pdf, format="pdf", bbox_inches="tight", dpi=300)
        show_and_close(fig_leg, show)

    pkl = metadata_dir / "figure_3f_data.pickle"
    write_pickle_with_meta(
        {
            "figure_name": "figure_3f",
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "animal_diffs": {a: np.asarray(v, dtype=float) for a, v in animal_diffs.items()},
            "color_map": color_map,
            "num_bins": num_bins,
            "x_range": x_range,
            "outlier_percentiles": outlier_percentiles,
            "exclude_animals": sorted(exclude_animals),
        },
        pkl,
        meta={
            "csv_choices": tables.csv_meta,
            "params": cfg,
            "figure": "3f",
            "animals_used": animals,
            "exclude_animals": sorted(exclude_animals),
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_3e_3f_pupil.export_figure_3f",
    )
    return pkl
