"""Fig 2f / S3 exports with multiple colormap variants (paper white0, turbo, hot)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml

from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
    Figure2fPoints,
    _draw_coupling_heatmap,
    collect_figure_2f_points,
    compact_axis_ticks,
    figure_2f_config,
    histogram2d_from_points,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import (
    begin_plot_bundle,
    finish_plot_bundle,
    infer_catalog_cohort,
)
from eye_tracking_system_tools.analysis.run_layout import assert_not_reproduction

COLORMAP_VARIANTS: tuple[str, ...] = ("paper_white0", "turbo", "hot", "hot_r")

COLORMAP_LABELS: dict[str, str] = {
    "paper_white0": "Paper style (turbo, white at zero)",
    "turbo": "Matplotlib turbo",
    "hot": "Hot (black → red → yellow → white; no white at zero)",
    "hot_r": "Reversed hot (white → yellow → red → black)",
}

FigureKind = Literal["2f", "s3"]


def colormap_trials_run_dir(
    out_root: Path | str,
    tag: str = "",
    *,
    when: datetime | None = None,
) -> Path:
    """Return ``outputs/2f_colormap_trials`` or a dated tagged snapshot folder."""
    when = when or datetime.now()
    out_root = Path(out_root)
    safe = "".join(c if (c.isalnum() or c in "-.") else "_" for c in str(tag).strip())
    safe = safe.strip("_")
    if not safe:
        name = "2f_colormap_trials"
    else:
        name = "_".join(
            [
                "2f_colormap_trials",
                safe,
                when.strftime("%Y%m%d"),
                when.strftime("%H"),
                when.strftime("%M"),
            ]
        )
    path = out_root / name
    assert_not_reproduction(path)
    return path


def resolve_figure_2f_colormap(variant: str):
    """Return a matplotlib colormap for the named variant."""
    key = str(variant).strip().lower()
    if key == "paper_white0":
        turbo = plt.get_cmap("turbo", 256)
        colors = turbo(np.linspace(0, 1, 256))
        colors[0] = np.array([1, 1, 1, 1])
        return mcolors.ListedColormap(colors)
    if key == "turbo":
        return plt.get_cmap("turbo")
    if key == "hot":
        return plt.get_cmap("hot")
    if key == "hot_r":
        # Low → high: white → yellow → red → black (matplotlib hot reversed).
        return plt.get_cmap("hot_r")
    raise ValueError(
        f"Unknown colormap variant {variant!r}; expected one of {COLORMAP_VARIANTS}"
    )


def _draw_2f_panels(
    macro: dict,
    micro: dict,
    *,
    macro_range: tuple[float, float],
    micro_range: tuple[float, float],
    macro_ticks: list[float],
    micro_ticks: list[float],
    vmax: float,
    cmap,
) -> plt.Figure:
    # Same canvas for lizard and mouse so auto-limit tick labels do not shrink panels.
    fig, axs = plt.subplots(1, 2, figsize=(3.2, 1.85), dpi=300, constrained_layout=True)
    for ax, hist, rng, title, ticks in (
        (axs[0], macro, macro_range, "Macro", macro_ticks),
        (axs[1], micro, micro_range, "Micro", micro_ticks),
    ):
        _draw_coupling_heatmap(ax, hist, rng, ticks, vmax=vmax, cmap=cmap)
        ax.set_title(title, fontsize=8)
    return fig


def _draw_colorbar(*, vmax: float, cmap) -> plt.Figure:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1))
    sm.set_array([])
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Probability", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    return fig_cbar


def _logic_md_for_bundle(*, plot_id: str, figure_kind: FigureKind, cohort: dict[str, Any]) -> str:
    cohort_name = str(cohort.get("cohort", "unspecified"))
    animals = cohort.get("animals") or []
    cmap_lines = "\n".join(f"- **{k}** — {COLORMAP_LABELS[k]}" for k in COLORMAP_VARIANTS)
    if figure_kind == "2f":
        body = (
            f"Inter-ocular peak-speed 2D histogram ({plot_id}, cohort={cohort_name}).\n"
            "Concurrent L/R pairs contribute one point from both detected event peaks. "
            "Unpaired events use ipsilateral peak vs contralateral ±51 ms window sample. "
            "Equal-animal weights.\n"
        )
        if cohort_name == "lizard":
            body += "Head-stationary filter applied (require_head_stationary).\n"
        else:
            body += "No head-stationary filter (mouse supplementary params).\n"
    else:
        body = (
            f"Peak-speed coupling still vs moving ({plot_id}, cohort={cohort_name}).\n"
            "Same collector as Fig 2f but all saccades (no head-stationary filter). "
            "Still and moving panels share vmax within each colormap variant.\n"
        )
    return (
        body
        + f"\nAnimals: {', '.join(map(str, animals)) or '(none)'}\n\n"
        + "Colormap variants:\n"
        + cmap_lines
        + "\n\nReplot reads the pickle and redraws every variant into plots/replot/.\n"
        + "Replot does not re-detect saccades or reload lab volumes.\n"
    )


def _collect_s3_histograms(tables: EventTables) -> tuple[Figure2fPoints, dict, dict, float, dict]:
    cfg = figure_2f_config(
        tables,
        {
            "event_mode": "all",
            "require_head_stationary": False,
            "exclude_animals": [],
        },
    )
    collected = collect_figure_2f_points(tables, cfg=cfg)
    points = collected.points
    if "head_movement" not in points.columns:
        raise ValueError("S3 colormap export needs head_movement labels on events")
    hm = points["head_movement"].fillna(False).astype(bool)
    still = Figure2fPoints(
        points=points.loc[~hm].reset_index(drop=True),
        macro_range=collected.macro_range,
        micro_range=collected.micro_range,
        bins=collected.bins,
        cfg=cfg,
    )
    moving = Figure2fPoints(
        points=points.loc[hm].reset_index(drop=True),
        macro_range=collected.macro_range,
        micro_range=collected.micro_range,
        bins=collected.bins,
        cfg=cfg,
    )
    hist_still = histogram2d_from_points(still, view="macro")
    hist_moving = histogram2d_from_points(moving, view="macro")
    vmax = float(
        max(
            np.nanmax(hist_still["norm_counts"]) if hist_still["norm_counts"].size else 0.0,
            np.nanmax(hist_moving["norm_counts"]) if hist_moving["norm_counts"].size else 0.0,
            1e-12,
        )
    )
    return collected, hist_still, hist_moving, vmax, cfg


def export_2f_colormap_bundle(
    tables: EventTables,
    out_dir: Path | str,
    *,
    plot_id: str,
    figure_kind: FigureKind = "2f",
    cohort_override: str | None = None,
    params_overrides: dict[str, Any] | None = None,
    colormap_variants: tuple[str, ...] | None = None,
    show: bool = False,
) -> dict[str, Path]:
    """
    Write a self-contained plot bundle with Fig 2f or S3 PDFs for each colormap variant.

    Produces ``plots/figure_*_{variant}.pdf`` (+ colorbars) and
    ``metadata/figure_*_colormap_trials.pickle``.
    """
    out_dir = Path(out_dir)
    variants = tuple(colormap_variants) if colormap_variants is not None else COLORMAP_VARIANTS
    for v in variants:
        if v not in COLORMAP_VARIANTS:
            raise ValueError(f"Unknown colormap variant {v!r}; expected one of {COLORMAP_VARIANTS}")
    cohort = infer_catalog_cohort(tables, override=cohort_override)
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="figure_2f_colormap_trials",
        tables=tables,
        cohort=cohort,
        logic_key="figure_2f_colormap_trials",
        params=dict(tables.params or {}),
        extra={"figure_kind": figure_kind, "colormap_variants": list(variants)},
    )
    plots_dir = bundle.plots_dir
    meta_dir = bundle.metadata_dir
    written: dict[str, Path] = {}

    if figure_kind == "2f":
        cfg = figure_2f_config(tables, params_overrides)
        collected = collect_figure_2f_points(tables, cfg=cfg)
        macro = histogram2d_from_points(collected, view="macro")
        micro = histogram2d_from_points(collected, view="micro")
        vmax = float(
            max(np.nanmax(macro["norm_counts"]), np.nanmax(micro["norm_counts"]), 1e-12)
        )
        pickle_data: dict[str, Any] = {
            "figure_name": f"{plot_id}_colormap_trials",
            "figure_kind": "2f",
            "plot_id": plot_id,
            "colormap_variants": list(variants),
            "macro": macro,
            "micro": micro,
            "macro_range": collected.macro_range,
            "micro_range": collected.micro_range,
            "macro_tick_list": list(
                compact_axis_ticks(
                    collected.cfg.get("macro_tick_list"),
                    float(collected.macro_range[0]),
                    float(collected.macro_range[1]),
                )
            ),
            "micro_tick_list": list(
                compact_axis_ticks(
                    collected.cfg.get("micro_tick_list"),
                    float(collected.micro_range[0]),
                    float(collected.micro_range[1]),
                )
            ),
            "vmax_all": vmax,
            "right_eye_speeds": collected.right_peak_v,
            "left_eye_speeds": collected.left_peak_v,
            "weights": collected.weights,
            "bins": collected.bins,
            "params_figure_2f": cfg,
            "n_points": int(len(collected.points)),
        }
        stem = "figure_2f" if plot_id == "figure_2f" else plot_id.replace("mouse_", "mouse_")
        if plot_id == "mouse_figure_2f":
            stem = "mouse_figure_2f"
        for variant in variants:
            cmap = resolve_figure_2f_colormap(variant)
            fig = _draw_2f_panels(
                macro,
                micro,
                macro_range=collected.macro_range,
                micro_range=collected.micro_range,
                macro_ticks=pickle_data["macro_tick_list"],
                micro_ticks=pickle_data["micro_tick_list"],
                vmax=vmax,
                cmap=cmap,
            )
            pdf = plots_dir / f"{stem}_{variant}.pdf"
            fig.savefig(pdf, bbox_inches="tight", dpi=300)
            show_and_close(fig, show)
            written[pdf.name] = pdf
            fig_cbar = _draw_colorbar(vmax=vmax, cmap=cmap)
            cbar_pdf = plots_dir / f"{stem}_colorbar_{variant}.pdf"
            fig_cbar.savefig(cbar_pdf, bbox_inches="tight", dpi=150)
            show_and_close(fig_cbar, show)
            written[cbar_pdf.name] = cbar_pdf
    else:
        collected, hist_still, hist_moving, vmax, cfg = _collect_s3_histograms(tables)
        hm = collected.points["head_movement"].fillna(False).astype(bool)
        rng = collected.macro_range
        ticks = list(
            compact_axis_ticks(
                cfg.get("macro_tick_list", [0.0, 0.25, 0.5]),
                float(rng[0]),
                float(rng[1]),
            )
        )
        pickle_data = {
            "figure_name": "figure_S3_colormap_trials",
            "figure_kind": "s3",
            "plot_id": plot_id,
            "colormap_variants": list(variants),
            "still": hist_still,
            "moving": hist_moving,
            "hist_still": hist_still,
            "hist_moving": hist_moving,
            "macro_range": rng,
            "range": rng,
            "ticks": ticks,
            "vmax": vmax,
            "vmax_all": vmax,
            "bins": collected.bins,
            "n_still": int((~hm).sum()),
            "n_moving": int(hm.sum()),
            "params_figure_2f": cfg,
        }
        for variant in variants:
            cmap = resolve_figure_2f_colormap(variant)
            for panel, hist, n, suffix in (
                ("head_still", hist_still, int((~hm).sum()), "head_still"),
                ("head_moving", hist_moving, int(hm.sum()), "head_moving"),
            ):
                fig, ax = plt.subplots(figsize=(1.85, 1.85), dpi=300)
                _draw_coupling_heatmap(ax, hist, rng, ticks, vmax=vmax, cmap=cmap)
                ax.set_title(f"n={n}", fontsize=8)
                pdf = plots_dir / f"figure_S3_{suffix}_{variant}.pdf"
                fig.savefig(pdf, bbox_inches="tight", dpi=300)
                show_and_close(fig, show)
                written[pdf.name] = pdf
            fig_cbar = _draw_colorbar(vmax=vmax, cmap=cmap)
            cbar_pdf = plots_dir / f"figure_S3_colorbar_{variant}.pdf"
            fig_cbar.savefig(cbar_pdf, bbox_inches="tight", dpi=150)
            show_and_close(fig_cbar, show)
            written[cbar_pdf.name] = cbar_pdf

    pkl = meta_dir / f"{plot_id}_colormap_trials.pickle"
    write_pickle_with_meta(
        pickle_data,
        pkl,
        meta={
            "figure": figure_kind,
            "plot_id": plot_id,
            "cohort": cohort,
            "colormap_variants": list(variants),
            "params": dict(tables.params or {}),
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_2f_colormap_export.export_2f_colormap_bundle",
    )
    written[pkl.name] = pkl

    finish_plot_bundle(bundle)
    logic = _logic_md_for_bundle(plot_id=plot_id, figure_kind=figure_kind, cohort=cohort)
    (meta_dir / "LOGIC.md").write_text(
        logic
        + "\n\nTo redraw without the PETS package:\n"
        "  python replot.py\n"
        "PDFs land in plots/replot/ (originals in plots/ are not overwritten).\n"
        "  python replot.py --overwrite   # replace plots/ in place\n",
        encoding="utf-8",
    )
    (bundle.bundle_dir / "LOGIC.md").write_text((meta_dir / "LOGIC.md").read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "plot_id": plot_id,
        "figure_kind": figure_kind,
        "cohort": cohort,
        "colormap_variants": list(COLORMAP_VARIANTS),
        "written": {k: str(v.resolve()) for k, v in written.items()},
    }
    with open(meta_dir / "export_manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return written


def verify_colormap_trials_run(run_dir: Path | str) -> dict[str, Any]:
    """Check that expected plot bundles and PDFs exist under a trials run folder."""
    run_dir = Path(run_dir)
    expected_bundles = ("figure_2f", "mouse_figure_2f", "figure_S3")
    report: dict[str, Any] = {"run_dir": str(run_dir.resolve()), "bundles": {}}
    for plot_id in expected_bundles:
        bundle_dir = run_dir / plot_id
        plots_dir = bundle_dir / "plots"
        meta_dir = bundle_dir / "metadata"
        bundle_report: dict[str, Any] = {
            "exists": bundle_dir.is_dir(),
            "plots_dir": plots_dir.is_dir(),
            "metadata_dir": meta_dir.is_dir(),
            "replot_py": (bundle_dir / "replot.py").is_file(),
            "logic_md": (meta_dir / "LOGIC.md").is_file() or (bundle_dir / "LOGIC.md").is_file(),
            "pdfs": [],
            "missing_pdfs": [],
        }
        if plots_dir.is_dir():
            pdfs = sorted(p.name for p in plots_dir.glob("*.pdf"))
            bundle_report["pdfs"] = pdfs
        kind = "s3" if plot_id == "figure_S3" else "2f"
        stem = plot_id if plot_id != "figure_2f" else "figure_2f"
        if plot_id == "mouse_figure_2f":
            stem = "mouse_figure_2f"
        expected: list[str] = []
        for variant in COLORMAP_VARIANTS:
            if kind == "2f":
                expected.extend([f"{stem}_{variant}.pdf", f"{stem}_colorbar_{variant}.pdf"])
            else:
                expected.extend(
                    [
                        f"figure_S3_head_still_{variant}.pdf",
                        f"figure_S3_head_moving_{variant}.pdf",
                        f"figure_S3_colorbar_{variant}.pdf",
                    ]
                )
        bundle_report["missing_pdfs"] = [p for p in expected if p not in bundle_report["pdfs"]]
        bundle_report["complete"] = (
            bundle_report["exists"]
            and bundle_report["replot_py"]
            and bundle_report["logic_md"]
            and not bundle_report["missing_pdfs"]
            and any(meta_dir.glob("*.pickle"))
        )
        report["bundles"][plot_id] = bundle_report
    report["all_complete"] = all(b.get("complete") for b in report["bundles"].values())
    return report
