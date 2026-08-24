"""Copy self-contained plot bundles into ``outputs/review_answers_<tag>/``."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from eye_tracking_system_tools.analysis.plot_bundle import is_plot_bundle, iter_plot_bundles
from eye_tracking_system_tools.analysis.run_layout import assert_not_reproduction

FOLDER_PREFIX = "review_answers"

# Legacy flat PDF names (pre-bundle runs). Still collected if found.
PDF_NAMES = (
    "jitter_modular_vs_rigid.pdf",
    "jitter_mouse.pdf",
    "jitter_turtle.pdf",
    "unified_jitter_quantification.pdf",
    "unified_noise_measure.pdf",
    "raw_interframe_delta.pdf",
    "raw_interframe_delta_logx.pdf",
    "yield_dlc_likelihood_modular_vs_rigid.pdf",
    "yield_dlc_likelihood_mouse.pdf",
    "yield_dlc_likelihood_turtle.pdf",
    "yield_ellipse_modular_vs_rigid.pdf",
    "yield_ellipse_mouse.pdf",
    "yield_ellipse_turtle.pdf",
    "figure_2b.pdf",
    "figure_2c.pdf",
    "figure_2d.pdf",
    "figure_2e.pdf",
    "figure_2e_per_animal_means.pdf",
    "figure_2e_per_animal_means_concurrent.pdf",
    "figure_2e_per_animal_means_monocular.pdf",
    "figure_2e_all_animals_scatter.pdf",
    "figure_2e_all_animals_density.pdf",
    "figure_2e_all_events_scatter.pdf",
    "figure_2e_concurrent_scatter.pdf",
    "figure_2e_monocular_scatter.pdf",
    "figure_2f.pdf",
    "figure_2f_colorbar.pdf",
    "fig_2g_top.pdf",
    "fig_2g_bot.pdf",
    "figure_2h.pdf",
    "figure_3a.pdf",
    "figure_3b.pdf",
    "figure_S3_head_still.pdf",
    "figure_S3_head_moving.pdf",
    "figure_S3_colorbar.pdf",
    "mouse_figure_2c.pdf",
    "mouse_figure_2d.pdf",
    "mouse_figure_2e.pdf",
    "mouse_figure_2e_per_animal_means.pdf",
    "mouse_figure_2e_per_animal_means_concurrent.pdf",
    "mouse_figure_2e_per_animal_means_monocular.pdf",
    "mouse_figure_2e_all_animals_scatter.pdf",
    "mouse_figure_2e_all_animals_density.pdf",
    "mouse_figure_2e_all_events_scatter.pdf",
    "mouse_figure_2e_concurrent_scatter.pdf",
    "mouse_figure_2e_monocular_scatter.pdf",
    "mouse_figure_2f.pdf",
    "mouse_figure_2f_colorbar.pdf",
    "mouse_fig_2g_top.pdf",
    "mouse_fig_2g_bot.pdf",
    "trace_lizard.pdf",
    "trace_mouse.pdf",
    "trace_turtle.pdf",
    "robustness_threshold_sweep.pdf",
    "robustness_window_sweep.pdf",
    "robustness_noise_floor.pdf",
    "robustness_shuffle.pdf",
    "ISI_histogram.pdf",
    "ISI_hist_linear_10_300ms.pdf",
    "ISI_log_active.pdf",
    "ISI_log_quiet.pdf",
    "ISI_linear_active.pdf",
    "ISI_linear_quiet.pdf",
    "epoch_durations_active_log.pdf",
    "epoch_durations_active_linear.pdf",
    "epoch_durations_quiet_log.pdf",
    "epoch_durations_quiet_linear.pdf",
    "reversal_rate_by_state.pdf",
    "reversal_example_traces.pdf",
    "noise_trace_quiet.pdf",
    "noise_trace_active.pdf",
    "noise_dframe_hist.pdf",
    "nictitating_peri_trace.pdf",
    "corrective_percent.pdf",
    "corrective_latency_hist.pdf",
    "head_peri_saccade.pdf",
    "s1_empirical_ratio_hist.pdf",
)


def review_answers_dir(
    out_root: Path | str,
    tag: str = "",
    *,
    when: datetime | None = None,
) -> Path:
    when = when or datetime.now()
    safe = "".join(c if (c.isalnum() or c in "-.") else "_" for c in str(tag).strip())
    safe = safe.strip("_")
    if not safe:
        name = f"{FOLDER_PREFIX}_latest"
    else:
        name = "_".join(
            [FOLDER_PREFIX, safe, when.strftime("%Y%m%d"), when.strftime("%H"), when.strftime("%M")]
        )
    path = Path(out_root) / name
    assert_not_reproduction(path)
    return path


def copy_plot_bundles(
    sources: list[Path],
    dest_run: Path,
) -> dict[str, Path]:
    """Copy ``plot_id/{plots,metadata,replot.py}`` folders into ``dest_run``."""
    dest_run.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for src in sources:
        src = Path(src)
        if not src.exists():
            continue
        roots = [src]
        if src.is_dir() and src.name in {"figures", "plots"}:
            roots.append(src.parent)
        for root in roots:
            if not root.is_dir():
                continue
            if is_plot_bundle(root):
                dst = dest_run / root.name
                if dst.resolve() != root.resolve():
                    shutil.copytree(root, dst, dirs_exist_ok=True)
                written[root.name] = dst
            for bundle in iter_plot_bundles(root):
                dst = dest_run / bundle.name
                if dst.resolve() != bundle.resolve():
                    shutil.copytree(bundle, dst, dirs_exist_ok=True)
                written[bundle.name] = dst
    return written


def copy_named_pdfs(
    sources: list[Path],
    dest_figures: Path,
    *,
    names: tuple[str, ...] = PDF_NAMES,
) -> dict[str, Path]:
    """Legacy: copy matching PDF basenames from source figure dirs."""
    dest_figures.mkdir(parents=True, exist_ok=True)
    wanted = set(names)
    written: dict[str, Path] = {}
    for src in sources:
        src = Path(src)
        candidates = []
        if src.is_dir():
            candidates.append(src)
            for child in ("figures", "plots"):
                if (src / child).is_dir():
                    candidates.append(src / child)
        for search in candidates:
            for path in search.glob("*.pdf"):
                if path.name in wanted:
                    dst = dest_figures / path.name
                    shutil.copy2(path, dst)
                    written[path.name] = dst
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect review-answer plot bundles.")
    parser.add_argument("--out-root", type=Path, default=Path("outputs"))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="Run folders containing plot bundles (and/or legacy figures/*.pdf)",
    )
    args = parser.parse_args(argv)
    dest = review_answers_dir(args.out_root, args.tag)
    dest.mkdir(parents=True, exist_ok=True)
    written = copy_plot_bundles(list(args.sources), dest)
    legacy = copy_named_pdfs(list(args.sources), dest / "figures")
    print(f"copied {len(written)} plot bundles → {dest}")
    for name in sorted(written):
        print(f"  {name}/")
    if legacy:
        print(f"copied {len(legacy)} legacy PDFs → {dest / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
