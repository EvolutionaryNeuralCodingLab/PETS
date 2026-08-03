"""
Registry of paper-figure exporters for the flexible figures notebook.

Each :class:`FigureSpec` declares what a panel needs (events, traces,
behavior state, …), which params section it reads, and the runner that
produces its PDFs. :func:`run_figure` filters the event tables, merges
params overrides, and dispatches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection

from eye_tracking_system_tools.analysis.pipeline import (
    EventTables,
    SaccadeFilter,
    apply_saccade_filter,
    filter_event_tables,
    with_params,
)

Runner = Callable[..., dict[str, Path] | Path | tuple[Path, ...]]


@dataclass(frozen=True)
class FigureSpec:
    fig_id: str
    label: str
    needs: tuple[str, ...]
    params_section: str | None
    outputs: tuple[str, ...]
    runner: Runner


def _as_path_dict(result: dict[str, Path] | Path | tuple[Path, ...], spec: FigureSpec) -> dict[str, Path]:
    """Normalize exporter return values to ``{name: Path}``."""
    if isinstance(result, dict):
        return {str(k): Path(v) for k, v in result.items()}
    if isinstance(result, Path):
        key = spec.outputs[0] if spec.outputs else spec.fig_id
        return {key: result}
    if isinstance(result, tuple):
        out: dict[str, Path] = {}
        for i, p in enumerate(result):
            key = spec.outputs[i] if i < len(spec.outputs) else f"{spec.fig_id}_{i}"
            out[key] = Path(p)
        return out
    raise TypeError(f"Unexpected runner return type: {type(result)!r}")


def _run_2c_2d(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_2c_2e import export_pos_vel_bundle
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    pkl = export_pos_vel_bundle(tables, out_dir, show=show)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    out: dict[str, Path] = {
        "figure_2c.pdf": figures_dir / "figure_2c.pdf",
        "figure_2d.pdf": figures_dir / "figure_2d.pdf",
        "pos_vel_by_amp_bins_bundle.pkl": pkl,
    }
    for pattern in (
        "figure_2c_legend.pdf",
        "figure_2d_legend.pdf",
        "figure_2c_*_legend.pdf",
        "figure_2d_*_legend.pdf",
        "figure_2c_*.pdf",
        "figure_2d_*.pdf",
    ):
        for path in sorted(figures_dir.glob(pattern)):
            # Avoid re-adding canonical names / double-counting legend globs.
            if path.name in out:
                continue
            if path.name in {"figure_2c.pdf", "figure_2d.pdf"}:
                continue
            out[path.name] = path
    return out


def _run_2e(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_2c_2e import export_amplitude_velocity_fit
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    pkl = export_amplitude_velocity_fit(tables, out_dir, show=show)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    return {
        "figure_2e.pdf": figures_dir / "figure_2e.pdf",
        "amplitude_velocity_linear_fit_bundle.pkl": pkl,
    }


def _run_2f(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
        export_archived_figure_2f,
        export_figure_2f,
    )
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)
    # Need traces: if any selected block has empty left/right, fall back to archived.
    has_traces = any(
        b.left is not None and not b.left.empty for b in tables.blocks
    )
    if tables.blocks and has_traces:
        pkl = export_figure_2f(tables, out_dir, show=show)
        return {
            "figure_2f.pdf": figures_dir / "figure_2f.pdf",
            "figure_2f_colorbar.pdf": figures_dir / "figure_2f_colorbar.pdf",
            "figure_2f_nodowncast.pickle": pkl,
        }
    pdf = export_archived_figure_2f(out_dir, show=show)
    return {
        "figure_2f.pdf": pdf,
        "figure_2f_colorbar.pdf": figures_dir / "figure_2f_colorbar.pdf",
    }


def _run_2g(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_2g_2j import export_figure_2g
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    top, bot = export_figure_2g(tables, out_dir, show=show)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    return {
        "fig_2g_top.pdf": figures_dir / "fig_2g_top.pdf",
        "fig_2g_bot.pdf": figures_dir / "fig_2g_bot.pdf",
        "fig_2g_top.pkl": top,
        "fig_2g_bot.pkl": bot,
    }


def _run_2h(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_2f_2h_2i import export_figure_2h
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    pkl = export_figure_2h(tables, out_dir, show=show)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    return {"figure_2h.pdf": figures_dir / "figure_2h.pdf", "figure_2h.pickle": pkl}


def _run_2i(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_2f_2h_2i import export_figure_2i
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    pkl = export_figure_2i(tables, out_dir, show=show)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    return {"figure_2i.pdf": figures_dir / "figure_2i.pdf", "figure_2i.pickle": pkl}


def _run_2j(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_2g_2j import export_figure_2j
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    pkl = export_figure_2j(tables, out_dir, show=show)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    return {"figure_2j.pdf": figures_dir / "figure_2j.pdf", "saccade_angles_data.pkl": pkl}


def _run_3d(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_3d_isi import export_figure_3d

    return export_figure_3d(tables, out_dir, show=show)


def _run_3e(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_3e_3f_pupil import export_figure_3e
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    pkl = export_figure_3e(tables, out_dir, show=show)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    return {"figure_3e.pdf": figures_dir / "figure_3e.pdf", "figure_3e.pickle": pkl}


def _run_3f(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_3e_3f_pupil import export_figure_3f
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    pkl = export_figure_3f(tables, out_dir, show=show)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    legend = figures_dir / "figure_3f_legend.pdf"
    out = {"figure_3f.pdf": figures_dir / "figure_3f.pdf", "figure_3f.pickle": pkl}
    if legend.exists():
        out["figure_3f_legend.pdf"] = legend
    return out


def _run_3a(tables: EventTables, out_dir: Path, *, show: bool = False, **kwargs) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_3a_3c_vignettes import export_figure_3a

    return export_figure_3a(tables, out_dir, show=show, **kwargs)


def _run_3b(tables: EventTables, out_dir: Path, *, show: bool = False, **kwargs) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_3a_3c_vignettes import export_figure_3b

    return export_figure_3b(tables, out_dir, show=show, **kwargs)


def _run_3c(tables: EventTables, out_dir: Path, *, show: bool = False, **kwargs) -> dict[str, Path]:
    from eye_tracking_system_tools.analysis.figures_3a_3c_vignettes import export_figure_3c

    return export_figure_3c(tables, out_dir, show=show, **kwargs)


def _run_1e(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
    jitter_bundle: Path | str | None = None,
    **kwargs,
) -> dict[str, Path]:
    """Replot Fig 1e from a finalized jitter export (µm displacement histogram)."""
    from eye_tracking_system_tools.analysis.jitter_export import load_jitter_bundle, plot_from_bundle
    from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

    if jitter_bundle is None:
        raise ValueError(
            "Fig 1e requires jitter_bundle= path to a finalized jitter export "
            "(folder or jitter_comparison_data.pickle)"
        )
    bundle = load_jitter_bundle(jitter_bundle)
    figures_dir, _ = resolve_figure_dirs(out_dir)
    written = plot_from_bundle(bundle, out_dir=figures_dir, show=show, **kwargs)
    # Prefer modular_vs_rigid as the paper 1e panel; also keep mouse if present.
    out: dict[str, Path] = {}
    if "modular_vs_rigid" in written:
        # Copy/alias to figure_1e.pdf name for the paper export.
        src = written["modular_vs_rigid"]
        dst = figures_dir / "figure_1e.pdf"
        if src.resolve() != dst.resolve():
            import shutil

            shutil.copy2(src, dst)
        out["figure_1e.pdf"] = dst
    for name, path in written.items():
        out[f"jitter_{name}.pdf" if not str(name).endswith(".pdf") else name] = path
    return out


CATALOG: dict[str, FigureSpec] = {
    "2c_2d": FigureSpec(
        fig_id="2c_2d",
        label="Fig 2c/2d — amp-binned position & velocity",
        needs=("events", "traces"),
        params_section="main_sequence",
        outputs=(
            "figure_2c.pdf",
            "figure_2d.pdf",
            "figure_2c_legend.pdf",
            "figure_2d_legend.pdf",
        ),
        runner=_run_2c_2d,
    ),
    "2e": FigureSpec(
        fig_id="2e",
        label="Fig 2e — amplitude–velocity linear fit",
        needs=("events",),
        params_section="main_sequence",
        outputs=("figure_2e.pdf",),
        runner=_run_2e,
    ),
    "2f": FigureSpec(
        fig_id="2f",
        label="Fig 2f — inter-ocular peak-speed coupling",
        needs=("events", "traces"),
        params_section="figure_2f",
        outputs=("figure_2f.pdf", "figure_2f_colorbar.pdf"),
        runner=_run_2f,
    ),
    "2g": FigureSpec(
        fig_id="2g",
        label="Fig 2g — amplitude distributions",
        needs=("events",),
        params_section="figure_2g",
        outputs=("fig_2g_top.pdf", "fig_2g_bot.pdf"),
        runner=_run_2g,
    ),
    "2h": FigureSpec(
        fig_id="2h",
        label="Fig 2h — endpoint heatmaps",
        needs=("events",),
        params_section="figure_2h",
        outputs=("figure_2h.pdf",),
        runner=_run_2h,
    ),
    "2i": FigureSpec(
        fig_id="2i",
        label="Fig 2i — polar direction histograms",
        needs=("events",),
        params_section="figure_2i",
        outputs=("figure_2i.pdf",),
        runner=_run_2i,
    ),
    "2j": FigureSpec(
        fig_id="2j",
        label="Fig 2j — orientation tuning",
        needs=("events",),
        params_section=None,
        outputs=("figure_2j.pdf",),
        runner=_run_2j,
    ),
    "3d": FigureSpec(
        fig_id="3d",
        label="Fig 3d — inter-saccade interval densities",
        needs=("events",),
        params_section="figure_3d",
        outputs=("ISI_histogram.pdf", "ISI_hist_linear_10_300ms.pdf"),
        runner=_run_3d,
    ),
    "3e": FigureSpec(
        fig_id="3e",
        label="Fig 3e — pupil diameter by behavioral state",
        needs=("traces", "behavior_state", "pix_size"),
        params_section="figure_3e",
        outputs=("figure_3e.pdf",),
        runner=_run_3e,
    ),
    "3f": FigureSpec(
        fig_id="3f",
        label="Fig 3f — z-scored pupil state difference",
        needs=("traces", "behavior_state", "pix_size"),
        params_section="figure_3f",
        outputs=("figure_3f.pdf",),
        runner=_run_3f,
    ),
    "3a": FigureSpec(
        fig_id="3a",
        label="Fig 3a — quiet vignette",
        needs=("traces",),
        params_section="figure_3a_3c",
        outputs=("figure_3a.pdf",),
        runner=_run_3a,
    ),
    "3b": FigureSpec(
        fig_id="3b",
        label="Fig 3b — active vignette",
        needs=("traces",),
        params_section="figure_3a_3c",
        outputs=("figure_3b.pdf",),
        runner=_run_3b,
    ),
    "3c": FigureSpec(
        fig_id="3c",
        label="Fig 3c — full vignette with state / rates",
        needs=("traces", "behavior_state"),
        params_section="figure_3a_3c",
        outputs=("figure_3c.pdf",),
        runner=_run_3c,
    ),
    "1e": FigureSpec(
        fig_id="1e",
        label="Fig 1e — camera jitter (from jitter export)",
        needs=("jitter_bundle",),
        params_section=None,
        outputs=("figure_1e.pdf",),
        runner=_run_1e,
    ),
}


def get_spec(fig_id: str) -> FigureSpec:
    if fig_id not in CATALOG:
        known = ", ".join(sorted(CATALOG))
        raise KeyError(f"Unknown figure {fig_id!r}. Known: {known}")
    return CATALOG[fig_id]


def run_figure(
    fig_id: str,
    tables: EventTables,
    out_dir: Path | str,
    *,
    block_keys: Collection[str] | None = None,
    animals: Collection[str] | None = None,
    saccade_filter: SaccadeFilter | dict[str, Any] | None = None,
    params_overrides: dict[str, Any] | None = None,
    show: bool = True,
    **runner_kwargs: Any,
) -> dict[str, Path]:
    """
    Filter ``tables``, merge params overrides, and run the catalogued exporter.

    ``saccade_filter`` optionally restricts events by kind (concurrent /
    monocular), ``head_movement``, column truth values, or a pandas query.

    Extra ``runner_kwargs`` are forwarded (e.g. ``jitter_bundle`` for Fig 1e,
    ``start_s`` / ``end_s`` for vignettes).
    """
    spec = get_spec(fig_id)
    out_dir = Path(out_dir)
    filtered = filter_event_tables(tables, block_keys=block_keys, animals=animals)
    filtered = apply_saccade_filter(filtered, saccade_filter)

    overrides = dict(params_overrides or {})
    if spec.params_section and overrides and spec.params_section not in overrides:
        # Allow passing the section body directly: {num_bins: 40} → {figure_3e: {...}}
        if not any(k in filtered.params for k in overrides):
            # Heuristic: if keys look like section knobs, wrap them.
            section_keys = set((filtered.params.get(spec.params_section) or {}).keys())
            if section_keys & set(overrides) or not section_keys:
                overrides = {spec.params_section: overrides}
    work = with_params(filtered, overrides) if overrides else filtered

    result = spec.runner(work, out_dir, show=show, **runner_kwargs)
    return _as_path_dict(result, spec)
