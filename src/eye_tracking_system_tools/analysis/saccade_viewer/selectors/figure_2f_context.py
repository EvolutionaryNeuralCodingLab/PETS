"""Prepare Fig 2f tables + display data for the ROI picker (matches paper Build)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eye_tracking_system_tools.analysis.event_cache import ensure_traces_for_blocks
from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
    Figure2fDisplayData,
    Figure2fPoints,
    collect_figure_2f_points,
    display_data_from_collected,
    figure_2f_config,
    load_figure_2f_display,
)
from eye_tracking_system_tools.analysis.paper_export import FigureBuildResult
from eye_tracking_system_tools.analysis.pipeline import (
    EventTables,
    SaccadeFilter,
    apply_saccade_filter,
    filter_event_tables,
    with_params,
)


@dataclass
class Figure2fRoiContext:
    """Tables, per-event speeds, and histogram panels for ROI picking."""

    tables: EventTables
    collected: Figure2fPoints
    display: Figure2fDisplayData
    cfg: dict[str, Any]
    sample_mode: str
    block_keys: list[str]


def _normalize_figure_2f_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    if "figure_2f" in overrides:
        return dict(overrides["figure_2f"])
    return dict(overrides)


def _figure_2f_cfg_from_build(result: FigureBuildResult) -> dict[str, Any]:
    used = dict(result.params_used or {})
    if "figure_2f" in used:
        return dict(used["figure_2f"])
    return _normalize_figure_2f_overrides(used)


def prepare_figure_2f_roi_context(
    tables: EventTables,
    *,
    block_keys: list[str],
    saccade_filter: SaccadeFilter | None = None,
    params_overrides: dict[str, Any] | None = None,
    sample_mode: str | None = None,
    build_pickle: Path | str | None = None,
    ensure_traces: bool = True,
) -> Figure2fRoiContext:
    """
    Mirror the Fig 2f **Build** path: blocks, saccade filter, params, traces.

    When ``build_pickle`` points at the export from the same ``sample_mode``,
    histogram panels are loaded from that pickle so the ROI view matches the PDF.
    """
    if not block_keys:
        raise ValueError("Select at least one block for Fig 2f ROI picking.")

    work = tables
    if ensure_traces and work.blocks:
        work = ensure_traces_for_blocks(work, block_keys)

    work = filter_event_tables(work, block_keys=block_keys)
    work = apply_saccade_filter(work, saccade_filter)

    overrides = _normalize_figure_2f_overrides(params_overrides)
    if overrides:
        nested = overrides if "figure_2f" in overrides else {"figure_2f": overrides}
        work = with_params(work, nested)

    cfg = figure_2f_config(work, overrides)
    mode = str(sample_mode or cfg.get("sample_mode", "contra_window")).lower()
    cfg = dict(cfg)
    cfg["sample_mode"] = mode

    collected = collect_figure_2f_points(work, cfg=cfg)

    display: Figure2fDisplayData | None = None
    if build_pickle is not None:
        pkl = Path(build_pickle)
        if pkl.is_file():
            display = load_figure_2f_display(pkl)
            if display.n_points != len(collected.points):
                print(
                    f"[2f ROI] WARNING: pickle n={display.n_points} != "
                    f"collected n={len(collected.points)} — using live histogram"
                )
                display = None

    if display is None:
        display = display_data_from_collected(collected)

    filt = saccade_filter.describe() if saccade_filter else "none"
    print(
        f"[2f ROI] blocks={len(block_keys)} events={len(work.all_saccades)} "
        f"sample_mode={mode} kept={len(collected.points)} filter=[{filt}] "
        f"display={display.source}"
    )
    return Figure2fRoiContext(
        tables=work,
        collected=collected,
        display=display,
        cfg=cfg,
        sample_mode=mode,
        block_keys=list(block_keys),
    )


def prepare_figure_2f_from_selector(
    selector: Any,
    build_result: FigureBuildResult | None = None,
    *,
    sample_mode: str | None = None,
    require_build: bool = True,
) -> Figure2fRoiContext:
    """
    Build ROI context from a ``PaperFigureSelector('2f', ctx)`` instance.

    Run ``selector.build()`` first so the export pickle matches the plotted figure.
    """
    block_keys = list(selector.selected)
    if not block_keys:
        raise RuntimeError("Tick at least one eligible block on the Fig 2f selector.")

    build_result = build_result or getattr(selector, "result", None)
    if require_build and build_result is None:
        raise RuntimeError(
            "Build Fig 2f from the selector first (click Build), then open the ROI picker."
        )

    build_pickle = None
    build_cfg: dict[str, Any] = {}
    if build_result is not None:
        build_cfg = _figure_2f_cfg_from_build(build_result)
        build_pickle = build_result.outputs.get("figure_2f_nodowncast.pickle")

    overrides = None
    try:
        raw = selector._parse_overrides()
        if raw:
            overrides = raw if "figure_2f" in raw else {"figure_2f": raw}
    except Exception:
        overrides = None

    mode = sample_mode
    if mode is None and hasattr(selector, "current_sample_mode"):
        mode = selector.current_sample_mode()
    if mode is None and build_cfg:
        mode = str(build_cfg.get("sample_mode", "contra_window"))
    build_mode = str(build_cfg.get("sample_mode", "contra_window")).lower()
    if mode is not None and str(mode).lower() != build_mode and build_pickle is not None:
        print(
            f"[2f ROI] sample_mode={mode!r} differs from build ({build_mode!r}); "
            "recomputing histogram from live points."
        )
        build_pickle = None

    saccade_filter = None
    if getattr(selector, "enable_saccade_filter", False):
        saccade_filter = selector.current_saccade_filter()

    return prepare_figure_2f_roi_context(
        selector.ctx.tables,
        block_keys=block_keys,
        saccade_filter=saccade_filter,
        params_overrides=overrides,
        sample_mode=mode,
        build_pickle=build_pickle,
        ensure_traces=True,
    )
