"""Shared data loading and matplotlib export for annotator plot scripts."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
    load_block_session,
)
from eye_tracking_system_tools.annotation.block_annotator.models import AnnotatorConfig
from eye_tracking_system_tools.annotation.event_explorer.catalog import (
    build_catalog,
    discover_annotation_files,
    event_types_in_catalog,
    load_events_from_file,
)
from eye_tracking_system_tools.annotation.event_explorer.eye_csv_resolver import (
    build_column_map,
    detect_degrees_column,
    load_eye_dataframe,
    pupil_values,
    resolve_eye_csv,
)
from eye_tracking_system_tools.annotation.event_explorer.models import (
    BlockDataCache,
    EventRecord,
    EventSnippet,
)
from eye_tracking_system_tools.annotation.event_explorer.snippet_extractor import (
    STREAM_EP,
    STREAM_L_DEG,
    STREAM_L_PUPIL,
    STREAM_R_DEG,
    STREAM_R_PUPIL,
    STREAM_LABELS,
    extract_ep_snippet,
    extract_eye_snippet,
    normalize_trials,
    resample_to_grid,
    stack_mean_sem,
)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PlotMode = Literal["average", "individual", "both"]
Normalization = Literal["none", "within_block_zscore", "per_trial_zscore"]

STREAM_COLORS = {
    STREAM_L_PUPIL: "#0072B2",
    STREAM_R_PUPIL: "#D55E00",
    STREAM_L_DEG: "#009E73",
    STREAM_R_DEG: "#CC79A7",
    STREAM_EP: "#000000",
}

PUPIL_STREAMS = (STREAM_L_PUPIL, STREAM_R_PUPIL)
DEG_STREAMS = (STREAM_L_DEG, STREAM_R_DEG)


class LoadLog(Protocol):
    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...


class StderrLoadLog:
    def info(self, msg: str) -> None:
        print(f"INFO: {msg}", file=sys.stderr)

    def warn(self, msg: str) -> None:
        print(f"WARN: {msg}", file=sys.stderr)


@dataclass
class AnnotationSummary:
    path: Path
    animal_call: str
    experiment_date: str | None
    block_num: str
    block_path: Path
    n_events: int
    event_types: list[str]


@dataclass
class PlotStyle:
    figsize: tuple[float, float] = (2.2, 1.8)
    dpi: int = 300
    label_fontsize: float = 9.0
    tick_fontsize: float = 8.0
    line_width: float = 1.2
    mean_line_width: float = 1.8
    trial_line_width: float = 0.6
    trial_alpha: float = 0.35
    sem_alpha: float = 0.18
    font_family: str = "Arial"


@dataclass
class PlotRequest:
    annotation_paths: list[Path]
    event_types: list[str]
    stream_ids: list[str]
    ep_channels: list[int] = field(default_factory=lambda: [1])
    half_window_ms: float = 100.0
    mode: PlotMode = "both"
    normalization: Normalization = "within_block_zscore"
    n_grid: int = 201
    style: PlotStyle = field(default_factory=PlotStyle)


@dataclass
class StreamPlotData:
    stream_id: str
    label: str
    color: str
    grid: np.ndarray
    trials: np.ndarray  # (n_trials, n_grid)
    mean: np.ndarray
    sem: np.ndarray
    n_trials: int
    ep_channel: int | None = None


def scan_annotations(output_folder: Path) -> list[AnnotationSummary]:
    paths = discover_annotation_files(scan_dirs=[output_folder], recursive=False)
    summaries: list[AnnotationSummary] = []
    for path in paths:
        records = load_events_from_file(path)
        types = sorted({r.event_type for r in records}, key=str.lower)
        block_path = records[0].block_path if records else Path()
        summaries.append(
            AnnotationSummary(
                path=path,
                animal_call=records[0].animal_call if records else "",
                experiment_date=records[0].experiment_date if records else None,
                block_num=records[0].block_num if records else "",
                block_path=block_path,
                n_events=len(records),
                event_types=types,
            )
        )
    return summaries


def annotation_label(summary: AnnotationSummary) -> str:
    date = summary.experiment_date or "—"
    return (
        f"{summary.animal_call}  {date}  block_{summary.block_num}  "
        f"({summary.n_events} events)"
    )


def load_block_cache_headless(
    record: EventRecord,
    log: LoadLog,
    cache_by_path: dict[Path, BlockDataCache],
) -> BlockDataCache | None:
    block_path = Path(record.block_path)
    if not block_path.exists():
        log.warn(f"block_path missing: {block_path} (event {record.event_id})")
        return None

    key = block_path.resolve()
    if key in cache_by_path:
        return cache_by_path[key]

    log.info(f"Loading block {block_path.name} ({record.animal_call})")

    session = load_block_session(
        block_path,
        block_path / "analysis",
        AnnotatorConfig(),
        animal_call=record.animal_call,
        experiment_date=record.experiment_date,
        block_num=record.block_num,
    )

    analysis = block_path / "analysis"
    le_path, le_cands = resolve_eye_csv(analysis, "left")
    re_path, re_cands = resolve_eye_csv(analysis, "right")
    le_df = load_eye_dataframe(le_path) if le_path else None
    re_df = load_eye_dataframe(re_path) if re_path else None
    col_map = build_column_map(le_df, re_df)

    le_degrees_df = None
    re_degrees_df = None
    if col_map.l_degrees is None or (le_df is not None and col_map.l_degrees not in le_df.columns):
        for cand in le_cands:
            if le_path and cand == le_path:
                continue
            try:
                extra = load_eye_dataframe(cand)
                deg = detect_degrees_column(extra, "left")
                if deg:
                    col_map.l_degrees = deg
                    le_degrees_df = extra
                    break
            except OSError:
                pass
    if col_map.r_degrees is None or (re_df is not None and col_map.r_degrees not in re_df.columns):
        for cand in re_cands:
            if re_path and cand == re_path:
                continue
            try:
                extra = load_eye_dataframe(cand)
                deg = detect_degrees_column(extra, "right")
                if deg:
                    col_map.r_degrees = deg
                    re_degrees_df = extra
                    break
            except OSError:
                pass

    cache = BlockDataCache(
        block_path=block_path,
        final_sync_df=session.final_sync_df,
        ms_axis=session.ms_axis,
        sample_rate_hz=session.sample_rate_hz,
        le_csv_path=le_path,
        re_csv_path=re_path,
        le_df=le_df,
        re_df=re_df,
        le_degrees_df=le_degrees_df,
        re_degrees_df=re_degrees_df,
        oe_rec=session.oe_rec,
        column_map=col_map,
    )
    cache_by_path[key] = cache
    return cache


def _block_series_for_stream(cache: BlockDataCache, stream_id: str) -> pd.Series | None:
    if stream_id == STREAM_L_PUPIL:
        df, spec = cache.le_df, cache.column_map.pupil
        if df is None or spec is None:
            return None
        return pupil_values(df, spec)
    if stream_id == STREAM_R_PUPIL:
        df, spec = cache.re_df, cache.column_map.pupil
        if df is None or spec is None:
            return None
        return pupil_values(df, spec)
    if stream_id == STREAM_L_DEG:
        df = cache.le_degrees_df if cache.le_degrees_df is not None else cache.le_df
        spec = cache.column_map.l_degrees
        if df is None or spec is None or spec not in df.columns:
            return None
        return df[spec].astype(float)
    if stream_id == STREAM_R_DEG:
        df = cache.re_degrees_df if cache.re_degrees_df is not None else cache.re_df
        spec = cache.column_map.r_degrees
        if df is None or spec is None or spec not in df.columns:
            return None
        return df[spec].astype(float)
    return None


def zscore_within_block(values: np.ndarray, block_series: pd.Series) -> np.ndarray:
    arr = np.asarray(block_series, dtype=np.float64)
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    y = np.asarray(values, dtype=np.float64)
    if sd < 1e-12:
        return y - mu
    return (y - mu) / sd


def apply_normalization(
    snippet: EventSnippet,
    block_series: pd.Series | None,
    mode: Normalization,
) -> np.ndarray:
    y = np.asarray(snippet.values, dtype=np.float64)
    if mode == "none":
        return y
    if mode == "within_block_zscore":
        if block_series is not None:
            return zscore_within_block(y, block_series)
        return normalize_trials([y], [snippet.time_rel_ms], "zscore")[0]
    if mode == "per_trial_zscore":
        return normalize_trials([y], [snippet.time_rel_ms], "zscore")[0]
    return y


def filter_catalog(
    catalog: list[EventRecord],
    *,
    annotation_paths: list[Path],
    event_types: list[str],
) -> list[EventRecord]:
    paths = {Path(p).resolve() for p in annotation_paths}
    types = set(event_types)
    out: list[EventRecord] = []
    for rec in catalog:
        if paths and Path(rec.annotation_path).resolve() not in paths:
            continue
        if types and rec.event_type not in types:
            continue
        out.append(rec)
    return out


def collect_snippets(
    records: list[EventRecord],
    stream_ids: list[str],
    ep_channels: list[int],
    half_window_ms: float,
    normalization: Normalization,
    log: LoadLog,
) -> dict[str, list[EventSnippet]]:
    """Key = stream_id or 'ep:chN' for electrophysiology."""
    cache_by_path: dict[Path, BlockDataCache] = {}
    out: dict[str, list[EventSnippet]] = {}

    for stream_id in stream_ids:
        if stream_id == STREAM_EP:
            for ch in ep_channels:
                out[f"ep:{int(ch)}"] = []
        else:
            out[stream_id] = []

    for record in records:
        cache = load_block_cache_headless(record, log, cache_by_path)
        if cache is None:
            continue

        for stream_id in stream_ids:
            if stream_id == STREAM_EP:
                for ch in ep_channels:
                    key = f"ep:{int(ch)}"
                    snip = extract_ep_snippet(record, cache, int(ch), half_window_ms, log)
                    if snip is None or len(snip.values) == 0:
                        continue
                    y = apply_normalization(snip, None, normalization)
                    out[key].append(
                        EventSnippet(
                            event_id=snip.event_id,
                            stream_id=STREAM_EP,
                            time_rel_ms=snip.time_rel_ms,
                            values=y,
                            source=snip.source,
                            meta={
                                **snip.meta,
                                "event_type": record.event_type,
                                "channel": int(ch),
                            },
                        )
                    )
                continue

            block_series = _block_series_for_stream(cache, stream_id)
            snip = extract_eye_snippet(record, cache, stream_id, half_window_ms, log)
            if snip is None or len(snip.values) == 0:
                continue
            y = apply_normalization(snip, block_series, normalization)
            out[stream_id].append(
                EventSnippet(
                    event_id=snip.event_id,
                    stream_id=snip.stream_id,
                    time_rel_ms=snip.time_rel_ms,
                    values=y,
                    source=snip.source,
                    meta={**snip.meta, "event_type": record.event_type},
                )
            )

    return out


def stack_stream_data(
    snippets: list[EventSnippet],
    stream_key: str,
    n_grid: int,
    half_window_ms: float | None = None,
) -> StreamPlotData | None:
    if not snippets:
        return None
    grid, stacked = resample_to_grid(
        snippets, n_points=n_grid, half_window_ms=half_window_ms
    )
    n = stacked.shape[0]
    mean, sem = stack_mean_sem(stacked)
    sid = snippets[0].stream_id
    if sid == STREAM_EP:
        ch = snippets[0].meta.get("channel", "")
        label = f"EP HS ch{ch}"
        color = STREAM_COLORS[STREAM_EP]
        ep_ch = int(ch) if ch != "" else None
    else:
        label = STREAM_LABELS.get(sid, sid)
        color = STREAM_COLORS.get(sid, "#333333")
        ep_ch = None
    return StreamPlotData(
        stream_id=stream_key,
        label=label,
        color=color,
        grid=grid,
        trials=stacked,
        mean=mean,
        sem=sem,
        n_trials=n,
        ep_channel=ep_ch,
    )


def build_plot_data(request: PlotRequest, log: LoadLog) -> list[StreamPlotData]:
    catalog = build_catalog([Path(p) for p in request.annotation_paths])
    catalog = filter_catalog(
        catalog,
        annotation_paths=request.annotation_paths,
        event_types=request.event_types,
    )
    if not catalog:
        return []

    snippets_by_key = collect_snippets(
        catalog,
        request.stream_ids,
        request.ep_channels,
        request.half_window_ms,
        request.normalization,
        log,
    )

    series: list[StreamPlotData] = []
    for stream_id in request.stream_ids:
        if stream_id == STREAM_EP:
            for ch in request.ep_channels:
                key = f"ep:{int(ch)}"
                data = stack_stream_data(
                    snippets_by_key.get(key, []), key, request.n_grid, request.half_window_ms
                )
                if data:
                    series.append(data)
        else:
            data = stack_stream_data(
                snippets_by_key.get(stream_id, []), stream_id, request.n_grid, request.half_window_ms
            )
            if data:
                series.append(data)
    return series


def _style_axes(ax: Axes, style: PlotStyle, ylabel: str) -> None:
    ax.axvline(0, color="#bbbbbb", linestyle=":", linewidth=0.7, zorder=0)
    ax.set_xlabel("Time from event (ms)", fontsize=style.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
    ax.tick_params(labelsize=style.tick_fontsize, width=0.6, length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.grid(False)


def ylabel_for_normalization(norm: Normalization) -> str:
    if norm == "within_block_zscore":
        return "Z-score (within block)"
    if norm == "per_trial_zscore":
        return "Z-score (per trial)"
    return "Value"


def render_stream_axis(
    ax: Axes,
    data: StreamPlotData,
    *,
    mode: PlotMode,
    style: PlotStyle,
) -> list:
    handles = []
    if mode in ("individual", "both"):
        for i in range(data.trials.shape[0]):
            (h,) = ax.plot(
                data.grid,
                data.trials[i],
                color=data.color,
                linewidth=style.trial_line_width,
                alpha=style.trial_alpha,
            )
            if i == 0:
                handles.append(h)
    if mode in ("average", "both"):
        (h_mean,) = ax.plot(
            data.grid,
            data.mean,
            color=data.color,
            linewidth=style.mean_line_width if mode == "both" else style.line_width,
            solid_capstyle="round",
            zorder=3,
        )
        handles.append(h_mean)
        sem_alpha = style.sem_alpha if mode == "average" else style.sem_alpha * 0.65
        lo = data.mean - data.sem
        hi = data.mean + data.sem
        band = np.isfinite(lo) & np.isfinite(hi)
        if band.any():
            ax.fill_between(
                data.grid[band],
                lo[band],
                hi[band],
                color=data.color,
                alpha=sem_alpha,
                linewidth=0,
                zorder=2,
            )
    return handles


def export_figures(
    series: list[StreamPlotData],
    *,
    mode: PlotMode,
    normalization: Normalization,
    style: PlotStyle,
    main_path: Path,
    legend_path: Path | None = None,
) -> None:
    if not series:
        raise ValueError("No data to plot")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [style.font_family, "DejaVu Sans"]

    n = len(series)
    h_each = style.figsize[1]
    fig_h = h_each * n if n > 1 else style.figsize[1]
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(style.figsize[0], fig_h),
        dpi=style.dpi,
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    ylabel = ylabel_for_normalization(normalization)
    legend_handles = []
    legend_labels = []

    for ax, data in zip(axes_flat, series):
        handles = render_stream_axis(ax, data, mode=mode, style=style)
        _style_axes(ax, style, ylabel)
        ax.set_xlim(float(data.grid[0]), float(data.grid[-1]))
        if mode == "individual":
            legend_handles.append(handles[0])
            legend_labels.append(f"{data.label} (N={data.n_trials})")
        elif mode == "average":
            legend_handles.append(handles[0])
            legend_labels.append(f"{data.label} mean ± SEM (N={data.n_trials})")
        else:
            legend_handles.append(handles[-1])
            legend_labels.append(f"{data.label} mean (N={data.n_trials})")

    fig.subplots_adjust(hspace=0.28 if n > 1 else 0.05)
    main_path = Path(main_path)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(main_path, format="pdf", bbox_inches="tight", dpi=style.dpi)
    plt.close(fig)

    if legend_path is not None and legend_handles:
        legend_path = Path(legend_path)
        legend_path.parent.mkdir(parents=True, exist_ok=True)
        n_leg = len(legend_labels)
        fig_leg = plt.figure(
            figsize=(2.4, 0.32 * max(1, n_leg) + 0.35),
            dpi=style.dpi,
        )
        fig_leg.legend(
            legend_handles,
            legend_labels,
            loc="center",
            frameon=False,
            ncol=1,
            prop={"size": style.tick_fontsize},
        )
        fig_leg.savefig(legend_path, format="pdf", bbox_inches="tight", dpi=style.dpi)
        plt.close(fig_leg)


def event_types_from_summaries(
    summaries: list[AnnotationSummary],
    selected_paths: list[Path] | None = None,
) -> list[str]:
    sel = {Path(p).resolve() for p in (selected_paths or [])}
    types: set[str] = set()
    for s in summaries:
        if sel and s.path.resolve() not in sel:
            continue
        types.update(s.event_types)
    return sorted(types, key=str.lower)


def catalog_from_folder(
    output_folder: Path,
    selected_jsons: list[Path] | None = None,
) -> list[EventRecord]:
    if selected_jsons:
        paths = [Path(p) for p in selected_jsons]
    else:
        paths = discover_annotation_files(scan_dirs=[output_folder], recursive=False)
    return build_catalog(paths)


def events_to_dataframe(catalog: list[EventRecord]) -> pd.DataFrame:
    """Flat event table for notebook filtering and inspection."""
    rows = []
    for rec in catalog:
        rows.append(
            {
                "event_id": rec.event_id,
                "event_type": rec.event_type,
                "animal_call": rec.animal_call,
                "experiment_date": rec.experiment_date,
                "block_num": rec.block_num,
                "timepoint_ms": rec.timepoint_ms,
                "start_ms": rec.start_ms,
                "end_ms": rec.end_ms,
                "l_eye_frame": rec.l_eye_frame,
                "r_eye_frame": rec.r_eye_frame,
                "note": rec.note,
                "annotation_file": rec.annotation_path.name,
                "annotation_path": str(rec.annotation_path),
                "block_path": str(rec.block_path),
                "block_exists": Path(rec.block_path).exists(),
            }
        )
    return pd.DataFrame(rows)


def summaries_to_dataframe(summaries: list[AnnotationSummary]) -> pd.DataFrame:
    """One row per annotation JSON (block export)."""
    rows = []
    for s in summaries:
        rows.append(
            {
                "annotation_file": s.path.name,
                "annotation_path": str(s.path),
                "animal_call": s.animal_call,
                "experiment_date": s.experiment_date,
                "block_num": s.block_num,
                "block_path": str(s.block_path),
                "block_exists": s.block_path.exists(),
                "n_events": s.n_events,
                "event_types": ", ".join(s.event_types),
            }
        )
    return pd.DataFrame(rows)


def records_from_dataframe(
    catalog: list[EventRecord],
    events_df: pd.DataFrame,
) -> list[EventRecord]:
    """Rebuild EventRecord list after boolean filtering on events_to_dataframe output."""
    ids = set(events_df["event_id"])
    return [rec for rec in catalog if rec.event_id in ids]
