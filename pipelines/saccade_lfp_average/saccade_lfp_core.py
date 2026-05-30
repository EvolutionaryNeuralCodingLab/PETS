"""Shared saccade-aligned LFP pipeline logic for CLI and GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from eye_tracking_system_tools.preprocessing.block_sync_core import load_final_sync_df
from eye_tracking_system_tools.preprocessing.utility_functions import block_generator

from eye_data_loader import EyeDataSource, attach_eye_data_to_block
from saccade_detection_legacy import annotate_events_sync_status, detect_legacy_block_events

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

EYE_COLORS = {
    "Concurrent": "#009E73",
    "Monocular": "#CC79A7",
    "L": "#0072B2",
    "R": "#D55E00",
    "All": "#000000",
}

EyeKey = Literal["Concurrent", "Monocular", "L", "R", "All"]
DetectionMode = Literal["velocity", "legacy"]


class LoadLog(Protocol):
    def info(self, msg: str) -> None: ...
    def warn(self, msg: str) -> None: ...


class NullLoadLog:
    def info(self, msg: str) -> None:
        pass

    def warn(self, msg: str) -> None:
        pass


@dataclass
class EyeAverage:
    eye: EyeKey
    label: str
    color: str
    grid_ms: np.ndarray
    mean: np.ndarray
    sem: np.ndarray
    n_trials: int
    n_trials_raw: int = 0
    n_trials_removed: int = 0


@dataclass
class ChannelTrialBundle:
    """Cached LFP snippets for one electrode × eye group."""

    channel: int
    eye: EyeKey
    label: str
    grid_ms: np.ndarray
    trials: np.ndarray


@dataclass
class PlotConfig:
    experiment_path: Path
    animal: str
    blocks: list[int]
    electrodes: list[int]
    query: str | None = None
    half_window_ms: float = 500.0
    saccade_threshold: float = 2.0
    min_saccade_frames: int = 1
    batch_size: int = 200
    dpi: int = 300
    figsize: tuple[float, float] = (2.4, 1.8)
    label_fontsize: float = 9.0
    tick_fontsize: float = 8.0
    sem_alpha: float = 0.15
    visible_eyes: frozenset[EyeKey] = frozenset({"Concurrent", "Monocular", "L", "R", "All"})
    plot_label: str | None = None
    analysis_folder: Path | None = None
    ep_noise_std_k: float = 0.0
    detection_mode: DetectionMode = "velocity"
    eye_data_source: EyeDataSource = "auto"
    legacy_speed_threshold: float = 2.0
    legacy_magnitude_calib: float = 1.0
    legacy_sync_diff_ms: float = 680.0
    legacy_use_pupil_diameter: bool = True


@dataclass
class ExportPaths:
    main_path: Path
    legend_path: Path
    output_dir: Path
    filename_stem: str
    collision_warning: str | None = None


@dataclass
class PipelineResult:
    config: PlotConfig
    block_labels: list[str]
    events: pd.DataFrame
    events_filtered: pd.DataFrame
    channel_averages: dict[int, list[EyeAverage]] = field(default_factory=dict)
    trial_bundles: list[ChannelTrialBundle] = field(default_factory=list)
    blocks_by_num: dict = field(default_factory=dict, repr=False)


def normalize_block_numbers(blocks: list[str | int]) -> list[int]:
    out: list[int] = []
    for b in blocks:
        try:
            out.append(int(b))
        except ValueError as exc:
            raise ValueError(f"Invalid block number: {b!r}") from exc
    return out


def parse_electrodes(text: str) -> list[int]:
    text = text.strip()
    if not text:
        raise ValueError("Provide at least one electrode channel.")
    return [int(x.strip()) for x in text.replace(" ", "").split(",") if x.strip()]


def discover_block_numbers(experiment_path: Path, animal: str) -> list[str]:
    """Return sorted 3-digit block numbers found under {experiment}/{animal}."""
    root = Path(experiment_path) / animal
    if not root.is_dir():
        return []
    found: set[str] = set()
    for date_path in root.iterdir():
        if not date_path.is_dir() or "block" in date_path.name.lower():
            continue
        for block_path in date_path.iterdir():
            if block_path.is_dir() and "block" in block_path.name.lower():
                found.add(block_path.name[-3:])
    return sorted(found)


def detect_saccades_chunked(
    eye_df: pd.DataFrame,
    *,
    threshold: float = 2.0,
    automatic: bool = True,
    min_length_frames: int = 1,
) -> pd.DataFrame:
    if "velocity" not in eye_df.columns:
        raise KeyError("Eye dataframe is missing 'velocity'; run pupil_speed_calc first.")

    vel = np.asarray(eye_df["velocity"], dtype=float)
    ms = np.asarray(eye_df["ms_axis"], dtype=float)
    valid = np.isfinite(vel) & np.isfinite(ms)
    if not np.any(valid):
        return pd.DataFrame(columns=["saccade_start_ms", "saccade_length_frames", "peak_velocity"])

    if automatic:
        base = float(np.nanmedian(vel[valid]))
        spread = float(np.nanstd(vel[valid]))
        if spread < 1e-12:
            spread = 1.0
        thr = base + threshold * spread
    else:
        thr = float(threshold)

    above = valid & (vel > thr)
    rows: list[dict[str, float | int]] = []
    in_saccade = False
    start_idx = 0

    for idx, is_above in enumerate(above):
        if is_above and not in_saccade:
            start_idx = idx
            in_saccade = True
        elif not is_above and in_saccade:
            length = idx - start_idx
            if length >= min_length_frames:
                rows.append(
                    {
                        "saccade_start_ms": float(ms[start_idx]),
                        "saccade_length_frames": int(length),
                        "peak_velocity": float(np.nanmax(vel[start_idx:idx])),
                    }
                )
            in_saccade = False

    if in_saccade:
        length = len(above) - start_idx
        if length >= min_length_frames:
            rows.append(
                {
                    "saccade_start_ms": float(ms[start_idx]),
                    "saccade_length_frames": int(length),
                    "peak_velocity": float(np.nanmax(vel[start_idx:])),
                }
            )

    return pd.DataFrame(rows)


def _load_behavior_df(block) -> pd.DataFrame | None:
    path = block.analysis_path / f"block_{block.block_num}_behavior_state.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not {"start_time", "end_time", "annotation"}.issubset(df.columns):
        return None
    return df


def _behavior_at_time(behavior_df: pd.DataFrame | None, t_ms: float) -> str | float:
    if behavior_df is None or behavior_df.empty:
        return np.nan
    hit = behavior_df.query("start_time <= @t_ms and end_time > @t_ms")
    if hit.empty:
        return np.nan
    return hit.iloc[0]["annotation"]


def _accel_sum(acc_df: pd.DataFrame | None, t0_ms: float, t1_ms: float) -> float:
    if acc_df is None or acc_df.empty:
        return np.nan
    seg = acc_df.query("t_mov_ms >= @t0_ms and t_mov_ms <= @t1_ms")
    if seg.empty:
        return np.nan
    return float(seg["movAll"].sum())


def _ensure_block_ep(block) -> None:
    if block.oe_rec is None:
        raise RuntimeError(f"Block {block.block_num}: missing Open Ephys recording (oe_rec).")


def _block_auxiliary_data(block) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    try:
        block.block_get_lizard_movement()
    except Exception:
        pass
    return getattr(block, "liz_mov_df", None), _load_behavior_df(block)


def prepare_block_velocity(
    block,
    *,
    saccade_threshold: float,
    min_saccade_frames: int,
    sync_diff_ms: float,
    eye_data_source: EyeDataSource = "auto",
    log: LoadLog | None = None,
) -> pd.DataFrame:
    load_final_sync_df(block, verbose=False)
    _ensure_block_ep(block)
    attach_eye_data_to_block(block, source=eye_data_source, log=log)

    block.pupil_speed_calc()
    acc_df, behavior_df = _block_auxiliary_data(block)

    rows: list[dict] = []
    for eye, eye_df in (("L", block.le_df), ("R", block.re_df)):
        saccades = detect_saccades_chunked(
            eye_df,
            threshold=saccade_threshold,
            automatic=True,
            min_length_frames=min_saccade_frames,
        )
        saccades = saccades[saccades["saccade_length_frames"] > 0]
        for _, sacc in saccades.iterrows():
            t_ms = float(sacc["saccade_start_ms"])
            rows.append(
                {
                    "block": block.block_num,
                    "eye": eye,
                    "saccade_start_ms": t_ms,
                    "saccade_length_frames": int(sacc["saccade_length_frames"]),
                    "peak_velocity": float(sacc["peak_velocity"]),
                    "accel": _accel_sum(acc_df, t_ms - 50.0, t_ms + 100.0),
                    "behavior": _behavior_at_time(behavior_df, t_ms),
                }
            )

    events = pd.DataFrame(rows)
    return annotate_events_sync_status(events, sync_diff_ms=sync_diff_ms)


def prepare_block_legacy(
    block,
    *,
    speed_threshold: float,
    magnitude_calib: float,
    sync_diff_ms: float,
    use_pupil_diameter: bool,
    eye_data_source: EyeDataSource = "auto",
    log: LoadLog | None = None,
) -> pd.DataFrame:
    load_final_sync_df(block, verbose=False)
    _ensure_block_ep(block)
    attach_eye_data_to_block(block, source=eye_data_source, log=log)
    acc_df, behavior_df = _block_auxiliary_data(block)

    return detect_legacy_block_events(
        block,
        speed_threshold=speed_threshold,
        magnitude_calib=magnitude_calib,
        sync_diff_ms=sync_diff_ms,
        use_pupil_diameter=use_pupil_diameter,
        acc_df=acc_df,
        behavior_df=behavior_df,
        accel_fn=_accel_sum,
        behavior_fn=_behavior_at_time,
    )


def prepare_block(block, config: PlotConfig, log: LoadLog | None = None) -> pd.DataFrame:
    if config.detection_mode == "legacy":
        return prepare_block_legacy(
            block,
            speed_threshold=config.legacy_speed_threshold,
            magnitude_calib=config.legacy_magnitude_calib,
            sync_diff_ms=config.legacy_sync_diff_ms,
            use_pupil_diameter=config.legacy_use_pupil_diameter,
            eye_data_source=config.eye_data_source,
            log=log,
        )
    return prepare_block_velocity(
        block,
        saccade_threshold=config.saccade_threshold,
        min_saccade_frames=config.min_saccade_frames,
        sync_diff_ms=config.legacy_sync_diff_ms,
        eye_data_source=config.eye_data_source,
        log=log,
    )


def filter_events(events: pd.DataFrame, query: str | None) -> pd.DataFrame:
    if events.empty or not query or not query.strip():
        return events
    expr = query.strip()
    try:
        filtered = events.query(expr, engine="python")
    except Exception as exc:
        raise ValueError(f"Invalid query expression: {expr!r} ({exc})") from exc
    return filtered


def _events_for_trace_group(events: pd.DataFrame, eye: EyeKey) -> pd.DataFrame:
    """Map pipeline events to trace groups (see README for definitions)."""
    if events.empty:
        return events
    if "sync_status" not in events.columns:
        return events.iloc[:0]

    synced = events["sync_status"] == "synced"
    mono = events["sync_status"] == "non_synced"
    concurrent = events[synced & (events["eye"] == "L")]  # one trial per pair

    if eye == "Concurrent":
        return concurrent
    if eye == "Monocular":
        return events[mono]
    if eye == "L":
        return events[mono & (events["eye"] == "L")]
    if eye == "R":
        return events[mono & (events["eye"] == "R")]
    if eye == "All":
        return pd.concat([concurrent, events[mono]], ignore_index=True)
    return events.iloc[:0]


def _time_grid_ms(oe_rec, n_samples: int, half_window_ms: float) -> np.ndarray:
    sample_ms = float(oe_rec.sample_ms)
    return np.arange(n_samples, dtype=float) * sample_ms - half_window_ms


def extract_trials_for_channel(
    block,
    events: pd.DataFrame,
    channel: int,
    *,
    half_window_ms: float,
    batch_size: int,
) -> np.ndarray:
    if events.empty:
        return np.empty((0, 0))

    window_ms = 2.0 * half_window_ms
    starts = events["saccade_start_ms"].to_numpy(dtype=float) - half_window_ms
    trials: list[np.ndarray] = []

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        start_2d = np.atleast_2d(batch_starts)
        data, _timestamps = block.oe_rec.get_data(
            [channel],
            start_2d,
            window_ms,
            convert_microvolts=True,
            return_timestamps=True,
            repress_output=True,
        )
        if data is None:
            continue
        channel_trials = np.asarray(data[0], dtype=float)
        if channel_trials.ndim == 1:
            channel_trials = channel_trials[np.newaxis, :]
        trials.append(channel_trials)

    if not trials:
        return np.empty((0, 0))
    return np.vstack(trials)


def trial_within_window_std(trials: np.ndarray) -> np.ndarray:
    """Per-trial std (µV) across the aligned snippet."""
    if trials.size == 0:
        return np.array([], dtype=float)
    return np.nanstd(trials, axis=1)


def filter_trials_by_noise_std(
    trials: np.ndarray,
    std_k: float,
) -> tuple[np.ndarray, int]:
    """Drop trials whose within-window std exceeds ``std_k × median(trial std)``.

    ``std_k <= 0`` disables filtering.
    """
    if std_k <= 0 or trials.size == 0:
        return trials, 0

    per_std = trial_within_window_std(trials)
    ref = float(np.nanmedian(per_std))
    if not np.isfinite(ref) or ref < 1e-12:
        ref = 1e-12

    keep = per_std <= (std_k * ref)
    n_removed = int(np.sum(~keep))
    if n_removed == 0:
        return trials, 0
    if int(np.sum(keep)) == 0:
        return trials[:0], n_removed
    return trials[keep], n_removed


def average_trials(
    bundle: ChannelTrialBundle,
    *,
    std_k: float,
) -> EyeAverage | None:
    raw_n = int(bundle.trials.shape[0])
    if raw_n == 0:
        return None

    trials, n_removed = filter_trials_by_noise_std(bundle.trials, std_k)
    if trials.size == 0:
        return None

    n = min(trials.shape[1], len(bundle.grid_ms))
    trials = trials[:, :n]
    grid_use = bundle.grid_ms[:n]
    mean = np.nanmean(trials, axis=0)
    sem = np.nanstd(trials, axis=0, ddof=1) / np.sqrt(max(1, trials.shape[0]))
    return EyeAverage(
        eye=bundle.eye,
        label=bundle.label,
        color=EYE_COLORS[bundle.eye],
        grid_ms=grid_use,
        mean=mean,
        sem=sem,
        n_trials=trials.shape[0],
        n_trials_raw=raw_n,
        n_trials_removed=n_removed,
    )

def extract_trial_bundles(
    blocks_by_num: dict[str, object],
    events: pd.DataFrame,
    channel: int,
    *,
    half_window_ms: float,
    batch_size: int,
) -> list[ChannelTrialBundle]:
    if events.empty:
        return []

    oe_rec = next(iter(blocks_by_num.values())).oe_rec
    sample_ms = float(oe_rec.sample_ms)
    n_samples = int(round((2.0 * half_window_ms) / sample_ms))
    grid = _time_grid_ms(oe_rec, n_samples, half_window_ms)

    trace_specs: list[tuple[EyeKey, str]] = [
        ("Concurrent", "Concurrent"),
        ("Monocular", "Monocular"),
        ("L", "Left"),
        ("R", "Right"),
        ("All", "All"),
    ]

    bundles: list[ChannelTrialBundle] = []
    for eye, label in trace_specs:
        group_events = _events_for_trace_group(events, eye)
        if group_events.empty:
            continue
        trials_parts: list[np.ndarray] = []
        for block_num, block_events in group_events.groupby("block"):
            block = blocks_by_num[str(block_num)]
            part = extract_trials_for_channel(
                block,
                block_events,
                channel,
                half_window_ms=half_window_ms,
                batch_size=batch_size,
            )
            if part.size:
                trials_parts.append(part)
        if not trials_parts:
            continue
        trials = np.vstack(trials_parts)
        n = min(trials.shape[1], len(grid))
        bundles.append(
            ChannelTrialBundle(
                channel=channel,
                eye=eye,
                label=label,
                grid_ms=grid[:n],
                trials=trials[:, :n],
            )
        )

    return bundles


def compute_averages_from_bundles(
    bundles: list[ChannelTrialBundle],
    *,
    std_k: float,
    log: LoadLog | None = None,
) -> dict[int, list[EyeAverage]]:
    log = log or NullLoadLog()
    channel_averages: dict[int, list[EyeAverage]] = {}
    for bundle in bundles:
        avg = average_trials(bundle, std_k=std_k)
        if avg is None:
            continue
        if std_k > 0 and avg.n_trials_removed:
            log.info(
                f"HS ch{bundle.channel} {bundle.label}: noise filter removed "
                f"{avg.n_trials_removed}/{avg.n_trials_raw} trials (k={std_k:g})"
            )
        channel_averages.setdefault(bundle.channel, []).append(avg)
    return channel_averages


def compute_eye_averages(
    blocks_by_num: dict[str, object],
    events: pd.DataFrame,
    channel: int,
    *,
    half_window_ms: float,
    batch_size: int,
    ep_noise_std_k: float = 0.0,
    log: LoadLog | None = None,
) -> tuple[list[ChannelTrialBundle], list[EyeAverage]]:
    bundles = extract_trial_bundles(
        blocks_by_num,
        events,
        channel,
        half_window_ms=half_window_ms,
        batch_size=batch_size,
    )
    avgs = compute_averages_from_bundles(bundles, std_k=ep_noise_std_k, log=log).get(
        channel, []
    )
    return bundles, avgs


def recompute_averages(result: PipelineResult, log: LoadLog | None = None) -> PipelineResult:
    """Re-average cached trials (e.g. after changing noise threshold)."""
    log = log or NullLoadLog()
    std_k = result.config.ep_noise_std_k
    channel_averages = compute_averages_from_bundles(
        result.trial_bundles, std_k=std_k, log=log
    )
    if not channel_averages:
        raise ValueError("No trials remain after applying the EP noise filter.")
    result.channel_averages = channel_averages
    return result


def run_pipeline(config: PlotConfig, log: LoadLog | None = None) -> PipelineResult:
    log = log or NullLoadLog()
    block_nums = normalize_block_numbers(config.blocks)
    block_labels = [f"{b:03d}" for b in block_nums]

    blocks = block_generator(
        block_numbers=block_nums,
        experiment_path=str(config.experiment_path),
        animal=config.animal,
    )
    if not blocks:
        raise ValueError(
            f"No blocks found for animal={config.animal!r} blocks={block_labels} "
            f"under {config.experiment_path}"
        )

    blocks_by_num = {str(b.block_num): b for b in blocks}
    event_parts: list[pd.DataFrame] = []
    for block in blocks:
        log.info(f"Detecting saccades in block {block.block_num}...")
        try:
            event_parts.append(prepare_block(block, config, log=log))
        except Exception as exc:
            log.warn(f"Skipping block {block.block_num}: {exc}")

    if not event_parts:
        raise ValueError("No saccade events collected from the selected blocks.")

    events = pd.concat(event_parts, ignore_index=True)
    log.info(f"Detected {len(events)} saccades before filtering.")
    events_filtered = filter_events(events, config.query)
    log.info(f"{len(events_filtered)} saccades after filter.")
    if events_filtered.empty:
        raise ValueError("No saccades left after applying the query filter.")

    channel_averages: dict[int, list[EyeAverage]] = {}
    all_bundles: list[ChannelTrialBundle] = []
    for ch in config.electrodes:
        log.info(f"Extracting LFP for channel {ch}...")
        bundles, avgs = compute_eye_averages(
            blocks_by_num,
            events_filtered,
            ch,
            half_window_ms=config.half_window_ms,
            batch_size=config.batch_size,
            ep_noise_std_k=config.ep_noise_std_k,
            log=log,
        )
        all_bundles.extend(bundles)
        if avgs:
            channel_averages[ch] = avgs
        else:
            log.warn(f"No trials for channel {ch}")

    if not channel_averages:
        raise ValueError("No LFP data extracted for the selected electrodes.")

    return PipelineResult(
        config=config,
        block_labels=block_labels,
        events=events,
        events_filtered=events_filtered,
        channel_averages=channel_averages,
        trial_bundles=all_bundles,
        blocks_by_num=blocks_by_num,
    )


def _style_axis(ax, *, ylabel: str, label_fontsize: float, tick_fontsize: float) -> None:
    ax.axvline(0, color="#bbbbbb", linestyle=":", linewidth=0.7, zorder=0)
    ax.set_xlabel("Time from saccade onset (ms)", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize, width=0.6, length=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_linewidth(0.6)
    ax.grid(False)


def build_figure(
    result: PipelineResult,
    *,
    visible_eyes: frozenset[EyeKey] | None = None,
) -> tuple[Figure, list, list]:
    """Build matplotlib figure; returns (fig, legend_handles, legend_labels)."""
    config = result.config
    visible = visible_eyes if visible_eyes is not None else config.visible_eyes
    channel_averages = result.channel_averages
    if not channel_averages:
        raise ValueError("No LFP averages to plot.")

    channels = list(channel_averages.keys())
    n = len(channels)
    fig_w, fig_h_each = config.figsize
    fig_h = fig_h_each * n if n > 1 else fig_h_each
    fig = Figure(figsize=(fig_w, fig_h), dpi=config.dpi)
    axes_flat = []
    for i in range(n):
        share = axes_flat[0] if axes_flat else None
        ax = fig.add_subplot(n, 1, i + 1, sharex=share)
        axes_flat.append(ax)

    legend_handles: list = []
    legend_labels: list[str] = []
    seen_eyes: set[EyeKey] = set()

    for ax, ch in zip(axes_flat, channels):
        for avg in channel_averages[ch]:
            if avg.eye not in visible:
                continue
            label = f"{avg.label} (n={avg.n_trials})"
            (line,) = ax.plot(
                avg.grid_ms,
                avg.mean,
                color=avg.color,
                linewidth=1.5 if avg.eye in ("All", "Concurrent", "Monocular") else 1.2,
                label=label,
                zorder=3,
            )
            ax.fill_between(
                avg.grid_ms,
                avg.mean - avg.sem,
                avg.mean + avg.sem,
                color=avg.color,
                alpha=config.sem_alpha,
                linewidth=0,
                zorder=2,
            )
            if avg.eye not in seen_eyes:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_eyes.add(avg.eye)
        ax.set_title(f"HS channel {ch}", fontsize=config.tick_fontsize, loc="left")
        _style_axis(
            ax,
            ylabel="LFP (µV)",
            label_fontsize=config.label_fontsize,
            tick_fontsize=config.tick_fontsize,
        )
        if visible:
            ax.legend(
                fontsize=max(6.0, config.tick_fontsize - 1.0),
                frameon=False,
                loc="upper right",
            )

    subtitle = f"{config.animal}  blocks {', '.join(result.block_labels)}"
    if config.query:
        subtitle += f"  filter: {config.query}"
    if config.detection_mode == "legacy":
        subtitle += (
            f"  legacy thr={config.legacy_speed_threshold:g}"
            f" sync±{config.legacy_sync_diff_ms:g}ms"
        )
    elif "sync_status" in result.events.columns:
        subtitle += f"  sync±{config.legacy_sync_diff_ms:g}ms"
    if config.ep_noise_std_k > 0:
        subtitle += f"  EP noise k={config.ep_noise_std_k:g}"
    fig.suptitle(subtitle, fontsize=config.label_fontsize + 1, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, legend_handles, legend_labels


def build_legend_figure(
    legend_handles: list,
    legend_labels: list[str],
    *,
    dpi: float,
) -> Figure:
    leg_fig = Figure(figsize=(2.0, 1.0), dpi=dpi)
    leg_ax = leg_fig.add_subplot(111)
    leg_ax.axis("off")
    leg_ax.legend(legend_handles, legend_labels, frameon=False, loc="center")
    leg_fig.tight_layout()
    return leg_fig


def save_figure_to_pdf(fig: Figure, path: Path) -> None:
    """Save a figure to PDF using its embedded dpi (no extra bbox crop)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    fig.savefig(path, format="pdf", dpi=fig.dpi)


def export_display_figures(
    fig: Figure,
    main_path: Path,
    legend_path: Path | None,
    legend_handles: list,
    legend_labels: list[str],
) -> None:
    """Write the on-screen figure (and matching legend) to PDF."""
    save_figure_to_pdf(fig, main_path)
    if legend_path and legend_handles:
        leg_fig = build_legend_figure(
            legend_handles, legend_labels, dpi=fig.dpi
        )
        save_figure_to_pdf(leg_fig, legend_path)


def export_figures(
    result: PipelineResult,
    main_path: Path,
    legend_path: Path | None = None,
    *,
    visible_eyes: frozenset[EyeKey] | None = None,
    display_figure: Figure | None = None,
    legend_handles: list | None = None,
    legend_labels: list[str] | None = None,
) -> None:
    """Export PDFs. Pass ``display_figure`` to save the exact preview figure."""
    if display_figure is not None:
        export_display_figures(
            display_figure,
            main_path,
            legend_path,
            legend_handles or [],
            legend_labels or [],
        )
        return

    fig, handles, labels = build_figure(result, visible_eyes=visible_eyes)
    export_display_figures(fig, main_path, legend_path, handles, labels)


def timestamped_export_paths(
    analysis_folder: Path,
    animal: str,
    block_labels: list[str],
    *,
    plot_label: str | None = None,
    when: datetime | None = None,
) -> tuple[Path, Path]:
    """Backward-compatible wrapper; prefer :func:`resolve_export_paths`."""
    resolved = resolve_export_paths(
        analysis_folder,
        animal,
        block_labels,
        plot_label=plot_label,
        when=when,
    )
    return resolved.main_path, resolved.legend_path


def sanitize_plot_label(label: str) -> str:
    """Make a user plot_label safe for filenames."""
    text = label.strip()
    if not text:
        return ""
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("._")


def build_export_stem(animal: str, block_labels: list[str], plot_label: str | None) -> str:
    blocks_tag = "-".join(block_labels) if block_labels else "blocks"
    stem = f"saccade_lfp_{animal}_blocks_{blocks_tag}"
    tag = sanitize_plot_label(plot_label or "")
    if tag:
        stem = f"{stem}_{tag}"
    return stem


def resolve_export_paths(
    analysis_folder: Path,
    animal: str,
    block_labels: list[str],
    *,
    plot_label: str | None = None,
    when: datetime | None = None,
) -> ExportPaths:
    """Resolve PDF paths under ``{analysis_folder}/{YYYY-MM-DD}/``.

    If the target filename already exists, appends ``_2``, ``_3``, … until unused.
    """
    when = when or datetime.now()
    output_dir = Path(analysis_folder) / when.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = build_export_stem(animal, block_labels, plot_label)
    candidate = stem
    collision_warning: str | None = None

    def _pair_exists(name: str) -> bool:
        return (output_dir / f"{name}.pdf").exists() or (
            output_dir / f"{name}_legend.pdf"
        ).exists()

    if _pair_exists(stem):
        n = 2
        while True:
            candidate = f"{stem}_{n}"
            if not _pair_exists(candidate):
                collision_warning = (
                    f"OUTPUT COLLISION: '{stem}.pdf' already exists in {output_dir}. "
                    f"Saving as '{candidate}.pdf' instead (no overwrite)."
                )
                break
            n += 1

    main_path = output_dir / f"{candidate}.pdf"
    legend_path = output_dir / f"{candidate}_legend.pdf"
    return ExportPaths(
        main_path=main_path,
        legend_path=legend_path,
        output_dir=output_dir,
        filename_stem=candidate,
        collision_warning=collision_warning,
    )


def default_export_paths(
    analysis_folder: Path,
    animal: str,
    block_labels: list[str] | None = None,
    *,
    plot_label: str | None = None,
) -> ExportPaths:
    return resolve_export_paths(
        analysis_folder,
        animal,
        block_labels or [],
        plot_label=plot_label,
    )
