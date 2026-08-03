"""
Vignette plotting helpers restored for reproduction scripts (Figs 2b, 3a–c).

``plot_zoomed_in_with_head_rate`` is ported from
``development/old_pipelines_for_ref/single_block_plots_clean.ipynb`` (cell 41),
simplified to the light theme only (the notebook also had an unused dark
theme).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from scipy.signal import medfilt

from eye_tracking_system_tools.analysis.behavior_state import normalize_label
from eye_tracking_system_tools.analysis.figure_display import show_and_close

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

TRACE_L = "#1f77b4"
TRACE_R = "#8c564b"
RATE_SACC = "#000000"
RATE_HEAD = "#7b3294"
STATE_QUIET = "#d62728"
STATE_ACTIVE = "#2ca02c"


def _as_array_ms(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return np.sort(arr)


def _merge_close_events(events_ms, merge_ms: float = 0.0, strategy: str = "first") -> np.ndarray:
    """Collapse events within ``merge_ms`` of each other into a single onset per ``strategy``."""
    ev = _as_array_ms(events_ms)
    if ev.size == 0 or merge_ms is None or merge_ms <= 0:
        return ev
    pick = {
        "first": lambda c: c[0],
        "last": lambda c: c[-1],
        "mean": lambda c: float(np.mean(c)),
        "median": lambda c: float(np.median(c)),
    }.get(strategy, lambda c: c[0])

    merged: list[float] = []
    cluster = [ev[0]]
    for t in ev[1:]:
        if (t - cluster[-1]) < merge_ms:
            cluster.append(t)
        else:
            merged.append(pick(cluster))
            cluster = [t]
    merged.append(pick(cluster))
    return np.asarray(merged, dtype=float)


def _calc_rate(
    events_ms: np.ndarray, start_time: float, end_time: float, window_size: float = 10000, bin_size: float = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-window event rate (Hz): each bin counts events in ``[t - window_size, t]``."""
    tb = np.arange(start_time * 1000.0, end_time * 1000.0 + bin_size, bin_size, dtype=float)
    if events_ms.size == 0:
        return tb, np.zeros_like(tb, dtype=float)
    rate = np.zeros_like(tb, dtype=float)
    for i, t in enumerate(tb):
        st, en = t - window_size, t
        rate[i] = np.sum((events_ms >= st) & (events_ms <= en))
    return tb, rate / (window_size / 1000.0)


def _calc_saccade_rate(
    left_ms,
    right_ms,
    start_time: float,
    end_time: float,
    window_size: float = 10000,
    bin_size: float = 1000,
    merge_ms: float = 30.0,
    strategy: str = "first",
    dedup_within_eye: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Binocular saccade rate: de-duplicate coincident L/R onsets before rating."""
    l = _as_array_ms(left_ms)
    r = _as_array_ms(right_ms)
    if dedup_within_eye and merge_ms and merge_ms > 0:
        l = _merge_close_events(l, merge_ms=merge_ms, strategy=strategy) if l.size else l
        r = _merge_close_events(r, merge_ms=merge_ms, strategy=strategy) if r.size else r
    if l.size or r.size:
        both = np.sort(np.concatenate([l, r])) if (l.size and r.size) else (l if l.size else r)
        all_sacc = (
            _merge_close_events(both, merge_ms=merge_ms, strategy=strategy)
            if merge_ms and merge_ms > 0
            else np.unique(both)
        )
    else:
        all_sacc = np.array([], dtype=float)
    return _calc_rate(all_sacc, start_time, end_time, window_size, bin_size)


def _calc_head_rate(
    head_ms,
    start_time: float,
    end_time: float,
    window_size: float = 10000,
    bin_size: float = 1000,
    merge_ms: float = 0.0,
    strategy: str = "first",
) -> tuple[np.ndarray, np.ndarray]:
    merged = _merge_close_events(head_ms, merge_ms=merge_ms, strategy=strategy)
    return _calc_rate(merged, start_time, end_time, window_size, bin_size)


def _apply_y_ticks(ax, spec) -> None:
    if spec is None:
        return
    if isinstance(spec, (int, np.integer)):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=int(spec)))
        return
    try:
        vals = np.asarray(list(spec), dtype=float)
    except Exception:
        return
    if vals.size:
        ax.set_yticks(vals)


def plot_zoomed_in_with_head_rate(
    start_time,
    end_time,
    traces=None,
    left_df=None,
    right_df=None,
    left_ms=None,
    right_ms=None,
    head_movements_ms=None,
    behavior_state_df=None,
    state_df=None,
    export_path=None,
    figure_size=(2.3, 1.8),
    std_multiplier=3,
    limit_margins=1,
    plot_state_y=0,
    window_size=10000,
    bin_size=1000,
    phi_ticks=None,
    theta_ticks=None,
    phi_theta_ticks=None,
    pupil_ticks=None,
    x_zero_origin=True,
    behavior_time_unit="ms",
    head_merge_ms=0,
    head_merge_strategy="first",
    saccade_merge_ms=30.0,
    saccade_merge_strategy="first",
    dedup_within_eye=True,
    show=False,
    **kwargs,
):
    """Plot φ/θ (+ optional pupil / saccade-rate) traces for a time window.

    ``traces`` controls both which panels are drawn and their row order — one
    row per entry, chosen from ``{'center_x', 'center_y', 'pupil_diameter',
    'saccade_frequency'}``. Behavior-state segments (when given) are drawn as
    a colored ``hlines`` strip on the φ (``center_x``) axis.
    """
    traces = list(traces) if traces else ["center_x", "center_y"]
    behavior_state_df = behavior_state_df if behavior_state_df is not None else state_df

    if left_df is not None and not left_df.empty and "ms_axis" in left_df.columns:
        x_axis_abs = left_df["ms_axis"] / 1000.0
        mask = ((x_axis_abs >= start_time) & (x_axis_abs <= end_time)).to_numpy()
        x_axis_abs = x_axis_abs.to_numpy()
    else:
        x_axis_abs = np.array([], dtype=float)
        mask = np.array([], dtype=bool)

    def _rel_time(x):
        return (np.asarray(x) - start_time) if x_zero_origin else np.asarray(x)

    zoomed_x = _rel_time(x_axis_abs[mask])

    time_bins_ms, sacc_rate = _calc_saccade_rate(
        left_ms,
        right_ms,
        start_time,
        end_time,
        window_size,
        bin_size,
        merge_ms=float(saccade_merge_ms) if saccade_merge_ms else 0.0,
        strategy=saccade_merge_strategy,
        dedup_within_eye=dedup_within_eye,
    )
    z_mask = (time_bins_ms / 1000.0 >= start_time) & (time_bins_ms / 1000.0 <= end_time)
    z_time = _rel_time(time_bins_ms[z_mask] / 1000.0)
    z_sacc_rate = sacc_rate[z_mask]

    _, head_rate = _calc_head_rate(
        head_movements_ms,
        start_time,
        end_time,
        window_size,
        bin_size,
        merge_ms=float(head_merge_ms) if head_merge_ms else 0.0,
        strategy=head_merge_strategy,
    )
    z_head_rate = head_rate[z_mask]

    num_traces = len(traces)
    fig, axes = plt.subplots(nrows=num_traces, ncols=1, figsize=figure_size, sharex=True, dpi=300)
    if num_traces <= 1:
        axes = [axes] if num_traces == 1 else []
    x_min = 0.0 if x_zero_origin else start_time
    x_max = (end_time - start_time) if x_zero_origin else end_time

    def _set_ylim_from_data(ax, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=float)
        if std_multiplier is not None:
            med_val, std_val = np.nanmedian(data), np.nanstd(data)
            if np.isfinite(med_val) and np.isfinite(std_val):
                ax.set_ylim(med_val - std_multiplier * std_val, med_val + std_multiplier * std_val)
                return
        ax.set_ylim(np.nanmin(data) - limit_margins, np.nanmax(data) + limit_margins)

    first_trace_axis = None
    for i, trace in enumerate(traces):
        ax = axes[i]
        if trace == "center_x":
            phi_left = left_df["k_phi"].to_numpy(dtype=float)[mask]
            phi_right = right_df["k_phi"].to_numpy(dtype=float)[mask]
            _set_ylim_from_data(ax, np.concatenate([phi_left, phi_right]))
            ax.plot(zoomed_x, phi_left, label="Left Eye", color=TRACE_L, linewidth=1, alpha=0.9)
            ax.plot(zoomed_x, phi_right, label="Right Eye", color=TRACE_R, linewidth=1, alpha=0.9)
            ax.set_ylabel("Phi (deg)")
            ax.set_xlim(left=x_min, right=x_max)
            _apply_y_ticks(ax, phi_ticks if phi_ticks is not None else phi_theta_ticks)
            first_trace_axis = ax

        elif trace == "center_y":
            theta_left = left_df["k_theta"].to_numpy(dtype=float)[mask]
            theta_right = right_df["k_theta"].to_numpy(dtype=float)[mask]
            _set_ylim_from_data(ax, np.concatenate([theta_left, theta_right]))
            ax.plot(zoomed_x, theta_left, label="Left Eye", color=TRACE_L, linewidth=1, alpha=0.9)
            ax.plot(zoomed_x, theta_right, label="Right Eye", color=TRACE_R, linewidth=1, alpha=0.9)
            ax.set_ylabel("Theta (deg)")
            ax.set_xlim(left=x_min, right=x_max)
            _apply_y_ticks(ax, theta_ticks if theta_ticks is not None else phi_theta_ticks)

        elif trace in ("pupil_diameter", "pupil"):
            left_pupil = medfilt(left_df["pupil_diameter"].to_numpy(dtype=float), 121)[mask]
            right_pupil = medfilt(right_df["pupil_diameter"].to_numpy(dtype=float), 121)[mask]
            _set_ylim_from_data(ax, np.concatenate([left_pupil, right_pupil]))
            ax.plot(zoomed_x, left_pupil, label="Left Eye", color=TRACE_L, linewidth=1, alpha=0.9)
            ax.plot(zoomed_x, right_pupil, label="Right Eye", color=TRACE_R, linewidth=1, alpha=0.9)
            ax.set_ylabel("Pupil (mm)")
            ax.set_xlim(left=x_min, right=x_max)
            _apply_y_ticks(ax, pupil_ticks)

        elif trace == "saccade_frequency":
            ax.plot(z_time, z_sacc_rate, label="Saccade Rate", color=RATE_SACC, linewidth=1.5)
            ax.plot(z_time, z_head_rate, label="Head Movement Rate", color=RATE_HEAD, linewidth=1.5, alpha=0.95)
            ax.set_ylabel("Rate (Hz)")
            ax.set_xlim(left=x_min, right=x_max)

        else:
            raise ValueError(f"Unknown trace {trace!r}")

        if i == num_traces - 1:
            ax.set_xlabel("Time (seconds from window start)" if x_zero_origin else "Time (seconds)")

    # ---------- behavior-state strip (φ axis only) ----------
    if first_trace_axis is not None and behavior_state_df is not None and not behavior_state_df.empty:
        df = behavior_state_df[["start_time", "end_time", "annotation"]].copy()
        if behavior_time_unit == "ms":
            df["start_s"] = df["start_time"].astype(float) / 1000.0
            df["end_s"] = df["end_time"].astype(float) / 1000.0
        elif behavior_time_unit == "s":
            df["start_s"] = df["start_time"].astype(float)
            df["end_s"] = df["end_time"].astype(float)
        else:
            raise ValueError("behavior_time_unit must be 'ms' or 's'")

        tol = 1e-12
        df = df[np.isfinite(df["start_s"]) & np.isfinite(df["end_s"])]
        df = df[df["end_s"] > df["start_s"] + tol].copy()
        df["start_s"] = df["start_s"].clip(lower=start_time, upper=end_time)
        df["end_s"] = df["end_s"].clip(lower=start_time, upper=end_time)
        df = df[df["end_s"] > df["start_s"] + tol].sort_values("start_s").reset_index(drop=True)
        df["ann2"] = df["annotation"].map(normalize_label)

        xs = (df["start_s"].to_numpy() - start_time) if x_zero_origin else df["start_s"].to_numpy()
        xe = (df["end_s"].to_numpy() - start_time) if x_zero_origin else df["end_s"].to_numpy()

        for x1, x2, label in zip(xs, xe, df["ann2"].to_numpy()):
            if x2 <= x1 + tol:
                continue
            first_trace_axis.hlines(
                y=plot_state_y,
                xmin=x1,
                xmax=x2,
                colors=STATE_QUIET if label == "quiet" else STATE_ACTIVE,
                linestyles="solid",
                linewidth=4,
                zorder=1000,
                clip_on=False,
            )

    # ---------- cosmetics ----------
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_ylabel(ax.get_ylabel(), fontsize=9)
        ax.set_xlabel(ax.get_xlabel(), fontsize=9)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        seen: set[str] = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        fig.legend(
            [h for h, _ in uniq],
            [l for _, l in uniq],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=7,
            frameon=False,
        )

    fig.tight_layout()

    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path, format="pdf", bbox_inches="tight")

    show_and_close(fig, show)
    return fig, axes


def plot_angle_mapping(
    start_time,
    end_time,
    left_df,
    right_df,
    figure_size=(2.7, 1.7),
    export_path=None,
    xy_span=10,
    create_colorbar=False,
):
    """2D φ–θ trajectory for a short window (saccade examples)."""
    left_df = left_df.copy()
    right_df = right_df.copy()
    for df in (left_df, right_df):
        df["t_s"] = df["ms_axis"] / 1000.0

    left_w = left_df[(left_df["t_s"] >= start_time) & (left_df["t_s"] <= end_time)]
    right_w = right_df[(right_df["t_s"] >= start_time) & (right_df["t_s"] <= end_time)]

    fig, axs = plt.subplots(1, 2, figsize=figure_size, dpi=300)
    for ax, df, title in (
        (axs[0], left_w, "Left"),
        (axs[1], right_w, "Right"),
    ):
        if len(df) == 0:
            ax.set_title(f"{title} (empty)")
            continue
        t = df["t_s"].to_numpy()
        t_norm = (t - t.min()) / max(t.max() - t.min(), 1e-9)
        ax.scatter(df["k_phi"], df["k_theta"], c=t_norm, cmap="viridis", s=8)
        if len(df):
            ax.plot(df["k_phi"].iloc[0], df["k_theta"].iloc[0], "o", color="black", ms=4)
        ax.set_aspect("equal")
        mid_x = float(df["k_phi"].mean()) if len(df) else 0
        mid_y = float(df["k_theta"].mean()) if len(df) else 0
        ax.set_xlim(mid_x - xy_span / 2, mid_x + xy_span / 2)
        ax.set_ylim(mid_y - xy_span / 2, mid_y + xy_span / 2)
        ax.set_xlabel("φ [deg]", fontsize=8)
        ax.set_ylabel("θ [deg]", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    if export_path:
        out = Path(export_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() != ".pdf":
            out = out.with_suffix(".pdf")
        fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return fig, axs


def create_colorbar_pdf(start_time, end_time, figure_height=1.7, export_path=None):
    """Standalone time colorbar for saccade example panels."""
    fig = plt.figure(figsize=(1.2, figure_height), dpi=150)
    cax = fig.add_axes([0.35, 0.1, 0.2, 0.8])
    sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=start_time, vmax=end_time))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Time [s]", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    if export_path:
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return fig
