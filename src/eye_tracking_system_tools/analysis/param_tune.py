"""
Headless helpers for the parameter-visualization / tuning notebook.

Prepare eye traces, run the production saccade detector, and build preview
figures for single-block threshold debugging (degrees vs pixels).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.figure import Figure
from scipy import stats

from eye_tracking_system_tools.analysis.binocular import find_synced_saccades_ms
from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.eye_trace_io import csv_choices_meta, load_block_eyes
from eye_tracking_system_tools.analysis.saccade_events import (
    create_saccade_events_with_direction_segmentation_robust,
)

Mode = Literal["deg", "px"]

PARAMS_HEADER = """\
# Analysis / figure parameters (edited via the parameter visualization desktop GUI).
# Units documented in configs/analysis_params.yaml.
#
"""

MAX_DRAW_POINTS = 30_000


@dataclass
class TraceBundle:
    """One block's left/right traces with speed columns precomputed."""

    spec: BlockSpec
    left: pd.DataFrame
    right: pd.DataFrame
    csv_meta: dict = field(default_factory=dict)

    def eye(self, which: str) -> pd.DataFrame:
        w = which.strip().upper()
        if w in {"L", "LEFT"}:
            return self.left
        if w in {"R", "RIGHT"}:
            return self.right
        raise KeyError(f"eye must be L or R, got {which!r}")


def prepare_traces(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure pixel and angular speed columns exist (same formulas as the detector)."""
    out = df.copy()
    if "speed_x" not in out.columns and "center_x" in out.columns:
        out["speed_x"] = out["center_x"].diff()
    if "speed_y" not in out.columns and "center_y" in out.columns:
        out["speed_y"] = out["center_y"].diff()
    if "speed_r" not in out.columns and {"speed_x", "speed_y"}.issubset(out.columns):
        out["speed_r"] = np.sqrt(out["speed_x"] ** 2 + out["speed_y"] ** 2)
    if "angular_speed_phi" not in out.columns and "k_phi" in out.columns:
        out["angular_speed_phi"] = out["k_phi"].diff()
    if "angular_speed_theta" not in out.columns and "k_theta" in out.columns:
        out["angular_speed_theta"] = out["k_theta"].diff()
    if "angular_speed_r" not in out.columns and {
        "angular_speed_phi",
        "angular_speed_theta",
    }.issubset(out.columns):
        out["angular_speed_r"] = np.sqrt(
            out["angular_speed_phi"] ** 2 + out["angular_speed_theta"] ** 2
        )
    return out


def load_trace_bundle(spec: BlockSpec, *, log: bool = True) -> TraceBundle:
    loaded = load_block_eyes(spec, log=log)
    return TraceBundle(
        spec=spec,
        left=prepare_traces(loaded.left),
        right=prepare_traces(loaded.right),
        csv_meta=csv_choices_meta(loaded),
    )


def saccade_params_from_dict(params: dict[str, Any]) -> dict[str, Any]:
    """Map YAML ``saccade:`` section → detector kwargs (matches pipeline._saccade_params)."""
    s = params.get("saccade", {}) if params else {}
    return {
        "speed_threshold": float(s.get("speed_threshold_deg_per_frame", 0.8)),
        "directional_delta_threshold_deg": float(
            s.get("directional_delta_threshold_deg", 90.0)
        ),
        "min_subsaccade_samples": int(s.get("min_subsaccade_samples", 2)),
        "min_net_disp": float(s.get("min_net_disp_deg", 0.5)),
        "speed_profile": bool(s.get("speed_profile", True)),
    }


def detect_eye(
    df: pd.DataFrame,
    saccade_params: dict[str, Any] | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the production angular saccade detector; return (enriched_df, events)."""
    if saccade_params is None:
        saccade_params = saccade_params_from_dict(params or {})
    prepared = prepare_traces(df)
    return create_saccade_events_with_direction_segmentation_robust(
        prepared, **saccade_params
    )


def raw_threshold_runs(
    df: pd.DataFrame,
    *,
    mode: Mode = "deg",
    threshold: float,
) -> list[tuple[float, float]]:
    """
    Contiguous above-threshold runs as ``(on_ms, off_ms)`` before length/disp filters.

    Used as a light overlay so the speed gate alone is visible.
    """
    speed_col = "angular_speed_r" if mode == "deg" else "speed_r"
    if speed_col not in df.columns or "ms_axis" not in df.columns:
        return []
    speed = df[speed_col].to_numpy(dtype=float)
    ms = df["ms_axis"].to_numpy(dtype=float)
    above = np.isfinite(speed) & (speed > float(threshold))
    if not above.any():
        return []
    runs: list[tuple[float, float]] = []
    in_run = False
    start_i = 0
    for i, flag in enumerate(above):
        if flag and not in_run:
            in_run = True
            start_i = i
        elif not flag and in_run:
            in_run = False
            runs.append((float(ms[start_i]), float(ms[i - 1])))
    if in_run:
        runs.append((float(ms[start_i]), float(ms[len(above) - 1])))
    return runs


def pixel_probe_events(
    df: pd.DataFrame,
    *,
    speed_threshold_px: float,
    min_subsaccade_samples: int = 2,
    min_net_disp_px: float = 0.5,
) -> pd.DataFrame:
    """
    Lightweight pixel-speed gate for the px-mode probe overlay (not written to YAML).

    Uses ``speed_r`` / ``center_x``/``center_y``; no direction segmentation.
    """
    prepared = prepare_traces(df)
    if "speed_r" not in prepared.columns:
        return pd.DataFrame()
    speed = prepared["speed_r"].to_numpy(dtype=float)
    above = np.isfinite(speed) & (speed > float(speed_threshold_px))
    on_off = above.astype(int) - np.r_[0, above[:-1].astype(int)]
    on_inds = np.where(on_off == 1)[0]
    off_inds = np.where(on_off == -1)[0]
    if len(on_inds) > len(off_inds):
        on_inds = on_inds[:-1]
    rows = []
    for start_ind, end_ind in zip(on_inds, off_inds):
        if end_ind - start_ind + 1 < min_subsaccade_samples:
            continue
        seg = prepared.iloc[start_ind : end_ind + 1]
        dx = float(seg["center_x"].iloc[-1] - seg["center_x"].iloc[0])
        dy = float(seg["center_y"].iloc[-1] - seg["center_y"].iloc[0])
        net = float(np.hypot(dx, dy))
        if net < min_net_disp_px:
            continue
        rows.append(
            {
                "saccade_on_ms": float(seg["ms_axis"].iloc[0]),
                "saccade_off_ms": float(seg["ms_axis"].iloc[-1]),
                "net_angular_disp": net,  # px displacement, probe only
                "peak_velocity": float(np.nanmax(seg["speed_r"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def _downsample_mask(n: int, max_points: int = MAX_DRAW_POINTS) -> np.ndarray:
    if n <= max_points:
        return np.ones(n, dtype=bool)
    step = int(np.ceil(n / max_points))
    mask = np.zeros(n, dtype=bool)
    mask[::step] = True
    mask[-1] = True
    return mask


def _window_slice(df: pd.DataFrame, t0_ms: float, t1_ms: float) -> pd.DataFrame:
    ms = df["ms_axis"]
    return df.loc[(ms >= t0_ms) & (ms <= t1_ms)]


def figure_saccade_preview(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    mode: Mode = "deg",
    threshold: float,
    t0_ms: float | None = None,
    t1_ms: float | None = None,
    show_raw_runs: bool = True,
    probe_events: pd.DataFrame | None = None,
    probe_threshold: float | None = None,
    figsize: tuple[float, float] = (9.0, 4.5),
) -> Figure:
    """Two-row position + speed preview with event / raw-run overlays."""
    df = prepare_traces(df)
    if t0_ms is None:
        t0_ms = float(np.nanmin(df["ms_axis"]))
    if t1_ms is None:
        t1_ms = float(np.nanmax(df["ms_axis"]))
    win = _window_slice(df, t0_ms, t1_ms)
    if win.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No samples in window", ha="center", va="center")
        ax.set_axis_off()
        return fig

    mask = _downsample_mask(len(win))
    drawn = win.iloc[mask]
    t = drawn["ms_axis"].to_numpy(dtype=float) / 1000.0  # seconds for readability

    if mode == "deg":
        y1a, y1b = drawn["k_phi"], drawn["k_theta"]
        speed = drawn["angular_speed_r"]
        pos_ylab, speed_ylab = "angle [deg]", "angular speed [deg/frame]"
        lab_a, lab_b = "φ", "θ"
    else:
        y1a, y1b = drawn["center_x"], drawn["center_y"]
        speed = drawn["speed_r"]
        pos_ylab, speed_ylab = "position [px]", "speed [px/frame]"
        lab_a, lab_b = "x", "y"

    fig, (ax_pos, ax_spd) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]}
    )
    ax_pos.plot(t, y1a.to_numpy(dtype=float), lw=0.8, color="#1f77b4", label=lab_a)
    ax_pos.plot(t, y1b.to_numpy(dtype=float), lw=0.8, color="#ff7f0e", label=lab_b)
    ax_pos.set_ylabel(pos_ylab, fontsize=9)
    ax_pos.legend(loc="upper right", fontsize=7, frameon=False)
    ax_pos.tick_params(labelsize=8)

    ax_spd.plot(t, speed.to_numpy(dtype=float), lw=0.7, color="#333333")
    ax_spd.axhline(threshold, color="#d62728", ls="-", lw=1.0, label=f"thr={threshold:g}")
    if probe_threshold is not None and mode == "px":
        ax_spd.axhline(
            probe_threshold, color="#9467bd", ls="--", lw=1.0, label=f"px probe={probe_threshold:g}"
        )
    ax_spd.set_ylabel(speed_ylab, fontsize=9)
    ax_spd.set_xlabel("time [s]", fontsize=9)
    ax_spd.tick_params(labelsize=8)

    # Raw above-threshold runs (light)
    if show_raw_runs:
        for on_ms, off_ms in raw_threshold_runs(win, mode=mode, threshold=threshold):
            if off_ms < t0_ms or on_ms > t1_ms:
                continue
            ax_spd.axvspan(
                max(on_ms, t0_ms) / 1000.0,
                min(off_ms, t1_ms) / 1000.0,
                color="#d62728",
                alpha=0.12,
                lw=0,
            )

    # Accepted degree-pipeline events (both modes overlay these times)
    if events is not None and not events.empty:
        for _, row in events.iterrows():
            on_ms = float(row["saccade_on_ms"])
            off_ms = float(row["saccade_off_ms"])
            if off_ms < t0_ms or on_ms > t1_ms:
                continue
            x0, x1 = max(on_ms, t0_ms) / 1000.0, min(off_ms, t1_ms) / 1000.0
            for ax in (ax_pos, ax_spd):
                ax.axvspan(x0, x1, color="#2ca02c", alpha=0.25, lw=0)
            ax_spd.axvline(on_ms / 1000.0, color="#2ca02c", lw=0.6, alpha=0.8)

    # Optional pixel probe events (dashed violet markers)
    if probe_events is not None and not probe_events.empty and mode == "px":
        for _, row in probe_events.iterrows():
            on_ms = float(row["saccade_on_ms"])
            off_ms = float(row["saccade_off_ms"])
            if off_ms < t0_ms or on_ms > t1_ms:
                continue
            ax_spd.axvspan(
                max(on_ms, t0_ms) / 1000.0,
                min(off_ms, t1_ms) / 1000.0,
                color="#9467bd",
                alpha=0.15,
                lw=0,
            )

    ax_spd.legend(loc="upper right", fontsize=7, frameon=False)
    for ax in (ax_pos, ax_spd):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def event_stats(events: pd.DataFrame, *, n_raw_runs: int = 0) -> dict[str, Any]:
    if events is None or events.empty:
        return {
            "n_raw_runs": int(n_raw_runs),
            "n_events": 0,
            "median_peak_speed": float("nan"),
            "median_amp": float("nan"),
        }
    peak = events["peak_velocity"].to_numpy(dtype=float) if "peak_velocity" in events else np.array([])
    amp = (
        events["net_angular_disp"].to_numpy(dtype=float)
        if "net_angular_disp" in events
        else np.array([])
    )
    return {
        "n_raw_runs": int(n_raw_runs),
        "n_events": int(len(events)),
        "median_peak_speed": float(np.nanmedian(peak)) if peak.size else float("nan"),
        "median_amp": float(np.nanmedian(amp)) if amp.size else float("nan"),
    }


def detect_block_both_eyes(
    bundle: TraceBundle,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (l_events, r_events, all_events with eye column)."""
    sp = saccade_params_from_dict(params)
    _, l_ev = detect_eye(bundle.left, sp)
    _, r_ev = detect_eye(bundle.right, sp)
    parts = []
    if not l_ev.empty:
        l = l_ev.copy()
        l["eye"] = "L"
        parts.append(l)
    if not r_ev.empty:
        r = r_ev.copy()
        r["eye"] = "R"
        parts.append(r)
    all_ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return l_ev, r_ev, all_ev


def figure_binocular_preview(
    l_ev: pd.DataFrame,
    r_ev: pd.DataFrame,
    sync_diff_ms: float,
    *,
    t0_ms: float | None = None,
    t1_ms: float | None = None,
    figsize: tuple[float, float] = (9.0, 3.2),
) -> tuple[Figure, dict[str, int]]:
    """Stem/raster of L/R onsets with sync pairing links."""
    parts = []
    for ev, eye in ((l_ev, "L"), (r_ev, "R")):
        if ev is None or ev.empty:
            continue
        tmp = ev.copy()
        tmp["eye"] = eye
        parts.append(tmp)
    if not parts:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_axis_off()
        return fig, {"n_synced_pairs": 0, "n_non_synced": 0, "n_l": 0, "n_r": 0}

    all_ev = pd.concat(parts, ignore_index=True)
    if t0_ms is not None:
        all_ev = all_ev[all_ev["saccade_on_ms"] >= t0_ms]
    if t1_ms is not None:
        all_ev = all_ev[all_ev["saccade_on_ms"] <= t1_ms]

    synced, non_synced = find_synced_saccades_ms(all_ev, sync_diff_ms=float(sync_diff_ms))
    n_pairs = int(synced["Main"].nunique()) if not synced.empty and "Main" in synced.columns else 0
    counts = {
        "n_synced_pairs": n_pairs,
        "n_non_synced": int(len(non_synced)),
        "n_l": int((all_ev["eye"] == "L").sum()) if not all_ev.empty else 0,
        "n_r": int((all_ev["eye"] == "R").sum()) if not all_ev.empty else 0,
    }

    fig, ax = plt.subplots(figsize=figsize)
    for eye, y, color in (("L", 1.0, "#1f77b4"), ("R", 0.0, "#ff7f0e")):
        sub = all_ev[all_ev["eye"] == eye]
        if sub.empty:
            continue
        t = sub["saccade_on_ms"].to_numpy(dtype=float) / 1000.0
        ax.vlines(t, y - 0.35, y + 0.35, colors=color, lw=0.9, label=eye)

    if not synced.empty and "Main" in synced.columns:
        for main, g in synced.groupby("Main"):
            if set(g["eye"].astype(str)) != {"L", "R"}:
                continue
            tl = float(g.loc[g["eye"] == "L", "saccade_on_ms"].iloc[0]) / 1000.0
            tr = float(g.loc[g["eye"] == "R", "saccade_on_ms"].iloc[0]) / 1000.0
            ax.plot([tl, tr], [1.0, 0.0], color="#2ca02c", lw=0.7, alpha=0.7)

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["R", "L"])
    ax.set_xlabel("onset time [s]", fontsize=9)
    ax.set_title(
        f"sync_diff_ms={sync_diff_ms:g}  pairs={counts['n_synced_pairs']}  "
        f"mono={counts['n_non_synced']}",
        fontsize=10,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, counts


def figure_main_sequence_preview(
    events: pd.DataFrame,
    main_sequence_cfg: dict[str, Any] | None = None,
    *,
    figsize: tuple[float, float] = (5.0, 4.0),
) -> Figure:
    """Amplitude vs peak-velocity scatter with optional linear fit."""
    cfg = dict(main_sequence_cfg or {})
    min_amp = float(cfg.get("min_amp_deg", 0.5))
    max_amp_pct = float(cfg.get("max_amp_pct", 99.5))
    frame_rate = float(cfg.get("frame_rate_fps", 60.0))
    velocity_unit = str(cfg.get("velocity_unit", "deg/sec"))

    fig, ax = plt.subplots(figsize=figsize)
    if events is None or events.empty:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    amp = events["net_angular_disp"].to_numpy(dtype=float)
    peak = events["peak_velocity"].to_numpy(dtype=float)  # deg/frame
    mask = np.isfinite(amp) & np.isfinite(peak) & (amp >= min_amp)
    if mask.any():
        hi = float(np.nanpercentile(amp[mask], max_amp_pct))
        mask &= amp <= hi

    amp_p, peak_p = amp[mask], peak[mask]
    if velocity_unit.replace(" ", "").lower() in {"deg/sec", "deg/s"}:
        peak_plot = peak_p * frame_rate
        v_label = "peak velocity [deg/s]"
    elif velocity_unit.replace(" ", "").lower() in {"deg/ms"}:
        peak_plot = peak_p * (frame_rate / 1000.0)
        v_label = "peak velocity [deg/ms]"
    else:
        peak_plot = peak_p
        v_label = "peak velocity [deg/frame]"

    ax.scatter(amp_p, peak_plot, s=8, alpha=0.45, color="#1f77b4", edgecolors="none")
    if amp_p.size >= 5:
        slope, intercept, r, p, _ = stats.linregress(amp_p, peak_plot)
        xs = np.linspace(float(np.nanmin(amp_p)), float(np.nanmax(amp_p)), 50)
        ax.plot(xs, slope * xs + intercept, color="#d62728", lw=1.2,
                label=f"fit r={r:.2f}  n={amp_p.size}")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_xlabel("amplitude [deg]", fontsize=9)
    ax.set_ylabel(v_label, fontsize=9)
    ax.set_title(f"main sequence  n={amp_p.size} / {len(events)}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def figure_amp_hist_preview(
    events: pd.DataFrame,
    figure_2g_cfg: dict[str, Any] | None = None,
    *,
    figsize: tuple[float, float] = (5.0, 3.2),
    label: str = "all",
) -> Figure:
    """Amplitude histogram using Fig 2g knobs."""
    cfg = dict(figure_2g_cfg or {})
    n_bins = int(cfg.get("amp_bins", 40))
    amp_max = float(cfg.get("amp_max_deg", 25.0))
    bins = np.linspace(0.0, amp_max, n_bins + 1)

    fig, ax = plt.subplots(figsize=figsize)
    if events is None or events.empty or "net_angular_disp" not in events.columns:
        ax.text(0.5, 0.5, "No events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    amp = events["net_angular_disp"].to_numpy(dtype=float)
    amp = amp[np.isfinite(amp)]
    ax.hist(amp, bins=bins, color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("amplitude [deg]", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(f"amp hist ({label})  n={amp.size}  bins={n_bins}  max={amp_max:g}", fontsize=10)
    ax.set_xlim(0.0, amp_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def deep_set(params: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``params['a']['b'] = value`` from dotted key ``a.b``."""
    parts = dotted_key.split(".")
    cur: dict[str, Any] = params
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def deep_get(params: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = params
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def flatten_params(params: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dict to ``(dotted_key, value)`` for leaf scalars/lists."""
    rows: list[tuple[str, Any]] = []
    for k, v in (params or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            rows.extend(flatten_params(v, key))
        else:
            rows.append((key, v))
    return rows


def save_params_yaml(path: Path | str, params: dict[str, Any], *, header: bool = True) -> Path:
    """Write params dict as YAML (clean dump; comments not preserved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(deepcopy(params), sort_keys=False, allow_unicode=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(PARAMS_HEADER)
        f.write(body)
    return path
