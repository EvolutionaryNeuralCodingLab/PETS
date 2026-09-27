#!/usr/bin/env python3
"""Redraw every deposited panel from one S1_Data.xlsx workbook.

Standalone: this file does not import the PETS package, does not read YAML,
and does not read pickle/CSV sidecars. The only data file is the workbook
passed on the command line (or S1_Data.xlsx next to this script).

Python packages required at runtime: numpy, pandas, matplotlib, scipy, openpyxl.

Usage:
  python plot_s1_main.py
  python plot_s1_main.py PATH/S1_Data.xlsx
  python plot_s1_main.py PATH/S1_Data.xlsx OUTDIR

With no arguments, reads S1_Data.xlsx next to this script and writes PDFs
into the same folder.
"""
from __future__ import annotations

import numbers
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.stats import gaussian_kde

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
COLOR_TEMPLATES = {"okabeito": OKABE}
TRACE_L = "#1f77b4"
TRACE_R = "#8c564b"
RATE_SACC = "#000000"
RATE_HEAD = "#7b3294"
STATE_QUIET = "#d62728"
STATE_ACTIVE = "#2ca02c"
_QUIET_LABELS = {"quiet", "stationary", "quiescent", "quite", "rest", "still"}
S9_BIN_WIDTHS = {"modular": 45.0, "rigid": 50.5, "mouse": 51.3284737549652, "turtle": 40.0}
S9_XMAX = 379.0960956761859
S10_THRESHOLD = {"Figure S10A": 0.8, "Figure S10B": 3.23, "Figure S10C": 2.0}
FRAME_MS = 16.67
FRAME_MS_3F = 16.650000000023283
LOG_EXPONENT_3F = 0.8
MS_PARAMS = {"figsize": (1.5, 1.7), "dpi": 300, "lw": 1.0}
FALLBACK_FRAME_MS = 16.67
_DEFAULT_LOG_PARAMS = {
    "figure_size": (2.5, 2.2),
    "num_bins": 20,
    "high_ms": 20000.0,
    "first_bin_factor": 0.5,
    "add_leading_zero_bin": False,
    "log_exponent": 1.0,
    "xlim": (100.0, 20000.0),
    "ylim": (0.0, 0.15),
    "linewidth": 1.5,
    "combined_linewidth": 2.0,
    "combined_color": "k",
    "combined_linestyle": "-",
    "xlabel": "ISI [ms]",
    "ylabel": "Probability",
}
_DEFAULT_LINEAR_PARAMS = {
    "figure_size": (2.0, 1.5),
    "num_bins": 1176,
    "low_ms": 10.0,
    "high_ms": 20000.0,
    "add_leading_zero_bin": False,
    "xlim": (10.0, 300.0),
    "xlim_policy": "slice",
    "ylim": None,
    "linewidth": 1.2,
    "combined_linewidth": 2.0,
    "combined_color": "k",
    "combined_linestyle": "-",
    "xlabel": "ISI [ms]",
    "ylabel": "Probability (per linear bin)",
}
_EPOCH_DURATION_COLORS = {"active": "#D55E00", "quiet": "#0072B2"}
UNIFIED_JITTER_MAX_BINS = 80
UNIFIED_JITTER_PANELS = (
    ("rigid", "rigid lizard", "#D55E00"),
    ("modular", "modular lizard", "#0072B2"),
    ("mouse", "modular mouse", "#009E73"),
    ("turtle", "modular turtle", "#CC79A7"),
)
RAYLEIGH_MEDIAN_FACTOR = float(np.sqrt(2.0 * np.log(2.0)))
S4_CORR_THR_POS = 0.2
S4_CORR_THR_NEG = -0.2
K = 70
METRIC = "total"
R_BINS = 20
R_RANGE = (0.0, 60.0)
MIN_PER_BIN = 0
RING_HALFWIDTH_DEG = 0.5
FIGSIZE = (2.5, 2.0)
DPI = 300
Y_RANGE = (0.0, 25.0)
A_EC = 640 // 2
B_EC = 480 // 2

XLSX = Path("S1_Data.xlsx")
OUT = Path(".")


def apply_paper_style() -> None:
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


def show_and_close(fig, show: bool = False) -> None:
    plt.close(fig)

def build_color_map(
    labels: Iterable,
    template: str = "okabeito",
    custom: Sequence | None = None,
    order: Sequence | None = None,
) -> dict:
    """Map each label to a color, cycling the chosen template."""
    labs = list(order) if order is not None else list(labels)
    base = (
        list(custom)
        if custom is not None
        else list(COLOR_TEMPLATES.get(template, COLOR_TEMPLATES["okabeito"]))
    )
    if not base:
        raise ValueError("Empty color template")
    return {lab: base[i % len(base)] for i, lab in enumerate(labs)}

def _bin_viridis_colors(n: int, cmap_name: str = "viridis") -> list[tuple]:
    """Paper Fig 2c: viridis sampled at linspace(0, 1, n)."""
    cmap = plt.get_cmap(cmap_name)
    if n <= 0:
        return []
    if n == 1:
        return [tuple(float(x) for x in cmap(0.0))]
    return [tuple(float(x) for x in cmap(i / (n - 1))) for i in range(n)]

def _empty_linear_fit(*, n: int = 0) -> dict[str, Any]:
    return {
        "n": int(n),
        "slope": float("nan"),
        "intercept": float("nan"),
        "R2": float("nan"),
        "r": float("nan"),
        "se_slope": float("nan"),
        "t_slope": float("nan"),
        "p_slope": float("nan"),
        "se_intercept": float("nan"),
        "t_intercept": float("nan"),
        "p_intercept": float("nan"),
    }

def _linregress_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return _empty_linear_fit(n=int(x.size))
    slope, intercept, r, pval, se = stats.linregress(x, y)
    return {
        "n": int(x.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "R2": float(r**2),
        "se_slope": float(se),
        "t_slope": float(slope / se) if se else float("nan"),
        "p_slope": float(pval),
        "se_intercept": float("nan"),
        "t_intercept": float("nan"),
        "p_intercept": float("nan"),
    }

def _main_sequence_stats(
    df: pd.DataFrame,
    *,
    amp_col: str,
    edges: np.ndarray,
    min_events: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """Amp-binned per-animal means + per-animal and pooled linear fits."""
    per_animal_stats: dict[str, pd.DataFrame] = {}
    linear_rows: list[dict[str, Any]] = []
    if df is None or df.empty:
        return per_animal_stats, pd.DataFrame(linear_rows), _empty_linear_fit(n=0)

    for animal, adf in df.groupby("animal"):
        rows = []
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            sub = adf[(adf[amp_col] >= lo) & (adf[amp_col] < hi)]
            if len(sub) < min_events:
                continue
            rows.append(
                {
                    "amp_lo": lo,
                    "amp_hi": hi,
                    "n": int(len(sub)),
                    "mean_peak_v": float(sub["peak_velocity"].mean()),
                }
            )
        per_animal_stats[str(animal)] = pd.DataFrame(rows)
        fit = _linregress_fit(adf[amp_col].to_numpy(float), adf["peak_velocity"].to_numpy(float))
        if fit["n"] >= 3:
            linear_rows.append({"animal": str(animal), **fit})

    global_fit = _linregress_fit(df[amp_col].to_numpy(float), df["peak_velocity"].to_numpy(float))
    return per_animal_stats, pd.DataFrame(linear_rows), global_fit

def _annotate_linear_fit(ax, fit: dict[str, Any] | None, *, fontsize: int = 7) -> None:
    """OLS slope and Pearson r for the dashed line (``scipy.stats.linregress``)."""
    if not fit:
        return
    slope = fit.get("slope")
    r = fit.get("r")
    if r is None or not np.isfinite(r):
        r2 = fit.get("R2")
        r = float(np.sqrt(r2)) if r2 is not None and np.isfinite(r2) else float("nan")
    if slope is None or not np.isfinite(slope) or not np.isfinite(r):
        return
    ax.text(
        0.04,
        0.97,
        f"slope = {float(slope):.3f}\nPearson r = {float(r):.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        color="0.1",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
    )

def _style_2e_axes(ax, *, xlim=None, ylim=None, labelsize: int = 8, ticksize: int = 7) -> None:
    ax.set_xlabel("Amplitude [deg]", fontsize=labelsize)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

def _plot_per_animal_means(
    per_animal_stats: dict[str, pd.DataFrame],
    global_fit: dict[str, Any],
    *,
    color_map: dict[str, Any],
    animal_order: list[str],
    params: dict[str, Any],
    xlim=None,
    ylim=None,
    x_fit_span=None,
    title: str | None = None,
):
    fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
    x_vals: list[float] = []
    for animal in animal_order:
        sdf = per_animal_stats.get(animal)
        if sdf is None or getattr(sdf, "empty", True):
            continue
        x = (sdf["amp_lo"] + sdf["amp_hi"]) / 2
        x_vals.extend(np.asarray(x, dtype=float).tolist())
        ax.plot(
            x,
            sdf["mean_peak_v"],
            "o-",
            color=color_map.get(animal, "0.3"),
            ms=3,
            lw=params["lw"],
            label=animal,
        )
    slope = global_fit.get("slope")
    intercept = global_fit.get("intercept")
    if (
        slope is not None
        and intercept is not None
        and np.isfinite(slope)
        and np.isfinite(intercept)
    ):
        if xlim is not None:
            x0, x1 = float(xlim[0]), float(xlim[1])
        elif x_fit_span is not None:
            x0, x1 = float(x_fit_span[0]), float(x_fit_span[1])
        elif x_vals:
            x0, x1 = float(min(x_vals)), float(max(x_vals))
        else:
            x0, x1 = 0.0, 1.0
        xs = np.linspace(x0, x1, 50)
        ax.plot(xs, float(slope) * xs + float(intercept), "k--", lw=params["lw"])
    if title:
        ax.set_title(title, fontsize=7)
    _style_2e_axes(ax, xlim=xlim, ylim=ylim)
    _annotate_linear_fit(ax, global_fit, fontsize=6)
    fig.tight_layout()
    return fig, ax

def _plot_2e_scatter(
    amp: np.ndarray,
    vel: np.ndarray,
    *,
    fit: dict[str, Any] | None,
    params: dict[str, Any],
    xlim,
    ylim,
    title: str,
    out_pdf: Path,
    show: bool,
    figsize=None,
    labelsize: int = 8,
    ticksize: int = 7,
    annot_size: int = 6,
    title_size: int = 7,
) -> None:
    fig, ax = plt.subplots(figsize=figsize or params["figsize"], dpi=params["dpi"])
    amp = np.asarray(amp, dtype=float)
    vel = np.asarray(vel, dtype=float)
    m = np.isfinite(amp) & np.isfinite(vel)
    amp, vel = amp[m], vel[m]
    if amp.size:
        ax.scatter(
            amp,
            vel,
            s=4,
            alpha=0.12,
            color="0.45",
            edgecolors="none",
            rasterized=True,
        )
    if fit is not None and np.isfinite(fit.get("slope", np.nan)) and amp.size >= 3:
        xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
        ax.plot(xs, float(fit["slope"]) * xs + float(fit["intercept"]), "k--", lw=params["lw"])
    ax.set_title(title, fontsize=title_size)
    _style_2e_axes(ax, xlim=xlim, ylim=ylim, labelsize=labelsize, ticksize=ticksize)
    _annotate_linear_fit(ax, fit, fontsize=annot_size)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)

def _ticks_for_span(lo: float, hi: float) -> list[float]:
    """Return exactly three ticks ``[lo, mid, hi]`` (paper-style; avoids dense labels)."""
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [0.0, 0.5]
    mid = round((lo + hi) / 2.0, 4)
    if abs(mid - lo) < 1e-12 or abs(mid - hi) < 1e-12:
        return [lo, hi]
    return [lo, mid, hi]

def compact_axis_ticks(
    ticks: object,
    lo: float,
    hi: float,
    *,
    max_ticks: int = 5,
) -> list[float]:
    """Keep an explicit tick list when it is short; otherwise ``[lo, mid, hi]``."""
    if isinstance(ticks, (list, tuple)) and 2 <= len(ticks) <= max_ticks:
        try:
            out = [float(t) for t in ticks]
        except (TypeError, ValueError):
            return _ticks_for_span(lo, hi)
        if all(np.isfinite(t) for t in out):
            return out
    return _ticks_for_span(lo, hi)

def nice_axis_max(value: float, *, step: float | None = None) -> float:
    """Round ``value`` up onto a short tick so the plotted percentile is not clipped."""
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        return 0.5
    if step is None:
        step = 0.05 if x < 2.0 else 0.1
    return float(np.ceil((x / step) - 1e-12) * step)

def histogram2d_xy(
    right: np.ndarray,
    left: np.ndarray,
    weights: np.ndarray | None,
    rng: tuple[float, float],
    bins: int,
) -> dict[str, np.ndarray]:
    """Normalized 2f histogram on a square ``rng`` grid."""
    n_edge = max(int(bins), 2)
    xbins = np.linspace(float(rng[0]), float(rng[1]), n_edge)
    ybins = np.linspace(float(rng[0]), float(rng[1]), n_edge)
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    if right.size == 0:
        zeros = np.zeros((n_edge - 1, n_edge - 1), dtype=float)
        return {
            "xedges": xbins.astype(float),
            "yedges": ybins.astype(float),
            "norm_counts": zeros,
        }
    if weights is None:
        w = np.ones(right.size, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    counts, xedges, yedges = np.histogram2d(right, left, bins=[xbins, ybins], weights=w)
    norm = counts / counts.sum() if counts.sum() > 0 else counts
    return {
        "xedges": xedges.astype(float),
        "yedges": yedges.astype(float),
        "norm_counts": norm.astype(float),
    }

def _turbo_white0():
    turbo = plt.get_cmap("turbo", 256)
    colors = turbo(np.linspace(0, 1, 256))
    colors[0] = np.array([1, 1, 1, 1])
    return mcolors.ListedColormap(colors)

def _draw_coupling_heatmap(ax, hist, rng, ticks, *, vmax: float, cmap) -> None:
    n_ticks = len(ticks) if isinstance(ticks, (list, tuple)) else 3
    ticks = compact_axis_ticks(
        ticks, float(rng[0]), float(rng[1]), max_ticks=max(5, n_ticks)
    )
    ax.pcolormesh(
        hist["xedges"],
        hist["yedges"],
        hist["norm_counts"].T,
        cmap=cmap,
        vmin=0,
        vmax=vmax if vmax > 0 else 1,
        shading="flat",
    )
    ax.set_xlim(*rng)
    ax.set_ylim(*rng)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.plot([rng[0], rng[1]], [rng[0], rng[1]], ls="--", color="gray", lw=1)
    ax.set_xlabel("Right max V [deg/ms]", fontsize=8)
    ax.set_ylabel("Left max V [deg/ms]", fontsize=8)
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if ax.collections:
        ax.collections[0].set_rasterized(True)

def percentile_speed_max(
    right: np.ndarray,
    left: np.ndarray,
    *,
    pct: float = 99.5,
) -> float:
    """``pct`` percentile of pooled finite L/R peak speeds."""
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    finite = np.concatenate(
        [right[np.isfinite(right)], left[np.isfinite(left)]]
    )
    if finite.size == 0:
        return 1.0
    return float(np.nanpercentile(finite, float(pct)))

def half_step_ticks(xmax: float) -> list[float]:
    """Even 0.5-step ticks from 0 up to the last 0.5-grid point at or below ``xmax``.

    If ``xmax`` itself sits on the 0.5 grid it is included. Otherwise the axis
    may run a little past the last tick (e.g. 0, 0.5, 1.0 on a 0–1.25 square).
    """
    hi = float(xmax)
    if not np.isfinite(hi) or hi <= 0:
        return [0.0, 0.5, 1.0]
    last = float(np.floor((hi + 1e-12) / 0.5) * 0.5)
    if last < 0.5:
        last = 0.5
    n = int(round(last / 0.5))
    ticks = [round(i * 0.5, 10) for i in range(n + 1)]
    if abs(hi - last) <= 1e-9 and ticks[-1] != hi:
        ticks.append(float(hi))
    return ticks

def publication_square_limits(raw_max: float, *, snap: bool = True) -> tuple[float, list[float]]:
    """Ceil ``raw_max`` onto a 0.5 grid (unless ``snap=False``) with even 0.5 ticks."""
    x = float(raw_max)
    if not np.isfinite(x) or x <= 0:
        return 1.0, [0.0, 0.5, 1.0]
    if snap:
        hi = float(np.ceil((x / 0.5) - 1e-12) * 0.5)
        if hi < 0.5:
            hi = 0.5
    else:
        hi = x
    return hi, half_step_ticks(hi)

def event_square_percentile(
    right: np.ndarray,
    left: np.ndarray,
    *,
    pct: float = 50.0,
) -> float:
    """Percentile of per-event ``max(right, left)``.

    The square ``[0, x] × [0, x]`` then contains about ``pct`` percent of events
    (so ``100 - pct`` percent are cropped outside the axes).
    """
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    m = np.fmax(right, left)
    m = m[np.isfinite(m)]
    if m.size == 0:
        return 0.2
    return float(np.nanpercentile(m, float(pct)))

def even_round_ticks(hi: float) -> list[float]:
    """Even ticks on a round ``hi`` using 0.5 / 0.2 / 0.1 / 0.05 steps."""
    hi = float(hi)
    if not np.isfinite(hi) or hi <= 0:
        return [0.0, 0.5, 1.0]
    for step in (0.5, 0.2, 0.1, 0.05):
        n = int(round(hi / step))
        if 2 <= n <= 3 and abs(n * step - hi) < 1e-9:
            return [round(i * step, 10) for i in range(n + 1)]
    return [0.0, round(hi, 10)]

def zoom_square_limits(raw_max: float) -> tuple[float, list[float]]:
    """Ceil a small xmax onto 0.1/0.2/… with even round ticks (zoom panel)."""
    x = float(raw_max)
    if not np.isfinite(x) or x <= 0:
        return 0.2, [0.0, 0.1, 0.2]
    for cand in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0):
        if x <= cand + 1e-12:
            return float(cand), even_round_ticks(cand)
    hi = float(np.ceil((x / 0.1) - 1e-12) * 0.1)
    return hi, even_round_ticks(hi)

def _s8j_hist_panel(
    right,
    left,
    weights,
    rng: tuple[float, float],
    ticks: list[float],
    bins: int,
) -> dict:
    hist = histogram2d_xy(right, left, weights, rng, bins)
    vmax = float(np.nanmax(hist["norm_counts"]) if hist["norm_counts"].size else 0.0) or 1e-12
    r = np.asarray(right, dtype=float)
    l = np.asarray(left, dtype=float)
    n_in = int(np.sum((r <= rng[1]) & (l <= rng[1]) & np.isfinite(r) & np.isfinite(l)))
    return {
        "range": rng,
        "ticks": list(ticks),
        "vmax": vmax,
        "hist": hist,
        "n_in": n_in,
    }

def build_s8j_views(
    liz_right,
    liz_left,
    liz_weights,
    mou_right,
    mou_left,
    mou_weights,
    *,
    bins: int = 60,
    full_pct: float = 99.5,
    zoom_keep_pct: float = 50.0,
    xmax_mouse_full: float | None = None,
) -> dict:
    """Full (matched high percentile) and inner-50% zoom views. Same bin count."""
    bins = int(max(int(bins), 60))
    if xmax_mouse_full is None or not np.isfinite(float(xmax_mouse_full)) or float(xmax_mouse_full) <= 0:
        xmax_m = nice_axis_max(percentile_speed_max(mou_right, mou_left, pct=full_pct))
    else:
        xmax_m = float(xmax_mouse_full)
    ticks_m = half_step_ticks(xmax_m)
    raw_l = percentile_speed_max(liz_right, liz_left, pct=full_pct)
    xmax_l, ticks_l = publication_square_limits(raw_l, snap=True)
    full_l = _s8j_hist_panel(liz_right, liz_left, liz_weights, (0.0, xmax_l), ticks_l, bins)
    full_m = _s8j_hist_panel(mou_right, mou_left, mou_weights, (0.0, xmax_m), ticks_m, bins)

    z_raw_l = event_square_percentile(liz_right, liz_left, pct=zoom_keep_pct)
    z_raw_m = event_square_percentile(mou_right, mou_left, pct=zoom_keep_pct)
    zmax_l, zticks_l = zoom_square_limits(z_raw_l)
    zmax_m, zticks_m = zoom_square_limits(z_raw_m)
    zoom_l = _s8j_hist_panel(liz_right, liz_left, liz_weights, (0.0, zmax_l), zticks_l, bins)
    zoom_m = _s8j_hist_panel(mou_right, mou_left, mou_weights, (0.0, zmax_m), zticks_m, bins)
    return {
        "full_lizard": full_l,
        "full_mouse": full_m,
        "zoom_lizard": zoom_l,
        "zoom_mouse": zoom_m,
        "vmax_joint_full": float(max(full_l["vmax"], full_m["vmax"], 1e-12)),
        "vmax_joint_zoom": float(max(zoom_l["vmax"], zoom_m["vmax"], 1e-12)),
        "bins": bins,
        "full_pct": float(full_pct),
        "zoom_keep_pct": float(zoom_keep_pct),
    }

def calculate_orientation_tuning(saccade_angles) -> float:
    saccade_angles = np.asarray(saccade_angles) % 360
    if saccade_angles.size == 0:
        return float("nan")
    horizontal_ranges = [(315, 360), (0, 45), (135, 225)]
    is_horizontal = np.logical_or.reduce(
        [
            (saccade_angles >= low) & (saccade_angles <= high)
            if low < high
            else (saccade_angles >= low) | (saccade_angles <= high)
            for low, high in horizontal_ranges
        ]
    )
    p_horizontal = float(np.sum(is_horizontal) / len(saccade_angles))
    p_vertical = 1.0 - p_horizontal
    return float((p_horizontal - p_vertical) / (p_horizontal + p_vertical))

def _resolve_font_family(preferred: str) -> str:
    """Use ``preferred`` (e.g. Arial) when installed, else fall back gracefully."""
    try:
        import matplotlib.font_manager as fm

        available = {f.name.lower() for f in fm.fontManager.ttflist}
        if preferred.lower() in available:
            return preferred
    except Exception:
        pass
    return "DejaVu Sans"

def _snap_edges(raw: np.ndarray, *, frame_ms: float, high_ms: float, add_leading_zero_bin: bool) -> np.ndarray:
    k_edges = np.floor(raw / frame_ms) + 0.5
    kmax = int(np.floor(high_ms / frame_ms))
    k_edges = np.clip(k_edges, 0.5, kmax + 0.5)
    k_edges = np.unique(k_edges)

    if add_leading_zero_bin and (k_edges.size == 0 or not np.isclose(k_edges[0], 0.5)):
        k_edges = np.r_[0.25, k_edges]
    if k_edges[-1] < (kmax + 0.5):
        k_edges = np.r_[k_edges, (kmax + 0.5)]
    return k_edges * frame_ms

def _snapped_log_bins(
    num_bins: int,
    frame_ms: float,
    low_ms: float,
    high_ms: float,
    add_leading_zero_bin: bool = True,
    log_exponent: float = 1.0,
) -> np.ndarray:
    """Geometric bin edges between ``low_ms``/``high_ms``, snapped to ``(k + 0.5) * frame_ms``."""
    num_bins = int(num_bins)
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if not (np.isfinite(low_ms) and np.isfinite(high_ms)) or not (0 < low_ms < high_ms):
        raise ValueError("low_ms and high_ms must be finite and satisfy 0 < low_ms < high_ms")
    if not np.isfinite(frame_ms) or frame_ms <= 0:
        raise ValueError("frame_ms must be a positive finite number")
    if not np.isfinite(log_exponent) or log_exponent <= 0:
        raise ValueError("log_exponent must be a positive finite number")

    t = np.linspace(0.0, 1.0, num_bins + 1)
    ratio = float(high_ms) / float(low_ms)
    raw = float(low_ms) * (ratio ** (t ** float(log_exponent)))
    return _snap_edges(raw, frame_ms=frame_ms, high_ms=high_ms, add_leading_zero_bin=add_leading_zero_bin)

def _snapped_linear_bins(
    num_bins: int,
    frame_ms: float,
    low_ms: float,
    high_ms: float,
    add_leading_zero_bin: bool = False,
) -> np.ndarray:
    """Linear bin edges between ``low_ms``/``high_ms``, snapped to ``(k + 0.5) * frame_ms``."""
    num_bins = int(num_bins)
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if not (np.isfinite(low_ms) and np.isfinite(high_ms)) or not (0 < low_ms < high_ms):
        raise ValueError("low_ms and high_ms must be finite and satisfy 0 < low_ms < high_ms")
    if not np.isfinite(frame_ms) or frame_ms <= 0:
        raise ValueError("frame_ms must be a positive finite number")

    raw = np.linspace(low_ms, high_ms, num_bins + 1)
    return _snap_edges(raw, frame_ms=frame_ms, high_ms=high_ms, add_leading_zero_bin=add_leading_zero_bin)

def _fold_to_xlim_log(
    counts: np.ndarray, edges: np.ndarray, xlim: tuple[float, float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Fold mass outside ``xlim`` into the first/last displayed bin; geometric centers."""
    if xlim is None:
        centers = np.sqrt(edges[:-1] * edges[1:])
        return counts, centers

    low, high = float(xlim[0]), float(xlim[1])
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        centers = np.sqrt(edges[:-1] * edges[1:])
        return counts, centers

    idx_lo = np.searchsorted(edges, low, side="right") - 1
    idx_hi = np.searchsorted(edges, high, side="left") - 1
    idx_lo = max(0, min(idx_lo, len(counts) - 1))
    idx_hi = max(0, min(idx_hi, len(counts) - 1))

    folded = counts.copy()
    if idx_lo > 0:
        folded[idx_lo] += folded[:idx_lo].sum()
        folded[:idx_lo] = 0.0
    if idx_hi < len(folded) - 1:
        folded[idx_hi] += folded[idx_hi + 1 :].sum()
        folded[idx_hi + 1 :] = 0.0

    disp_counts = folded[idx_lo : idx_hi + 1]
    disp_edges = edges[idx_lo : idx_hi + 2]
    disp_centers = np.sqrt(disp_edges[:-1] * disp_edges[1:])
    return disp_counts, disp_centers

def _slice_to_xlim_linear(
    counts: np.ndarray,
    edges: np.ndarray,
    xlim: tuple[float, float] | None,
    *,
    renormalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice bins to ``xlim`` (no folding); arithmetic centers; optional renormalization."""
    if xlim is None:
        centers = 0.5 * (edges[:-1] + edges[1:])
        return counts, centers

    low, high = float(xlim[0]), float(xlim[1])
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        centers = 0.5 * (edges[:-1] + edges[1:])
        return counts, centers

    idx_lo = np.searchsorted(edges, low, side="right") - 1
    idx_hi = np.searchsorted(edges, high, side="left") - 1
    idx_lo = max(0, min(idx_lo, len(counts) - 1))
    idx_hi = max(0, min(idx_hi, len(counts) - 1))

    disp_counts = counts[idx_lo : idx_hi + 1].copy()
    disp_edges = edges[idx_lo : idx_hi + 2]
    disp_centers = 0.5 * (disp_edges[:-1] + disp_edges[1:])

    if renormalize:
        total = disp_counts.sum()
        if total > 0:
            disp_counts = disp_counts / total
    return disp_counts, disp_centers

def _plot_isi_traces(
    *,
    bins: np.ndarray,
    isi_all: dict[str, np.ndarray],
    animals: list[str],
    color_map: dict[str, str],
    fold_fn,
    figure_size: tuple[float, float],
    xscale: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    xlabel: str,
    ylabel: str,
    linewidth: float,
    combined_linewidth: float,
    combined_color: str,
    combined_linestyle: str,
    combined_label: str,
    font_family: str,
) -> tuple[plt.Figure, list, list[str], dict[str, Any]]:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family]

    fig, ax = plt.subplots(figsize=figure_size, dpi=300)
    legend_handles: list = []
    legend_labels: list[str] = []

    total_counts = np.zeros(len(bins) - 1, dtype=float)
    per_animal: dict[str, dict[str, Any]] = {}
    valid_any = False

    for animal in animals:
        isi = isi_all.get(animal, np.array([]))
        if isi.size == 0:
            continue
        hist, _ = np.histogram(isi, bins=bins)
        if hist.sum() == 0:
            continue
        valid_any = True
        total_counts += hist
        dens = hist.astype(float) / hist.sum()
        y_plot, x_plot = fold_fn(dens, bins)

        per_animal[animal] = {"hist": hist, "density": dens, "x": x_plot, "y": y_plot}
        (h,) = ax.plot(
            x_plot, y_plot, linewidth=linewidth, color=color_map.get(animal), label=str(animal)
        )
        legend_handles.append(h)
        legend_labels.append(str(animal))

    combined: dict[str, Any] = {}
    if valid_any and total_counts.sum() > 0:
        total_dens = total_counts / total_counts.sum()
        y_all, x_all = fold_fn(total_dens, bins)
        combined = {
            "total_counts": total_counts,
            "total_density": total_dens,
            "x": x_all,
            "y": y_all,
            "label": combined_label,
        }
        (h_all,) = ax.plot(
            x_all,
            y_all,
            combined_linestyle,
            color=combined_color,
            linewidth=combined_linewidth,
            label=combined_label,
            zorder=10,
        )
        legend_handles.insert(0, h_all)
        legend_labels.insert(0, combined_label)

    ax.set_xscale(xscale)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        ax.set_xlim(float(bins[0]), float(bins[-1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=8, length=5, width=1, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()

    return fig, legend_handles, legend_labels, {"per_animal": per_animal, "combined": combined}

def epoch_duration_zoom_xmax_s(
    active_vals: np.ndarray,
    *,
    round_to: float = 5.0,
    pad_s: float = 0.0,
) -> float:
    """Linear zoom upper limit from the longest active epoch (rounded up)."""
    vals = np.asarray(active_vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return float(round_to)
    hi = float(np.nanmax(vals)) + float(pad_s)
    step = max(float(round_to), 1e-9)
    return float(np.ceil(hi / step) * step)

def epoch_duration_linear_edges(
    *,
    xmax_s: float,
    xmin_s: float = 0.0,
    n_bins: int | None = None,
    bin_width_s: float | None = None,
) -> np.ndarray:
    """Linear histogram edges; prefer integer-aligned ``bin_width_s`` when set."""
    lo = float(xmin_s)
    hi = float(xmax_s)
    if hi <= lo:
        hi = lo + 1.0
    if bin_width_s is not None:
        width = float(bin_width_s)
        if width <= 0:
            raise ValueError("bin_width_s must be > 0")
        n = max(1, int(np.ceil((hi - lo) / width)))
        return lo + width * np.arange(n + 1, dtype=float)
    n_bins = 15 if n_bins is None else int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    return np.linspace(lo, hi, n_bins + 1)

def _draw_epoch_duration_hist(
    ax,
    vals: np.ndarray,
    *,
    color: str,
    title: str,
    scale: str,
    bins: np.ndarray,
    xmin: float = 0.0,
    xmax: float | None = None,
    continuous: bool = False,
) -> None:
    vals = np.asarray(vals, dtype=float)
    if vals.size:
        hist_kw: dict[str, Any] = {"bins": bins, "color": color, "alpha": 1.0}
        if continuous:
            hist_kw.update(histtype="stepfilled", edgecolor="none", linewidth=0)
        else:
            hist_kw["edgecolor"] = "black"
            hist_kw["linewidth"] = 0.4
        ax.hist(vals, **hist_kw)
        if scale == "log":
            ax.set_xscale("log")
        else:
            right = float(xmax) if xmax is not None else (float(bins[-1]) if bins.size else None)
            ax.set_xlim(float(xmin), right)
    ax.set_xlabel("Epoch duration [s]", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _overlay_epoch_duration_kde(
    ax,
    vals: np.ndarray,
    *,
    bins: np.ndarray,
    color: str,
    xmin: float,
    xmax: float,
) -> None:
    """Draw a count-scaled Gaussian KDE over an existing histogram."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2 or bins.size < 2:
        return
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return
    width = float(np.mean(np.diff(bins)))
    if width <= 0:
        return
    xs = np.linspace(float(xmin), float(xmax), 256)
    dens = gaussian_kde(vals)(xs)
    ax.plot(xs, dens * vals.size * width, color="0.15", linewidth=1.2, zorder=3)

def plot_epoch_duration_triptych(
    quiet: np.ndarray,
    active: np.ndarray,
    *,
    zoom_xmax_s: float,
    quiet_full_bins: np.ndarray,
    zoom_bins: np.ndarray,
    title: str | None = None,
    kde: bool = False,
) -> Any:
    """Quiet full-range | quiet zoomed | active zoomed (shared zoom x-limit)."""
    quiet = np.asarray(quiet, dtype=float)
    active = np.asarray(active, dtype=float)
    quiet_full_hi = float(np.nanmax(quiet)) if quiet.size else float(quiet_full_bins[-1])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9), dpi=300)
    panels = (
        (
            axes[0],
            quiet,
            _EPOCH_DURATION_COLORS["quiet"],
            f"quiet full n={quiet.size}",
            quiet_full_bins,
            0.0,
            quiet_full_hi,
        ),
        (
            axes[1],
            quiet,
            _EPOCH_DURATION_COLORS["quiet"],
            f"quiet 0–{zoom_xmax_s:g}s n={(quiet <= zoom_xmax_s).sum()}",
            zoom_bins,
            0.0,
            float(zoom_xmax_s),
        ),
        (
            axes[2],
            active,
            _EPOCH_DURATION_COLORS["active"],
            f"active 0–{zoom_xmax_s:g}s n={active.size}",
            zoom_bins,
            0.0,
            float(zoom_xmax_s),
        ),
    )
    for ax, vals, color, panel_title, bins, xmin, xmax in panels:
        _draw_epoch_duration_hist(
            ax,
            vals,
            color=color,
            title=panel_title,
            scale="linear",
            bins=bins,
            xmin=xmin,
            xmax=xmax,
            continuous=False,
        )
        if kde:
            _overlay_epoch_duration_kde(
                ax, vals, bins=bins, color=color, xmin=xmin, xmax=xmax
            )
    if title:
        fig.suptitle(title, fontsize=9, y=1.05)
    fig.tight_layout()
    return fig

def per_animal_spans(per_eye: pd.DataFrame) -> pd.DataFrame:
    """Average the two eyes, which is the only averaging the published run did."""
    if per_eye.empty:
        return pd.DataFrame(columns=["animal", "main_span_mean", "perp_span_mean"])
    wide = per_eye.pivot_table(index="animal", columns="eye", values=["main_span_deg", "perp_span_deg"])
    out = pd.DataFrame(
        {
            "main_span_mean": per_eye.groupby("animal")["main_span_deg"].mean(),
            "perp_span_mean": per_eye.groupby("animal")["perp_span_deg"].mean(),
        }
    )
    for eye in ("left", "right"):
        for kind in ("main", "perp"):
            col = (f"{kind}_span_deg", eye)
            out[f"{kind}_span_{eye}"] = wide[col] if col in wide.columns else np.nan
    return out.reset_index()

def coupling_fractions(
    rolling: pd.DataFrame,
    *,
    corr_thr_pos: float | None = None,
    corr_thr_neg: float | None = None,
) -> pd.DataFrame:
    """Per-animal time fractions and r summary, straight from the deposited rows."""
    thr_pos = float(S4_CORR_THR_POS if corr_thr_pos is None else corr_thr_pos)
    thr_neg = float(S4_CORR_THR_NEG if corr_thr_neg is None else corr_thr_neg)
    if rolling.empty:
        return pd.DataFrame(
            columns=["animal", "n_blocks", "n_valid_points", "frac_corr", "frac_weak", "frac_anti", "median_r", "iqr_r"]
        )

    rows = []
    for animal, g in rolling.groupby("animal", sort=True):
        v = pd.to_numeric(g["rolling_r"], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q25, q50, q75 = np.percentile(v, [25, 50, 75])
        rows.append(
            {
                "animal": animal,
                "n_blocks": int(g["block"].nunique()),
                "n_valid_points": int(v.size),
                "frac_corr": float(np.sum(v > thr_pos)) / v.size,
                "frac_weak": float(np.sum((v >= thr_neg) & (v <= thr_pos))) / v.size,
                "frac_anti": float(np.sum(v < thr_neg)) / v.size,
                "median_r": float(q50),
                "iqr_r": float(q75 - q25),
            }
        )
    return pd.DataFrame(rows)

def unit_label(units: str) -> str:
    """Axis label for a units code (``um`` → ``µm``)."""
    return "µm" if str(units).lower() == "um" else str(units)

def _jitter_stat_annotation(med: float, p95: float, units: str) -> str:
    """Compact median / P95 label; one decimal in pixels, integers in µm."""
    fmt = ".1f" if str(units).lower() == "px" else ".0f"
    return f"median {med:{fmt}}\nP95 {p95:{fmt}}"

def _hist_percent_frames(
    distances: np.ndarray,
    *,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return bins[:-1], np.zeros(len(bins) - 1)
    hist, edges = np.histogram(distances, bins=bins)
    pct = (hist / distances.size) * 100.0
    return edges[:-1], pct

def shared_jitter_xmax(
    pools: dict[str, np.ndarray],
    *,
    percentile: float = 99.5,
) -> tuple[float, str | None]:
    """Xmax that fits the 99.5th-percentile bin of the most jittery mount.

    Returns ``(xmax, mount_that_set_it)``. Empty pools yield ``(1e-6, None)``.
    """
    best = 1e-6
    source: str | None = None
    for mount, arr in pools.items():
        values = np.asarray(arr, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        xmax = float(np.nanpercentile(values, percentile))
        if xmax >= best:
            best = xmax
            source = str(mount)
    return max(best, 1e-6), source

def unified_jitter_bin_edges(
    xmax: float,
    *,
    bin_width: float | None = None,
    n_bins: int | None = None,
    max_bins: int = UNIFIED_JITTER_MAX_BINS,
) -> np.ndarray:
    """Edges covering ``[0, xmax]``. ``bin_width`` wins over ``n_bins`` when set."""
    xmax = max(float(xmax), 1e-6)
    if bin_width is not None and np.isfinite(bin_width) and float(bin_width) > 0:
        width = max(float(bin_width), xmax / int(max_bins))
        edges = np.arange(0.0, xmax, width)
        if edges.size == 0 or edges[-1] < xmax:
            edges = np.append(edges, xmax)
        return edges
    n = int(n_bins if n_bins is not None else 15)
    return np.linspace(0.0, xmax, n + 1)

def figure_unified_jitter(
    pools: dict[str, np.ndarray],
    *,
    xmax: float | None = None,
    n_bins: int | None = None,
    bin_widths: dict[str, float] | None = None,
    units: str = "um",
):
    """2×2 histograms with a shared x-limit and per-panel bin widths.

    ``xmax`` is the 99.5th percentile of the most jittery mount. When
    ``bin_widths`` maps mount type → bar width (typically one coarsest pixel in
    that panel), each histogram is binned independently. Otherwise all panels
    share ``n_bins`` equal-width bins (legacy).
    """
    if xmax is None:
        xmax, _ = shared_jitter_xmax(pools)
    xmax = max(float(xmax), 1e-6)
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.8), dpi=150, sharex=True)
    for ax, (mount, label, color) in zip(axes.ravel(), UNIFIED_JITTER_PANELS):
        values = np.asarray(pools.get(mount, np.array([])), dtype=float)
        n = int(np.isfinite(values).sum()) if values.size else 0
        width = None if not bin_widths else bin_widths.get(mount)
        bins = unified_jitter_bin_edges(xmax, bin_width=width, n_bins=n_bins)
        x, y = _hist_percent_frames(values, bins=bins)
        ax.bar(
            x,
            y,
            width=np.diff(bins),
            align="edge",
            color=color,
            edgecolor="black",
            alpha=0.7,
        )
        finite = values[np.isfinite(values)]
        if finite.size:
            med = float(np.median(finite))
            p95 = float(np.percentile(finite, 95))
            ax.axvline(med, color="0.15", ls="-", lw=0.9, zorder=3)
            ax.axvline(p95, color="0.15", ls=":", lw=0.9, zorder=3)
            ax.text(
                0.98,
                0.96,
                _jitter_stat_annotation(med, p95, units),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                color="0.15",
            )
        ax.set_xlim(0, xmax)
        ax.set_title(f"{label} (n={n})", fontsize=8)
        ax.set_ylabel("% frames", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ulabel = unit_label(units)
    xlabel = (
        f"Displacement [{ulabel}] (eye plane)"
        if str(units).lower() == "um"
        else f"Displacement [{ulabel}]"
    )
    for ax in axes[1]:
        ax.set_xlabel(xlabel, fontsize=8)
    fig.tight_layout()
    return fig

def animal_level_from_blocks(per_block: pd.DataFrame) -> pd.DataFrame:
    """Animal-level mean |Δ| (unweighted mean of that animal's blocks)."""
    rows: list[dict[str, Any]] = []
    for (animal, eye, axis_name), g in per_block.groupby(["animal", "eye", "axis"], sort=False):
        mean_abs = pd.to_numeric(g["mean_abs_deg"], errors="coerce")
        rows.append(
            {
                "animal": animal,
                "eye": eye,
                "axis": axis_name,
                "n_blocks": int(g["block_key"].nunique()) if "block_key" in g.columns else int(len(g)),
                "mean_abs_deg": float(mean_abs.mean()),
                "std_of_block_mean_abs_deg": float(mean_abs.std(ddof=1)) if len(g) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)

def figure_s1c_overall(across: pd.DataFrame) -> plt.Figure:
    """Two-bar overall φ/θ mean |Δ| (animal as unit), manuscript 0.84 / 1.19 values."""
    both = across[(across["eye"] == "both") & (across["axis"].isin(["phi", "theta"]))].copy()
    both["axis"] = pd.Categorical(both["axis"], ["phi", "theta"], ordered=True)
    both = both.sort_values("axis")
    labels = ["φ", "θ"]
    y = both["mean_of_mean_abs_deg"].to_numpy(dtype=float)
    yerr = both["std_of_mean_abs_deg"].to_numpy(dtype=float)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(2.6, 2.8), dpi=150)
    x = np.arange(len(labels))
    ax.bar(
        x,
        y,
        yerr=np.where(np.isfinite(yerr), yerr, 0.0),
        color=["#0072B2", "#D55E00"],
        alpha=0.85,
        capsize=3,
        error_kw={"lw": 0.9},
        width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("mean |Δ| (deg)\nerror = SD across animals", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    n = int(both["n_animals"].iloc[0]) if "n_animals" in both.columns and len(both) else 0
    ax.set_title(f"Eye-pooled leftover error  (n = {n} animals)", fontsize=9)
    fig.tight_layout()
    return fig

def _finite_d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a) & (a >= 0)]

def rayleigh_pdf(x: np.ndarray, B: float) -> np.ndarray:
    """Rayleigh PDF, loc=0, scale B."""
    b = max(float(B), 1e-15)
    x = np.asarray(x, dtype=float)
    return (x / (b * b)) * np.exp(-0.5 * (x / b) ** 2)

def rayleigh_scale_from_median(arr: np.ndarray) -> float:
    """High-breakdown scale: B = median(D) / sqrt(2 ln 2). No cutoff."""
    a = _finite_d(arr)
    if a.size < 8:
        return float("nan")
    med = float(np.median(a))
    if med <= 0:
        return float("nan")
    return med / RAYLEIGH_MEDIAN_FACTOR

def resolve_noise_core_xmax(
    arr: np.ndarray,
    B: float,
    threshold: float | None,
) -> float:
    """X-limit showing the noise body/tail and the detection threshold.

    Takes the farther of the 99.9th percentile of ``D`` and ``6 B`` (Rayleigh
    tail), then extends to ``1.08 × threshold`` when a detector floor is set.
    """
    pos = np.asarray(arr, dtype=float)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    thr = (
        float(threshold)
        if threshold is not None and np.isfinite(threshold) and float(threshold) > 0
        else None
    )
    if pos.size == 0:
        return 1.08 * thr if thr is not None else 1.0
    p999 = float(np.percentile(pos, 99.9))
    fit_hi = 6.0 * float(B) if np.isfinite(B) and B > 0 else p999
    xmax = max(p999, fit_hi)
    if thr is not None:
        xmax = max(xmax, thr * 1.08)
    return float(xmax)

def resolve_noise_core_n_bins(
    arr: np.ndarray,
    B: float,
    xmax: float,
    n_bins: int = 40,
) -> int:
    """Keep histogram resolution on the noise body when xmax extends to a far threshold."""
    pos = np.asarray(arr, dtype=float)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    base = max(int(n_bins), 8)
    if pos.size == 0 or not np.isfinite(xmax) or xmax <= 0:
        return base
    p999 = float(np.percentile(pos, 99.9))
    fit_hi = 6.0 * float(B) if np.isfinite(B) and B > 0 else p999
    core = max(p999, fit_hi, 1e-12)
    return max(base, int(round(base * float(xmax) / core)))

def figure_rayleigh_noise_core_window(
    d: np.ndarray,
    *,
    B: float | None = None,
    threshold: float | None = None,
    title: str = "",
    n_bins: int = 40,
    color: str = "0.70",
    hist_label: str = "quiet segments",
    noise_marker: str = "2B",
) -> plt.Figure:
    """Quiet-segment hist + median-matched Rayleigh.

    ``noise_marker``:
      - ``\"2B\"`` — 2 axis-σ radius (legacy supplement style)
      - ``\"median\"`` — sample median of ``D`` (media / Methods style)
    """
    apply_paper_style()
    marker = str(noise_marker).strip().lower()
    if marker not in {"2b", "median"}:
        raise ValueError(f"noise_marker must be '2B' or 'median', got {noise_marker!r}")

    arr = np.asarray(d, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if B is None or not np.isfinite(B):
        B = rayleigh_scale_from_median(arr)
    B = float(B) if np.isfinite(B) else float("nan")
    median_d = float(np.median(arr)) if arr.size else float("nan")
    xmax = resolve_noise_core_xmax(arr, B, threshold)
    n_bins_use = resolve_noise_core_n_bins(arr, B, xmax, n_bins)

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    if arr.size:
        core = arr[arr > 0]
        plot_arr = core if core.size >= 8 else arr
        # Same opaque gray + black edge for every species (Illustrator-clean).
        _ = color
        ax.hist(
            plot_arr,
            bins=int(n_bins_use),
            range=(0.0, xmax),
            density=True,
            color="0.70",
            edgecolor="black",
            linewidth=0.4,
            alpha=1.0,
            histtype="bar",
            label=hist_label,
        )
        xs = np.linspace(0.0, xmax, 400)
        if np.isfinite(B) and B > 0:
            ax.plot(xs, rayleigh_pdf(xs, B), color="0.1", lw=1.8, label=f"Rayleigh B={B:.3g}")
        if marker == "2b" and np.isfinite(B) and B > 0:
            floor_2b = 2.0 * B
            ax.axvline(
                floor_2b,
                color="#E69F00",
                ls="--",
                lw=1.2,
                label=f"2B (2σ)={floor_2b:.3g}",
            )
        elif marker == "median" and np.isfinite(median_d) and median_d > 0:
            ax.axvline(
                median_d,
                color="#E69F00",
                ls="--",
                lw=1.2,
                label=f"median={median_d:.3g}",
            )
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(
            float(threshold),
            color="#D55E00",
            ls="--",
            lw=1.2,
            label=f"threshold={float(threshold):g}",
        )
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("D = hypot(Δφ, Δθ)  [deg/frame]")
    ax.set_ylabel("Probability density")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

def normalize_label(annotation) -> str:
    """Map a raw annotation string to the canonical ``'quiet'`` / ``'active'``.

    ``quiet``/``stationary`` (and close synonyms: quiescent, quite, rest,
    still) → ``'quiet'``; ``active``/``explores`` → ``'active'``; anything
    else also defaults to ``'active'`` (unrecognized labels are treated as
    non-quiescent).
    """
    label = str(annotation).strip().lower()
    return "quiet" if label in _QUIET_LABELS else "active"

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
            ax.set_xlabel("[s]")

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

def create_figure(distances: np.ndarray, output_path: Path = None):
    """
    Create Figure 1e: Histogram of camera movements (jitter quantification).
    
    Parameters
    ----------
    distances : np.ndarray
        Array of displacement distances in micrometers
    output_path : Path, optional
        Where to save the figure. If None, displays instead.
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    # Create figure with specific size
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.6), dpi=150)
    
    # Plot the histogram
    # Bins: 15 bins from 0 to 500 μm
    bins = np.linspace(0, 500, 15)
    hist, bins = np.histogram(distances, bins=bins)
    percentage = (hist / len(distances)) * 100
    
    # Use 'gray' for bin fill and 'black' for edges
    ax.bar(bins[:-1], percentage, width=np.diff(bins), 
           color='gray', edgecolor='black', align='edge')
    
    # Set labels
    ax.set_xlabel('Displacement [$\mu$m]', fontsize=10)
    ax.set_ylabel('% frames', fontsize=10)
    
    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='y', which='both', length=3, color='black')
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.tick_params(axis='y', which='major', length=3, width=1, color='black')
    
    # Set white background and black text
    ax.set_facecolor('white')
    ax.title.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='black')
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='y', which='both', length=3.5, color='black')
    ax.tick_params(axis='x', which='both', length=3.5)
    
    # Set x-axis and y-axis limits
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 40)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
    
    return fig, ax

def find_roundest_ellipse_in_df(df: pd.DataFrame) -> int:
    s = df.ratio2
    closest_ind = np.nanargmin(np.abs(s - 1))  # find the index of the value closest to 1
    return int(closest_ind)

def kerr(df: pd.DataFrame, aEC=np.nan, bEC=np.nan):
    if aEC != aEC:
        idx = find_roundest_ellipse_in_df(df)
        dx = df.loc[df.index[idx], "center_x"]
        dy = df.loc[df.index[idx], "center_y"]
        if not isinstance(dx, numbers.Number):
            dx = dx.iloc[0]
        if not isinstance(dy, numbers.Number):
            dy = dy.iloc[0]
        aEC = int(dx)
        bEC = int(dy)

    theta_values = np.full(len(df), np.nan)  # Initialize theta column with NaNs
    phi_values = np.full(len(df), np.nan)  # Initialize phi column with NaNs
    r_values = np.full(len(df), np.nan)  # Initialize r column with NaNs

    # Convert columns to NumPy arrays for faster access
    hw_values = df["ratio2"].values
    aPC_values = df["center_x"].values
    bPC_values = df["center_y"].values

    # Mask for valid `hw` values (to ignore NaNs)
    valid_mask = ~np.isnan(hw_values)

    # Vectorized computation for `top` and `bot`
    sqrt_component = np.sqrt(1 - hw_values[valid_mask] ** 2)
    distances = np.sqrt((aPC_values[valid_mask] - aEC) ** 2 + (bPC_values[valid_mask] - bEC) ** 2)

    top_values = sqrt_component * distances
    bot_values = 1 - hw_values[valid_mask] ** 2

    top = np.sum(top_values)
    bot = np.sum(bot_values)

    f_z = top / bot

    # Compute `r` for all rows where `major_ax` is valid
    valid_major_ax = ~np.isnan(df["major_ax"].values)
    max_axes = np.maximum(df["major_ax"].values, df["minor_ax"].values)
    r_values[valid_major_ax] = (2 * max_axes[valid_major_ax]) / f_z

    # Compute `theta` and `phi` in a vectorized way
    valid_positions = ~np.isnan(aPC_values) & ~np.isnan(bPC_values)

    with np.errstate(invalid="ignore"):
        comp_p = np.arcsin((aPC_values[valid_positions] - aEC) / f_z)
        comp_t = np.arcsin((bPC_values[valid_positions] - bEC) / (np.cos(comp_p) * f_z))

    theta_values[valid_positions] = np.degrees(comp_t)
    phi_values[valid_positions] = np.degrees(comp_p)

    # Create output DataFrame
    output_df = pd.DataFrame({"r": r_values, "theta": theta_values, "phi": phi_values}, index=df.index)
    output_df = pd.concat([df[["frame", "x_angle", "y_angle", "ratio2", "ratio1"]], output_df], axis=1)
    return f_z, output_df

def plot_error_vs_ecc_with_ratio_axis_pdf(
    df,
    k=40,  # span (deg) -> vertical line at R=k/2 on bottom axis
    metric="total",  # "total" | "phi" | "theta"
    r_bins=30,  # number of eccentricity bins (x points are bin centers)
    r_range=None,  # (r_min, r_max) in deg for bottom axis, e.g. (0, 60)
    min_per_bin=20,  # drop curve bins with < this many samples
    ring_halfwidth_deg=0.5,  # for reporting ratio at the span ring (optional)
    figsize=(3.8, 2.6),
    dpi=300,
    export_pdf_path=None,
    font_family="Arial",
    y_range=None,
):
    """
    Mean angular error vs eccentricity (bottom x-axis in degrees), with a top x-axis
    showing the corresponding ratio (minor/major) learned from the dataset.

    • Curve is computed by binning on eccentricity r = hypot(x_angle, y_angle).
      Each plotted point is the mean error in one r-bin, placed at that bin's CENTER.
    • Bottom x-axis extent is controlled via r_range=(r_min, r_max) in degrees.
    • The dashed vertical line is drawn at R = k/2 on the degree axis.
    • The top x-axis labels (ratio) are computed from the dataset by taking the
      median ratio in each r-bin and interpolating to the bottom tick positions.
    • PDF export uses TrueType fonts (editable text); Arial if available.
    """
    plt.style.use("default")
    rcParams["pdf.fonttype"] = 42  # Ensure fonts are embedded and editable
    rcParams["ps.fonttype"] = 42  # Ensure compatibility with vector outputs
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [font_family, "Helvetica", "DejaVu Sans"]

    # ---------- Data prep ----------
    req = {"ratio2", "x_angle", "y_angle", "phi", "theta"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    d = df[list(req)].copy().dropna(subset=list(req))
    d["ratio2"] = pd.to_numeric(d["ratio2"], errors="coerce").clip(0.0, 1.0)
    d = d.dropna(subset=["ratio2"])
    d["r_xy"] = np.hypot(d["x_angle"], d["y_angle"])

    # Errors
    d["err_phi"] = d["phi"] - d["x_angle"]
    d["err_theta"] = d["theta"] - d["y_angle"]
    if metric == "total":
        d["error"] = np.sqrt(d["err_phi"] ** 2 + d["err_theta"] ** 2)
        ylab = "Mean error (deg)"
    elif metric == "phi":
        d["error"] = np.abs(d["err_phi"])
        ylab = "Mean |φ − x_angle| (deg)"
    elif metric == "theta":
        d["error"] = np.abs(d["err_theta"])
        ylab = "Mean |θ − y_angle| (deg)"
    else:
        raise ValueError("metric must be 'total', 'phi', or 'theta'")

    # ---------- Eccentricity binning for the curve ----------
    if r_range is None:
        r_min = float(np.nanmin(d["r_xy"]))
        r_max = float(np.nanmax(d["r_xy"]))
    else:
        r_min, r_max = map(float, r_range)

    # Build r-bins across the requested extent (even if edges are empty)
    r_edges = np.linspace(r_min, r_max, int(r_bins) + 1)
    r_cats = pd.cut(d["r_xy"], r_edges, include_lowest=True)
    centers_r = np.array([iv.mid for iv in r_cats.cat.categories])

    # Aggregate per r-bin: mean error, n, and median ratio (for top-axis mapping)
    gb = d.groupby(r_cats, observed=False).agg(
        mean_error=("error", "mean"),
        n=("error", "size"),
        ratio_med=("ratio2", "median"),
    ).reset_index(drop=True)
    gb = gb.assign(r_center=centers_r)

    # Curve points (drop sparse bins)
    curve = gb[gb["n"] >= int(min_per_bin)].copy()

    # ---------- Span line at R = k/2 (degrees) ----------
    R = float(k) / 2.0

    # For reporting only: dataset-based ratio at that ring (use ring or fallback to interp)
    ring_mask = np.abs(d["r_xy"] - R) <= float(ring_halfwidth_deg)
    if ring_mask.any():
        ratio_at_R = float(d.loc[ring_mask, "ratio2"].mean())
    else:
        valid = gb["n"].to_numpy() > 0
        ratio_at_R = float(np.interp(R, gb.loc[valid, "r_center"], gb.loc[valid, "ratio_med"]))

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Draw the curve (points are at r-bin CENTERS)
    ax.plot(curve["r_center"], curve["mean_error"], linewidth=1.5, color="black")
    ax.scatter(curve["r_center"], curve["mean_error"], s=3, color="black")

    # Bottom axis (degrees)
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.set_xlim(r_min, r_max)
    ax.set_xlabel("Eccentricity (deg)", fontsize=10)
    ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(axis="both", labelsize=8, width=1.0, length=4, direction="out")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", width=0.8, length=2, direction="out")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Vertical span line at R (degrees)
    ax.axvline(R, linestyle="--", linewidth=1.0, color="black")

    # Top x-axis: ratio at the same degree ticks
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())  # share transform
    bottom_ticks = ax.get_xticks()
    valid_map = gb["n"].to_numpy() > 0
    r_map = gb.loc[valid_map, "r_center"].to_numpy()
    q_map = gb.loc[valid_map, "ratio_med"].to_numpy()
    ratio_labels = np.interp(
        bottom_ticks,
        r_map,
        q_map,
        left=q_map[0] if q_map.size else np.nan,
        right=q_map[-1] if q_map.size else np.nan,
    )
    ax_top.set_xticks(bottom_ticks)
    ax_top.set_xticklabels([f"{v:.2f}" if np.isfinite(v) else "" for v in ratio_labels], fontsize=8)
    ax_top.set_xlabel("Minor/Major axis ratio", fontsize=10)
    ax_top.tick_params(axis="x", width=1.0, length=4, direction="out")
    ax_top.spines["top"].set_visible(True)
    ax_top.spines["top"].set_linewidth(1.0)
    ax_top.spines["right"].set_visible(False)

    plt.tight_layout()

    # ---------- Export ----------
    if export_pdf_path:
        Path(export_pdf_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_pdf_path, bbox_inches="tight")

    in_span = d["r_xy"] <= R
    mean_span = float(d.loc[in_span, "error"].mean()) if in_span.any() else np.nan

    return {
        "curve": curve[["r_center", "mean_error", "n", "ratio_med"]].reset_index(drop=True),
        "r_limits": (r_min, r_max),
        "R": R,
        "ratio_at_R": ratio_at_R,
        "within_span_mean": mean_span,
        "figure": fig,
    }


def _bind_paths(argv: list[str]) -> None:
    """Workbook is the only data file. Plots go next to this script unless OUTDIR is given."""
    global XLSX, OUT
    args = [Path(a).expanduser() for a in argv[1:] if not str(a).startswith("--")]
    here = Path(__file__).resolve().parent
    if not args:
        XLSX = here / "S1_Data.xlsx"
        OUT = here
        return
    if args[0].suffix.lower() == ".xlsx":
        XLSX = args[0].resolve()
        OUT = (args[1] if len(args) > 1 else here).resolve()
        return
    OUT = args[0].resolve()
    if len(args) > 1 and args[1].suffix.lower() == ".xlsx":
        XLSX = args[1].resolve()
    else:
        XLSX = here / "S1_Data.xlsx"



def sheets() -> set[str]:
    return set(pd.ExcelFile(XLSX).sheet_names)


def load(name: str) -> pd.DataFrame:
    frame = pd.read_excel(XLSX, sheet_name=name)
    return frame.drop(columns=frame.columns[0])


def finish(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def unsplit(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Join ``col_part1``, ``col_part2``, ... back into one row per sample."""
    if all(c in frame.columns for c in columns) and f"{columns[0]}_part1" not in frame.columns:
        return frame
    chunks = []
    i = 1
    while f"{columns[0]}_part{i}" in frame.columns:
        sub = frame[[f"{c}_part{i}" for c in columns]].copy()
        sub.columns = columns
        sub = sub.dropna(subset=[columns[-1]])
        chunks.append(sub)
        i += 1
    if not chunks:
        return frame
    return pd.concat(chunks, ignore_index=True)


def concat_parts(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [c for c in frame.columns if str(c) == prefix or str(c).startswith(prefix + "_part")]
    if not cols:
        return np.array([])
    return np.concatenate([frame[c].dropna().to_numpy(float) for c in cols])



def eye_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df.dropna(subset=["time_s"]).copy()
    t_ms = sub["time_s"].to_numpy(float) * 1000.0

    def one(phi: str, theta: str, pupil: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ms_axis": t_ms,
                "k_phi": sub[phi].to_numpy(float),
                "k_theta": sub[theta].to_numpy(float),
                "pupil_diameter": sub[pupil].to_numpy(float) if pupil in sub.columns else np.nan,
            }
        )

    return one("phi_left_deg", "theta_left_deg", "pupil_left_mm"), one(
        "phi_right_deg", "theta_right_deg", "pupil_right_mm"
    )


def event_times(df: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in df.columns:
        return None
    values = df[column].dropna().to_numpy(float)
    return values if values.size else None


def behavior_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    """State intervals. Stored in seconds; the 3C drawer expects milliseconds."""
    needed = {"behavior_start_s", "behavior_end_s", "behavior_annotation"}
    if not needed.issubset(df.columns):
        return None
    sub = df.dropna(subset=["behavior_start_s", "behavior_end_s"])
    if sub.empty:
        return None
    start = sub["behavior_start_s"].to_numpy(float)
    end = sub["behavior_end_s"].to_numpy(float)
    if np.nanmax(start) < 10000:
        start = start * 1000.0
        end = end * 1000.0
    return pd.DataFrame(
        {
            "start_time": start,
            "end_time": end,
            "annotation": sub["behavior_annotation"].astype(str).to_numpy(),
        }
    )


def draw_vignette(
    df: pd.DataFrame,
    name: str,
    *,
    start_s: float,
    end_s: float,
    traces: list[str],
    figure_size: tuple[float, float],
    with_rates: bool = False,
    **kwargs,
) -> None:
    left, right = eye_tables(df)
    use = list(traces)
    state = behavior_frame(df) if with_rates else None
    left_ms = event_times(df, "left_saccade_ms") if with_rates else None
    right_ms = event_times(df, "right_saccade_ms") if with_rates else None
    head_ms = event_times(df, "head_movements_ms") if with_rates else None
    if "saccade_frequency" in use and left_ms is None and right_ms is None and head_ms is None:
        use = [t for t in use if t != "saccade_frequency"]
    OUT.mkdir(parents=True, exist_ok=True)
    plot_zoomed_in_with_head_rate(
        start_s,
        end_s,
        traces=use,
        left_df=left,
        right_df=right,
        left_ms=left_ms,
        right_ms=right_ms,
        head_movements_ms=head_ms,
        behavior_state_df=state,
        behavior_time_unit="ms",
        figure_size=figure_size,
        export_path=OUT / f"{name}.pdf",
        show=False,
        **kwargs,
    )
    print("wrote", name)


def coupling_hist(right, left, weights, rng, bins: int) -> dict:
    mask = np.isfinite(right) & np.isfinite(left)
    if weights is not None:
        mask &= np.isfinite(weights)
        w = np.asarray(weights, float)[mask]
    else:
        w = None
    return histogram2d_xy(np.asarray(right, float)[mask], np.asarray(left, float)[mask], w, rng, bins)


def save_coupling(panels: list[tuple], name: str, figsize: tuple[float, float]) -> None:
    cmap = _turbo_white0()
    vmax = max(float(np.nanmax(hist["norm_counts"])) for hist, *_ in panels)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, dpi=300, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (hist, rng, ticks, title) in zip(axes, panels):
        _draw_coupling_heatmap(ax, hist, rng, ticks, vmax=vmax, cmap=cmap)
        ax.set_title(title, fontsize=8)
    finish(fig, name)


def mean_curves(
    frame: pd.DataFrame,
    ycol: str,
    name: str,
    ylabel: str,
    *,
    animal: str | None = "PV_106",
    min_events: int = 30,
    speed_to_per_ms: bool = True,
    title: str | None = "PV_106",
) -> None:
    """Amplitude-bin means: 5° bins, 2 ms grid, 10 ms Gaussian on occupancy."""
    sub = frame if animal is None else frame[frame["animal"].astype(str) == animal].copy()
    if sub.empty:
        sub = frame
    keys = ["animal", "block", "eye", "saccade_on_ms"]
    events = sub.groupby(keys, sort=False)
    amps = events["amplitude_deg"].first().to_numpy(float)
    amps = amps[np.isfinite(amps) & (amps >= 0.5)]
    if amps.size == 0:
        print("skip", name, "(no events)")
        return
    max_amp = float(np.nanpercentile(amps, 99.5))
    edges = np.arange(0.0, max_amp + 5.0, 5.0)
    if len(edges) > 9:
        edges = edges[:9]
    t_grid = np.arange(-100.0, 100.0 + 1e-9, 2.0)
    curves = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        chosen = []
        for _, g in events:
            amp = float(g["amplitude_deg"].iloc[0])
            if lo <= amp < hi or (i == len(edges) - 2 and amp == hi):
                chosen.append(g)
        if len(chosen) < min_events:
            continue
        num = np.zeros(len(t_grid))
        den = np.zeros(len(t_grid))
        for g in chosen:
            t = g["time_from_align_ms"].to_numpy(float)
            y = g[ycol].to_numpy(float)
            if ycol == "speed" and speed_to_per_ms:
                y = y / 1000.0
            m = np.isfinite(t) & np.isfinite(y)
            idx = np.floor((t[m] - t_grid[0]) / 2.0).astype(int)
            ok = (idx >= 0) & (idx < len(t_grid))
            np.add.at(num, idx[ok], y[m][ok])
            np.add.at(den, idx[ok], 1.0)
        num_s = gaussian_filter1d(num, sigma=5.0, mode="nearest")
        den_s = gaussian_filter1d(den, sigma=5.0, mode="nearest")
        curve = np.divide(num_s, den_s, out=np.full_like(num, np.nan), where=den_s > 1e-6)
        if ycol == "speed":
            curve = np.clip(curve, 0.0, None)
        curves.append((f"{int(lo)}-{int(hi)}° (n={len(chosen)})", curve))
    fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=300)
    colors = _bin_viridis_colors(len(curves))
    for (label, curve), color in zip(curves, colors):
        ax.plot(t_grid, curve, color=color, lw=1.0, label=label)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time from peak (ms)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-100, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    finish(fig, name)


def main_sequence(df: pd.DataFrame, name: str, *, xlim=None, ylim=None, title: str | None = None):
    work = df.copy()
    work["peak_velocity"] = pd.to_numeric(work["peak_velocity_deg_per_ms"], errors="coerce")
    work["amplitude_deg"] = pd.to_numeric(work["amplitude_deg"], errors="coerce")
    work = work[np.isfinite(work["amplitude_deg"]) & np.isfinite(work["peak_velocity"])]
    work = work[work["amplitude_deg"] >= 0.5]
    if work.empty:
        print("skip", name)
        return None, None
    amp_hi = float(work["amplitude_deg"].max())
    edges = np.arange(0.0, max(25.0, np.ceil(amp_hi / 5.0) * 5.0) + 1e-9, 5.0)
    per_animal, _, gfit = _main_sequence_stats(
        work, amp_col="amplitude_deg", edges=edges, min_events=15
    )
    animals = list(per_animal.keys())
    color_map = build_color_map(animals, template="okabeito", order=animals)
    fig, ax = _plot_per_animal_means(
        per_animal,
        gfit,
        color_map=color_map,
        animal_order=animals,
        params=MS_PARAMS,
        xlim=xlim,
        ylim=ylim,
        x_fit_span=(float(work["amplitude_deg"].min()), float(work["amplitude_deg"].max())),
        title=title,
    )
    kept_x, kept_y = ax.get_xlim(), ax.get_ylim()
    finish(fig, name)
    return (kept_x if xlim is None else xlim), (kept_y if ylim is None else ylim)


def isi_by_animal(df: pd.DataFrame, *, state: str | None = None) -> dict[str, np.ndarray]:
    sub = df if state is None else df[df["state"].astype(str) == state]
    out = {}
    for animal, g in sub.groupby(sub["animal"].astype(str)):
        vals = g["isi_ms"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[str(animal)] = vals
    return out


def draw_isi(
    isi_all: dict[str, np.ndarray],
    name: str,
    *,
    scale: str = "log",
    frame_ms: float = FRAME_MS,
    log_exponent: float | None = None,
) -> None:
    animals = sorted(isi_all)
    color_map = build_color_map(animals, template="okabeito", order=animals)
    font = _resolve_font_family("Arial")
    if scale == "log":
        cfg = _DEFAULT_LOG_PARAMS
        low = max(1e-6, float(cfg["first_bin_factor"]) * frame_ms)
        bins = _snapped_log_bins(
            int(cfg["num_bins"]),
            frame_ms,
            low,
            float(cfg["high_ms"]),
            add_leading_zero_bin=bool(cfg["add_leading_zero_bin"]),
            log_exponent=float(cfg["log_exponent"] if log_exponent is None else log_exponent),
        )
        xlim = tuple(cfg["xlim"])
        fold = lambda counts, edges: _fold_to_xlim_log(counts, edges, xlim)
        xscale = "log"
    else:
        cfg = _DEFAULT_LINEAR_PARAMS
        bins = _snapped_linear_bins(
            int(cfg["num_bins"]),
            frame_ms,
            float(cfg["low_ms"]),
            float(cfg["high_ms"]),
            add_leading_zero_bin=bool(cfg["add_leading_zero_bin"]),
        )
        xlim = tuple(cfg["xlim"])
        fold = lambda counts, edges: _slice_to_xlim_linear(
            counts, edges, xlim, renormalize=str(cfg["xlim_policy"]) == "conditional"
        )
        xscale = "linear"
    fig, *_ = _plot_isi_traces(
        bins=bins,
        isi_all=isi_all,
        animals=animals,
        color_map=color_map,
        fold_fn=fold,
        figure_size=tuple(cfg["figure_size"]),
        xscale=xscale,
        xlim=xlim,
        ylim=tuple(cfg["ylim"]) if cfg.get("ylim") is not None else None,
        xlabel=str(cfg["xlabel"]),
        ylabel=str(cfg["ylabel"]),
        linewidth=float(cfg["linewidth"]),
        combined_linewidth=float(cfg["combined_linewidth"]),
        combined_color=cfg["combined_color"],
        combined_linestyle=str(cfg["combined_linestyle"]),
        combined_label="All (combined)",
        font_family=font,
    )
    finish(fig, name)


def eye_pooled_across(per_block: pd.DataFrame) -> pd.DataFrame:
    """Animal-mean leftover error, eyes pooled, for the two-bar S1 panel."""
    an = animal_level_from_blocks(per_block)
    rows = []
    for axis in ("phi", "theta"):
        sub = an[an["axis"].astype(str) == axis]
        per_animal = sub.groupby(sub["animal"].astype(str))["mean_abs_deg"].mean()
        rows.append(
            {
                "eye": "both",
                "axis": axis,
                "n_animals": int(per_animal.size),
                "mean_of_mean_abs_deg": float(per_animal.mean()),
                "std_of_mean_abs_deg": float(per_animal.std(ddof=1)) if per_animal.size > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def span_scatter(labels, main, perp, name: str) -> None:
    """One point per animal: main-axis span vs perpendicular span."""
    labels = [str(x) for x in labels]
    main = np.asarray(main, float)
    perp = np.asarray(perp, float)
    color_map = build_color_map(labels, template="okabeito", order=labels)
    lim = float(max(np.nanmax(main), np.nanmax(perp))) * 1.1
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    for animal, x, y in zip(labels, main, perp):
        ax.scatter(x, y, s=40, color=color_map[animal], edgecolor="black", linewidth=0.5, label=animal)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Main-axis span (°)", fontsize=8)
    ax.set_ylabel("Perpendicular span (°)", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7, direction="out", length=3, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    finish(fig, name)


def draw_s2(per_eye: pd.DataFrame, per_block: pd.DataFrame | None = None) -> None:
    """Published animal means, with the deposited per-eye and per-block spans under them."""
    means = per_animal_spans(per_eye)
    animals = means["animal"].astype(str).tolist()
    color_map = build_color_map(animals, template="okabeito", order=animals)
    xs = [per_eye["main_span_deg"].to_numpy(float)]
    ys = [per_eye["perp_span_deg"].to_numpy(float)]
    if per_block is not None and not per_block.empty:
        xs.append(per_block["main_span_deg"].to_numpy(float))
        ys.append(per_block["perp_span_deg"].to_numpy(float))
    xs.append(means["main_span_mean"].to_numpy(float))
    ys.append(means["perp_span_mean"].to_numpy(float))
    lim = float(max(np.nanmax(np.concatenate(xs)), np.nanmax(np.concatenate(ys)))) * 1.1
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    if per_block is not None and not per_block.empty:
        for animal, g in per_block.groupby("animal"):
            ax.scatter(
                g["main_span_deg"],
                g["perp_span_deg"],
                s=10,
                color=color_map.get(str(animal), "0.5"),
                alpha=0.35,
                linewidth=0,
                zorder=1,
            )
    for animal, g in per_eye.groupby("animal"):
        ax.scatter(
            g["main_span_deg"],
            g["perp_span_deg"],
            s=22,
            facecolors="none",
            edgecolor=color_map.get(str(animal), "0.2"),
            linewidth=0.8,
            zorder=2,
        )
    for _, row in means.iterrows():
        animal = str(row["animal"])
        ax.scatter(
            row["main_span_mean"],
            row["perp_span_mean"],
            s=44,
            color=color_map[animal],
            edgecolor="black",
            linewidth=0.5,
            label=animal,
            zorder=3,
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Main-axis span (°)", fontsize=8)
    ax.set_ylabel("Perpendicular span (°)", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7, direction="out", length=3, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    finish(fig, "Figure_S2")


def span_bars(labels, main, perp, name: str, *, sort_mean: bool = False, title: str | None = None) -> None:
    labels = [str(x) for x in labels]
    main = np.asarray(main, float)
    perp = np.asarray(perp, float)
    if sort_mean:
        order = np.argsort(-(main + perp) / 2.0)
        labels = [labels[i] for i in order]
        main, perp = main[order], perp[order]
    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=300)
    x = np.arange(len(labels))
    width = 0.36
    for shift, values, color, label in (
        (-width / 2, main, "#0072B2", "Main Axis"),
        (width / 2, perp, "#E69F00", "Perpendicular"),
    ):
        bars = ax.bar(x + shift, values, width, color=color, edgecolor="black", linewidth=0.6, label=label)
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_xticks(x, labels, rotation=0, fontsize=8)
    ax.set_ylabel("Oculomotor span (°)", fontsize=8)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    finish(fig, name)


def probability_mats(values_by_animal: list[np.ndarray], bins: np.ndarray) -> np.ndarray:
    rows = []
    for values in values_by_animal:
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        hist, _ = np.histogram(values, bins=bins)
        rows.append(hist / hist.sum() if hist.sum() else hist)
    return np.vstack(rows) if rows else np.zeros((1, len(bins) - 1))


def s13_slope_bars(df: pd.DataFrame) -> None:
    """Paired concurrent vs monocular OLS slopes, one fit per animal on the S13A rows."""
    from scipy import stats

    animals = sorted(df["animal"].astype(str).unique())
    concurrent, monocular = [], []
    for animal in animals:
        group = df[df["animal"].astype(str) == animal]

        def slope(sub: pd.DataFrame) -> float:
            x = sub["amplitude_deg"].to_numpy(float)
            y = sub["peak_velocity_deg_per_ms"].to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(y)
            if int(keep.sum()) < 3:
                return float("nan")
            return float(stats.linregress(x[keep], y[keep]).slope)

        concurrent.append(slope(group[group["pairing"].astype(str) == "concurrent"]))
        monocular.append(slope(group[group["pairing"].astype(str) == "monocular"]))
    concurrent_a = np.asarray(concurrent, float)
    monocular_a = np.asarray(monocular, float)
    t_stat, p_val = stats.ttest_rel(concurrent_a, monocular_a)
    fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=150)
    x = np.arange(len(animals))
    width = 0.35
    ax.bar(x - width / 2, concurrent_a, width, color="#0072B2", edgecolor="0.2", label="concurrent")
    ax.bar(x + width / 2, monocular_a, width, color="#D55E00", edgecolor="0.2", label="monocular")
    ax.set_xticks(x)
    ax.set_xticklabels(animals, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("OLS slope (°/ms per °)")
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"paired t={t_stat:.2f}  p={p_val:.3g}  df={len(animals) - 1}", fontsize=8)
    fig.tight_layout()
    finish(fig, "Figure_S13E")


def plot_all(present: set[str]) -> None:
    if "Figure 1E" in present:
        distances = load("Figure 1E")["displacement_um"].dropna().to_numpy(float)
        fig, _ = create_figure(distances, output_path=None)
        finish(fig, "Figure_1E")

    if "Figure 2B" in present:
        # Window and ticks are the ones stored on figure_2b.pickle (240–320 s).
        draw_vignette(
            load("Figure 2B"),
            "Figure_2B",
            start_s=240.0,
            end_s=320.0,
            traces=["center_x", "center_y"],
            figure_size=(4.3, 1.8),
            std_multiplier=2,
            phi_ticks=[-15, 0, 15],
            theta_ticks=[-15, 0, 15],
            x_zero_origin=True,
        )

    # Figure 2B examples are not deposited.

    if "Figure 2C" in present:
        df = load("Figure 2C")
        mean_curves(df, "speed", "Figure_2C", "Speed [°/ms]")
        mean_curves(df, "position_deg", "Figure_2D", "Position (deg)")
    else:
        print("skip Figure 2C")
        print("skip Figure 2D")

    if "Figure 2E" in present:
        main_sequence(load("Figure 2E"), "Figure_2E")
    else:
        print("skip Figure 2E")

    if "Figure 2F" in present:
        df = load("Figure 2F")
        right = df["right_peak_speed_deg_per_ms"].to_numpy(float)
        left = df["left_peak_speed_deg_per_ms"].to_numpy(float)
        weights = df["animal_weight"].to_numpy(float)
        macro = coupling_hist(right, left, weights, (0.0, 0.5), 60)
        micro = coupling_hist(right, left, weights, (0.0, 0.1), 60)
        save_coupling(
            [
                (macro, (0.0, 0.5), [0.0, 0.25, 0.5], "Macro"),
                (micro, (0.0, 0.1), [0.0, 0.05, 0.1], "Micro"),
            ],
            "Figure_2F",
            (3.0, 1.7),
        )

    if "Figure 2G" in present:
        df = load("Figure 2G")
        amp_col = "magnitude_raw_angular" if "magnitude_raw_angular" in df.columns else "net_angular_disp"
        animals = sorted(df["animal"].astype(str).unique())
        bins = np.linspace(0, 25, 40)
        centers = 0.5 * (bins[:-1] + bins[1:])
        synced, mono = [], []
        for animal in animals:
            g = df[df["animal"].astype(str) == animal]
            synced.append(g.loc[g["pairing"] == "concurrent", amp_col].to_numpy(float))
            mono.append(g.loc[g["pairing"] == "monocular", amp_col].to_numpy(float))
        synced_m = probability_mats(synced, bins)
        mono_m = probability_mats(mono, bins)
        fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300)
        sem = lambda mat: mat.std(axis=0, ddof=1) / np.sqrt(max(len(mat), 1))
        ax.plot(centers, synced_m.mean(0), color="green", lw=1.5, label="Synchronized")
        ax.fill_between(centers, synced_m.mean(0) - sem(synced_m), synced_m.mean(0) + sem(synced_m), color="green", alpha=0.3)
        ax.plot(centers, mono_m.mean(0), color="blue", lw=1.5, label="Monocular")
        ax.fill_between(centers, mono_m.mean(0) - sem(mono_m), mono_m.mean(0) + sem(mono_m), color="blue", alpha=0.3)
        ax.set_xlabel("Saccade Amplitude [deg]", fontsize=8)
        ax.set_ylabel("Probability", fontsize=8)
        ax.legend(fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        finish(fig, "Figure_2G_top")

        bins_b = np.linspace(0, 25, 20)
        centers_b = 0.5 * (bins_b[:-1] + bins_b[1:])
        color_map = build_color_map(animals, template="okabeito", order=animals)
        fig, ax = plt.subplots(figsize=(4.4, 2.6), dpi=300)
        for animal, s, n in zip(animals, synced, mono):
            hs = probability_mats([s], bins_b)[0]
            hn = probability_mats([n], bins_b)[0]
            ax.plot(centers_b, hs - hn, color=color_map[animal], lw=1.5, label=animal)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Amplitude [deg]", fontsize=8)
        ax.set_ylabel("Diff [probability]", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(left=0)
        fig.tight_layout()
        finish(fig, "Figure_2G_bottom")
        old = OUT / "Figure_2G.pdf"
        if old.is_file():
            old.unlink()
    else:
        print("skip Figure 2G")

    if "Figure 2I" in present:
        # Polar recipe matches figure_2i.py; densities are histogrammed from the
        # deposited per-saccade rotated angles, not from stored bin counts.
        df = load("Figure 2I")
        animals = np.array(sorted(df["animal"].astype(str).unique()))
        color_map = build_color_map(animals, template="okabeito", order=animals)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        fig, axs = plt.subplots(1, 2, figsize=(5, 4), dpi=300, subplot_kw=dict(projection="polar"))

        def eye_code(value: str) -> str:
            text = str(value).strip().upper()
            return "L" if text.startswith("L") else "R" if text.startswith("R") else text

        def one(ax, eye: str) -> None:
            sub = df[df["eye"].map(eye_code) == eye]
            for animal in animals:
                ang = sub.loc[sub["animal"].astype(str) == animal, "rotated_angle_deg"].to_numpy(float)
                ang = ang[np.isfinite(ang)] % 360
                if ang.size == 0:
                    continue
                counts, edges = np.histogram(ang, bins=36, range=(0, 360), density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                theta = np.deg2rad(np.r_[centers, centers[0]])
                rho = np.r_[counts, counts[0]]
                ax.plot(theta, rho, lw=1.2, color=color_map[animal], label=f"{animal} (n={ang.size})")
            ring = np.linspace(0, 2 * np.pi, 512)
            ax.fill_between(ring, 0, 0.005, alpha=0.15, color="gray", zorder=0)
            ax.set_facecolor("white")
            ax.set_yticks([])
            ax.grid(False)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=6, frameon=False)

        one(axs[0], "R")
        one(axs[1], "L")
        axs[0].set_title("Right eye (rotated, axis-based)", fontsize=9, pad=10)
        axs[1].set_title("Left eye (rotated, axis-based)", fontsize=9, pad=10)
        fig.tight_layout()
        finish(fig, "Figure_2I")

    if "Figure 2J" in present:
        df = load("Figure 2J")
        animals = sorted(df["animal"].astype(str).unique())
        color_map = build_color_map(animals, template="okabeito", order=animals)
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
        xs, ys = [], []
        for animal in animals:
            g = df[df["animal"].astype(str) == animal]
            sx = calculate_orientation_tuning(g.loc[g["pairing"] == "concurrent", "overall_angle_deg"])
            sy = calculate_orientation_tuning(g.loc[g["pairing"] == "monocular", "overall_angle_deg"])
            xs.append(sx)
            ys.append(sy)
            ax.scatter(sx, sy, color=color_map[animal], s=30, label=animal)
        ax.axhline(0, color="gray", ls="--", lw=0.7)
        ax.axvline(0, color="gray", ls="--", lw=0.7)
        finite = [v for v in xs + ys if np.isfinite(v)]
        lim = (max(abs(min(finite)), abs(max(finite))) * 1.1) if finite else 1.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Concurrent [A.U]", fontsize=8)
        ax.set_ylabel("Monocular [A.U]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        finish(fig, "Figure_2J")

    if "Figure 3A" in present:
        draw_vignette(
            load("Figure 3A"), "Figure_3A",
            start_s=210.0, end_s=240.0, traces=["center_x", "center_y"],
            figure_size=(2.3, 1.8), std_multiplier=3,
            phi_ticks=[-15, 0, 15], theta_ticks=[-15, 0, 15], x_zero_origin=False,
        )
    if "Figure 3B" in present:
        draw_vignette(
            load("Figure 3B"), "Figure_3B",
            start_s=310.0, end_s=340.0, traces=["center_x", "center_y"],
            figure_size=(2.3, 1.8), std_multiplier=3,
            phi_ticks=[-15, 0, 15], theta_ticks=[-15, 0, 15], x_zero_origin=False,
        )
    if "Figure 3C" in present:
        draw_vignette(
            load("Figure 3C"),
            "Figure_3C",
            start_s=200.0,
            end_s=415.0,
            traces=["center_x", "center_y", "pupil_diameter", "saccade_frequency"],
            figure_size=(4.7, 3.7),
            with_rates=True,
            std_multiplier=3.5,
            phi_ticks=3,
            theta_ticks=3,
            pupil_ticks=[1.5, 2.0, 2.5],
            x_zero_origin=True,
            plot_state_y=20,
            head_merge_ms=20,
            head_merge_strategy="first",
        )

    if "Figure 3D" in present:
        # Same bin span as fig_3e.py. Manuscript panel is the KDE traces only.
        df = load("Figure 3D")
        quiet = df["pupil_quiet_mm"].dropna().to_numpy(float)
        active = df["pupil_active_mm"].dropna().to_numpy(float)
        pooled = np.concatenate([quiet, active])
        x_range = (1.0, 2.35)
        in_range = pooled[(pooled >= x_range[0]) & (pooled <= x_range[1])]
        lo, hi = np.percentile(in_range, (0.001, 99.999))
        start = max(min(lo, hi), x_range[0])
        stop = min(max(lo, hi), x_range[1])
        edges = np.linspace(start, stop, 41)
        fig, ax = plt.subplots(figsize=(2.2, 1.5), dpi=200)
        from scipy.stats import gaussian_kde

        width = float(np.mean(np.diff(edges)))
        xs = np.linspace(float(edges[0]), float(edges[-1]), 256)
        for label, data, color, style in (
            ("Quiet", quiet, "0.45", "--"),
            ("Active", active, "black", "-"),
        ):
            kept = data[(data >= edges[0]) & (data <= edges[-1])]
            if kept.size > 2:
                ax.plot(
                    xs,
                    gaussian_kde(kept)(xs) * width,
                    color=color,
                    ls=style,
                    lw=1.5,
                    label=label,
                )
        ax.set_xlabel("Pupil [mm]", fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.tick_params(axis="both", labelsize=8, direction="out", length=5, width=1, colors="black")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        finish(fig, "Figure_3D")

    if "Figure 3E" in present:
        df = unsplit(load("Figure 3E"), ["animal", "state", "pupil_mm"])
        edges = np.linspace(-3.0, 3.0, 16)
        centers = 0.5 * (edges[:-1] + edges[1:])
        animals = sorted(df["animal"].astype(str).unique())
        color_map = build_color_map(
            animals,
            template="okabeito",
            order=animals,
        )
        # fig_3f.py Okabe list is the same template; force that hex order.
        okabe = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
        color_map = {lab: okabe[i % len(okabe)] for i, lab in enumerate(animals)}
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        fig, ax = plt.subplots(figsize=(2.2, 1.7), dpi=300)
        overall = df["pupil_mm"].to_numpy(float)
        lo, hi = np.nanpercentile(overall, [0.1, 99.9])
        for animal in animals:
            g = df[df["animal"].astype(str) == animal]
            a = g.loc[g["state"] == "active", "pupil_mm"].to_numpy(float)
            q = g.loc[g["state"] == "quiet", "pupil_mm"].to_numpy(float)
            a = a[(a >= lo) & (a <= hi)]
            q = q[(q >= lo) & (q <= hi)]
            comb = np.concatenate([a, q])
            mu, sd = float(np.nanmean(comb)), float(np.nanstd(comb) or 1.0)
            ha, _ = np.histogram((a - mu) / sd, bins=edges)
            hq, _ = np.histogram((q - mu) / sd, bins=edges)
            pa = ha / ha.sum() if ha.sum() else ha.astype(float)
            pq = hq / hq.sum() if hq.sum() else hq.astype(float)
            ax.plot(centers, pa - pq, lw=1.5, color=color_map[animal], label=animal)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlim(-3, 3)
        ax.set_xlabel("Z-scored Pupil Diameter", fontsize=10)
        ax.set_ylabel("Probability Difference", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        fig.tight_layout()
        finish(fig, "Figure_3E")
    else:
        print("skip Figure 3E")

    if "Figure 3F" in present:
        df = load("Figure 3F")
        draw_isi(
            isi_by_animal(df),
            "Figure_3F",
            scale="log",
            frame_ms=FRAME_MS_3F,
            log_exponent=LOG_EXPONENT_3F,
        )
        draw_isi(isi_by_animal(df), "Figure_3F_linear", scale="linear", frame_ms=FRAME_MS_3F)
    else:
        print("skip Figure 3F")

    if "Figure S1A" in present:
        sim = load("Figure S1A")
        sim["major_ax"] = np.maximum(sim["axs_1"], sim["axs_2"])
        sim["minor_ax"] = np.minimum(sim["axs_1"], sim["axs_2"])
        sim["ratio2"] = sim["minor_ax"] / sim["major_ax"]
        sim["ratio1"] = sim["major_ax"] / sim["minor_ax"]
        _, kerr_df = kerr(sim, aEC=A_EC, bEC=B_EC)
        result = plot_error_vs_ecc_with_ratio_axis_pdf(
            kerr_df,
            k=K,
            metric=METRIC,
            r_bins=R_BINS,
            r_range=R_RANGE,
            min_per_bin=MIN_PER_BIN,
            ring_halfwidth_deg=RING_HALFWIDTH_DEG,
            figsize=FIGSIZE,
            dpi=DPI,
            export_pdf_path=None,
            font_family="Arial",
            y_range=Y_RANGE,
        )
        finish(result["figure"], "Figure_S1A")

    if "Figure S1B" in present:
        across = eye_pooled_across(load("Figure S1B"))
        finish(figure_s1c_overall(across), "Figure_S1B")

    if "Figure S2" in present:
        s2 = load("Figure S2")
        if {"main_span_deg", "eye"}.issubset(s2.columns):
            means = per_animal_spans(s2)
            span_scatter(means["animal"], means["main_span_mean"], means["perp_span_mean"], "Figure_S2")
        else:
            span_scatter(s2["animal"], s2["main_span_mean"], s2["perp_span_mean"], "Figure_S2")
    else:
        print("skip Figure S2")

    if "Figure S3" in present:
        df = load("Figure S3")
        weight = df["weight"].to_numpy(float) if "weight" in df.columns else None
        still = ~df["head_movement"].astype(bool)
        moving = ~still
        panels = []
        for mask, title in ((still, "still"), (moving, "moving")):
            sub = df.loc[mask]
            w = None if weight is None else weight[mask.to_numpy()]
            hist = coupling_hist(
                sub["right_peak_v"].to_numpy(float),
                sub["left_peak_v"].to_numpy(float),
                w,
                (0.0, 0.2),
                100,
            )
            panels.append((hist, (0.0, 0.2), [0.0, 0.1, 0.2], title))
        save_coupling(panels, "Figure_S3", (3.6, 1.7))
    else:
        print("skip Figure S3")

    if "Figure S4" in present:
        raw = unsplit(load("Figure S4"), ["animal", "block", "time_ms", "rolling_r"])
        if "rolling_r" in raw.columns:
            df = coupling_fractions(raw)
        else:
            df = raw
        animals = df["animal"].astype(str).tolist()
        corr = df["frac_corr"].to_numpy(float) * 100.0
        weak = df["frac_weak"].to_numpy(float) * 100.0
        anti = df["frac_anti"].to_numpy(float) * 100.0
        fig, ax = plt.subplots(figsize=(5.5, 2.4), dpi=300)
        x = np.arange(len(animals))
        ax.bar(x, corr, 0.6, label="Correlated (> 0.2)", color="C0")
        ax.bar(x, weak, 0.6, bottom=corr, label="Weak ([−0.2, 0.2])", color="C1")
        ax.bar(x, anti, 0.6, bottom=corr + weak, label="Anti (< −0.2)", color="C2")
        if "n_valid_points" in df.columns:
            labels = [f"{a}\nn={int(n):,}" for a, n in zip(animals, df["n_valid_points"].to_numpy())]
        else:
            labels = animals
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Time (%)", fontsize=10)
        ax.set_xlabel("Animal", fontsize=10)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        fig.tight_layout()
        finish(fig, "Figure_S4")

    if "Figure S5" in present:
        df = load("Figure S5")
        span_bars(
            df["species"],
            df["main_axis_span"],
            df["perpendicular_span"],
            "Figure_S5",
            sort_mean=True,
            title="Oculomotor Range by Species",
        )

    if "Figure S8A" in present:
        df = load("Figure S8A")
        order = [s for s in ("lizard", "turtle", "mouse") if s in set(df["species"].astype(str))]
        fig, axes = plt.subplots(len(order) * 2, 1, figsize=(6.4, 2.0 * max(len(order), 1)), dpi=300, sharex=True)
        axes = np.atleast_1d(axes)
        for i, species in enumerate(order):
            g = df[df["species"].astype(str) == species].sort_values("time_s")
            t = g["time_s"].to_numpy(float)
            for row, (ylab, left, right) in enumerate(
                (
                    ("φ [deg]", "phi_left_deg", "phi_right_deg"),
                    ("θ [deg]", "theta_left_deg", "theta_right_deg"),
                )
            ):
                ax = axes[i * 2 + row]
                ax.plot(t, g[left].to_numpy(float), color="#1f77b4", lw=0.8, label="Left")
                ax.plot(t, g[right].to_numpy(float), color="#8c564b", lw=0.8, label="Right")
                ax.set_ylabel(ylab, fontsize=8)
                ax.tick_params(labelsize=7, direction="out")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if row == 0:
                    ax.set_title(species.capitalize(), loc="left", fontsize=9)
                    if i == 0:
                        ax.legend(frameon=False, fontsize=7, loc="upper right")
        axes[-1].set_xlabel("Time [s]", fontsize=8)
        axes[-1].set_xlim(0.0, 80.0)
        fig.tight_layout()
        finish(fig, "Figure_S8A")

    def s8_bin_curves(sheet: str, ycol: str, name: str, ylabel: str) -> None:
        if sheet not in present:
            return
        df = unsplit(
            load(sheet),
            ["animal", "block", "eye", "saccade_on_ms", "amplitude_deg", "time_from_align_ms", "speed", "position_deg"],
        )
        if {"saccade_on_ms", "amplitude_deg", ycol if ycol != "speed_deg_per_s" else "speed"}.issubset(df.columns):
            mean_curves(
                df,
                "speed" if ycol == "speed_deg_per_s" else ycol,
                name,
                ylabel,
                animal="M_002",
                min_events=15,
                speed_to_per_ms=False,
                title="M_002",
            )
            return
        labels = list(dict.fromkeys(df["amplitude_bin"].astype(str)))
        colors = _bin_viridis_colors(len(labels))
        fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=300)
        for label, color in zip(labels, colors):
            g = df[df["amplitude_bin"].astype(str) == label].sort_values("time_from_peak_ms")
            n_events = int(g["n_events"].iloc[0]) if len(g) else 0
            ax.plot(
                g["time_from_peak_ms"].to_numpy(float),
                g[ycol].to_numpy(float),
                color=color,
                lw=1.0,
                label=f"{label} (n={n_events})",
            )
        ax.set_xlabel("Time from peak (ms)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(-100, 100)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        finish(fig, name)

    s8_src = "Figure S8B" if "Figure S8B" in present else "Figure S8C"
    s8_bin_curves(s8_src, "speed_deg_per_s", "Figure_S8B", "Speed [deg/s]")
    s8_bin_curves("Figure S8C" if "Figure S8C" in present else s8_src, "position_deg", "Figure_S8C", "Position [deg]")

    if "Figure S8D" in present:
        main_sequence(load("Figure S8D"), "Figure_S8D")

    if "Figure S8E" in present:
        df = load("Figure S8E")

        def pair(species: str) -> tuple[np.ndarray, np.ndarray]:
            right = df[f"{species}_right_deg_per_ms"].to_numpy(float)
            left = df[f"{species}_left_deg_per_ms"].to_numpy(float)
            n = min(right.size, left.size)
            right, left = right[:n], left[:n]
            keep = np.isfinite(right) & np.isfinite(left)
            return right[keep], left[keep]

        lizard_r, lizard_l = pair("lizard")
        mouse_r, mouse_l = pair("mouse")
        views = build_s8j_views(lizard_r, lizard_l, None, mouse_r, mouse_l, None)
        fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.2), dpi=300)
        cmap = _turbo_white0()
        for ax, key, title, vmax_key in (
            (axes[0, 0], "full_lizard", "lizard", "vmax_joint_full"),
            (axes[0, 1], "full_mouse", "mouse", "vmax_joint_full"),
            (axes[1, 0], "zoom_lizard", "lizard", "vmax_joint_zoom"),
            (axes[1, 1], "zoom_mouse", "mouse", "vmax_joint_zoom"),
        ):
            panel = views[key]
            _draw_coupling_heatmap(
                ax, panel["hist"], panel["range"], panel["ticks"], vmax=views[vmax_key], cmap=cmap
            )
            ax.set_title(title, fontsize=8)
        fig.tight_layout()
        finish(fig, "Figure_S8E")

    if "Figure S9" in present:
        df = load("Figure S9")
        pools = {
            "rigid": concat_parts(df, "rigid_lizard_um"),
            "modular": concat_parts(df, "modular_lizard_um"),
            "mouse": concat_parts(df, "mouse_um"),
            "turtle": concat_parts(df, "turtle_um"),
        }
        fig = figure_unified_jitter(pools, xmax=S9_XMAX, bin_widths=S9_BIN_WIDTHS, units="um")
        finish(fig, "Figure_S9")

    for sheet, tag in (("Figure S10A", "S10A"), ("Figure S10B", "S10B"), ("Figure S10C", "S10C")):
        if sheet not in present:
            continue
        d = load(sheet)["D_deg_per_frame"].dropna().to_numpy(float)
        thr = S10_THRESHOLD[sheet]
        fig = figure_rayleigh_noise_core_window(
            d,
            threshold=None if thr is None or not np.isfinite(thr) else thr,
            title=tag,
            n_bins=40,
            noise_marker="median",
        )
        finish(fig, f"Figure_{tag}")

    if "Figure S11A" in present:
        df = load("Figure S11A")
        draw_isi(isi_by_animal(df, state="active"), "Figure_S11A_active", scale="log")
        draw_isi(isi_by_animal(df, state="quiet"), "Figure_S11A_quiet", scale="log")
        draw_isi(isi_by_animal(df, state="active"), "Figure_S11A_active_linear", scale="linear")
        draw_isi(isi_by_animal(df, state="quiet"), "Figure_S11A_quiet_linear", scale="linear")
    else:
        print("skip Figure S11A")

    if "Figure S11B" in present:
        df = load("Figure S11B")
        quiet = df["quiet_duration_s"].dropna().to_numpy(float)
        active = df["active_duration_s"].dropna().to_numpy(float)
        zoom = epoch_duration_zoom_xmax_s(active, round_to=5.0)
        quiet_hi = float(np.nanmax(quiet)) if quiet.size else zoom
        fig = plot_epoch_duration_triptych(
            quiet,
            active,
            zoom_xmax_s=zoom,
            quiet_full_bins=epoch_duration_linear_edges(xmax_s=quiet_hi, bin_width_s=50.0),
            zoom_bins=epoch_duration_linear_edges(xmax_s=zoom, bin_width_s=5.0),
            title="zoom 5 s bins; quiet-full 50 s",
        )
        finish(fig, "Figure_S11B")

    if "Figure S12A" in present:
        df = load("Figure S12A")
        left, right = eye_tables(df)
        t = df["time_s"].dropna().to_numpy(float)
        OUT.mkdir(parents=True, exist_ok=True)
        plot_zoomed_in_with_head_rate(
            float(np.nanmin(t)),
            float(np.nanmax(t)),
            traces=["center_x", "center_y"],
            left_df=left,
            right_df=right,
            figure_size=(2.3, 1.8),
            x_zero_origin=False,
            std_multiplier=3.5,
            export_path=OUT / "Figure_S12A.pdf",
            show=False,
        )
        print("wrote Figure_S12A")

    if "Figure S12B" in present:
        df = load("Figure S12B")
        flag = df["has_back_and_forth"]
        if flag.dtype == object:
            positive = flag.astype(str).str.lower().isin(["true", "1"])
        else:
            positive = flag.fillna(False).astype(bool)
        n_events = int(len(df))
        n_positive = int(positive.sum())
        rate = 100.0 * n_positive / n_events if n_events else float("nan")
        fig, ax = plt.subplots(figsize=(2.2, 2.4), dpi=300)
        ax.bar([0], [rate], width=0.55, color="0.45", edgecolor="0.2")
        ax.set_xticks([0], ["Back-and-Forth"])
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of large saccades", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0, rate + 3, f"{rate:.1f}%\nn={n_events}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        finish(fig, "Figure_S12B")

    shared = (None, None)
    if "Figure S13A" in present:
        df = load("Figure S13A")
        shared = main_sequence(df, "Figure_S13A")
        main_sequence(df[df["pairing"] == "concurrent"], "Figure_S13B", xlim=shared[0], ylim=shared[1], title="concurrent")
        main_sequence(df[df["pairing"] == "monocular"], "Figure_S13C", xlim=shared[0], ylim=shared[1], title="monocular")
        s13_slope_bars(df)
    else:
        print("skip Figure S13A")
        print("skip Figure S13B")
        print("skip Figure S13C")

    if "Figure S13D" in present:
        df = load("Figure S13D")
        amp = df["amplitude_deg"].to_numpy(float)
        vel = df["peak_velocity_deg_per_ms"].to_numpy(float)
        m = np.isfinite(amp) & np.isfinite(vel)
        fit = _linregress_fit(amp[m], vel[m])
        xlim = shared[0] or (float(np.nanmin(amp[m])), float(np.nanmax(amp[m])))
        ylim = shared[1] or (float(np.nanmin(vel[m])), float(np.nanmax(vel[m])))
        _plot_2e_scatter(
            amp[m],
            vel[m],
            fit=fit,
            params=MS_PARAMS,
            xlim=xlim,
            ylim=ylim,
            title=f"pooled n={int(m.sum())}",
            out_pdf=OUT / "Figure_S13D.pdf",
            show=False,
        )
        print("wrote Figure_S13D")

def main() -> int:
    argv = [a for a in sys.argv if not a.startswith("--only=")]
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = {s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()}
    _bind_paths(argv)
    if not XLSX.is_file():
        print("missing", XLSX, file=sys.stderr)
        return 1
    present = sheets()
    if only:
        present = {s for s in present if s in only}
    plot_all(present)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
