"""Diagnostic Fig 2e scatter: peak V vs amplitude colored by event ``length``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import stats

from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.figures_2c_2e import (
    _peak_velocity_deg_per_ms,
    _pool_pogona,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PLOT_ID = "diagnostics_2e"
PDF_NAME = "figure_2e_scatter_by_length.pdf"

# Detector ``length`` = last_index - first_index (samples in the event = length + 1).
LENGTH_BINS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("1", (1, 1)),
    ("2", (2, 2)),
    ("3", (3, 3)),
    ("4", (4, 4)),
    ("5", (5, 5)),
    ("6–8", (6, 8)),
    ("9+", (9, 10_000)),
)
# Draw long events first so the short-event ridge stays visible on top.
LENGTH_DRAW_ORDER: tuple[str, ...] = ("9+", "6–8", "5", "4", "3", "2", "1")
LENGTH_COLORS: dict[str, str] = {
    "1": "#D55E00",
    "2": "#E69F00",
    "3": "#F0E442",
    "4": "#009E73",
    "5": "#0072B2",
    "6–8": "#56B4E9",
    "9+": "#000000",
}


def _length_label(length: np.ndarray) -> np.ndarray:
    out = np.full(length.shape, "9+", dtype=object)
    for label, (lo, hi) in LENGTH_BINS:
        out[(length >= lo) & (length <= hi)] = label
    return out


def _event_length(df: pd.DataFrame, *, frame_ms: float) -> np.ndarray:
    if "length" in df.columns and df["length"].notna().any():
        return pd.to_numeric(df["length"], errors="coerce").to_numpy(dtype=float)
    if {"saccade_on_ms", "saccade_off_ms"}.issubset(df.columns):
        dur = (
            pd.to_numeric(df["saccade_off_ms"], errors="coerce")
            - pd.to_numeric(df["saccade_on_ms"], errors="coerce")
        ).to_numpy(dtype=float)
        n_intervals = np.round(dur / float(frame_ms))
        return np.clip(n_intervals, 1.0, None)
    return np.full(len(df), np.nan, dtype=float)


def collect_2e_length_points(tables: EventTables) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pogona (or current cohort) events after the same amp/V filters as Fig 2e."""
    ms = dict((tables.params or {}).get("main_sequence", {}))
    sacc = dict((tables.params or {}).get("saccade", {}))
    amp_col = str(ms.get("amp_col", "net_angular_disp"))
    min_amp = float(ms.get("min_amp_deg", 0.5))
    max_amp_pct = float(ms.get("max_amp_pct", 99.5))
    frame_rate = float(ms.get("frame_rate_fps", 60.0))
    speed_thr = float(sacc.get("speed_threshold_deg_per_frame", 0.8))
    frame_ms = 1000.0 / frame_rate

    df = tables.all_saccades.copy()
    if df.empty:
        raise ValueError("No saccades for 2e length diagnostic")
    df["peak_velocity_deg_per_ms"] = _peak_velocity_deg_per_ms(df, frame_rate)
    df = df[
        np.isfinite(df[amp_col])
        & np.isfinite(df["peak_velocity_deg_per_ms"])
        & (df[amp_col] >= min_amp)
    ].copy()
    amp_hi = float(np.nanpercentile(df[amp_col], max_amp_pct))
    df = df[df[amp_col] <= amp_hi].copy()
    df, cohort = _pool_pogona(df)
    df["length"] = _event_length(df, frame_ms=frame_ms)
    df = df[np.isfinite(df["length"])].copy()
    df["length"] = df["length"].astype(int)
    df["length_bin"] = _length_label(df["length"].to_numpy())
    params = {
        "amp_col": amp_col,
        "min_amp_deg": min_amp,
        "max_amp_pct": max_amp_pct,
        "frame_rate_fps": frame_rate,
        "frame_ms": frame_ms,
        "speed_threshold_deg_per_frame": speed_thr,
        "speed_threshold_deg_per_ms": speed_thr / frame_ms,
        "one_frame_slope": 1.0 / frame_ms,
        "cohort": cohort,
        "figsize": (5.2, 4.2),
        "dpi": 300,
    }
    return df, params


def _draw_length_scatter(
    amp: np.ndarray,
    vel: np.ndarray,
    length_bin: np.ndarray,
    *,
    params: dict[str, Any],
    xlim,
    ylim,
    counts: dict[str, int],
    slope: float | None = None,
    intercept: float | None = None,
) -> Any:
    figsize = tuple(params.get("figsize", (5.2, 4.2)))
    dpi = int(params.get("dpi", 300))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    n = int(amp.size)
    for label in LENGTH_DRAW_ORDER:
        mask = length_bin == label
        if not np.any(mask):
            continue
        ax.scatter(
            amp[mask],
            vel[mask],
            s=6,
            alpha=0.35,
            color=LENGTH_COLORS[label],
            edgecolors="none",
            rasterized=True,
            label=f"{label}  n={counts.get(label, int(mask.sum()))}",
            zorder=10 + LENGTH_DRAW_ORDER.index(label),
        )
    x0, x1 = float(xlim[0]), float(xlim[1])
    xs = np.linspace(max(x0, 0.0), x1, 80)
    one_slope = float(params["one_frame_slope"])
    thr = float(params["speed_threshold_deg_per_ms"])
    ax.plot(xs, one_slope * xs, color="0.15", lw=1.4, ls="-", label="1-frame  V=amp/16.7 ms")
    ax.axhline(thr, color="0.15", lw=1.2, ls=":", label=f"detect floor  {thr:.3f} deg/ms")
    if slope is not None and intercept is not None and np.isfinite(slope) and np.isfinite(intercept):
        ax.plot(xs, slope * xs + intercept, "k--", lw=1.2, label=f"OLS  slope={slope:.3f}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Amplitude [deg]", fontsize=11)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cohort = params.get("cohort", "all")
    ax.set_title(f"all animals ({cohort}) n={n}  colored by length", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.6)
    fig.tight_layout()
    return fig


def export_diagnostics_2e(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
) -> dict[str, Path]:
    """Write ``diagnostics_2e/`` plot bundle: length-colored 2e scatter + pickle."""
    df, params = collect_2e_length_points(tables)
    amp_col = params["amp_col"]
    amp = df[amp_col].to_numpy(dtype=float)
    vel = df["peak_velocity_deg_per_ms"].to_numpy(dtype=float)
    length = df["length"].to_numpy(dtype=int)
    length_bin = df["length_bin"].to_numpy()
    counts = {lab: int((length_bin == lab).sum()) for lab, _ in LENGTH_BINS}

    slope = intercept = r = float("nan")
    if amp.size >= 3:
        slope, intercept, r, _, _ = stats.linregress(amp, vel)

    xmax = float(np.nanpercentile(amp, 99.8)) if amp.size else 1.0
    ymax = float(np.nanpercentile(vel, 99.8)) if vel.size else 0.5
    xlim = (0.0, max(xmax * 1.05, 1.0))
    ylim = (0.0, max(ymax * 1.08, 0.1))

    payload = {
        "params": params,
        "amp": amp.astype(np.float32),
        "vel": vel.astype(np.float32),
        "length": length.astype(np.int16),
        "length_bin": np.asarray(length_bin, dtype="U8"),
        "animal": df["animal"].astype(str).to_numpy() if "animal" in df.columns else np.array([], dtype=object),
        "counts": counts,
        "xlim": xlim,
        "ylim": ylim,
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "length_bins": [lab for lab, _ in LENGTH_BINS],
        "length_colors": dict(LENGTH_COLORS),
        "length_draw_order": list(LENGTH_DRAW_ORDER),
    }

    bundle = begin_plot_bundle(
        out_dir,
        PLOT_ID,
        kind=PLOT_ID,
        tables=tables,
        logic_key=PLOT_ID,
        params=params,
        extra={"n": int(amp.size), "counts": counts, "slope": float(slope), "pearson_r": float(r)},
    )
    pkl = bundle.metadata_dir / "diagnostics_2e.pkl"
    write_pickle_with_meta(
        payload,
        pkl,
        meta={
            "figure": PLOT_ID,
            "params": params,
            "n": int(amp.size),
            "counts": counts,
            "slope": float(slope),
            "pearson_r": float(r),
        },
        entrypoint="eye_tracking_system_tools.analysis.diagnostics_2e.export_diagnostics_2e",
    )
    pd.DataFrame(
        [{"length_bin": lab, "n": counts[lab]} for lab, _ in LENGTH_BINS]
    ).to_csv(bundle.metadata_dir / "length_bin_counts.csv", index=False)

    fig = _draw_length_scatter(
        amp,
        vel,
        length_bin,
        params=params,
        xlim=xlim,
        ylim=ylim,
        counts=counts,
        slope=float(slope) if np.isfinite(slope) else None,
        intercept=float(intercept) if np.isfinite(intercept) else None,
    )
    pdf = bundle.plots_dir / PDF_NAME
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    finish_plot_bundle(bundle)
    return {PDF_NAME: pdf, pkl.name: pkl}
