"""Diagnostic Fig 2e scatter: peak V vs amplitude, length / class variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from eye_tracking_system_tools.analysis.colors import build_color_map
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.figures_2c_2e import (
    _annotate_linear_fit,
    _event_ids,
    _main_sequence_stats,
    _peak_velocity_deg_per_ms,
    _plot_2e_scatter,
    _plot_per_animal_means,
    _pool_pogona,
    _serialize_color_map,
    _subset_by_event_ids,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables, _row_block_key
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PLOT_ID = "diagnostics_2e"
OMISSION_PLOT_ID = "omission_trial"
PDF_LENGTH = "figure_2e_scatter_by_length.pdf"
PDF_GRAY_ALL = "2e_all_animals_gray.pdf"
PDF_GRAY_ANIMAL = "2e_per_animal_gray.pdf"
PDF_MONO_CONC_ALL = "fig_2e_mono_conc_all.pdf"
GRAY_COLOR = "0.45"
GRAY_ALPHA = 0.5
GRAY_SIZE = 6
# Drop detector length ≤ this value in omission_trial (length 1–2 = short-event ridge).
OMISSION_MAX_LENGTH = 2

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


def on_length_identity_mask(
    df: pd.DataFrame,
    *,
    amp_col: str,
    vel_col: str = "peak_velocity_deg_per_ms",
    length_col: str = "length",
    frame_ms: float,
    rel_tol: float = 0.01,
    abs_tol: float = 0.0,
) -> pd.Series:
    """
    Boolean mask (index-aligned): True if the event sits on its length identity.

    For detector length L (inter-frame intervals):

        V_id = amplitude / (L * frame_ms)

    On-line when ``|V - V_id| <= max(abs_tol, rel_tol * V_id)``.
    Invalid rows (non-finite / length < 1 / amp ≤ 0) are False.
    """
    amp = pd.to_numeric(df[amp_col], errors="coerce")
    vel = pd.to_numeric(df[vel_col], errors="coerce")
    length = pd.to_numeric(df[length_col], errors="coerce")
    ok = amp.notna() & vel.notna() & length.notna() & (length >= 1) & (amp > 0)
    v_id = amp / (length * float(frame_ms))
    tol = np.maximum(float(abs_tol), float(rel_tol) * v_id.to_numpy(dtype=float))
    on = ok.to_numpy(dtype=bool) & (np.abs(vel.to_numpy(dtype=float) - v_id.to_numpy(dtype=float)) <= tol)
    return pd.Series(on, index=df.index, name="on_length_identity")


def length_identity_summary(
    df: pd.DataFrame,
    *,
    amp_col: str,
    vel_col: str = "peak_velocity_deg_per_ms",
    length_col: str = "length",
    frame_ms: float,
    rel_tol: float = 0.05,
    abs_tol: float = 0.002,
) -> pd.DataFrame:
    """
    Per detector-length cohort: how many events sit on the duration identity.

    Identity for an event of detector ``length`` L (inter-frame intervals):

        V_id = amplitude / (L * frame_ms)

    An event is "on" the line when ``|V - V_id| <= max(abs_tol, rel_tol * V_id)``.
    Rows use the same length bins as the diagnostic scatter; ``6–8`` / ``9+`` still
    use each event's own L for the identity check.
    """
    on_line = on_length_identity_mask(
        df,
        amp_col=amp_col,
        vel_col=vel_col,
        length_col=length_col,
        frame_ms=frame_ms,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ).to_numpy(dtype=bool)
    length = pd.to_numeric(df[length_col], errors="coerce").to_numpy(dtype=float)
    amp = pd.to_numeric(df[amp_col], errors="coerce").to_numpy(dtype=float)
    vel = pd.to_numeric(df[vel_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(amp) & np.isfinite(vel) & np.isfinite(length) & (length >= 1) & (amp > 0)
    length_ok = length[ok]
    on_ok = on_line[ok]
    n_all = int(ok.sum())

    rows: list[dict[str, Any]] = []
    for label, (lo, hi) in LENGTH_BINS:
        mask = (length_ok >= lo) & (length_ok <= hi)
        n = int(mask.sum())
        n_on = int(on_ok[mask].sum()) if n else 0
        rows.append(
            {
                "length_bin": label,
                "length_lo": int(lo),
                "length_hi": int(hi) if hi < 10_000 else None,
                "identity_slope_deg_per_ms": (
                    1.0 / (float(lo) * float(frame_ms)) if lo == hi else float("nan")
                ),
                "n": n,
                "n_on_identity": n_on,
                "pct_of_bin": (100.0 * n_on / n) if n else float("nan"),
                "pct_of_all": (100.0 * n_on / n_all) if n_all else float("nan"),
                "pct_bin_of_all": (100.0 * n / n_all) if n_all else float("nan"),
            }
        )
    total_on = int(on_ok.sum())
    rows.append(
        {
            "length_bin": "ALL",
            "length_lo": None,
            "length_hi": None,
            "identity_slope_deg_per_ms": float("nan"),
            "n": n_all,
            "n_on_identity": total_on,
            "pct_of_bin": (100.0 * total_on / n_all) if n_all else float("nan"),
            "pct_of_all": (100.0 * total_on / n_all) if n_all else float("nan"),
            "pct_bin_of_all": 100.0 if n_all else float("nan"),
        }
    )
    return pd.DataFrame(rows)


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


def collect_2e_length_points(
    tables: EventTables,
    *,
    omit_length_le: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pogona (or current cohort) events after the same amp/V filters as Fig 2e."""
    ms = dict((tables.params or {}).get("main_sequence", {}))
    sacc = dict((tables.params or {}).get("saccade", {}))
    amp_col = str(ms.get("amp_col", "net_angular_disp"))
    min_amp = float(ms.get("min_amp_deg", 0.5))
    max_amp_pct = float(ms.get("max_amp_pct", 99.5))
    frame_rate = float(ms.get("frame_rate_fps", 60.0))
    speed_thr = float(sacc.get("speed_threshold_deg_per_frame", 0.8))
    frame_ms = 1000.0 / frame_rate
    bin_width = float(ms.get("bin_width_deg", 5.0))
    min_events = int(ms.get("min_events_per_bin", 15))

    df = tables.all_saccades.copy()
    if df.empty:
        raise ValueError("No saccades for 2e length diagnostic")
    df["peak_velocity_deg_per_ms"] = _peak_velocity_deg_per_ms(df, frame_rate)
    # Alias used by Fig 2e main-sequence stats (already in deg/ms).
    df["peak_velocity"] = df["peak_velocity_deg_per_ms"]
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
    if omit_length_le is not None:
        df = df[df["length"] > int(omit_length_le)].copy()
    df["length_bin"] = _length_label(df["length"].to_numpy())

    conc = _subset_by_event_ids(df, tables.synced)
    mono = _subset_by_event_ids(df, tables.non_synced)
    pairing = np.full(len(df), "other", dtype=object)
    if not df.empty:
        ids = _event_ids(df)
        if not conc.empty:
            pairing[ids.isin(set(_event_ids(conc))).to_numpy()] = "concurrent"
        if not mono.empty:
            pairing[ids.isin(set(_event_ids(mono))).to_numpy()] = "monocular"
    df["pairing"] = pairing

    params = {
        "amp_col": amp_col,
        "min_amp_deg": min_amp,
        "max_amp_pct": max_amp_pct,
        "frame_rate_fps": frame_rate,
        "frame_ms": frame_ms,
        "speed_threshold_deg_per_frame": speed_thr,
        "speed_threshold_deg_per_ms": speed_thr / frame_ms,
        "one_frame_slope": 1.0 / frame_ms,
        "bin_width_deg": bin_width,
        "min_events_per_bin": min_events,
        "cohort": cohort,
        "figsize": (5.2, 4.2),
        "dpi": 300,
        "lw": float(ms.get("lw", 1.0)),
        "means_figsize": (1.5, 1.7),
        "triptych_figsize": (5.0, 1.9),
        "gray_color": GRAY_COLOR,
        "gray_alpha": GRAY_ALPHA,
        "gray_size": GRAY_SIZE,
        "omit_length_le": omit_length_le,
    }
    return df, params


def attach_pre_event_amplitude(df: pd.DataFrame, tables: EventTables) -> pd.DataFrame:
    """
    Add ``net_angular_disp_pre``: net displacement from the frame before onset to offset.

    Uses the eye trace row immediately before ``saccade_start_ind`` (the same pre-onset
    sample that enters ``speed_profile_angular[0]``) and ``phi_end_pos`` / ``theta_end_pos``
    at event offset. Requires ``tables`` blocks to carry left/right traces.
    """
    from eye_tracking_system_tools.analysis.pipeline import _row_block_key

    out = df.copy()
    amp_pre = np.full(len(out), np.nan, dtype=float)
    required = {"saccade_start_ind", "phi_end_pos", "theta_end_pos", "animal", "block", "eye"}
    if not required.issubset(out.columns):
        out["net_angular_disp_pre"] = amp_pre
        return out

    pos_map = {idx: i for i, idx in enumerate(out.index)}
    for (_animal, _block, _eye), grp in out.groupby(["animal", "block", "eye"], sort=False):
        key = _row_block_key(_animal, _block)
        bundle = tables.block_dict.get(key)
        if bundle is None:
            continue
        eye_u = str(_eye).upper()
        if eye_u.startswith("L"):
            trace = bundle.left
        elif eye_u.startswith("R"):
            trace = bundle.right
        else:
            continue
        if trace is None or trace.empty or not {"k_phi", "k_theta"}.issubset(trace.columns):
            continue
        for row_idx, row in grp.iterrows():
            try:
                start = int(row["saccade_start_ind"])
                loc = trace.index.get_loc(start)
                if isinstance(loc, slice):
                    loc = int(loc.start)
                elif isinstance(loc, np.ndarray):
                    loc = int(np.asarray(loc).flat[0])
                else:
                    loc = int(loc)
            except (KeyError, TypeError, ValueError):
                continue
            if loc < 1:
                continue
            pre = trace.iloc[loc - 1]
            phi_pre = float(pre["k_phi"])
            theta_pre = float(pre["k_theta"])
            phi_end = float(row["phi_end_pos"])
            theta_end = float(row["theta_end_pos"])
            if not all(np.isfinite([phi_pre, theta_pre, phi_end, theta_end])):
                continue
            amp_pre[pos_map[row_idx]] = float(np.hypot(phi_end - phi_pre, theta_end - theta_pre))

    out["net_angular_disp_pre"] = amp_pre
    return out


def attach_s13_amplitude(
    df: pd.DataFrame,
    tables: EventTables,
    *,
    frame_rate_fps: float = 60.0,
) -> pd.DataFrame:
    """Onset→offset A, except length-1 events which use pre-onset→offset A.

    Length 1 is one inter-frame interval (two samples). Peak V can be the step
    *into* the event (frame before onset → onset). Onset→offset A then misses
    that step. Longer events have interior samples, so peak V and onset→offset
    A already describe the same movement.
    """
    out = attach_pre_event_amplitude(df, tables)
    frame_ms = 1000.0 / float(frame_rate_fps)
    length = _event_length(out, frame_ms=frame_ms)
    out["length"] = length
    onset = pd.to_numeric(out.get("net_angular_disp"), errors="coerce").to_numpy(float)
    pre = pd.to_numeric(out["net_angular_disp_pre"], errors="coerce").to_numpy(float)
    mixed = onset.copy()
    is_len1 = np.isfinite(length) & (np.round(length) == 1)
    use_pre = is_len1 & np.isfinite(pre)
    mixed[use_pre] = pre[use_pre]
    out["amp_s13"] = mixed
    out["amp_s13_used_preonset"] = use_pre
    return out


def _export_animal_color_legend(
    color_map: dict[str, Any],
    animals: list[str],
    out_pdf: Path,
    *,
    show: bool = False,
) -> Path:
    """Standalone animal-color legend for Illustrator."""
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=color_map.get(a, "0.3"),
            marker="o",
            ms=5,
            lw=1.4,
            label=str(a),
        )
        for a in animals
    ]
    fig = plt.figure(figsize=(1.8, 0.32 * max(1, len(animals)) + 0.35), dpi=150)
    fig.legend(handles, animals, loc="center", frameon=False, fontsize=8)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    return out_pdf


def _overlay_guides(
    ax,
    *,
    params: dict[str, Any],
    xlim,
    slope: float | None = None,
    intercept: float | None = None,
) -> None:
    """Detection floor + OLS only (no 1-frame identity line)."""
    thr = float(params["speed_threshold_deg_per_ms"])
    ax.axhline(thr, color="0.15", lw=1.2, ls=":", label=f"detect floor  {thr:.3f} deg/ms")
    x0, x1 = float(xlim[0]), float(xlim[1])
    xs = np.linspace(max(x0, 0.0), x1, 80)
    if slope is not None and intercept is not None and np.isfinite(slope) and np.isfinite(intercept):
        ax.plot(xs, slope * xs + intercept, "k--", lw=1.2, label=f"OLS  slope={slope:.3f}")


def _style_2e_axes(ax, xlim, ylim) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Amplitude [deg]", fontsize=11)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    title_suffix: str = "",
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
    _overlay_guides(ax, params=params, xlim=xlim, slope=slope, intercept=intercept)
    _style_2e_axes(ax, xlim, ylim)
    cohort = params.get("cohort", "all")
    ax.set_title(f"all animals ({cohort}) n={n}  colored by length{title_suffix}", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.6)
    fig.tight_layout()
    return fig


def _draw_gray_scatter(
    amp: np.ndarray,
    vel: np.ndarray,
    *,
    params: dict[str, Any],
    xlim,
    ylim,
    title: str,
    slope: float | None = None,
    intercept: float | None = None,
) -> Any:
    figsize = tuple(params.get("figsize", (5.2, 4.2)))
    dpi = int(params.get("dpi", 300))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if amp.size:
        ax.scatter(
            amp,
            vel,
            s=float(params.get("gray_size", GRAY_SIZE)),
            alpha=float(params.get("gray_alpha", GRAY_ALPHA)),
            color=str(params.get("gray_color", GRAY_COLOR)),
            edgecolors="none",
            rasterized=True,
        )
    _overlay_guides(ax, params=params, xlim=xlim, slope=slope, intercept=intercept)
    _style_2e_axes(ax, xlim, ylim)
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def _animal_order(animals: np.ndarray) -> list[str]:
    return list(dict.fromkeys(str(a) for a in animals if str(a) and str(a) != "nan"))


def _fit_ols(amp: np.ndarray, vel: np.ndarray) -> tuple[float, float, float]:
    if amp.size < 3:
        return float("nan"), float("nan"), float("nan")
    slope, intercept, r, _, _ = stats.linregress(amp, vel)
    return float(slope), float(intercept), float(r)


def _save_per_animal_gray_pdf(
    path: Path,
    amp: np.ndarray,
    vel: np.ndarray,
    animals: np.ndarray,
    *,
    params: dict[str, Any],
    xlim,
    ylim,
) -> list[str]:
    order = _animal_order(animals)
    labels = np.asarray(animals, dtype=str)
    with PdfPages(path) as pdf:
        if not order:
            fig = _draw_gray_scatter(
                np.array([], dtype=float),
                np.array([], dtype=float),
                params=params,
                xlim=xlim,
                ylim=ylim,
                title="(no animals)",
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return order
        for animal in order:
            mask = labels == animal
            xa, ya = amp[mask], vel[mask]
            slope, intercept, _ = _fit_ols(xa, ya)
            fig = _draw_gray_scatter(
                xa,
                ya,
                params=params,
                xlim=xlim,
                ylim=ylim,
                title=f"{animal} n={int(xa.size)}",
                slope=slope if np.isfinite(slope) else None,
                intercept=intercept if np.isfinite(intercept) else None,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return order


def _draw_means_on_ax(
    ax,
    per_animal_stats: dict[str, pd.DataFrame],
    global_fit: dict[str, Any],
    *,
    color_map: dict[str, Any],
    animal_order: list[str],
    lw: float,
    xlim,
    ylim,
    title: str | None = None,
) -> None:
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
            lw=lw,
            label=animal,
        )
    slope = global_fit.get("slope")
    intercept = global_fit.get("intercept")
    if (
        slope is not None
        and intercept is not None
        and np.isfinite(slope)
        and np.isfinite(intercept)
        and xlim is not None
    ):
        xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
        ax.plot(xs, float(slope) * xs + float(intercept), "k--", lw=lw)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xlabel("Amplitude [deg]", fontsize=8)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _annotate_linear_fit(ax, global_fit, fontsize=6)


def _class_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all": df,
        "concurrent": df.loc[df["pairing"] == "concurrent"].copy() if "pairing" in df.columns else df.iloc[0:0].copy(),
        "monocular": df.loc[df["pairing"] == "monocular"].copy() if "pairing" in df.columns else df.iloc[0:0].copy(),
    }


def _build_main_sequence_classes(
    df: pd.DataFrame,
    *,
    params: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any], Any, Any]:
    amp_col = params["amp_col"]
    bin_width = float(params["bin_width_deg"])
    min_events = int(params["min_events_per_bin"])
    amp = df[amp_col].to_numpy(dtype=float) if not df.empty else np.array([], dtype=float)
    amp_hi = float(np.nanmax(amp)) if amp.size else 25.0
    edges = np.arange(0.0, max(25.0, np.ceil(amp_hi / bin_width) * bin_width) + 1e-9, bin_width)

    class_stats: dict[str, dict[str, Any]] = {}
    for label, sub in _class_frames(df).items():
        per_animal, _, gfit = _main_sequence_stats(
            sub, amp_col=amp_col, edges=edges, min_events=min_events
        )
        class_stats[label] = {
            "per_animal_stats": per_animal,
            "global_fit": gfit,
            "n": int(len(sub)),
        }
    animals = list(class_stats["all"]["per_animal_stats"].keys())
    color_map = build_color_map(animals, template="okabeito", order=animals)

    # Shared limits from the "all" panel (same convention as Fig 2e).
    means_params = {
        "figsize": tuple(params.get("means_figsize", (1.5, 1.7))),
        "dpi": int(params.get("dpi", 300)),
        "lw": float(params.get("lw", 1.0)),
    }
    fig0, ax0 = plt.subplots(figsize=means_params["figsize"], dpi=means_params["dpi"])
    _draw_means_on_ax(
        ax0,
        class_stats["all"]["per_animal_stats"],
        class_stats["all"]["global_fit"],
        color_map=color_map,
        animal_order=animals,
        lw=means_params["lw"],
        xlim=None,
        ylim=None,
    )
    shared_xlim = ax0.get_xlim()
    shared_ylim = ax0.get_ylim()
    plt.close(fig0)
    return class_stats, animals, color_map, shared_xlim, shared_ylim


def _draw_mono_conc_all(
    class_stats: dict[str, dict[str, Any]],
    *,
    animals: list[str],
    color_map: dict[str, Any],
    params: dict[str, Any],
    xlim,
    ylim,
) -> Any:
    dpi = int(params.get("dpi", 300))
    lw = float(params.get("lw", 1.0))
    figsize = tuple(params.get("triptych_figsize", (5.0, 1.9)))
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharex=True, sharey=True)
    for ax, label in zip(axes, ("all", "concurrent", "monocular")):
        body = class_stats[label]
        _draw_means_on_ax(
            ax,
            body["per_animal_stats"],
            body["global_fit"],
            color_map=color_map,
            animal_order=animals,
            lw=lw,
            xlim=xlim,
            ylim=ylim,
            title=f"{label} n={body['n']}",
        )
    fig.tight_layout()
    return fig


def _write_diagnostics_bundle(
    tables: EventTables,
    out_dir: Path,
    *,
    plot_id: str,
    logic_key: str,
    omit_length_le: int | None = None,
    show: bool = False,
) -> dict[str, Path]:
    df, params = collect_2e_length_points(tables, omit_length_le=omit_length_le)
    amp_col = params["amp_col"]
    amp = df[amp_col].to_numpy(dtype=float)
    vel = df["peak_velocity_deg_per_ms"].to_numpy(dtype=float)
    length = df["length"].to_numpy(dtype=int) if not df.empty else np.array([], dtype=int)
    length_bin = df["length_bin"].to_numpy() if not df.empty else np.array([], dtype=object)
    animals = (
        df["animal"].astype(str).to_numpy()
        if "animal" in df.columns and not df.empty
        else np.array([], dtype=object)
    )
    pairing = (
        df["pairing"].astype(str).to_numpy()
        if "pairing" in df.columns and not df.empty
        else np.array([], dtype=object)
    )
    counts = {lab: int((length_bin == lab).sum()) for lab, _ in LENGTH_BINS}

    slope, intercept, r = _fit_ols(amp, vel)

    xmax = float(np.nanpercentile(amp, 99.8)) if amp.size else 1.0
    ymax = float(np.nanpercentile(vel, 99.8)) if vel.size else 0.5
    xlim = (0.0, max(xmax * 1.05, 1.0))
    ylim = (0.0, max(ymax * 1.08, 0.1))

    class_stats, animal_order_ms, color_map, means_xlim, means_ylim = _build_main_sequence_classes(
        df, params=params
    )
    title_suffix = ""
    if omit_length_le is not None:
        title_suffix = f"  (omit length≤{int(omit_length_le)})"

    payload = {
        "params": params,
        "amp": amp.astype(np.float32),
        "vel": vel.astype(np.float32),
        "length": length.astype(np.int16),
        "length_bin": np.asarray(length_bin, dtype="U8"),
        "animal": np.asarray(animals, dtype="U16"),
        "pairing": np.asarray(pairing, dtype="U12"),
        "counts": counts,
        "xlim": xlim,
        "ylim": ylim,
        "means_xlim": means_xlim,
        "means_ylim": means_ylim,
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "length_bins": [lab for lab, _ in LENGTH_BINS],
        "length_colors": dict(LENGTH_COLORS),
        "length_draw_order": list(LENGTH_DRAW_ORDER),
        "animal_order": _animal_order(animals),
        "classes": {
            k: {
                "n": v["n"],
                "global_fit": v["global_fit"],
                "per_animal_stats": {
                    a: sdf.to_dict(orient="list") if hasattr(sdf, "to_dict") else {}
                    for a, sdf in (v["per_animal_stats"] or {}).items()
                },
            }
            for k, v in class_stats.items()
        },
        "color_map": _serialize_color_map(color_map),
        "animal_order_means": animal_order_ms,
    }

    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="diagnostics_2e",
        tables=tables,
        logic_key=logic_key,
        params=params,
        extra={
            "n": int(amp.size),
            "counts": counts,
            "slope": float(slope),
            "pearson_r": float(r),
            "omit_length_le": omit_length_le,
            "class_n": {k: v["n"] for k, v in class_stats.items()},
        },
    )
    pkl = bundle.metadata_dir / "diagnostics_2e.pkl"
    write_pickle_with_meta(
        payload,
        pkl,
        meta={
            "figure": plot_id,
            "params": params,
            "n": int(amp.size),
            "counts": counts,
            "slope": float(slope),
            "pearson_r": float(r),
            "omit_length_le": omit_length_le,
        },
        entrypoint="eye_tracking_system_tools.analysis.diagnostics_2e.export_diagnostics_2e",
    )
    pd.DataFrame(
        [{"length_bin": lab, "n": counts[lab]} for lab, _ in LENGTH_BINS]
    ).to_csv(bundle.metadata_dir / "length_bin_counts.csv", index=False)
    pd.DataFrame(
        [{"class": k, "n": v["n"]} for k, v in class_stats.items()]
    ).to_csv(bundle.metadata_dir / "class_counts.csv", index=False)

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
        title_suffix=title_suffix,
    )
    pdf = bundle.plots_dir / PDF_LENGTH
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)

    gray_all = bundle.plots_dir / PDF_GRAY_ALL
    fig_gray = _draw_gray_scatter(
        amp,
        vel,
        params=params,
        xlim=xlim,
        ylim=ylim,
        title=f"all animals ({params.get('cohort', 'all')}) n={int(amp.size)}{title_suffix}",
        slope=float(slope) if np.isfinite(slope) else None,
        intercept=float(intercept) if np.isfinite(intercept) else None,
    )
    fig_gray.savefig(gray_all, format="pdf", bbox_inches="tight")
    plt.close(fig_gray)

    gray_animal = bundle.plots_dir / PDF_GRAY_ANIMAL
    _save_per_animal_gray_pdf(
        gray_animal,
        amp,
        vel,
        animals,
        params=params,
        xlim=xlim,
        ylim=ylim,
    )

    triptych = bundle.plots_dir / PDF_MONO_CONC_ALL
    fig_t = _draw_mono_conc_all(
        class_stats,
        animals=animal_order_ms,
        color_map=color_map,
        params=params,
        xlim=means_xlim,
        ylim=means_ylim,
    )
    fig_t.savefig(triptych, format="pdf", bbox_inches="tight")
    plt.close(fig_t)

    finish_plot_bundle(bundle)
    return {
        PDF_LENGTH: pdf,
        PDF_GRAY_ALL: gray_all,
        PDF_GRAY_ANIMAL: gray_animal,
        PDF_MONO_CONC_ALL: triptych,
        pkl.name: pkl,
        "_bundle_dir": bundle.bundle_dir,
    }


def export_diagnostics_2e(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
) -> dict[str, Path]:
    """Write ``diagnostics_2e/`` plus nested ``omission_trial/`` (length > 2)."""
    written = _write_diagnostics_bundle(
        tables,
        out_dir,
        plot_id=PLOT_ID,
        logic_key=PLOT_ID,
        omit_length_le=None,
        show=show,
    )
    bundle_dir = written.pop("_bundle_dir")
    omitted = _write_diagnostics_bundle(
        tables,
        bundle_dir,
        plot_id=OMISSION_PLOT_ID,
        logic_key=OMISSION_PLOT_ID,
        omit_length_le=OMISSION_MAX_LENGTH,
        show=False,
    )
    omitted.pop("_bundle_dir", None)
    for name, path in omitted.items():
        written[f"omission_trial/{name}"] = path
    return written


S13_PLOT_ID = "s13_preonset_2e"


def export_s13_preonset_main_sequence(
    tables: EventTables,
    out_dir: Path | str,
    *,
    show: bool = False,
) -> dict[str, Path]:
    """Paper-style per-animal means + mixed-A scatter + paired slope test.

    Amplitude is onset→offset except for detector length-1 events, which use
    pre-onset→offset. Peak V is unchanged. Identity points are kept.
    """
    from eye_tracking_system_tools.analysis.event_cache import ensure_traces_for_blocks

    ms = dict(tables.params.get("main_sequence", {}))
    amp_col = "amp_s13"
    bin_width = float(ms.get("bin_width_deg", 5.0))
    min_amp = float(ms.get("min_amp_deg", 0.5))
    max_amp_pct = float(ms.get("max_amp_pct", 99.5))
    min_events = int(ms.get("min_events_per_bin", 15))
    frame_rate = float(ms.get("frame_rate_fps", 60.0))

    df0 = tables.all_saccades.copy()
    if df0.empty:
        raise ValueError("No saccades for S13 export")
    keys = sorted(
        {
            _row_block_key(a, b)
            for a, b in zip(df0["animal"], df0["block"])
        }
    )
    tables_tr = ensure_traces_for_blocks(tables, keys)
    df = attach_s13_amplitude(df0, tables_tr, frame_rate_fps=frame_rate)
    n_len1 = int((np.round(df["length"].to_numpy(float)) == 1).sum())
    n_pre = int(df["amp_s13_used_preonset"].to_numpy(bool).sum())
    if n_len1 >= 100 and n_pre < 50:
        raise RuntimeError(
            f"pre-onset A finite for {n_pre} / {n_len1} length-1 events; "
            "lab traces are required (ensure_traces_for_blocks)."
        )
    df["peak_velocity"] = _peak_velocity_deg_per_ms(df, frame_rate)
    df = df[np.isfinite(df[amp_col]) & np.isfinite(df["peak_velocity"]) & (df[amp_col] >= min_amp)]
    amp_hi = float(np.nanpercentile(df[amp_col], max_amp_pct))
    df = df[df[amp_col] <= amp_hi].copy()

    edges = np.arange(0.0, max(25.0, np.ceil(amp_hi / bin_width) * bin_width) + 1e-9, bin_width)
    concurrent_df = _subset_by_event_ids(df, tables.synced)
    monocular_df = _subset_by_event_ids(df, tables.non_synced)
    class_frames = {"all": df, "concurrent": concurrent_df, "monocular": monocular_df}
    class_stats: dict[str, dict[str, Any]] = {}
    for label, sub in class_frames.items():
        per_animal, linear_df, gfit = _main_sequence_stats(
            sub, amp_col=amp_col, edges=edges, min_events=min_events
        )
        class_stats[label] = {
            "per_animal_stats": per_animal,
            "linear_stats_df": linear_df,
            "global_fit": gfit,
            "n": int(len(sub)),
        }

    animals = list(class_stats["all"]["per_animal_stats"].keys())
    color_map = build_color_map(animals, template="okabeito", order=animals)
    params = {
        "amp_col": amp_col,
        "figsize": tuple(ms.get("figsize", (1.5, 1.7))),
        "dpi": int(ms.get("dpi", 300)),
        "lw": float(ms.get("lw", 1.0)),
        "density_figsize": (3.8, 3.2),
    }
    x_fit_span = (float(df[amp_col].min()), float(df[amp_col].max())) if not df.empty else None
    fig, ax = _plot_per_animal_means(
        class_stats["all"]["per_animal_stats"],
        class_stats["all"]["global_fit"],
        color_map=color_map,
        animal_order=animals,
        params=params,
        x_fit_span=x_fit_span,
        title=f"all n={class_stats['all']['n']}",
    )
    shared_xlim = ax.get_xlim()
    shared_ylim = ax.get_ylim()

    bundle = begin_plot_bundle(
        out_dir,
        S13_PLOT_ID,
        kind="s13_preonset_2e",
        logic_key="s13_preonset_2e",
        tables=tables,
        params={
            "amp_col": amp_col,
            "amp_definition": "onset_to_offset; length1_preonset_to_offset",
        },
    )
    pdf_all = bundle.plots_dir / "S13a_all.pdf"
    fig.savefig(pdf_all, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)

    pdfs: dict[str, Path] = {"S13a_all.pdf": pdf_all}
    for label, name in (
        ("concurrent", "S13b_concurrent.pdf"),
        ("monocular", "S13c_monocular.pdf"),
    ):
        body = class_stats[label]
        fig, _ = _plot_per_animal_means(
            body["per_animal_stats"],
            body["global_fit"],
            color_map=color_map,
            animal_order=animals,
            params=params,
            xlim=shared_xlim,
            ylim=shared_ylim,
            title=f"{label} n={body['n']}",
        )
        out = bundle.plots_dir / name
        fig.savefig(out, format="pdf", bbox_inches="tight")
        show_and_close(fig, show)
        pdfs[name] = out

    pooled_df, pool_cohort = _pool_pogona(df)
    pooled_amp = pooled_df[amp_col].to_numpy(float) if not pooled_df.empty else np.array([], dtype=float)
    pooled_vel = pooled_df["peak_velocity"].to_numpy(float) if not pooled_df.empty else np.array([], dtype=float)
    scatter_pdf = bundle.plots_dir / "S13d_preonset_scatter.pdf"
    _plot_2e_scatter(
        pooled_amp,
        pooled_vel,
        fit=class_stats["all"]["global_fit"],
        params=params,
        xlim=shared_xlim,
        ylim=shared_ylim,
        title=f"mixed A  all animals ({pool_cohort}) n={int(pooled_amp.size)}",
        out_pdf=scatter_pdf,
        show=show,
        figsize=params["density_figsize"],
        labelsize=11,
        ticksize=9,
        annot_size=9,
        title_size=11,
    )
    pdfs["S13d_preonset_scatter.pdf"] = scatter_pdf

    conc_lin = class_stats["concurrent"]["linear_stats_df"].set_index("animal")
    mono_lin = class_stats["monocular"]["linear_stats_df"].set_index("animal")
    shared = sorted(set(conc_lin.index) & set(mono_lin.index))
    conc_s = conc_lin.loc[shared, "slope"].to_numpy(float)
    mono_s = mono_lin.loc[shared, "slope"].to_numpy(float)
    n_an = int(len(shared))
    if n_an >= 2:
        t_stat, p_val = stats.ttest_rel(conc_s, mono_s)
    else:
        t_stat, p_val = float("nan"), float("nan")
    slope_summary = {
        "amp_definition": "onset_to_offset; length1_preonset_to_offset",
        "n_animals": n_an,
        "concurrent_mean_slope": float(np.mean(conc_s)) if n_an else float("nan"),
        "concurrent_sem_slope": float(np.std(conc_s, ddof=1) / np.sqrt(n_an)) if n_an > 1 else float("nan"),
        "monocular_mean_slope": float(np.mean(mono_s)) if n_an else float("nan"),
        "monocular_sem_slope": float(np.std(mono_s, ddof=1) / np.sqrt(n_an)) if n_an > 1 else float("nan"),
        "paired_t": float(t_stat),
        "paired_p": float(p_val),
        "df": n_an - 1,
        "slope_units": "deg_per_ms_per_deg",
        "animals": shared,
        "concurrent_slopes": [float(x) for x in conc_s],
        "monocular_slopes": [float(x) for x in mono_s],
        "n_events_all": class_stats["all"]["n"],
        "n_events_concurrent": class_stats["concurrent"]["n"],
        "n_events_monocular": class_stats["monocular"]["n"],
        "n_length1": n_len1,
        "n_length1_used_preonset": n_pre,
    }
    import yaml

    yaml_path = bundle.metadata_dir / "slope_comparison.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(slope_summary, f, sort_keys=False)

    fig_s, ax_s = plt.subplots(figsize=(4.2, 2.4), dpi=150)
    x = np.arange(n_an)
    w = 0.35
    ax_s.bar(x - w / 2, conc_s, w, color="#0072B2", edgecolor="0.2", label="concurrent")
    ax_s.bar(x + w / 2, mono_s, w, color="#D55E00", edgecolor="0.2", label="monocular")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(shared, rotation=30, ha="right", fontsize=7)
    ax_s.set_ylabel("OLS slope (°/ms per °)")
    ax_s.legend(fontsize=7, frameon=False)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    p_txt = f"p={p_val:.3g}" if np.isfinite(p_val) else "p=nan"
    ax_s.set_title(
        f"paired t={t_stat:.2f}  {p_txt}  df={n_an - 1}" if n_an else "no animals",
        fontsize=8,
    )
    fig_s.tight_layout()
    test_pdf = bundle.plots_dir / "S13e_slope_test.pdf"
    fig_s.savefig(test_pdf, format="pdf", bbox_inches="tight")
    show_and_close(fig_s, show)
    pdfs["S13e_slope_test.pdf"] = test_pdf

    legend_pdf = bundle.plots_dir / "S13f_animal_legend.pdf"
    _export_animal_color_legend(color_map, animals, legend_pdf, show=show)
    pdfs["S13f_animal_legend.pdf"] = legend_pdf

    for label, body in class_stats.items():
        suffix = "" if label == "all" else f"_{label}"
        body["linear_stats_df"].to_csv(
            bundle.metadata_dir / f"amplitude_velocity_linear_stats{suffix}.csv",
            index=False,
        )

    pkl = bundle.metadata_dir / "s13_preonset_2e.pkl"
    write_pickle_with_meta(
        {
            "params": params,
            "amp_col": amp_col,
            "xlim": (float(shared_xlim[0]), float(shared_xlim[1])),
            "ylim": (float(shared_ylim[0]), float(shared_ylim[1])),
            "color_map": _serialize_color_map(color_map),
            "animal_order": animals,
            "classes": {
                k: {
                    "n": v["n"],
                    "global_fit": v["global_fit"],
                    "linear_stats_df": v["linear_stats_df"],
                    "per_animal_stats": v["per_animal_stats"],
                }
                for k, v in class_stats.items()
            },
            "slope_summary": slope_summary,
            "pooled": {"amp": pooled_amp.astype(np.float32), "vel": pooled_vel.astype(np.float32)},
        },
        pkl,
        meta={"amp_col": amp_col, "paired_p": slope_summary["paired_p"], "n_length1": n_len1},
        entrypoint="eye_tracking_system_tools.analysis.diagnostics_2e.export_s13_preonset_main_sequence",
    )
    finish_plot_bundle(bundle)
    pdfs["s13_preonset_2e.pkl"] = pkl
    pdfs["_bundle_dir"] = bundle.bundle_dir
    return pdfs
