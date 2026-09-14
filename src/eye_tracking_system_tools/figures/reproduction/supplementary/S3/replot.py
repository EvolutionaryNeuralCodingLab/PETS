#!/usr/bin/env python3
"""Redraw PDFs from this folder's metadata/ into plots/replot/. No PETS install required."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

ROOT = Path(__file__).resolve().parent
META = ROOT / "metadata"
PLOTS = ROOT / "plots"
OUT = PLOTS if "--overwrite" in sys.argv else (PLOTS / "replot")
KIND = "figure_s3"
TRACE_L = "#1f77b4"
TRACE_R = "#8c564b"


def _pickle_arg():
    for i, a in enumerate(sys.argv):
        if a == "--pickle" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--pickle="):
            return a.split("=", 1)[1]
    return None


def _load_pickle():
    want = _pickle_arg()
    if want:
        p = Path(want)
        if not p.is_file():
            p = META / want
        if p.is_file():
            with open(p, "rb") as f:
                return pickle.load(f), p
        return None, None
    for p in sorted(META.glob("*.pickle")) + sorted(META.glob("*.pkl")):
        with open(p, "rb") as f:
            return pickle.load(f), p
    return None, None


def _savefig(fig, name: str) -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _style_spines(ax, tick_size=7):
    ax.tick_params(axis="both", labelsize=tick_size, direction="out", colors="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _turbo_white0():
    import matplotlib.colors as mcolors

    turbo = plt.get_cmap("turbo", 256)
    colors = turbo(np.linspace(0, 1, 256))
    colors[0] = np.array([1, 1, 1, 1])
    return mcolors.ListedColormap(colors)


def _draw_coupling_heatmap(ax, hist, rng, ticks, cmap, vmax, title=None):
    mesh = ax.pcolormesh(
        hist["xedges"], hist["yedges"], hist["norm_counts"].T,
        cmap=cmap, vmin=0, vmax=vmax if vmax > 0 else 1,
        shading="flat", antialiased=False, edgecolors="none", linewidth=0,
    )
    mesh.set_rasterized(True)
    ax.set_xlim(*rng)
    ax.set_ylim(*rng)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.plot([rng[0], rng[1]], [rng[0], rng[1]], ls="--", color="gray", lw=1)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xlabel("Right max V [deg/ms]", fontsize=9)
    ax.set_ylabel("Left max V [deg/ms]", fontsize=9)
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _save_cbar(cmap, vmax, name, label="Probability"):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    _savefig(fig_cbar, name)


def _nice_axis_max(value, *, step=None):
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        return 0.5
    if step is None:
        step = 0.05 if x < 2.0 else 0.1
    return float(np.ceil((x / step) - 1e-12) * step)


def _ticks_for_span(lo, hi):
    lo, hi = float(lo), float(hi)
    if hi <= 0.3:
        step = 0.1
    elif hi <= 0.6:
        step = 0.25
    elif hi <= 1.5:
        step = 0.5
    else:
        step = 1.0
    ticks = []
    t = lo
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi - 1e-9:
        ticks.append(round(hi, 10))
    return ticks


def _hist2d(x, y, w, rng, bins):
    n_edge = max(int(bins), 2)
    edges = np.linspace(float(rng[0]), float(rng[1]), n_edge)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        z = np.zeros((n_edge - 1, n_edge - 1), dtype=float)
        return {"xedges": edges, "yedges": edges, "norm_counts": z}
    counts, xe, ye = np.histogram2d(x, y, bins=[edges, edges], weights=w)
    norm = counts / counts.sum() if counts.sum() > 0 else counts
    return {"xedges": xe, "yedges": ye, "norm_counts": norm}


def _rebin_figure_2f(data: dict) -> dict:
    """Rebuild macro/micro hists from stored speeds so axis-limit edits take effect."""
    right = np.asarray(data.get("right_eye_speeds", []), dtype=float)
    left = np.asarray(data.get("left_eye_speeds", []), dtype=float)
    if right.size == 0 or left.size == 0:
        return data
    try:
        from eye_tracking_system_tools.analysis.figures_2f_2h_2i import rebin_figure_2f_data

        return rebin_figure_2f_data(data)
    except Exception:
        pass
    auto = bool(data.get("auto_view_limits"))
    pct = float(data.get("macro_pct", 99.5))
    bins = int(data.get("bins", 60))
    weights = data.get("weights")
    data = dict(data)
    if auto:
        finite_r = right[np.isfinite(right)]
        finite_l = left[np.isfinite(left)]
        hi = _nice_axis_max(
            max(float(np.nanpercentile(finite_r, pct)), float(np.nanpercentile(finite_l, pct)))
        )
        frac = float(data.get("micro_frac_of_macro", 0.2))
        micro_hi = _nice_axis_max(hi * frac, step=0.05)
        data["macro_range"] = (0.0, hi)
        data["micro_range"] = (0.0, micro_hi)
        data["macro_tick_list"] = _ticks_for_span(0.0, hi)
        data["micro_tick_list"] = _ticks_for_span(0.0, micro_hi)
    macro_range = tuple(data["macro_range"])
    micro_range = tuple(data["micro_range"])
    data["macro"] = _hist2d(right, left, weights, macro_range, bins)
    data["micro"] = _hist2d(right, left, weights, micro_range, bins)
    data["vmax_all"] = float(
        max(
            np.nanmax(data["macro"]["norm_counts"]),
            np.nanmax(data["micro"]["norm_counts"]),
            1e-12,
        )
    )
    return data


def replot_figure_2f(data: dict) -> None:
    data = _rebin_figure_2f(data)
    cmap = _turbo_white0()
    vmax = float(data.get("vmax_all", 1.0) or 1.0)
    fig, axs = plt.subplots(1, 2, figsize=(3, 1.7), dpi=300, constrained_layout=True)
    for ax, key, rng, title, ticks in (
        (axs[0], "macro", tuple(data["macro_range"]), "Macro", data.get("macro_tick_list", [0, 0.25, 0.5])),
        (axs[1], "micro", tuple(data["micro_range"]), "Micro", data.get("micro_tick_list", [0, 0.05, 0.1])),
    ):
        _draw_coupling_heatmap(ax, data[key], rng, ticks, cmap, vmax, title=title)
    _savefig(fig, "figure_2f.pdf")
    _save_cbar(cmap, vmax, "figure_2f_colorbar.pdf")


def replot_figure_s3(data: dict) -> None:
    cmap = _turbo_white0()
    vmax = float(data.get("vmax", data.get("vmax_all", 1.0)) or 1.0)
    rng = tuple(data.get("range", data.get("macro_range", (0.0, 0.2))))
    ticks = data.get("ticks") or data.get("macro_tick_list") or [0.0, 0.1, 0.2]
    for name, hist in (
        ("figure_S3_head_still.pdf", data.get("hist_still") or data.get("still")),
        ("figure_S3_head_moving.pdf", data.get("hist_moving") or data.get("moving")),
    ):
        if not hist:
            continue
        fig, ax = plt.subplots(figsize=(1.7, 1.7), dpi=300)
        _draw_coupling_heatmap(ax, hist, rng, ticks, cmap, vmax)
        _savefig(fig, name)
    _save_cbar(cmap, vmax, "figure_S3_colorbar.pdf")


def replot_jitter_histogram(data: dict) -> None:
    import matplotlib.pyplot as plt

    values = np.asarray(data.get("values", []), dtype=float)
    values = values[np.isfinite(values)]
    n_bins = int(data.get("n_bins", 15))
    xmax = float(data.get("xmax") or (np.nanpercentile(values, 99.5) if values.size else 1.0))
    bins = np.linspace(0, max(xmax, 1e-6), n_bins + 1)
    fig, ax = plt.subplots(figsize=(2, 1.6), dpi=150)
    if values.size:
        counts, edges = np.histogram(values, bins=bins)
        y = 100.0 * counts / counts.sum() if counts.sum() else counts
        ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color="gray", edgecolor="black")
    ax.set_xlim(0, xmax)
    ax.set_xlabel(f"Displacement [{data.get('units', 'um')}] (eye plane)", fontsize=10)
    ax.set_ylabel("% frames", fontsize=10)
    _style_spines(ax, tick_size=8)
    name = str(data.get("pdf_name", "histogram.pdf"))
    _savefig(fig, name)


def replot_unified_jitter(data: dict) -> None:
    import matplotlib.pyplot as plt

    pools = data.get("pools") or {}
    raw_n = data.get("n_bins")
    n_bins = int(raw_n) if raw_n is not None else 15
    xmax = float(data.get("xmax") or 1.0)
    units = str(data.get("units", "um"))
    panels = data.get("panels") or [
        ["rigid", "rigid lizard", "#D55E00"],
        ["modular", "modular lizard", "#0072B2"],
        ["mouse", "modular mouse", "#009E73"],
        ["turtle", "modular turtle", "#CC79A7"],
    ]
    stored_edges = data.get("bin_edges") or {}
    bin_widths = data.get("bin_widths") or {}
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.8), dpi=150, sharex=True)
    for ax, panel in zip(axes.ravel(), panels):
        mount, label, color = panel[0], panel[1], panel[2]
        values = np.asarray(pools.get(mount, []), dtype=float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        if mount in stored_edges and stored_edges[mount] is not None:
            bins = np.asarray(stored_edges[mount], dtype=float)
        elif mount in bin_widths and bin_widths[mount]:
            width = float(bin_widths[mount])
            bins = np.arange(0.0, max(xmax, 1e-6), width)
            if bins.size == 0 or bins[-1] < xmax:
                bins = np.append(bins, xmax)
        else:
            bins = np.linspace(0, max(xmax, 1e-6), n_bins + 1)
        if values.size:
            counts, edges = np.histogram(values, bins=bins)
            y = 100.0 * counts / counts.sum() if counts.sum() else counts
            ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color=color, edgecolor="black", alpha=0.7)
            med = float(np.median(values))
            p95 = float(np.percentile(values, 95))
            ax.axvline(med, color="0.15", ls="-", lw=0.9, zorder=3)
            ax.axvline(p95, color="0.15", ls=":", lw=0.9, zorder=3)
            ax.text(
                0.98,
                0.96,
                f"median {med:.0f}\nP95 {p95:.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                color="0.15",
            )
        ax.set_xlim(0, xmax)
        ax.set_title(f"{label} (n={n})", fontsize=8)
        ax.set_ylabel("% frames", fontsize=10)
        _style_spines(ax, tick_size=8)
    ulabel = "µm" if units.lower() == "um" else units
    for ax in axes[1]:
        ax.set_xlabel(f"Displacement [{ulabel}] (eye plane)", fontsize=10)
    name = str(data.get("pdf_name", "unified_jitter_quantification.pdf"))
    _savefig(fig, name)


def replot_unified_noise(data: dict) -> None:
    import matplotlib.pyplot as plt

    pools = data.get("pools") or {}
    n_bins = int(data.get("n_bins", 60))
    xmax = float(data.get("xmax") or 1.0)
    xlabel = str(data.get("xlabel") or "Inter-frame Δangle [deg/frame]")
    ylabel = str(data.get("ylabel") or "% samples")
    panels = data.get("panels") or [
        ["rigid", "rigid lizard", "#D55E00"],
        ["modular", "modular lizard", "#0072B2"],
        ["turtle", "turtle", "#CC79A7"],
        ["mouse", "mouse", "#009E73"],
    ]
    bins = np.linspace(0, max(xmax, 1e-6), n_bins + 1)
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.8), dpi=150, sharex=True)
    for ax, panel in zip(axes.ravel(), panels):
        mount, label, color = panel[0], panel[1], panel[2]
        values = np.asarray(pools.get(mount, []), dtype=float)
        values = values[np.isfinite(values)]
        n = int(values.size)
        if values.size:
            counts, edges = np.histogram(values, bins=bins)
            y = 100.0 * counts / counts.sum() if counts.sum() else counts
            ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color=color, edgecolor="black", alpha=0.7)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{label} (n={n})", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(xlabel, fontsize=8)
    name = str(data.get("pdf_name", "unified_noise_measure.pdf"))
    _savefig(fig, name)


def replot_csv_lines() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    for csv in sorted(META.glob("*.csv")):
        df = pd.read_csv(csv)
        num = df.select_dtypes(include=["number"])
        if num.shape[1] < 2:
            continue
        x, y = num.iloc[:, 0], num.iloc[:, 1]
        fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
        ax.plot(x, y, color="#1f77b4", lw=1.2)
        ax.set_xlabel(num.columns[0], fontsize=8)
        ax.set_ylabel(num.columns[1], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _savefig(fig, csv.stem + ".pdf")


def replot_pos_vel(data: dict) -> None:
    import matplotlib.pyplot as plt

    t = np.asarray(data.get("t_grid", data.get("t_ms", [])), dtype=float)
    animals = data.get("animals") or data.get("curves") or {}
    params = data.get("params") or {}
    isolation = str(params.get("isolation") or "none").lower()
    series_lists = []
    if isinstance(animals, dict) and animals:
        first = next(iter(animals.values()))
        if isinstance(first, dict) and isinstance(first.get("series"), list):
            for name, body in animals.items():
                series_lists.append((str(name), list(body.get("series") or [])))
        elif "vel_center" in first if isinstance(first, dict) else False:
            series_lists = [(str(k), [v]) for k, v in animals.items()]
    if not series_lists and "vel_center" in data:
        series_lists = [("all", [data])]

    def _plot_key(key, ylabel, pdf_name):
        fig, ax = plt.subplots(figsize=(1.8, 1.8), dpi=300)
        for animal, series_list in series_lists:
            for series in series_list:
                y = np.asarray(series.get(key, []), dtype=float)
                tax = np.asarray(series.get("time_axis", t), dtype=float)
                if tax.size and y.size:
                    n = min(tax.size, y.size)
                    color = series.get("color_rgba")
                    ax.plot(tax[:n], y[:n], lw=1.0, color=color, label=series.get("label", animal))
        align_to = str(params.get("align_to", "peak"))
        ax.set_xlabel(f"Time from {align_to} (ms)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _style_spines(ax, tick_size=7)
        if t.size:
            ax.set_xlim(float(t[0]), float(t[-1]))
        _savefig(fig, pdf_name)

    vel_unit = str(params.get("velocity_unit", "deg/ms"))
    if str(vel_unit).lower() in {"deg/ms", "dpms"}:
        vel_ylabel = "Angular speed (deg/ms)"
    else:
        vel_ylabel = f"Angular speed ({vel_unit})"
    if params.get("normalize_position_to_amp"):
        pos_ylabel = "Position (norm. amp)"
    else:
        pos_ylabel = "Position (deg)"
    _plot_key("vel_center", vel_ylabel, "figure_2c.pdf")
    _plot_key("pos_center", pos_ylabel, "figure_2d.pdf")
    if isolation == "nan_mask_neighbors":
        _plot_key("vel_center", vel_ylabel, "figure_2c_isolated.pdf")
        _plot_key("pos_center", pos_ylabel, "figure_2d_isolated.pdf")


def replot_species_trace(data: dict) -> None:
    import matplotlib.pyplot as plt

    t = np.asarray(data.get("t_s", []), dtype=float)
    duration = float(data.get("duration_s") or (float(t[-1]) if t.size else 80.0))
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 2.0), dpi=300, sharex=True)
    for ax, key, ylab in (
        (axes[0], "phi", "φ [deg]"),
        (axes[1], "theta", "θ [deg]"),
    ):
        body = data.get(key, {})
        if "L" in body:
            ax.plot(t, np.asarray(body["L"], dtype=float), color=TRACE_L, lw=0.8, label="Left")
        if "R" in body:
            ax.plot(t, np.asarray(body["R"], dtype=float), color=TRACE_R, lw=0.8, label="Right")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlim(0.0, duration)
        _style_spines(ax, tick_size=7)
    axes[-1].set_xlabel("[s]", fontsize=8)
    _savefig(fig, str(data.get("pdf_name", "trace.pdf")))


def _epoch_duration_bins(vals: np.ndarray, scale: str, n_bins: int = 30) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n_edges = int(n_bins) + 1
    if vals.size == 0:
        return np.logspace(-2, 1, n_edges) if scale == "log" else np.linspace(0.0, 1.0, n_edges)
    hi = float(np.nanmax(vals))
    if scale == "log":
        lo = max(float(np.nanmin(vals)), 1e-2)
        if hi <= lo:
            hi = lo * 10.0
        return np.logspace(np.log10(lo), np.log10(hi), n_edges)
    return np.linspace(0.0, hi if hi > 0 else 1.0, n_edges)


def replot_epoch_durations(data: dict) -> None:
    import matplotlib.pyplot as plt

    stored = data.get("bins") or {}
    n_bins = int(data.get("n_bins") or 30)
    xmin = data.get("linear_xmin_s")
    xmin_f = 0.0 if xmin is None else float(xmin)
    for lab, color in (("active", "#D55E00"), ("quiet", "#0072B2")):
        vals = np.asarray(data.get(lab, []), dtype=float)
        for scale in ("log", "linear"):
            fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
            body = stored.get(lab) if isinstance(stored, dict) else None
            bins = None
            if isinstance(body, dict) and scale in body:
                bins = np.asarray(body[scale], dtype=float)
            if bins is None or bins.size < 2:
                bins = _epoch_duration_bins(vals, scale, n_bins=n_bins)
            if vals.size:
                ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.8)
                if scale == "log":
                    ax.set_xscale("log")
                else:
                    ax.set_xlim(xmin_f, float(bins[-1]))
            ax.set_xlabel("Epoch duration [s]", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title(f"{lab} n={vals.size} ({scale})", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _savefig(fig, f"epoch_durations_{lab}_{scale}.pdf")


def _replot_2e_axes(ax, xlim, ylim):
    ax.set_xlabel("Amplitude [deg]", fontsize=8)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _replot_2e_means(per_animal_stats, global_fit, color_map, animal_order, params, xlim, ylim, title, name):
    import matplotlib.pyplot as plt

    figsize = tuple(params.get("figsize", (1.5, 1.7)))
    dpi = int(params.get("dpi", 300))
    lw = float(params.get("lw", 1.0))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for animal in animal_order:
        sdf = per_animal_stats.get(animal) if isinstance(per_animal_stats, dict) else None
        if sdf is None:
            continue
        try:
            empty = sdf.empty
        except AttributeError:
            continue
        if empty:
            continue
        x = (sdf["amp_lo"] + sdf["amp_hi"]) / 2
        ax.plot(x, sdf["mean_peak_v"], "o-", color=color_map.get(animal, "0.3"), ms=3, lw=lw, label=str(animal))
    slope = (global_fit or {}).get("slope")
    intercept = (global_fit or {}).get("intercept")
    if slope is not None and intercept is not None and np.isfinite(slope) and np.isfinite(intercept) and xlim is not None:
        xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
        ax.plot(xs, float(slope) * xs + float(intercept), "k--", lw=lw)
    if title:
        ax.set_title(title, fontsize=7)
    _replot_2e_axes(ax, xlim, ylim)
    slope = (global_fit or {}).get("slope")
    r = (global_fit or {}).get("r")
    if r is None or not np.isfinite(r):
        r2 = (global_fit or {}).get("R2")
        r = float(np.sqrt(r2)) if r2 is not None and np.isfinite(r2) else float("nan")
    if slope is not None and np.isfinite(slope) and np.isfinite(r):
        ax.text(
            0.04, 0.97, f"slope = {float(slope):.3f}\nPearson r = {float(r):.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6, color="0.1",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
        )
    fig.tight_layout()
    _savefig(fig, name)


def replot_figure_2e(data: dict) -> None:
    import matplotlib.pyplot as plt

    params = data.get("params") or {}
    xlim = data.get("xlim")
    ylim = data.get("ylim")
    if xlim is not None:
        xlim = (float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ylim = (float(ylim[0]), float(ylim[1]))
    color_map = data.get("color_map") or {}
    animal_order = list(data.get("animal_order") or (data.get("per_animal_stats") or {}).keys())
    classes = data.get("classes") or {}
    all_body = classes.get("all") or {
        "per_animal_stats": data.get("per_animal_stats") or {},
        "global_fit": data.get("global_fit") or {},
        "n": (data.get("global_fit") or {}).get("n"),
    }
    _replot_2e_means(
        all_body.get("per_animal_stats") or {},
        all_body.get("global_fit") or {},
        color_map,
        animal_order,
        params,
        xlim,
        ylim,
        None,
        "figure_2e_per_animal_means.pdf",
    )
    _replot_2e_means(
        all_body.get("per_animal_stats") or {},
        all_body.get("global_fit") or {},
        color_map,
        animal_order,
        params,
        xlim,
        ylim,
        None,
        "figure_2e.pdf",
    )
    for label, name in (
        ("concurrent", "figure_2e_per_animal_means_concurrent.pdf"),
        ("monocular", "figure_2e_per_animal_means_monocular.pdf"),
    ):
        body = classes.get(label) or {}
        n = body.get("n")
        title = f"{label} n={n}" if n is not None else label
        _replot_2e_means(
            body.get("per_animal_stats") or {},
            body.get("global_fit") or {},
            color_map,
            animal_order,
            params,
            xlim,
            ylim,
            title,
            name,
        )
    pooled = data.get("pooled") or {}
    amp = np.asarray(pooled.get("amp", data.get("amp", data.get("x", []))), dtype=float)
    vel = np.asarray(pooled.get("vel", data.get("vel", data.get("y", []))), dtype=float)
    n = int(min(amp.size, vel.size))
    amp, vel = amp[:n], vel[:n]
    figsize = tuple(params.get("figsize", (1.5, 1.7)))
    dens_figsize = tuple(params.get("density_figsize", (3.8, 3.2)))
    dpi = int(params.get("dpi", 300))
    lw = float(params.get("lw", 1.0))
    fit = all_body.get("global_fit") or data.get("global_fit") or {}
    cohort = pooled.get("cohort", "all")

    def _annot(ax, body, fs=9):
        sl = (body or {}).get("slope")
        rr = (body or {}).get("r")
        if rr is None or not np.isfinite(rr):
            r2 = (body or {}).get("R2")
            rr = float(np.sqrt(r2)) if r2 is not None and np.isfinite(r2) else float("nan")
        if sl is None or not np.isfinite(sl) or not np.isfinite(rr):
            return
        ax.text(
            0.04, 0.97, f"slope = {float(sl):.3f}\nPearson r = {float(rr):.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=fs, color="0.1",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
        )

    if n:
        fig, ax = plt.subplots(figsize=dens_figsize, dpi=dpi)
        ax.scatter(amp, vel, s=4, alpha=0.12, color="0.45", edgecolors="none", rasterized=True)
        if np.isfinite(fit.get("slope", np.nan)) and xlim is not None:
            xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
            ax.plot(xs, float(fit["slope"]) * xs + float(fit["intercept"]), "k--", lw=lw)
        ax.set_title(f"all animals ({cohort}) n={n}", fontsize=11)
        _replot_2e_axes(ax, xlim, ylim)
        ax.set_xlabel("Amplitude [deg]", fontsize=11)
        ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
        ax.tick_params(labelsize=9)
        _annot(ax, fit, 9)
        fig.tight_layout()
        _savefig(fig, "figure_2e_all_animals_scatter.pdf")
        fig, ax = plt.subplots(figsize=dens_figsize, dpi=dpi)
        dens = data.get("density") or {}
        zi = dens.get("zi")
        extent = dens.get("extent")
        if extent is None and xlim and ylim:
            extent = (float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
        cmap = str(dens.get("cmap") or params.get("density_cmap", "turbo"))
        if zi is not None:
            im = ax.imshow(
                np.asarray(zi).T, extent=extent, origin="lower", cmap=cmap,
                aspect="auto", interpolation="bilinear",
            )
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Likelihood", fontsize=10)
            cb.ax.tick_params(labelsize=9)
        if np.isfinite(fit.get("slope", np.nan)) and xlim is not None:
            xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
            ax.plot(xs, float(fit["slope"]) * xs + float(fit["intercept"]), "k--", lw=max(lw, 1.2))
        ax.set_title(f"all animals ({cohort}) n={n}", fontsize=11)
        _replot_2e_axes(ax, xlim, ylim)
        ax.set_xlabel("Amplitude [deg]", fontsize=11)
        ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
        ax.tick_params(labelsize=9)
        _annot(ax, fit, 9)
        fig.tight_layout()
        _savefig(fig, "figure_2e_all_animals_density.pdf")
    animals_p = np.asarray(pooled.get("animal", []), dtype=str)
    pairing = np.asarray(pooled.get("pairing", []), dtype=str)
    scatter_animal = params.get("scatter_animal")
    if scatter_animal is None and animals_p.size:
        scatter_animal = str(animals_p[0])
    if scatter_animal is not None and amp.size and animals_p.size == amp.size:
        animal_mask = animals_p == str(scatter_animal)
        class_specs = (
            ("all_events", animal_mask, "figure_2e_all_events_scatter.pdf"),
            ("concurrent", animal_mask & (pairing == "concurrent") if pairing.size == amp.size else animal_mask, "figure_2e_concurrent_scatter.pdf"),
            ("monocular", animal_mask & (pairing == "monocular") if pairing.size == amp.size else animal_mask, "figure_2e_monocular_scatter.pdf"),
        )
        for label, mask, name in class_specs:
            xa, ya = amp[mask], vel[mask]
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            if xa.size:
                ax.scatter(xa, ya, s=4, alpha=0.12, color="0.45", edgecolors="none", rasterized=True)
            title = f"{label} {scatter_animal} (empty)" if xa.size == 0 else f"{label} {scatter_animal} n={int(xa.size)}"
            if xa.size >= 3 and xlim is not None:
                sl = np.polyfit(xa, ya, 1)
                xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
                ax.plot(xs, sl[0] * xs + sl[1], "k--", lw=lw)
                r = np.corrcoef(xa, ya)[0, 1]
                title = f"{label} {scatter_animal} n={int(xa.size)}  R²={float(r*r):.2f}"
            ax.set_title(title, fontsize=7)
            _replot_2e_axes(ax, xlim, ylim)
            fig.tight_layout()
            _savefig(fig, name)


def replot_s13_preonset(data: dict) -> None:
    import matplotlib.pyplot as plt

    params = data.get("params") or {}
    xlim = data.get("xlim")
    ylim = data.get("ylim")
    if xlim is not None:
        xlim = (float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ylim = (float(ylim[0]), float(ylim[1]))
    color_map = data.get("color_map") or {}
    animal_order = list(data.get("animal_order") or [])
    classes = data.get("classes") or {}
    names = {
        "all": ("S13a_all.pdf", "all"),
        "concurrent": ("S13b_concurrent.pdf", "concurrent"),
        "monocular": ("S13c_monocular.pdf", "monocular"),
    }
    for label, (pdf_name, title_stub) in names.items():
        body = classes.get(label) or {}
        n = body.get("n")
        title = f"{title_stub} n={n}" if n is not None else title_stub
        _replot_2e_means(
            body.get("per_animal_stats") or {},
            body.get("global_fit") or {},
            color_map,
            animal_order,
            params,
            xlim,
            ylim,
            title,
            pdf_name,
        )
    pooled = data.get("pooled") or {}
    amp = np.asarray(pooled.get("amp", []), dtype=float)
    vel = np.asarray(pooled.get("vel", []), dtype=float)
    n = int(min(amp.size, vel.size))
    amp, vel = amp[:n], vel[:n]
    dens_figsize = tuple(params.get("density_figsize", (3.8, 3.2)))
    dpi = int(params.get("dpi", 300))
    lw = float(params.get("lw", 1.0))
    fit = (classes.get("all") or {}).get("global_fit") or {}
    fig, ax = plt.subplots(figsize=dens_figsize, dpi=dpi)
    if n:
        ax.scatter(amp, vel, s=4, alpha=0.12, color="0.45", edgecolors="none", rasterized=True)
    if np.isfinite(fit.get("slope", np.nan)) and xlim is not None:
        xs = np.linspace(float(xlim[0]), float(xlim[1]), 50)
        ax.plot(xs, float(fit["slope"]) * xs + float(fit["intercept"]), "k--", lw=lw)
    ax.set_title(f"mixed A  all animals n={n}", fontsize=11)
    _replot_2e_axes(ax, xlim, ylim)
    ax.set_xlabel("Amplitude [deg]", fontsize=11)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
    ax.tick_params(labelsize=9)
    slope = fit.get("slope")
    r = fit.get("r")
    if r is None or not np.isfinite(r):
        r2 = fit.get("R2")
        r = float(np.sqrt(r2)) if r2 is not None and np.isfinite(r2) else float("nan")
    if slope is not None and np.isfinite(slope) and np.isfinite(r):
        ax.text(
            0.04, 0.97, f"slope = {float(slope):.3f}\nPearson r = {float(r):.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, color="0.1",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
        )
    fig.tight_layout()
    _savefig(fig, "S13d_preonset_scatter.pdf")

    sl = data.get("slope_summary") or {}
    animals = list(sl.get("animals") or [])
    conc_s = np.asarray(sl.get("concurrent_slopes") or [], dtype=float)
    mono_s = np.asarray(sl.get("monocular_slopes") or [], dtype=float)
    n_an = int(min(len(animals), conc_s.size, mono_s.size))
    animals, conc_s, mono_s = animals[:n_an], conc_s[:n_an], mono_s[:n_an]
    t_stat = sl.get("paired_t", float("nan"))
    p_val = sl.get("paired_p", float("nan"))
    fig_s, ax_s = plt.subplots(figsize=(4.2, 2.4), dpi=150)
    x = np.arange(n_an)
    w = 0.35
    ax_s.bar(x - w / 2, conc_s, w, color="#0072B2", edgecolor="0.2", label="concurrent")
    ax_s.bar(x + w / 2, mono_s, w, color="#D55E00", edgecolor="0.2", label="monocular")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(animals, rotation=30, ha="right", fontsize=7)
    ax_s.set_ylabel("OLS slope (°/ms per °)")
    ax_s.legend(fontsize=7, frameon=False)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    p_txt = f"p={float(p_val):.3g}" if np.isfinite(float(p_val)) else "p=nan"
    t_txt = f"{float(t_stat):.2f}" if np.isfinite(float(t_stat)) else "nan"
    ax_s.set_title(f"paired t={t_txt}  {p_txt}  df={n_an - 1}" if n_an else "no animals", fontsize=8)
    fig_s.tight_layout()
    _savefig(fig_s, "S13e_slope_test.pdf")
    handles = [
        plt.Line2D([0], [0], color=color_map.get(a, "0.3"), marker="o", ms=5, lw=1.4, label=str(a))
        for a in animal_order
    ]
    fig_leg = plt.figure(figsize=(1.8, 0.32 * max(1, len(animal_order)) + 0.35), dpi=150)
    fig_leg.legend(handles, animal_order, loc="center", frameon=False, fontsize=8)
    _savefig(fig_leg, "S13f_animal_legend.pdf")


def replot_diagnostics_2e(data: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    params = data.get("params") or {}
    amp = np.asarray(data.get("amp", []), dtype=float)
    vel = np.asarray(data.get("vel", []), dtype=float)
    length_bin = np.asarray(data.get("length_bin", []), dtype=object).astype(str)
    n = int(min(amp.size, vel.size, length_bin.size))
    amp, vel, length_bin = amp[:n], vel[:n], length_bin[:n]
    counts = data.get("counts") or {}
    colors = data.get("length_colors") or {
        "1": "#D55E00", "2": "#E69F00", "3": "#F0E442", "4": "#009E73",
        "5": "#0072B2", "6–8": "#56B4E9", "9+": "#000000",
    }
    order = list(data.get("length_draw_order") or ["9+", "6–8", "5", "4", "3", "2", "1"])
    xlim = data.get("xlim") or (0.0, 1.0)
    ylim = data.get("ylim") or (0.0, 0.5)
    figsize = tuple(params.get("figsize", (5.2, 4.2)))
    dpi = int(params.get("dpi", 300))
    thr = float(params.get("speed_threshold_deg_per_ms") or 0.048)
    slope, intercept = data.get("slope"), data.get("intercept")
    omit = params.get("omit_length_le")
    title_suffix = f"  (omit length≤{int(omit)})" if omit is not None else ""
    cohort = params.get("cohort", "all")

    def _guides(ax, sl, ic):
        ax.axhline(thr, color="0.15", lw=1.2, ls=":", label=f"detect floor  {thr:.3f} deg/ms")
        x0, x1 = float(xlim[0]), float(xlim[1])
        xs = np.linspace(max(x0, 0.0), x1, 80)
        if sl is not None and ic is not None and np.isfinite(sl) and np.isfinite(ic):
            ax.plot(xs, float(sl) * xs + float(ic), "k--", lw=1.2, label=f"OLS  slope={float(sl):.3f}")

    def _axes(ax, title):
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
        ax.set_xlabel("Amplitude [deg]", fontsize=11)
        ax.set_ylabel("Peak V [deg/ms]", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for label in order:
        mask = length_bin == str(label)
        if not np.any(mask):
            continue
        ax.scatter(
            amp[mask], vel[mask], s=6, alpha=0.35, color=colors.get(label, "0.4"),
            edgecolors="none", rasterized=True,
            label=f"{label}  n={counts.get(label, int(mask.sum()))}",
        )
    _guides(ax, slope, intercept)
    _axes(ax, f"all animals ({cohort}) n={n}  colored by length{title_suffix}")
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.6)
    fig.tight_layout()
    _savefig(fig, "figure_2e_scatter_by_length.pdf")

    gray = str(params.get("gray_color", "0.45"))
    galpha = float(params.get("gray_alpha", 0.5))
    gs = float(params.get("gray_size", 6))
    animals = np.asarray(data.get("animal", []), dtype=str)
    n_a = int(min(amp.size, vel.size, animals.size)) if animals.size else 0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if n:
        ax.scatter(amp, vel, s=gs, alpha=galpha, color=gray, edgecolors="none", rasterized=True)
    _guides(ax, slope, intercept)
    _axes(ax, f"all animals ({cohort}) n={n}{title_suffix}")
    fig.tight_layout()
    _savefig(fig, "2e_all_animals_gray.pdf")

    animal_order = list(data.get("animal_order") or [])
    if not animal_order and n_a:
        animal_order = list(dict.fromkeys(animals[:n_a].tolist()))
    pdf_path = OUT / "2e_per_animal_gray.pdf"
    OUT.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        pages = animal_order or ["(no animals)"]
        for animal in pages:
            if animal == "(no animals)":
                xa = ya = np.array([], dtype=float)
                sl = ic = None
            else:
                mask = animals[:n_a] == str(animal) if n_a else np.array([], dtype=bool)
                xa, ya = (amp[:n_a][mask], vel[:n_a][mask]) if n_a else (np.array([]), np.array([]))
                sl = ic = None
                if xa.size >= 3:
                    coef = np.polyfit(xa, ya, 1)
                    sl, ic = float(coef[0]), float(coef[1])
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            if xa.size:
                ax.scatter(xa, ya, s=gs, alpha=galpha, color=gray, edgecolors="none", rasterized=True)
            _guides(ax, sl, ic)
            _axes(ax, f"{animal} n={int(xa.size)}" if animal != "(no animals)" else animal)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    classes = data.get("classes") or {}
    if classes:
        color_map = data.get("color_map") or {}
        means_order = list(data.get("animal_order_means") or animal_order)
        means_xlim = data.get("means_xlim") or xlim
        means_ylim = data.get("means_ylim") or ylim
        lw = float(params.get("lw", 1.0))
        fig, axes = plt.subplots(
            1, 3, figsize=tuple(params.get("triptych_figsize", (5.0, 1.9))), dpi=dpi, sharex=True, sharey=True,
        )
        for ax, label in zip(axes, ("all", "concurrent", "monocular")):
            body = classes.get(label) or {}
            per = body.get("per_animal_stats") or {}
            fit = body.get("global_fit") or {}
            for animal in means_order:
                sdf = per.get(animal) or {}
                amp_lo = np.asarray(sdf.get("amp_lo", []), dtype=float)
                amp_hi = np.asarray(sdf.get("amp_hi", []), dtype=float)
                mean_v = np.asarray(sdf.get("mean_peak_v", []), dtype=float)
                if amp_lo.size == 0 or mean_v.size == 0:
                    continue
                x = (amp_lo + amp_hi) / 2
                ax.plot(x, mean_v, "o-", color=color_map.get(animal, "0.3"), ms=3, lw=lw, label=animal)
            sl, ic = fit.get("slope"), fit.get("intercept")
            if sl is not None and ic is not None and np.isfinite(sl) and np.isfinite(ic):
                xs = np.linspace(float(means_xlim[0]), float(means_xlim[1]), 50)
                ax.plot(xs, float(sl) * xs + float(ic), "k--", lw=lw)
            rr = fit.get("r")
            if rr is None and fit.get("R2") is not None and np.isfinite(fit.get("R2")):
                rr = float(np.sqrt(fit["R2"]))
            if sl is not None and np.isfinite(sl) and rr is not None and np.isfinite(rr):
                ax.text(
                    0.04, 0.97, f"slope = {float(sl):.3f}\nPearson r = {float(rr):.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=6, color="0.1",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
                )
            ax.set_title(f"{label} n={body.get('n', 0)}", fontsize=8)
            ax.set_xlabel("Amplitude [deg]", fontsize=8)
            ax.set_ylabel("Peak V [deg/ms]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(float(means_xlim[0]), float(means_xlim[1]))
            ax.set_ylim(float(means_ylim[0]), float(means_ylim[1]))
        fig.tight_layout()
        _savefig(fig, "fig_2e_mono_conc_all.pdf")


def replot_figure_2f_colormap_trials(data: dict) -> None:
    """Redraw all colormap variants stored in a colormap_trials pickle."""
    import matplotlib.pyplot as plt

    from eye_tracking_system_tools.analysis.figures_2f_colormap_export import (
        COLORMAP_VARIANTS,
        resolve_figure_2f_colormap,
    )
    from eye_tracking_system_tools.analysis.figures_2f_2h_2i import _draw_coupling_heatmap

    kind = str(data.get("figure_kind", "2f")).lower()
    plot_id = str(data.get("plot_id", "figure_2f"))
    variants = list(data.get("colormap_variants") or COLORMAP_VARIANTS)

    if kind == "s3":
        hist_still = data.get("hist_still") or data.get("still")
        hist_moving = data.get("hist_moving") or data.get("moving")
        vmax = float(data.get("vmax", data.get("vmax_all", 1.0)) or 1.0)
        rng = tuple(data.get("range", data.get("macro_range", (0.0, 0.5))))
        ticks = data.get("ticks", [0.0, 0.25, 0.5])
        n_still = int(data.get("n_still", 0))
        n_moving = int(data.get("n_moving", 0))
        for variant in variants:
            cmap = resolve_figure_2f_colormap(variant)
            for suffix, hist, n in (
                ("head_still", hist_still, n_still),
                ("head_moving", hist_moving, n_moving),
            ):
                if not hist:
                    continue
                fig, ax = plt.subplots(figsize=(1.85, 1.85), dpi=300)
                _draw_coupling_heatmap(ax, hist, rng, ticks, vmax=vmax, cmap=cmap)
                ax.set_title(f"n={n}", fontsize=8)
                _savefig(fig, f"figure_S3_{suffix}_{variant}.pdf")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
            sm.set_array([])
            fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
            cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
            plt.colorbar(sm, cax=cax, orientation="vertical")
            _savefig(fig_cbar, f"figure_S3_colorbar_{variant}.pdf")
        return

    macro = data.get("macro")
    micro = data.get("micro")
    if not macro or not micro:
        replot_figure_2f(data)
        return
    vmax = float(data.get("vmax_all", 1.0) or 1.0)
    macro_range = tuple(data.get("macro_range", (0.0, 0.5)))
    micro_range = tuple(data.get("micro_range", (0.0, 0.1)))
    macro_ticks = data.get("macro_tick_list", [0.0, 0.25, 0.5])
    micro_ticks = data.get("micro_tick_list", [0.0, 0.05, 0.1])
    if plot_id == "mouse_figure_2f":
        stem = "mouse_figure_2f"
    elif plot_id == "figure_2f":
        stem = "figure_2f"
    else:
        stem = plot_id
    for variant in variants:
        cmap = resolve_figure_2f_colormap(variant)
        fig, axs = plt.subplots(1, 2, figsize=(3.2, 1.85), dpi=300, constrained_layout=True)
        for ax, hist, rng, title, ticks in (
            (axs[0], macro, macro_range, "Macro", macro_ticks),
            (axs[1], micro, micro_range, "Micro", micro_ticks),
        ):
            _draw_coupling_heatmap(ax, hist, rng, ticks, vmax=vmax, cmap=cmap)
            ax.set_title(title, fontsize=8)
        _savefig(fig, f"{stem}_{variant}.pdf")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
        cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
        plt.colorbar(sm, cax=cax, orientation="vertical")
        _savefig(fig_cbar, f"{stem}_colorbar_{variant}.pdf")


def replot_saccade_head_timing(data: dict) -> None:
    import matplotlib.pyplot as plt

    offsets = np.asarray(data.get("offsets_ms", []), dtype=float)
    offsets = offsets[np.isfinite(offsets)]
    window_ms = float(data.get("window_ms", 500.0))
    bin_ms = float(data.get("bin_ms", 10.0))
    summary = data.get("summary") or {}
    n_edge = int(np.round(2.0 * window_ms / bin_ms))
    edges = np.linspace(-window_ms, window_ms, n_edge + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = []
    for c in centers:
        if c < -1e-12:
            colors.append("#0072B2")
        elif c > 1e-12:
            colors.append("#D55E00")
        else:
            colors.append("0.55")
    counts, _ = np.histogram(offsets, bins=edges)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=colors, edgecolor="0.25", linewidth=0.3)
    ax.axvline(0.0, color="k", lw=0.9, ls="--")
    ax.set_xlim(-window_ms, window_ms)
    ax.set_xlabel("Head onset − saccade onset [ms]")
    ax.set_ylabel("Count")
    ax.set_title("Head-bout onset relative to saccade onset")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    n = int(summary.get("n_in_window") or offsets.size)
    med = summary.get("median_dt_ms")
    med_s = f"{med:.1f}" if med is not None and np.isfinite(float(med)) else "n/a"
    ax.text(
        0.02,
        0.98,
        f"n = {n}\nmedian = {med_s} ms",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
    )
    _savefig(fig, str(data.get("pdf_pooled", "head_onset_relative_to_saccade.pdf")))

    per_animal = data.get("per_animal") or {}
    animals = sorted(per_animal)
    if animals:
        ncols = 3 if len(animals) > 2 else max(len(animals), 1)
        nrows = int(np.ceil(len(animals) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows), dpi=150, sharex=True, squeeze=False)
        for i, animal in enumerate(animals):
            ax = axes[i // ncols][i % ncols]
            dt = np.asarray(per_animal[animal], dtype=float)
            dt = dt[np.isfinite(dt)]
            c, _ = np.histogram(dt, bins=edges)
            ax.bar(edges[:-1], c, width=np.diff(edges), align="edge", color=colors, edgecolor="0.25", linewidth=0.3)
            ax.axvline(0.0, color="k", lw=0.9, ls="--")
            ax.set_xlim(-window_ms, window_ms)
            med_a = float(np.median(dt)) if dt.size else float("nan")
            med_as = f"{med_a:.0f} ms" if np.isfinite(med_a) else "n/a"
            ax.set_title(f"{animal}  n={dt.size}  median={med_as}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if i // ncols == nrows - 1:
                ax.set_xlabel("Head onset − saccade onset [ms]")
            if i % ncols == 0:
                ax.set_ylabel("Count")
        for j in range(len(animals), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        _savefig(fig, str(data.get("pdf_by_animal", "head_onset_relative_to_saccade_by_animal.pdf")))

    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=150)
    head = float(summary.get("pct_head_leads") or 0.0)
    sacc = float(summary.get("pct_saccade_leads") or 0.0)
    simul = float(summary.get("pct_simultaneous") or 0.0)
    ax.bar([0, 1, 2], [head, sacc, simul], color=["#0072B2", "#D55E00", "0.55"], width=0.7, edgecolor="0.2")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["head leads", "saccade leads", "simultaneous"], fontsize=8)
    ax.set_ylabel("% of paired events")
    ax.set_ylim(0, 100)
    ax.set_title(f"Who starts first?  n={n}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(fig, str(data.get("pdf_lead_lag", "lead_lag_summary.pdf")))

    edges = np.asarray(data.get("peth_edges_ms", []), dtype=float)
    occ = np.asarray(data.get("peth_occupancy", []), dtype=float)
    n_peth = int(data.get("peth_n_saccades") or 0)
    if edges.size >= 2 and occ.size == edges.size - 1:
        fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=150)
        ax.bar(edges[:-1], 100.0 * occ, width=np.diff(edges), align="edge", color="#0072B2", edgecolor="0.25", linewidth=0.3)
        ax.axvline(0.0, color="k", lw=0.9, ls="--")
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        ax.set_xlabel("Time relative to saccade onset [ms]")
        ax.set_ylabel("% of saccades with head motion")
        ax.set_title(f"Head-movement occupancy around saccades  n={n_peth}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _savefig(fig, str(data.get("pdf_peth", "head_motion_peth.pdf")))


def replot_secondary_after_saccade(data: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    summary = data.get("summary") or {}
    amps = np.asarray(data.get("stationary_amp_deg", []), dtype=float)
    amps = amps[np.isfinite(amps)]
    thr = float(data.get("threshold_deg", summary.get("amp_threshold_deg") or 5.0))
    amp_stats = data.get("amp_stats") or {}
    xmax = max(float(np.percentile(amps, 99.5)) if amps.size else 10.0, thr * 1.15, 10.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=150)
    ax.hist(amps, bins=np.linspace(0.0, xmax, 40), color="#0072B2", edgecolor="0.25", linewidth=0.4, alpha=0.85)
    ax.axvline(thr, color="#D55E00", ls="--", lw=1.4, label=f"threshold {thr:.1f}°")
    ax.set_xlabel("Unique-event amplitude [deg]")
    ax.set_ylabel("Count")
    ax.set_title(f"Head-stationary-at-onset amplitudes  n={amps.size}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(fig, str(data.get("pdf_amp", "amplitude_distribution.pdf")))

    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=150)
    vals = [
        float(summary.get("pct_no_secondary") or 0.0),
        float(summary.get("pct_first_same") or 0.0),
        float(summary.get("pct_first_reverse") or 0.0),
    ]
    ax.bar([0, 1, 2], vals, color=["0.55", "#009E73", "#CC79A7"], edgecolor="0.2", width=0.72)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["no follow-up", "first: same dir.", "first: reverse"])
    ax.set_ylabel("% of large, still-head saccades")
    ax.set_ylim(0, 100)
    ax.set_title(f"Follow-up detected eye movements  n={int(summary.get('n_qualifying') or 0)}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(fig, str(data.get("pdf_fractions", "secondary_fractions.pdf")))

    by = summary.get("head_by_secondary_class") or {}
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    vals = [float((by.get(k) or {}).get("pct_subsequent_head") or 0.0) for k in ("none", "same", "reverse")]
    ax.bar([0, 1, 2], vals, color=["0.55", "#009E73", "#CC79A7"], edgecolor="0.2", width=0.72)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["no follow-up", "first: same", "first: reverse"])
    ax.set_ylabel("% with head-bout onset in post window")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(fig, str(data.get("pdf_head", "head_association.pdf")))

    grid = np.asarray(data.get("grid_ms", []), dtype=float)
    pos = np.asarray(data.get("pos_along"))
    classes = np.asarray(data.get("secondary_class", []))
    if grid.size and pos.size and pos.ndim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), dpi=150, sharex=True, sharey=True)
        for ax, lab, color in zip(axes, ("none", "same", "reverse"), ("0.55", "#009E73", "#CC79A7")):
            mask = classes == lab if classes.size == pos.shape[0] else np.ones(pos.shape[0], dtype=bool)
            mat = pos[mask]
            segs = []
            n = int(mat.shape[0])
            use = np.arange(n)
            if n > 250:
                use = np.linspace(0, n - 1, 250).astype(int)
            for i in use:
                y = mat[i]
                m = np.isfinite(y)
                if int(m.sum()) < 2:
                    continue
                segs.append(np.column_stack([grid[m], y[m]]))
            if segs:
                ax.add_collection(LineCollection(segs, colors=color, linewidths=0.4, alpha=0.22))
                ax.plot(grid, np.nanmedian(mat[use], axis=0), color="k", lw=1.3)
            ax.axvline(0.0, color="k", ls="--", lw=0.8)
            ax.set_xlim(float(grid[0]), float(grid[-1]))
            ax.set_title(f"{lab} n={n}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Time from onset [ms]")
        axes[0].set_ylabel("Pos. along primary axis [deg]")
        _savefig(fig, str(data.get("pdf_pop", "population_traces.pdf")))


def replot_kerr_component_error(data: dict) -> None:
    import pandas as pd

    from eye_tracking_system_tools.analysis.kerr_component_error_export import (
        figure_s1b_per_animal,
        figure_s1c_overall,
    )

    per_block = pd.DataFrame(data["per_block"])
    animal_level = pd.DataFrame(data["animal_level"])
    across = pd.DataFrame(data["across"])
    fig_b = figure_s1b_per_animal(per_block, animal_level)
    _savefig(fig_b, str(data.get("s1b_pdf", "cohort_component_error_across_animals.pdf")))
    fig_c = figure_s1c_overall(across)
    _savefig(fig_c, str(data.get("s1c_pdf", "cohort_component_error_overall_phi_theta.pdf")))


def replot_figure_2f_lizard_mouse(data: dict) -> None:
    cmap = _turbo_white0()
    liz = data.get("lizard") or {}
    mou = data.get("mouse") or {}
    vmax_j = float(data.get("vmax_joint") or max(float(liz.get("vmax") or 1e-12), float(mou.get("vmax") or 1e-12)))

    def _two(hist_a, rng_a, ticks_a, title_a, hist_b, rng_b, ticks_b, title_b, vmax_l, vmax_m, pdf_name):
        fig, axs = plt.subplots(1, 2, figsize=(3.0, 1.7), dpi=300, constrained_layout=True)
        _draw_coupling_heatmap(axs[0], hist_a, rng_a, ticks_a, cmap, vmax_l, title=title_a)
        _draw_coupling_heatmap(axs[1], hist_b, rng_b, ticks_b, cmap, vmax_m, title=title_b)
        _savefig(fig, pdf_name)

    _two(
        liz["hist"], tuple(liz["range"]), liz.get("ticks") or [0, liz["range"][1]], "lizard",
        mou["hist"], tuple(mou["range"]), mou.get("ticks") or [0, mou["range"][1]], "mouse",
        vmax_j, vmax_j, "S8j_lizard_mouse_all_2f.pdf",
    )
    _two(
        liz["hist"], tuple(liz["range"]), liz.get("ticks") or [0, liz["range"][1]], "lizard",
        mou["hist"], tuple(mou["range"]), mou.get("ticks") or [0, mou["range"][1]], "mouse",
        float(liz.get("vmax") or vmax_j), float(mou.get("vmax") or vmax_j),
        "S8j_lizard_mouse_all_2f_separate.pdf",
    )
    zoom = data.get("zoom") or {}
    zliz = zoom.get("lizard") or {}
    zmou = zoom.get("mouse") or {}
    if zliz.get("hist") and zmou.get("hist"):
        zvj = float(zoom.get("vmax_joint") or max(float(zliz.get("vmax") or 1e-12), float(zmou.get("vmax") or 1e-12)))
        _two(
            zliz["hist"], tuple(zliz["range"]), zliz.get("ticks") or [0, zliz["range"][1]],
            f"lizard inner 50% n={zliz.get('n_in', '')}",
            zmou["hist"], tuple(zmou["range"]), zmou.get("ticks") or [0, zmou["range"][1]],
            f"mouse inner 50% n={zmou.get('n_in', '')}",
            zvj, zvj, "S8j_lizard_mouse_zoom_2f.pdf",
        )
        _two(
            zliz["hist"], tuple(zliz["range"]), zliz.get("ticks") or [0, zliz["range"][1]],
            f"lizard inner 50% n={zliz.get('n_in', '')}",
            zmou["hist"], tuple(zmou["range"]), zmou.get("ticks") or [0, zmou["range"][1]],
            f"mouse inner 50% n={zmou.get('n_in', '')}",
            float(zliz.get("vmax") or zvj), float(zmou.get("vmax") or zvj),
            "S8j_lizard_mouse_zoom_2f_separate.pdf",
        )
        _save_cbar(cmap, zvj, "S8j_colorbar_zoom.pdf")
        _save_cbar(cmap, float(zliz.get("vmax") or zvj), "S8j_colorbar_lizard_zoom.pdf")
        _save_cbar(cmap, float(zmou.get("vmax") or zvj), "S8j_colorbar_mouse_zoom.pdf")
    _save_cbar(cmap, vmax_j, "S8j_colorbar.pdf")
    _save_cbar(cmap, float(liz.get("vmax") or vmax_j), "S8j_colorbar_lizard.pdf")
    _save_cbar(cmap, float(mou.get("vmax") or vmax_j), "S8j_colorbar_mouse.pdf")


def replot_epoch_duration_bin_trials(data: dict) -> None:
    from eye_tracking_system_tools.analysis.figures_3d_isi import plot_epoch_duration_triptych

    quiet = np.asarray(data.get("quiet", []), dtype=float)
    active = np.asarray(data.get("active", []), dtype=float)
    for trial in data.get("trials") or []:
        name = str(trial.get("name") or "trial")
        fig = plot_epoch_duration_triptych(
            quiet,
            active,
            zoom_xmax_s=float(trial.get("zoom_xmax_s") or data.get("zoom_xmax_s") or 1.0),
            quiet_full_bins=np.asarray(trial["quiet_full_bins"], dtype=float),
            zoom_bins=np.asarray(trial["zoom_bins"], dtype=float),
            title=str(trial.get("label") or name),
            kde="kde" in name.lower(),
        )
        _savefig(fig, f"epoch_duration_triptych_{name}.pdf")


def replot_isi_plotdata(data: dict) -> None:
    params = data.get("params") or {}
    animals = list(data.get("animals") or [])
    color_map = data.get("color_map") or {}
    per_animal = data.get("per_animal") or {}
    combined = data.get("combined") or {}
    figsize = tuple(params.get("figure_size") or (2.5, 2.2))
    xscale = "log"
    xlim = params.get("xlim")
    if params.get("xlim_policy") or "linear" in str(params.get("ylabel", "")).lower():
        xscale = "linear"
    elif xlim is not None and float(xlim[1]) <= 500:
        xscale = "linear"
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    handles, labels = [], []
    if combined:
        x = np.asarray(combined.get("x", []), dtype=float)
        y = np.asarray(combined.get("y", []), dtype=float)
        if x.size and y.size:
            h, = ax.plot(
                x, y,
                str(params.get("combined_linestyle", "-")),
                color=params.get("combined_color", "k"),
                linewidth=float(params.get("combined_linewidth", 2.0)),
                label=str(combined.get("label") or "All (combined)"),
                zorder=10,
            )
            handles.append(h)
            labels.append(str(combined.get("label") or "All (combined)"))
    for animal in animals:
        body = per_animal.get(animal) or {}
        x = np.asarray(body.get("x", []), dtype=float)
        y = np.asarray(body.get("y", []), dtype=float)
        if not x.size or not y.size:
            continue
        h, = ax.plot(
            x, y,
            linewidth=float(params.get("linewidth", 1.5)),
            color=color_map.get(animal),
            label=str(animal),
        )
        handles.append(h)
        labels.append(str(animal))
    xlim = params.get("xlim")
    ylim = params.get("ylim")
    ax.set_xscale(xscale)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_xlabel(str(params.get("xlabel", "ISI [ms]")), fontsize=10)
    ax.set_ylabel(str(params.get("ylabel", "Probability")), fontsize=10)
    _style_spines(ax, tick_size=8)
    stem = str(data.get("pdf_name") or "ISI_histogram.pdf")
    if not stem.endswith(".pdf"):
        stem = stem + ".pdf"
    _savefig(fig, stem)
    if handles:
        fig_leg = plt.figure(figsize=(2.0, 0.28 * max(1, len(labels)) + 0.4), dpi=300)
        fig_leg.legend(handles, labels, loc="center", frameon=False, ncol=1, prop={"size": 8})
        legend_name = "legend_" + stem
        _savefig(fig_leg, legend_name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data, pkl = _load_pickle()
    kind = KIND
    if kind == "figure_2f" and data is not None:
        replot_figure_2f(data)
    elif kind == "figure_s3" and data is not None:
        replot_figure_s3(data)
    elif kind == "figure_2f_lizard_mouse" and data is not None:
        replot_figure_2f_lizard_mouse(data)
    elif kind == "kerr_component_error" and data is not None:
        replot_kerr_component_error(data)
    elif kind == "figure_2f_colormap_trials" and data is not None:
        replot_figure_2f_colormap_trials(data)
    elif kind == "jitter_histogram" and data is not None:
        replot_jitter_histogram(data)
    elif kind == "yield_histogram" and data is not None:
        replot_jitter_histogram(data)
    elif kind == "pos_vel" and data is not None:
        replot_pos_vel(data)
    elif kind == "species_trace" and data is not None:
        replot_species_trace(data)
    elif kind == "unified_jitter" and data is not None:
        replot_unified_jitter(data)
    elif kind == "rayleigh_noise_core" and data is not None:
        from eye_tracking_system_tools.analysis.rayleigh_noise_core_export import (
            figure_rayleigh_noise_core_window,
        )

        values = np.asarray(data.get("d", []), dtype=float)
        fig = figure_rayleigh_noise_core_window(
            values,
            B=float(data.get("B_med", float("nan"))),
            threshold=data.get("threshold"),
            title=str(data.get("title", "")),
            n_bins=int(data.get("n_bins", 40)),
            color=str(data.get("color", "#0072B2")),
            hist_label=str(data.get("hist_label", "quiet segments")),
        )
        _savefig(fig, str(data.get("pdf_name", "rayleigh_noise_core.pdf")))
    elif kind == "raw_interframe_delta" and data is not None:
        import matplotlib.pyplot as plt

        pools = data.get("pools") or {}
        n_bins = int(data.get("n_bins", 50))
        fits = data.get("fits") or {}
        panels = data.get("panels") or [
            ["rigid", "rigid lizard", "#D55E00"],
            ["modular", "modular lizard", "#0072B2"],
            ["turtle", "turtle", "#CC79A7"],
            ["mouse", "mouse", "#009E73"],
        ]
        fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0), dpi=150)
        for ax, panel in zip(axes.ravel(), panels):
            mount, label, color = panel[0], panel[1], panel[2]
            values = np.asarray(pools.get(mount, []), dtype=float)
            values = values[np.isfinite(values) & (values >= 0)]
            n = int(values.size)
            hi = float(values.max()) if n else 1.0
            if n:
                ax.hist(values, bins=n_bins, range=(0, hi), density=True, color=color, edgecolor="0.3", alpha=0.7)
                scale = (fits.get(mount) or {}).get("B")
                if scale is not None and np.isfinite(float(scale)) and float(scale) > 0:
                    xs = np.linspace(0, hi, 400)
                    b = float(scale)
                    ax.plot(xs, (xs / (b * b)) * np.exp(-0.5 * (xs / b) ** 2), color="0.15", lw=1.5)
                    ax.set_title(f"{label}  B={b:.3f}  n={n}", fontsize=8)
                else:
                    ax.set_title(f"{label} (n={n})", fontsize=8)
            ax.set_xlim(0, hi)
            ax.set_xlabel("Inter-frame Δangle [deg/frame]", fontsize=8)
            ax.set_ylabel("Probability density", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        _savefig(fig, str(data.get("pdf_name", "raw_interframe_delta.pdf")))
        logx_name = data.get("logx_pdf_name")
        if logx_name:
            xmax = float(data.get("logx_xmax") or 1.0)
            lo = 1e-3
            fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0), dpi=150, sharex=True)
            bins = np.logspace(np.log10(lo), np.log10(max(xmax, lo * 10)), n_bins + 1)
            for ax, panel in zip(axes.ravel(), panels):
                mount, label, color = panel[0], panel[1], panel[2]
                values = np.asarray(pools.get(mount, []), dtype=float)
                values = values[np.isfinite(values)]
                n = int(values.size)
                if values.size:
                    counts, edges = np.histogram(values, bins=bins)
                    y = 100.0 * counts / counts.sum() if counts.sum() else counts
                    ax.bar(edges[:-1], y, width=np.diff(edges), align="edge", color=color, edgecolor="black", alpha=0.7)
                ax.set_xscale("log")
                ax.set_xlim(lo, xmax)
                ax.set_title(f"{label} (n={n})", fontsize=8)
                ax.set_ylabel("% samples", fontsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            for ax in axes[1]:
                ax.set_xlabel("Inter-frame Δangle [deg/frame]", fontsize=8)
            _savefig(fig, str(logx_name))
    elif kind == "unified_noise" and data is not None:
        replot_unified_noise(data)
    elif kind == "epoch_durations" and data is not None:
        replot_epoch_durations(data)
    elif kind == "epoch_duration_bin_trials" and data is not None:
        replot_epoch_duration_bin_trials(data)
    elif kind == "isi_plotdata" and data is not None:
        payload = dict(data)
        if pkl is not None and not payload.get("pdf_name"):
            stem = pkl.name
            if stem.startswith("ISI_") and stem[4:].startswith("ISI_"):
                stem = stem[4:]
            stem = stem.replace("_plotdata.pickle", "").replace("_plotdata.pkl", "")
            payload["pdf_name"] = stem + ".pdf"
        replot_isi_plotdata(payload)
    elif kind == "figure_2e" and data is not None:
        replot_figure_2e(data)
    elif kind == "s13_preonset_2e" and data is not None:
        replot_s13_preonset(data)
    elif kind == "diagnostics_2e" and data is not None:
        replot_diagnostics_2e(data)
    elif kind == "saccade_head_timing" and data is not None:
        replot_saccade_head_timing(data)
    elif kind == "secondary_after_saccade" and data is not None:
        replot_secondary_after_saccade(data)
    elif kind == "movement_associated_nystagmus" and data is not None:
        import matplotlib.pyplot as plt

        summary = data.get("summary") or {}
        overall = summary.get("overall") or {}
        still = summary.get("head_still") or {}
        moving = summary.get("head_moving") or {}
        fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=150)
        vals = [float(still.get("pct") or 0.0), float(moving.get("pct") or 0.0)]
        ns = [int(still.get("n") or 0), int(moving.get("n") or 0)]
        ax.bar([0, 1], vals, color=["#0072B2", "#D55E00"], edgecolor="0.2", width=0.7)
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + 1.4, f"{v:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["head stationary", "head moving"])
        ax.set_ylabel("% of large saccades")
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _savefig(fig, str(data.get("pdf_by_head", "reverse_followup_by_head.pdf")))
        fig, ax = plt.subplots(figsize=(3.4, 3.6), dpi=150)
        ax.bar([0], [float(overall.get("pct") or 0.0)], color="0.35", edgecolor="0.2", width=0.7)
        ax.set_xticks([0])
        ax.set_xticklabels(["reverse follow-up\nwithin 100 ms"])
        ax.set_ylabel("% of large saccades")
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _savefig(fig, str(data.get("pdf_overall", "reverse_followup_overall.pdf")))
    elif kind == "double_steps_rev_verbatim" and data is not None:
        import matplotlib.pyplot as plt

        summary = data.get("summary") or {}
        overall = summary.get("overall") or {}
        still = summary.get("head_still") or {}
        moving = summary.get("head_moving") or {}
        fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=150)
        vals = [float(still.get("pct") or 0.0), float(moving.get("pct") or 0.0)]
        ns = [int(still.get("n") or 0), int(moving.get("n") or 0)]
        ax.bar([0, 1], vals, color=["#0072B2", "#D55E00"], edgecolor="0.2", width=0.7)
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + 1.4, f"{v:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["head stationary", "head moving"])
        ax.set_ylabel("% of large saccades")
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _savefig(fig, str(data.get("pdf_by_head", "back_and_forth_by_head.pdf")))
        fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=150)
        ax.bar([0], [float(overall.get("pct") or 0.0)], color="0.35", edgecolor="0.2", width=0.7)
        ax.set_xticks([0])
        ax.set_xticklabels(["back-and-forth\nsequence"])
        ax.set_ylabel("% of large saccades")
        ax.set_ylim(0, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _savefig(fig, str(data.get("pdf_overall", "back_and_forth_overall.pdf")))
    elif kind == "double_steps_move_only" and data is not None:
        import matplotlib.pyplot as plt

        summary = data.get("summary") or {}
        overall = summary.get("overall") or {}
        fig, ax = plt.subplots(figsize=(2.15, 3.6), dpi=150)
        v = float(overall.get("pct") or 0.0)
        n = int(overall.get("n") or 0)
        ax.bar([0], [v], color="0.35", edgecolor="0.2", width=0.42)
        ax.text(0, v + 1.4, f"{v:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0])
        ax.set_xticklabels(["back-and-forth"])
        ax.set_ylabel("% of large saccades")
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.7, 0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _savefig(fig, str(data.get("pdf_overall", "back_and_forth_overall.pdf")))
    elif kind in {"robustness", "s1_occupancy", "corrective_head", "ISI_by_state", "qc_reversals_noise_blinks", "figure_2g", "isi_hist"}:
        replot_csv_lines()
        if data is None:
            print(f"replot kind={kind}: drew CSV panels (pickle optional)")
    elif data is None:
        print("No pickle/CSV found in metadata/; nothing to replot.")
        return 1
    else:
        print(f"kind={kind} pickle={None if pkl is None else pkl.name}: generic CSV/line fallback")
        replot_csv_lines()
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
