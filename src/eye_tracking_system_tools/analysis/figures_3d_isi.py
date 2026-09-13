"""Figure 3d: inter-saccade interval (ISI) density traces, log- and linear-scale.

Ports ``export_inter_saccade_intervals_density_traces_from_blocks`` (log-scale) and
``export_inter_saccade_intervals_density_traces_from_blocks_LINEAR`` (linear-scale)
from ``development/old_pipelines_for_ref/inter_saccade_interval_clean.ipynb`` into the
analysis package, driven by :class:`EventTables` instead of a raw ``block_collection``.

Algorithm summary (see notebook for the original derivation):

1. Collect saccade onset events (animal, block, eye, saccade_on_ms, head_movement)
   from ``tables.all_saccades``.
2. Estimate the video frame period from the median of ``np.diff(ms_axis)`` across the
   per-block eye traces (``tables.blocks``); fall back to a constant when traces are
   unavailable (e.g. event-pickle mode).
3. Deduplicate coincident left/right onsets per ``(animal, block)`` with a greedy
   two-pointer merge (``pair_threshold_ms``); this is intentionally *not*
   :func:`eye_tracking_system_tools.analysis.binocular.find_synced_saccades_ms`.
4. Compute inter-saccade intervals per ``(animal, block)`` (dropping sub-frame
   intervals below ``min_isi_ms``) and concatenate across blocks *within* an animal.
5. Bin the pooled per-animal ISIs into "snapped" bin edges -- edges land on
   ``(k + 0.5) * frame_ms`` so every bin can contain a realizable ISI -- either
   geometrically spaced (log panel) or linearly spaced (linear panel).
6. Normalize each animal's histogram to a probability mass function (not a KDE), plus
   a pooled "All (combined)" trace built from summed counts (not the mean of the
   per-animal pmfs).
7. Apply the configured display-window policy: the log panel folds out-of-window mass
   into the first/last displayed bin (geometric centers); the linear panel slices to
   the window with an optional renormalization (arithmetic centers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.colors import build_color_map
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

FALLBACK_FRAME_MS = 16.67

_DEFAULT_SHARED: dict[str, Any] = {
    "font_family": "Arial",
    "color_template": "okabeito",
    "combined_label": "All (combined)",
    "frame_ms_fallback": FALLBACK_FRAME_MS,
}

_DEFAULT_LOG_PARAMS: dict[str, Any] = {
    "figure_size": (2.5, 2.2),
    "num_bins": 20,
    "high_ms": 20000.0,
    "pair_threshold_ms": 60.0,
    "min_isi_ms": None,
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

_DEFAULT_LINEAR_PARAMS: dict[str, Any] = {
    "figure_size": (2.0, 1.5),
    "num_bins": 1176,
    "low_ms": 10.0,
    "high_ms": 20000.0,
    "pair_threshold_ms": 60.0,
    "min_isi_ms": None,
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


def _collect_events(tables: EventTables) -> pd.DataFrame:
    """Pull ``animal, block, eye, saccade_on_ms, head_movement`` from ``all_saccades``."""
    cols = ["animal", "block", "eye", "saccade_on_ms", "head_movement"]
    df = tables.all_saccades
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df[[c for c in cols if c in df.columns]].copy()
    if "head_movement" not in out.columns:
        out["head_movement"] = False
    out["saccade_on_ms"] = pd.to_numeric(out["saccade_on_ms"], errors="coerce")
    out["head_movement"] = out["head_movement"].fillna(False).astype(bool)
    out["animal"] = out["animal"].astype(str)
    return out.dropna(subset=["saccade_on_ms", "eye"])


def _estimate_frame_ms(tables: EventTables, *, fallback_ms: float) -> float:
    """Median of ``np.diff(ms_axis)`` across per-block eye traces; fall back if unavailable."""
    dts: list[float] = []
    for bundle in tables.blocks:
        for eye_df in (bundle.left, bundle.right):
            if eye_df is None or eye_df.empty or "ms_axis" not in eye_df.columns:
                continue
            ms = pd.to_numeric(eye_df["ms_axis"], errors="coerce").dropna().to_numpy()
            if ms.size > 3:
                dt = np.diff(ms)
                dt = dt[(dt > 0) & np.isfinite(dt)]
                if dt.size:
                    dts.append(float(np.median(dt)))
    return float(np.median(dts)) if dts else float(fallback_ms)


def _dedupe_lr_pairs(events_df: pd.DataFrame, *, pair_threshold_ms: float) -> pd.DataFrame:
    """Greedy two-pointer merge of coincident L/R onsets per ``(animal, block)``."""
    out_cols = ["animal", "block", "event_time_ms", "head_movement"]
    if events_df.empty:
        return pd.DataFrame(columns=out_cols)

    parts: list[pd.DataFrame] = []
    for (animal, block), g in events_df.groupby(["animal", "block"], dropna=False):
        left = g[g["eye"] == "L"].sort_values("saccade_on_ms")
        right = g[g["eye"] == "R"].sort_values("saccade_on_ms")
        lt = left["saccade_on_ms"].to_numpy(float)
        rt = right["saccade_on_ms"].to_numpy(float)
        lhm = left["head_movement"].to_numpy(bool) if len(left) else np.array([], dtype=bool)
        rhm = right["head_movement"].to_numpy(bool) if len(right) else np.array([], dtype=bool)

        i = j = 0
        ev_t: list[float] = []
        ev_hm: list[bool] = []
        while i < len(lt) and j < len(rt):
            dt = lt[i] - rt[j]
            if abs(dt) <= pair_threshold_ms:
                ev_t.append(min(lt[i], rt[j]))
                ev_hm.append(bool(lhm[i] or rhm[j]))
                i += 1
                j += 1
            elif lt[i] < rt[j] - pair_threshold_ms:
                ev_t.append(lt[i])
                ev_hm.append(bool(lhm[i]))
                i += 1
            else:
                ev_t.append(rt[j])
                ev_hm.append(bool(rhm[j]))
                j += 1
        while i < len(lt):
            ev_t.append(lt[i])
            ev_hm.append(bool(lhm[i]))
            i += 1
        while j < len(rt):
            ev_t.append(rt[j])
            ev_hm.append(bool(rhm[j]))
            j += 1

        parts.append(
            pd.DataFrame(
                {"animal": animal, "block": block, "event_time_ms": ev_t, "head_movement": ev_hm}
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=out_cols)


def _compute_blockwise_isis(dedup_df: pd.DataFrame, *, min_isi_ms: float) -> dict[str, np.ndarray]:
    """Per-``(animal, block)`` ISIs (>= ``min_isi_ms``), concatenated within each animal."""
    isi_all: dict[str, list[np.ndarray]] = {}
    for (animal, _block), g in dedup_df.groupby(["animal", "block"], dropna=False):
        t = np.sort(g["event_time_ms"].to_numpy())
        if t.size > 1:
            d = np.diff(t)
            d = d[np.isfinite(d) & (d >= min_isi_ms)]
            if d.size:
                isi_all.setdefault(animal, []).append(d)
    return {a: (np.concatenate(v) if v else np.array([], dtype=float)) for a, v in isi_all.items()}


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


def _save_legend(
    handles: list, labels: list[str], out_pdf: Path, *, show: bool = False
) -> Path | None:
    if not handles:
        return None
    fig_leg = plt.figure(figsize=(2.0, 0.28 * max(1, len(labels)) + 0.4), dpi=300)
    fig_leg.legend(handles, labels, loc="center", frameon=False, ncol=1, prop={"size": 8})
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig_leg.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    show_and_close(fig_leg, show)
    return out_pdf


def export_figure_3d(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    """
    Export the Fig 3d ISI density traces (log- and linear-scale panels).

    Reads ``tables.params["figure_3d"]`` (with ``log`` / ``linear`` sub-sections) for
    overrides; paper defaults are baked in as fallbacks. Writes:

    - ``figures/ISI_histogram.pdf`` + ``figures/legend_ISI_histogram.pdf``
    - ``figures/ISI_hist_linear_10_300ms.pdf`` + ``figures/legend_ISI_hist_linear_10_300ms.pdf``
    - matching ``metadata/*_plotdata.pickle`` (+ ``.meta.yaml``) sidecars
    """
    cfg = dict(tables.params.get("figure_3d", {}))
    shared = {**_DEFAULT_SHARED, **{k: cfg[k] for k in _DEFAULT_SHARED if k in cfg}}
    log_cfg = {**_DEFAULT_LOG_PARAMS, **dict(cfg.get("log") or {})}
    linear_cfg = {**_DEFAULT_LINEAR_PARAMS, **dict(cfg.get("linear") or {})}

    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)

    events = _collect_events(tables)
    animals = sorted(events["animal"].unique()) if not events.empty else []
    frame_ms = _estimate_frame_ms(tables, fallback_ms=float(shared["frame_ms_fallback"]))
    color_map = build_color_map(animals, template=shared["color_template"], order=animals)
    font_family = _resolve_font_family(str(shared["font_family"]))

    written: dict[str, Path] = {}

    # --- Log-scale panel ---
    log_pair_threshold_ms = float(log_cfg["pair_threshold_ms"])
    log_dedup = _dedupe_lr_pairs(events, pair_threshold_ms=log_pair_threshold_ms)
    log_min_isi_ms = (
        0.75 * frame_ms if log_cfg["min_isi_ms"] is None else float(log_cfg["min_isi_ms"])
    )
    log_isi_all = _compute_blockwise_isis(log_dedup, min_isi_ms=log_min_isi_ms)

    low_ms = max(1e-6, float(log_cfg["first_bin_factor"]) * frame_ms)
    log_bins = _snapped_log_bins(
        int(log_cfg["num_bins"]),
        frame_ms,
        low_ms,
        float(log_cfg["high_ms"]),
        add_leading_zero_bin=bool(log_cfg["add_leading_zero_bin"]),
        log_exponent=float(log_cfg["log_exponent"]),
    )
    log_xlim = tuple(log_cfg["xlim"]) if log_cfg["xlim"] is not None else None

    log_fig, log_handles, log_labels, log_data = _plot_isi_traces(
        bins=log_bins,
        isi_all=log_isi_all,
        animals=animals,
        color_map=color_map,
        fold_fn=lambda counts, edges: _fold_to_xlim_log(counts, edges, log_xlim),
        figure_size=tuple(log_cfg["figure_size"]),
        xscale="log",
        xlim=log_xlim,
        ylim=tuple(log_cfg["ylim"]) if log_cfg["ylim"] is not None else None,
        xlabel=str(log_cfg["xlabel"]),
        ylabel=str(log_cfg["ylabel"]),
        linewidth=float(log_cfg["linewidth"]),
        combined_linewidth=float(log_cfg["combined_linewidth"]),
        combined_color=log_cfg["combined_color"],
        combined_linestyle=str(log_cfg["combined_linestyle"]),
        combined_label=str(shared["combined_label"]),
        font_family=font_family,
    )
    log_pdf = figures_dir / "ISI_histogram.pdf"
    log_fig.savefig(log_pdf, format="pdf", bbox_inches="tight", dpi=300)
    show_and_close(log_fig, show)
    written["ISI_histogram"] = log_pdf

    log_legend_pdf = _save_legend(
        log_handles, log_labels, figures_dir / "legend_ISI_histogram.pdf", show=show
    )
    if log_legend_pdf is not None:
        written["legend_ISI_histogram"] = log_legend_pdf

    log_plot_data = {
        "params": {
            **log_cfg,
            "font_family": font_family,
            "color_template": shared["color_template"],
        },
        "frame_ms": frame_ms,
        "bins": log_bins,
        "animals": animals,
        "color_map": color_map,
        "per_animal": log_data["per_animal"],
        "combined": log_data["combined"],
    }
    log_pickle = metadata_dir / "ISI_histogram_plotdata.pickle"
    write_pickle_with_meta(
        log_plot_data,
        log_pickle,
        meta={
            "csv_choices": tables.csv_meta,
            "params": log_cfg,
            "figure": "3d_log",
            "n_animals": len(animals),
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_3d_isi.export_figure_3d",
    )
    written["ISI_histogram_plotdata"] = log_pickle

    # --- Linear-scale panel ---
    lin_pair_threshold_ms = float(linear_cfg["pair_threshold_ms"])
    if lin_pair_threshold_ms == log_pair_threshold_ms:
        lin_dedup = log_dedup
    else:
        lin_dedup = _dedupe_lr_pairs(events, pair_threshold_ms=lin_pair_threshold_ms)
    lin_min_isi_ms = (
        0.75 * frame_ms if linear_cfg["min_isi_ms"] is None else float(linear_cfg["min_isi_ms"])
    )
    lin_isi_all = _compute_blockwise_isis(lin_dedup, min_isi_ms=lin_min_isi_ms)

    lin_bins = _snapped_linear_bins(
        int(linear_cfg["num_bins"]),
        frame_ms,
        float(linear_cfg["low_ms"]),
        float(linear_cfg["high_ms"]),
        add_leading_zero_bin=bool(linear_cfg["add_leading_zero_bin"]),
    )
    lin_xlim = tuple(linear_cfg["xlim"]) if linear_cfg["xlim"] is not None else None
    xlim_policy = str(linear_cfg["xlim_policy"])
    if xlim_policy not in ("slice", "conditional", "fold"):
        raise ValueError("xlim_policy must be 'slice', 'conditional', or 'fold'")

    if xlim_policy == "fold":
        lin_fold_fn = lambda counts, edges: _fold_to_xlim_log(counts, edges, lin_xlim)  # noqa: E731
    else:
        renormalize = xlim_policy == "conditional"
        lin_fold_fn = lambda counts, edges: _slice_to_xlim_linear(  # noqa: E731
            counts, edges, lin_xlim, renormalize=renormalize
        )

    lin_fig, lin_handles, lin_labels, lin_data = _plot_isi_traces(
        bins=lin_bins,
        isi_all=lin_isi_all,
        animals=animals,
        color_map=color_map,
        fold_fn=lin_fold_fn,
        figure_size=tuple(linear_cfg["figure_size"]),
        xscale="linear",
        xlim=lin_xlim,
        ylim=tuple(linear_cfg["ylim"]) if linear_cfg["ylim"] is not None else None,
        xlabel=str(linear_cfg["xlabel"]),
        ylabel=str(linear_cfg["ylabel"]),
        linewidth=float(linear_cfg["linewidth"]),
        combined_linewidth=float(linear_cfg["combined_linewidth"]),
        combined_color=linear_cfg["combined_color"],
        combined_linestyle=str(linear_cfg["combined_linestyle"]),
        combined_label=str(shared["combined_label"]),
        font_family=font_family,
    )
    lin_pdf = figures_dir / "ISI_hist_linear_10_300ms.pdf"
    lin_fig.savefig(lin_pdf, format="pdf", bbox_inches="tight", dpi=300)
    show_and_close(lin_fig, show)
    written["ISI_hist_linear_10_300ms"] = lin_pdf

    lin_legend_pdf = _save_legend(
        lin_handles,
        lin_labels,
        figures_dir / "legend_ISI_hist_linear_10_300ms.pdf",
        show=show,
    )
    if lin_legend_pdf is not None:
        written["legend_ISI_hist_linear_10_300ms"] = lin_legend_pdf

    lin_plot_data = {
        "params": {
            **linear_cfg,
            "font_family": font_family,
            "color_template": shared["color_template"],
        },
        "frame_ms": frame_ms,
        "bins": lin_bins,
        "animals": animals,
        "color_map": color_map,
        "per_animal": lin_data["per_animal"],
        "combined": lin_data["combined"],
    }
    lin_pickle = metadata_dir / "ISI_hist_linear_10_300ms_plotdata.pickle"
    write_pickle_with_meta(
        lin_plot_data,
        lin_pickle,
        meta={
            "csv_choices": tables.csv_meta,
            "params": linear_cfg,
            "figure": "3d_linear",
            "n_animals": len(animals),
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_3d_isi.export_figure_3d",
    )
    written["ISI_hist_linear_10_300ms_plotdata"] = lin_pickle

    return written


def _epoch_index_for_times(state_df: pd.DataFrame, times_ms: np.ndarray) -> np.ndarray:
    """Index of the behavior-state row containing each time, or -1 if none."""
    out = np.full(times_ms.shape, -1, dtype=int)
    if state_df is None or state_df.empty:
        return out
    starts = state_df["start_time"].to_numpy(dtype=float)
    ends = state_df["end_time"].to_numpy(dtype=float)
    for i, t in enumerate(times_ms):
        if not np.isfinite(t):
            continue
        hit = np.where((starts <= t) & (t <= ends))[0]
        if hit.size:
            out[i] = int(hit[0])
    return out


def _compute_blockwise_isis_same_epoch(
    dedup_df: pd.DataFrame,
    tables: EventTables,
    *,
    min_isi_ms: float,
    require_label: str | None,
) -> dict[str, np.ndarray]:
    """ISIs whose two bounding events fall in the same behavior-state epoch.

    Cross-epoch intervals are dropped. ``require_label`` is ``'active'`` / ``'quiet'``
    or None for all same-epoch ISIs.
    """
    from eye_tracking_system_tools.analysis.behavior_state import (
        label_by_time,
        normalize_label,
        read_behavior_state,
    )

    isi_all: dict[str, list[np.ndarray]] = {}
    block_map = tables.block_dict
    for (animal, block), g in dedup_df.groupby(["animal", "block"], dropna=False):
        t = np.sort(g["event_time_ms"].to_numpy(float))
        if t.size < 2:
            continue
        key = f"{animal}_block_{''.join(c for c in str(block) if c.isdigit()).zfill(3)}"
        bundle = block_map.get(key)
        if bundle is None:
            # tolerate unpadded block numbers
            for cand, b in block_map.items():
                if cand.startswith(f"{animal}_") and str(block) in cand:
                    bundle = b
                    break
        if bundle is None:
            continue
        state_df = read_behavior_state(bundle.spec)
        if state_df is None or state_df.empty:
            continue
        epoch_idx = _epoch_index_for_times(state_df, t)
        labels = np.asarray(label_by_time(state_df, t), dtype=object)
        keep: list[float] = []
        for i in range(len(t) - 1):
            dt = float(t[i + 1] - t[i])
            if not np.isfinite(dt) or dt < min_isi_ms:
                continue
            if epoch_idx[i] < 0 or epoch_idx[i] != epoch_idx[i + 1]:
                continue
            lab = str(labels[i])
            if require_label is not None and normalize_label(lab) != require_label:
                continue
            keep.append(dt)
        if keep:
            isi_all.setdefault(str(animal), []).append(np.asarray(keep, dtype=float))
    return {a: np.concatenate(v) if v else np.array([], dtype=float) for a, v in isi_all.items()}


def export_isi_by_state(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    """Fig 3d plotters for all-ISI plus active-only and quiet-only (same bins)."""
    bundle = begin_plot_bundle(
        out_dir,
        "ISI_by_state",
        kind="ISI_by_state",
        tables=tables,
        logic_key="ISI_by_state",
        params=dict(getattr(tables, "params", {}) or {}),
    )
    all_written = export_figure_3d(tables, bundle.bundle_dir, show=show)
    cfg = dict(tables.params.get("figure_3d", {}))
    shared = {**_DEFAULT_SHARED, **{k: cfg[k] for k in _DEFAULT_SHARED if k in cfg}}
    log_cfg = {**_DEFAULT_LOG_PARAMS, **dict(cfg.get("log") or {})}
    linear_cfg = {**_DEFAULT_LINEAR_PARAMS, **dict(cfg.get("linear") or {})}
    figures_dir, metadata_dir = bundle.plots_dir, bundle.metadata_dir

    events = _collect_events(tables)
    animals = sorted(events["animal"].unique()) if not events.empty else []
    frame_ms = _estimate_frame_ms(tables, fallback_ms=float(shared["frame_ms_fallback"]))
    color_map = build_color_map(animals, template=shared["color_template"], order=animals)
    font_family = _resolve_font_family(str(shared["font_family"]))
    pair_ms = float(log_cfg["pair_threshold_ms"])
    dedup = _dedupe_lr_pairs(events, pair_threshold_ms=pair_ms)
    min_isi = 0.75 * frame_ms if log_cfg["min_isi_ms"] is None else float(log_cfg["min_isi_ms"])

    low_ms = max(1e-6, float(log_cfg["first_bin_factor"]) * frame_ms)
    log_bins = _snapped_log_bins(
        int(log_cfg["num_bins"]),
        frame_ms,
        low_ms,
        float(log_cfg["high_ms"]),
        add_leading_zero_bin=bool(log_cfg["add_leading_zero_bin"]),
        log_exponent=float(log_cfg["log_exponent"]),
    )
    log_xlim = tuple(log_cfg["xlim"]) if log_cfg["xlim"] is not None else None
    lin_bins = _snapped_linear_bins(
        int(linear_cfg["num_bins"]),
        frame_ms,
        float(linear_cfg["low_ms"]),
        float(linear_cfg["high_ms"]),
        add_leading_zero_bin=bool(linear_cfg["add_leading_zero_bin"]),
    )
    lin_xlim = tuple(linear_cfg["xlim"]) if linear_cfg["xlim"] is not None else None
    renormalize = str(linear_cfg["xlim_policy"]) == "conditional"

    written = dict(all_written)
    for label, log_name, lin_name in (
        ("active", "ISI_log_active.pdf", "ISI_linear_active.pdf"),
        ("quiet", "ISI_log_quiet.pdf", "ISI_linear_quiet.pdf"),
    ):
        isi_all = _compute_blockwise_isis_same_epoch(
            dedup, tables, min_isi_ms=min_isi, require_label=label
        )
        log_fig, *_rest = _plot_isi_traces(
            bins=log_bins,
            isi_all=isi_all,
            animals=animals,
            color_map=color_map,
            fold_fn=lambda counts, edges: _fold_to_xlim_log(counts, edges, log_xlim),
            figure_size=tuple(log_cfg["figure_size"]),
            xscale="log",
            xlim=log_xlim,
            ylim=tuple(log_cfg["ylim"]) if log_cfg["ylim"] is not None else None,
            xlabel=str(log_cfg["xlabel"]),
            ylabel=str(log_cfg["ylabel"]),
            linewidth=float(log_cfg["linewidth"]),
            combined_linewidth=float(log_cfg["combined_linewidth"]),
            combined_color=log_cfg["combined_color"],
            combined_linestyle=str(log_cfg["combined_linestyle"]),
            combined_label=str(shared["combined_label"]),
            font_family=font_family,
        )
        log_pdf = figures_dir / log_name
        log_fig.savefig(log_pdf, format="pdf", bbox_inches="tight", dpi=300)
        show_and_close(log_fig, show)
        written[log_name] = log_pdf

        lin_fig, *_r2 = _plot_isi_traces(
            bins=lin_bins,
            isi_all=isi_all,
            animals=animals,
            color_map=color_map,
            fold_fn=lambda counts, edges, _xlim=lin_xlim: _slice_to_xlim_linear(
                counts, edges, _xlim, renormalize=renormalize
            ),
            figure_size=tuple(linear_cfg["figure_size"]),
            xscale="linear",
            xlim=lin_xlim,
            ylim=tuple(linear_cfg["ylim"]) if linear_cfg["ylim"] is not None else None,
            xlabel=str(linear_cfg["xlabel"]),
            ylabel=str(linear_cfg["ylabel"]),
            linewidth=float(linear_cfg["linewidth"]),
            combined_linewidth=float(linear_cfg["combined_linewidth"]),
            combined_color=linear_cfg["combined_color"],
            combined_linestyle=str(linear_cfg["combined_linestyle"]),
            combined_label=str(shared["combined_label"]),
            font_family=font_family,
        )
        lin_pdf = figures_dir / lin_name
        lin_fig.savefig(lin_pdf, format="pdf", bbox_inches="tight", dpi=300)
        show_and_close(lin_fig, show)
        written[lin_name] = lin_pdf
    _ = metadata_dir
    finish_plot_bundle(bundle)
    return written


EPOCH_DURATION_N_BINS = 30
_EPOCH_DURATION_COLORS = {"active": "#D55E00", "quiet": "#0072B2"}

# Binning trials for the quiet-full / quiet-zoom / active-zoom layout. Durations are
# nearly integer seconds, so fixed-width integer bins avoid empty-bin combing.
EPOCH_DURATION_BIN_TRIALS: tuple[dict[str, Any], ...] = (
    {
        "name": "width5s_quiet50s",
        "zoom_bin_width_s": 5.0,
        "quiet_full_bin_width_s": 50.0,
        "label": "zoom 5 s bins; quiet-full 50 s",
    },
    {
        "name": "width10s_quiet100s",
        "zoom_bin_width_s": 10.0,
        "quiet_full_bin_width_s": 100.0,
        "label": "zoom 10 s bins; quiet-full 100 s",
    },
    {
        "name": "width2s_quiet30s",
        "zoom_bin_width_s": 2.0,
        "quiet_full_bin_width_s": 30.0,
        "label": "zoom 2 s bins; quiet-full 30 s",
    },
    {
        "name": "nbins12_quiet15",
        "zoom_n_bins": 12,
        "quiet_full_n_bins": 15,
        "label": "zoom 12 bins; quiet-full 15 bins",
    },
    {
        "name": "nbins15_quiet20",
        "zoom_n_bins": 15,
        "quiet_full_n_bins": 20,
        "label": "zoom 15 bins; quiet-full 20 bins",
    },
    {
        "name": "kde_width5s_quiet50s",
        "zoom_bin_width_s": 5.0,
        "quiet_full_bin_width_s": 50.0,
        "kde": True,
        "label": "KDE + 5 s hist; quiet-full 50 s",
    },
)


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


def epoch_duration_bin_edges(
    vals: np.ndarray,
    *,
    scale: str,
    n_bins: int = EPOCH_DURATION_N_BINS,
    xmin: float | None = None,
) -> np.ndarray:
    """Return histogram edges for epoch durations in seconds (``log`` or ``linear``)."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    n_edges = int(n_bins) + 1
    if vals.size == 0:
        lo = 1e-2 if xmin is None else float(xmin)
        hi = max(lo * 10.0, lo + 1.0)
        if scale == "log":
            lo = max(lo, 1e-2)
            return np.logspace(np.log10(lo), np.log10(hi), n_edges)
        return np.linspace(0.0 if xmin is None else float(xmin), hi, n_edges)
    hi = float(np.nanmax(vals))
    if scale == "log":
        lo = max(float(np.nanmin(vals)), 1e-2)
        if xmin is not None:
            lo = max(lo, float(xmin))
        if hi <= lo:
            hi = lo * 10.0
        return np.logspace(np.log10(lo), np.log10(hi), n_edges)
    lo_lin = 0.0 if xmin is None else float(xmin)
    if hi <= lo_lin:
        hi = lo_lin + 1.0
    return np.linspace(lo_lin, hi, n_edges)


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


def collect_epoch_durations(
    tables: EventTables,
    *,
    smooth_state: bool = False,
    min_duration_s: float = 0.0,
    min_active_ms: float = 5000.0,
    min_quiet_ms: float = 5000.0,
    gap_bridge_ms: float = 3000.0,
) -> dict[str, list[float]]:
    """Per-state epoch lengths in seconds from behavior-state CSVs."""
    frame = collect_epoch_table(
        tables,
        smooth_state=smooth_state,
        min_duration_s=min_duration_s,
        min_active_ms=min_active_ms,
        min_quiet_ms=min_quiet_ms,
        gap_bridge_ms=gap_bridge_ms,
    )
    durations = {"active": [], "quiet": []}
    if frame.empty:
        return durations
    for lab in durations:
        durations[lab] = frame.loc[frame["state"] == lab, "duration_s"].astype(float).tolist()
    return durations


def collect_epoch_table(
    tables: EventTables,
    *,
    smooth_state: bool = False,
    min_duration_s: float = 0.0,
    min_active_ms: float = 5000.0,
    min_quiet_ms: float = 5000.0,
    gap_bridge_ms: float = 3000.0,
) -> pd.DataFrame:
    """Per-epoch table with animal/block/timestamps (ms) and duration (s)."""
    from eye_tracking_system_tools.analysis.behavior_state import (
        normalize_label,
        read_behavior_state,
        smooth_behavior_state,
    )

    rows: list[dict[str, Any]] = []
    min_s = float(min_duration_s)
    for block in tables.blocks:
        spec = block.spec
        state = read_behavior_state(spec)
        if state is None or state.empty:
            continue
        if smooth_state:
            state = smooth_behavior_state(
                state,
                min_active_ms=float(min_active_ms),
                min_quiet_ms=float(min_quiet_ms),
                gap_bridge_ms=float(gap_bridge_ms),
            )
        for _, row in state.iterrows():
            lab = normalize_label(row["annotation"])
            if lab not in {"active", "quiet"}:
                continue
            start_ms = float(row["start_time"])
            end_ms = float(row["end_time"])
            dur_s = (end_ms - start_ms) / 1000.0
            if not (np.isfinite(dur_s) and dur_s > 0 and dur_s >= min_s):
                continue
            rows.append(
                {
                    "animal": spec.animal,
                    "block_num": spec.block_num,
                    "block_key": spec.block_key,
                    "state": lab,
                    "duration_s": dur_s,
                    "start_time_ms": start_ms,
                    "end_time_ms": end_ms,
                    "start_s": start_ms / 1000.0,
                    "end_s": end_ms / 1000.0,
                    "block_path": str(spec.block_path),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "animal",
                "block_num",
                "block_key",
                "state",
                "duration_s",
                "start_time_ms",
                "end_time_ms",
                "start_s",
                "end_s",
                "block_path",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["duration_s", "animal", "block_num"], ascending=[False, True, True]
    ).reset_index(drop=True)


def export_epoch_durations(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
    plot_id: str = "epoch_durations",
    smooth_state: bool = False,
    min_duration_s: float = 0.0,
    linear_xmin_s: float | None = None,
    min_active_ms: float = 5000.0,
    min_quiet_ms: float = 5000.0,
    gap_bridge_ms: float = 3000.0,
) -> dict[str, Path]:
    """Histograms of active/quiet epoch durations (log and linear x-axis)."""
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="epoch_durations",
        tables=tables,
        logic_key=plot_id if plot_id in {"epoch_durations", "epoch_durations_raw", "epoch_durations_smoothed"} else "epoch_durations",
        params=dict(getattr(tables, "params", {}) or {}),
        extra={
            "smooth_state": bool(smooth_state),
            "min_duration_s": float(min_duration_s),
            "linear_xmin_s": linear_xmin_s,
        },
    )
    figures_dir = bundle.plots_dir
    for stale in figures_dir.glob("epoch_durations_*.pdf"):
        stale.unlink()
    durations = collect_epoch_durations(
        tables,
        smooth_state=smooth_state,
        min_duration_s=min_duration_s,
        min_active_ms=min_active_ms,
        min_quiet_ms=min_quiet_ms,
        gap_bridge_ms=gap_bridge_ms,
    )
    xmin = 0.0 if linear_xmin_s is None else float(linear_xmin_s)

    written: dict[str, Path] = {}
    bins_out: dict[str, dict[str, np.ndarray]] = {}
    for lab, color in _EPOCH_DURATION_COLORS.items():
        vals = np.asarray(durations[lab], dtype=float)
        log_bins = epoch_duration_bin_edges(vals, scale="log", xmin=linear_xmin_s)
        lin_bins = epoch_duration_bin_edges(vals, scale="linear", xmin=linear_xmin_s)
        bins_out[lab] = {"log": log_bins, "linear": lin_bins}
        for scale, bins in (("log", log_bins), ("linear", lin_bins)):
            fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
            _draw_epoch_duration_hist(
                ax,
                vals,
                color=color,
                title=f"{lab} n={vals.size} ({scale})",
                scale=scale,
                bins=bins,
                xmin=xmin,
            )
            out = figures_dir / f"epoch_durations_{lab}_{scale}.pdf"
            fig.tight_layout()
            fig.savefig(out, format="pdf", bbox_inches="tight")
            show_and_close(fig, show)
            written[out.name] = out
    write_pickle_with_meta(
        {
            "active": np.asarray(durations["active"], dtype=float),
            "quiet": np.asarray(durations["quiet"], dtype=float),
            "bins": bins_out,
            "n_bins": EPOCH_DURATION_N_BINS,
            "smooth_state": bool(smooth_state),
            "min_duration_s": float(min_duration_s),
            "linear_xmin_s": linear_xmin_s,
        },
        bundle.metadata_dir / "epoch_durations.pkl",
        meta={
            "n_active": len(durations["active"]),
            "n_quiet": len(durations["quiet"]),
            "scales": ["log", "linear"],
            "n_bins": EPOCH_DURATION_N_BINS,
            "smooth_state": bool(smooth_state),
            "min_duration_s": float(min_duration_s),
            "plot_id": plot_id,
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_3d_isi.export_epoch_durations",
    )
    finish_plot_bundle(bundle)
    return written


def export_epoch_durations_raw_and_smoothed(
    tables: EventTables, out_dir: Path, *, show: bool = False
) -> dict[str, Path]:
    """Raw vs smoothed duration bundles (does not write ``epoch_durations/``)."""
    written: dict[str, Path] = {}
    written.update(
        export_epoch_durations(
            tables,
            out_dir,
            show=show,
            plot_id="epoch_durations_raw",
            smooth_state=False,
        )
    )
    written.update(
        export_epoch_durations(
            tables,
            out_dir,
            show=show,
            plot_id="epoch_durations_smoothed",
            smooth_state=True,
            min_duration_s=1.0,
            linear_xmin_s=1.0,
        )
    )
    return written


def export_epoch_duration_bin_trials(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
    plot_id: str = "epoch_duration_bin_trials",
    trials: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    zoom_round_to_s: float = 5.0,
    smooth_state: bool = True,
    min_duration_s: float = 1.0,
) -> dict[str, Path]:
    """Export quiet-full / quiet-zoom / active-zoom triptychs for several bin schemes.

    Defaults match ``epoch_durations_smoothed`` (bridge ≤3 s, min 5 s/state, drop
    epochs shorter than 1 s). Zoom xmax is
    ``ceil(max(active) / zoom_round_to_s) * zoom_round_to_s``.
    """
    trial_cfgs = list(EPOCH_DURATION_BIN_TRIALS if trials is None else trials)
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="epoch_duration_bin_trials",
        tables=tables,
        logic_key="epoch_duration_bin_trials",
        params=dict(getattr(tables, "params", {}) or {}),
        extra={
            "smooth_state": bool(smooth_state),
            "min_duration_s": float(min_duration_s),
            "zoom_round_to_s": float(zoom_round_to_s),
            "trials": [dict(t) for t in trial_cfgs],
        },
    )
    figures_dir = bundle.plots_dir
    for stale in figures_dir.glob("epoch_duration_*.pdf"):
        stale.unlink()
    # Also clear PNG previews from prior inspection runs.
    for stale in figures_dir.glob("epoch_duration_*.png"):
        stale.unlink()

    durations = collect_epoch_durations(
        tables,
        smooth_state=smooth_state,
        min_duration_s=min_duration_s,
    )
    active = np.asarray(durations["active"], dtype=float)
    quiet = np.asarray(durations["quiet"], dtype=float)
    zoom_xmax = epoch_duration_zoom_xmax_s(active, round_to=float(zoom_round_to_s))
    quiet_full_hi = float(np.nanmax(quiet)) if quiet.size else zoom_xmax

    written: dict[str, Path] = {}
    trial_meta: list[dict[str, Any]] = []
    for cfg in trial_cfgs:
        name = str(cfg["name"])
        zoom_bins = epoch_duration_linear_edges(
            xmax_s=zoom_xmax,
            n_bins=cfg.get("zoom_n_bins"),
            bin_width_s=cfg.get("zoom_bin_width_s"),
        )
        quiet_full_bins = epoch_duration_linear_edges(
            xmax_s=quiet_full_hi,
            n_bins=cfg.get("quiet_full_n_bins"),
            bin_width_s=cfg.get("quiet_full_bin_width_s"),
        )
        label = str(cfg.get("label") or name)
        fig = plot_epoch_duration_triptych(
            quiet,
            active,
            zoom_xmax_s=zoom_xmax,
            quiet_full_bins=quiet_full_bins,
            zoom_bins=zoom_bins,
            title=f"{label}  (zoom≤{zoom_xmax:g}s = ceil max active)",
            kde=bool(cfg.get("kde", False)),
        )
        out = figures_dir / f"epoch_duration_triptych_{name}.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")
        show_and_close(fig, show)
        written[out.name] = out
        trial_meta.append(
            {
                "name": name,
                "label": label,
                "zoom_xmax_s": zoom_xmax,
                "zoom_bins": zoom_bins,
                "quiet_full_bins": quiet_full_bins,
                "zoom_bin_width_s": cfg.get("zoom_bin_width_s"),
                "quiet_full_bin_width_s": cfg.get("quiet_full_bin_width_s"),
                "zoom_n_bins": cfg.get("zoom_n_bins"),
                "quiet_full_n_bins": cfg.get("quiet_full_n_bins"),
            }
        )

    write_pickle_with_meta(
        {
            "active": active,
            "quiet": quiet,
            "zoom_xmax_s": zoom_xmax,
            "max_active_s": float(np.nanmax(active)) if active.size else None,
            "max_quiet_s": float(np.nanmax(quiet)) if quiet.size else None,
            "trials": trial_meta,
            "smooth_state": bool(smooth_state),
            "min_duration_s": float(min_duration_s),
        },
        bundle.metadata_dir / "epoch_duration_bin_trials.pkl",
        meta={
            "n_active": int(active.size),
            "n_quiet": int(quiet.size),
            "zoom_xmax_s": zoom_xmax,
            "n_trials": len(trial_meta),
            "plot_id": plot_id,
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_3d_isi.export_epoch_duration_bin_trials",
    )
    finish_plot_bundle(bundle)
    return written


def export_state_isi_and_epochs(
    tables: EventTables, out_dir: Path, *, show: bool = False
) -> dict[str, Path]:
    """All-ISI Fig 3d copies, same-epoch active/quiet ISIs, and epoch-duration hists."""
    written = export_isi_by_state(tables, out_dir, show=show)
    written.update(export_epoch_durations(tables, out_dir, show=show))
    written.update(export_epoch_duration_bin_trials(tables, out_dir, show=show))
    return written

