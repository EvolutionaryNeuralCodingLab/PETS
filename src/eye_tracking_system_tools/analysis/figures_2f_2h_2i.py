"""Figures 2f (peak-speed coupling), 2h (endpoint heatmaps), 2i (polar dirs)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.stats import gaussian_kde

from eye_tracking_system_tools.analysis.binocular import find_synced_saccades_ms
from eye_tracking_system_tools.analysis.colors import build_color_map
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.pipeline import EventTables, _row_block_key
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def archived_figure_2f_pickle(repo: Path | None = None) -> Path:
    if repo is None:
        repo = Path(__file__).resolve().parents[3]
    return (
        repo
        / "src"
        / "eye_tracking_system_tools"
        / "figures"
        / "reproduction"
        / "main_figures"
        / "Fig_2_f"
        / "figure_2f_nodowncast.pickle"
    )


def export_archived_figure_2f(
    out_dir: Path,
    *,
    pickle_path: Path | None = None,
    repo: Path | None = None,
    show: bool = False,
) -> Path:
    """
    Plot paper Fig 2f from the archived nodowncast pickle into the run figures/.

    Used in ``--event-pickle`` mode (no eye traces). Does not mutate reproduction/.
    """
    import importlib.util

    import yaml

    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)
    pickle_path = Path(pickle_path) if pickle_path else archived_figure_2f_pickle(repo)
    if not pickle_path.exists():
        raise FileNotFoundError(pickle_path)

    script = pickle_path.parent / "figure_2f.py"
    spec = importlib.util.spec_from_file_location("fig2f_repro_helpers", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helpers from {script}")
    repro_2f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro_2f)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    cmap = repro_2f._make_custom_turbo()
    macro_hist = data["macro"]
    micro_hist = data["micro"]
    macro_range = tuple(data["macro_range"])
    micro_range = tuple(data["micro_range"])
    macro_n_ticks = int(data["macro_n_ticks"])
    micro_n_ticks = int(data["micro_n_ticks"])
    macro_tick_list = data.get("macro_tick_list")
    micro_tick_list = data.get("micro_tick_list")
    vmax_all = float(data.get("vmax_all", 1.0))

    fig, axs = plt.subplots(1, 2, figsize=(3, 1.7), dpi=300, constrained_layout=True)
    repro_2f._plot_panel(
        axs[0],
        macro_hist,
        macro_range,
        "Macro",
        macro_n_ticks,
        macro_tick_list,
        cmap,
        rasterize_mesh=True,
    )
    repro_2f._plot_panel(
        axs[1],
        micro_hist,
        micro_range,
        "Micro",
        micro_n_ticks,
        micro_tick_list,
        cmap,
        rasterize_mesh=True,
    )
    out_pdf = figures_dir / "figure_2f.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    show_and_close(fig, show)

    if not np.isfinite(vmax_all) or vmax_all <= 0:
        vmax_all = 1.0
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax_all))
    sm.set_array([])
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Probability", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    cbar_pdf = figures_dir / "figure_2f_colorbar.pdf"
    fig_cbar.savefig(cbar_pdf, bbox_inches="tight", dpi=150)
    show_and_close(fig_cbar, show)

    meta = {
        "figure": "2f",
        "source": "archived_reproduction_pickle",
        "source_pickle": str(pickle_path.resolve()),
        "pdf": str(out_pdf.resolve()),
        "colorbar_pdf": str(cbar_pdf.resolve()),
    }
    meta_path = metadata_dir / "figure_2f_archived.meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return out_pdf


def _iqr_bounds(arr: np.ndarray, mult: float) -> tuple[float, float]:
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return float(q1 - mult * iqr), float(q3 + mult * iqr)


def _is_auto_token(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"auto", "data"}


def _as_range(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if value is None or _is_auto_token(value):
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            lo, hi = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return default
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return (lo, hi)
    return default


def nice_axis_max(value: float, *, step: float | None = None) -> float:
    """Round ``value`` up onto a short tick so the plotted percentile is not clipped."""
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        return 0.5
    if step is None:
        step = 0.05 if x < 2.0 else 0.1
    return float(np.ceil((x / step) - 1e-12) * step)


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


def _as_tick_list(value: object, lo: float, hi: float) -> list[float]:
    if value is None or _is_auto_token(value):
        return _ticks_for_span(lo, hi)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            ticks = [float(v) for v in value]
        except (TypeError, ValueError):
            return _ticks_for_span(lo, hi)
        if all(np.isfinite(t) for t in ticks):
            return ticks
    return _ticks_for_span(lo, hi)


def resolve_figure_2f_view_limits(
    right: np.ndarray,
    left: np.ndarray,
    cfg: dict | None = None,
) -> dict[str, object]:
    """Macro/micro axis ranges for Fig 2f.

    With ``auto_view_limits`` (or ``macro_range: auto``), macro xmax is the
    ``macro_pct`` (default 99.5) percentile of the plotted L/R speeds, rounded
    up so the cloud is not clipped. Micro xmax is ``micro_frac_of_macro`` of
    that (default 0.2), a zoom on slower/near-threshold saccades.
    """
    cfg = dict(cfg or {})
    pct = float(cfg.get("macro_pct", 99.5))
    auto_macro = bool(cfg.get("auto_view_limits")) or _is_auto_token(cfg.get("macro_range"))
    auto_micro = bool(cfg.get("auto_view_limits")) or _is_auto_token(cfg.get("micro_range"))
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    finite_r = right[np.isfinite(right)]
    finite_l = left[np.isfinite(left)]

    if auto_macro and finite_r.size and finite_l.size:
        p_r = float(np.nanpercentile(finite_r, pct))
        p_l = float(np.nanpercentile(finite_l, pct))
        macro_range = (0.0, nice_axis_max(max(p_r, p_l)))
    else:
        macro_range = _as_range(cfg.get("macro_range"), (0.0, 0.5))

    if auto_micro:
        frac = float(cfg.get("micro_frac_of_macro", 0.2))
        micro_hi = nice_axis_max(macro_range[1] * frac, step=0.05)
        if micro_hi >= macro_range[1]:
            micro_hi = macro_range[1]
        micro_range = (0.0, float(micro_hi))
    else:
        micro_range = _as_range(cfg.get("micro_range"), (0.0, 0.1))

    if auto_macro:
        macro_ticks = _ticks_for_span(*macro_range)
    else:
        macro_ticks = _as_tick_list(cfg.get("macro_tick_list"), *macro_range)
    if auto_micro:
        micro_ticks = _ticks_for_span(*micro_range)
    else:
        micro_ticks = _as_tick_list(cfg.get("micro_tick_list"), *micro_range)

    return {
        "macro_range": macro_range,
        "micro_range": micro_range,
        "macro_tick_list": [float(t) for t in macro_ticks],
        "micro_tick_list": [float(t) for t in micro_ticks],
    }


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


def rebin_figure_2f_data(data: dict, cfg: dict | None = None) -> dict:
    """Recompute 2f histograms from stored speeds. Does not re-detect events."""
    data = dict(data)
    right = np.asarray(data.get("right_eye_speeds", []), dtype=float)
    left = np.asarray(data.get("left_eye_speeds", []), dtype=float)
    if right.size == 0 or left.size == 0:
        return data
    weights = data.get("weights")
    bins = int((cfg or {}).get("bins", data.get("bins", 60)))
    merged = dict(data)
    if cfg:
        merged.update(cfg)
    limits = resolve_figure_2f_view_limits(right, left, merged)
    macro_range = tuple(limits["macro_range"])
    micro_range = tuple(limits["micro_range"])
    data["bins"] = bins
    data["macro_range"] = macro_range
    data["micro_range"] = micro_range
    data["macro_tick_list"] = list(limits["macro_tick_list"])
    data["micro_tick_list"] = list(limits["micro_tick_list"])
    data["macro"] = histogram2d_xy(right, left, weights, macro_range, bins)
    data["micro"] = histogram2d_xy(right, left, weights, micro_range, bins)
    data["vmax_all"] = float(
        max(
            np.nanmax(data["macro"]["norm_counts"]),
            np.nanmax(data["micro"]["norm_counts"]),
            1e-12,
        )
    )
    data["auto_view_limits"] = bool(merged.get("auto_view_limits"))
    data["macro_pct"] = float(merged.get("macro_pct", 99.5))
    data["micro_frac_of_macro"] = float(merged.get("micro_frac_of_macro", 0.2))
    return data


def _estimate_frame_period_ms(eye_df: pd.DataFrame, t_ms: float) -> float:
    if eye_df is None or eye_df.empty or "ms_axis" not in getattr(eye_df, "columns", []):
        return 17.0
    try:
        window = eye_df.query(
            "ms_axis >= @t_ms - 51 and ms_axis <= @t_ms + 51"
        )["ms_axis"].to_numpy(dtype=float)
        if window.size >= 3:
            diffs = np.diff(np.sort(window))
            fp = float(np.nanmedian(diffs))
            if np.isfinite(fp) and fp > 0:
                return fp
    except Exception:
        pass
    diffs = np.diff(eye_df["ms_axis"].to_numpy(dtype=float))
    diffs = diffs[np.isfinite(diffs)]
    fp = float(np.nanmedian(diffs)) if diffs.size else np.nan
    return fp if (np.isfinite(fp) and fp > 0) else 17.0


def _contra_has_event(
    other_onsets: np.ndarray, t_ms: float, win_ms: float
) -> bool:
    if other_onsets.size == 0:
        return False
    return bool(np.nanmin(np.abs(other_onsets - t_ms)) <= win_ms)


def _sample_contra_peak(
    contra_df: pd.DataFrame, t_ms: float, halfwin_ms: float, frame_ms: float
) -> float:
    if contra_df is None or contra_df.empty:
        return np.nan
    if "angular_speed_r" not in contra_df.columns or "ms_axis" not in contra_df.columns:
        return np.nan
    series = contra_df.query(
        "ms_axis >= @t_ms - @halfwin_ms and ms_axis <= @t_ms + @halfwin_ms"
    )["angular_speed_r"]
    if series.notna().sum() == 0:
        return np.nan
    return float(np.nanmax(series.to_numpy(dtype=float))) / frame_ms


def _sample_event_span_peak(
    eye_df: pd.DataFrame, t_on_ms: float, t_off_ms: float, frame_ms: float
) -> float:
    """Max ``angular_speed_r`` over the event's own [on, off] ms span (strict mode)."""
    if eye_df is None or eye_df.empty:
        return np.nan
    if "angular_speed_r" not in eye_df.columns or "ms_axis" not in eye_df.columns:
        return np.nan
    series = eye_df.query(
        "ms_axis >= @t_on_ms and ms_axis <= @t_off_ms"
    )["angular_speed_r"]
    if series.notna().sum() == 0:
        return np.nan
    return float(np.nanmax(series.to_numpy(dtype=float))) / frame_ms


def _event_peak_deg_per_ms(row: pd.Series, eye_df: pd.DataFrame) -> float:
    """Detected-saccade peak: max ``speed_profile_angular`` / frame period."""
    sp = row.get("speed_profile_angular", None)
    if sp is None:
        return float("nan")
    try:
        arr = np.asarray(sp, dtype=float)
    except (TypeError, ValueError):
        return float("nan")
    if arr.size == 0 or not np.isfinite(np.nanmax(arr)):
        return float("nan")
    t0 = float(row["saccade_on_ms"]) if "saccade_on_ms" in row.index else float("nan")
    frame_ms = _estimate_frame_period_ms(eye_df, t0)
    if not np.isfinite(frame_ms) or frame_ms <= 0:
        return float("nan")
    return float(np.nanmax(arr)) / frame_ms


def _resolve_contra_peak(
    contra_df: pd.DataFrame,
    row: pd.Series,
    *,
    sample_mode: str,
    contra_sample_ms: float,
    frame_ms: float,
) -> float:
    mode = str(sample_mode).lower()
    t0 = float(row["saccade_on_ms"])
    if mode == "event_span":
        if "saccade_off_ms" not in row.index or not np.isfinite(row["saccade_off_ms"]):
            return np.nan
        return _sample_event_span_peak(
            contra_df, t0, float(row["saccade_off_ms"]), frame_ms
        )
    if mode == "contra_window":
        return _sample_contra_peak(contra_df, t0, contra_sample_ms, frame_ms)
    raise ValueError(
        f"figure_2f sample_mode must be 'contra_window' or 'event_span', got {sample_mode!r}"
    )


@dataclass
class Figure2fDisplayData:
    """Histogram panels for Fig 2f display (from export pickle or live compute)."""

    macro: dict[str, np.ndarray]
    micro: dict[str, np.ndarray]
    macro_range: tuple[float, float]
    micro_range: tuple[float, float]
    macro_tick_list: list[float]
    micro_tick_list: list[float]
    vmax_all: float
    source: str = "computed"
    n_points: int = 0


def display_data_from_collected(collected: Figure2fPoints) -> Figure2fDisplayData:
    macro = histogram2d_from_points(collected, view="macro")
    micro = histogram2d_from_points(collected, view="micro")
    cfg = collected.cfg
    vmax_all = float(
        max(
            np.nanmax(macro["norm_counts"]),
            np.nanmax(micro["norm_counts"]),
            1e-12,
        )
    )
    return Figure2fDisplayData(
        macro=macro,
        micro=micro,
        macro_range=collected.macro_range,
        micro_range=collected.micro_range,
        macro_tick_list=list(cfg.get("macro_tick_list", [0.0, 0.25, 0.5])),
        micro_tick_list=list(cfg.get("micro_tick_list", [0.0, 0.05, 0.1])),
        vmax_all=vmax_all,
        source="computed",
        n_points=len(collected.points),
    )


def load_figure_2f_display(pickle_path: Path | str) -> Figure2fDisplayData:
    """Load macro/micro histograms from a ``figure_2f_nodowncast.pickle`` export."""
    pickle_path = Path(pickle_path)
    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)
    right = np.asarray(data["right_eye_speeds"], dtype=float)
    left = np.asarray(data["left_eye_speeds"], dtype=float)
    macro_range = tuple(data["macro_range"])
    micro_range = tuple(data["micro_range"])
    vmax_all = float(data.get("vmax_all", 1.0))
    if not np.isfinite(vmax_all) or vmax_all <= 0:
        vmax_all = 1.0
    return Figure2fDisplayData(
        macro=data["macro"],
        micro=data["micro"],
        macro_range=(float(macro_range[0]), float(macro_range[1])),
        micro_range=(float(micro_range[0]), float(micro_range[1])),
        macro_tick_list=list(data.get("macro_tick_list", [0.0, 0.25, 0.5])),
        micro_tick_list=list(data.get("micro_tick_list", [0.0, 0.05, 0.1])),
        vmax_all=vmax_all,
        source=str(pickle_path),
        n_points=int(right.size),
    )


@dataclass
class Figure2fPoints:
    """Per-event inter-ocular peak speeds used for Fig 2f and ROI selection."""

    points: pd.DataFrame
    macro_range: tuple[float, float]
    micro_range: tuple[float, float]
    bins: int
    cfg: dict[str, object] = field(default_factory=dict)
    n_skip_mode: int = 0
    n_skip_profile: int = 0

    @property
    def right_peak_v(self) -> np.ndarray:
        return self.points["right_peak_v"].to_numpy(dtype=float)

    @property
    def left_peak_v(self) -> np.ndarray:
        return self.points["left_peak_v"].to_numpy(dtype=float)

    @property
    def weights(self) -> np.ndarray:
        return self.points["weight"].to_numpy(dtype=float)


def figure_2f_config(tables: EventTables, overrides: dict | None = None) -> dict:
    cfg = dict(tables.params.get("figure_2f", {}))
    if overrides:
        cfg.update(overrides)
    return cfg


def _apply_figure_2f_row_filters(
    df: pd.DataFrame, cfg: dict, *, verbose: bool = True
) -> pd.DataFrame:
    """Head-stationary and animal exclusions shared by all 2f collectors."""
    exclude = {str(a) for a in cfg.get("exclude_animals", [])}
    out = df.copy()
    if exclude:
        out = out[~out["animal"].astype(str).isin(exclude)]
    if cfg.get("require_head_stationary"):
        if "head_movement" not in out.columns or out["head_movement"].notna().sum() == 0:
            if verbose:
                print("[2f] WARNING: require_head_stationary but no labels; keeping all rows")
        else:
            before = len(out)
            out = out[out["head_movement"] == False]  # noqa: E712
            if verbose:
                print(f"[2f] head_stationary filter: {before} → {len(out)}")
    head_filter = str(cfg.get("head_filter", "all")).lower()
    if head_filter in {"moving", "head_moving"} and "head_movement" in out.columns:
        before = len(out)
        out = out[out["head_movement"] == True]  # noqa: E712
        if verbose:
            print(f"[2f] head_moving filter: {before} → {len(out)}")
    elif head_filter in {"still", "stationary"} and "head_movement" in out.columns:
        before = len(out)
        out = out[out["head_movement"] == False]  # noqa: E712
        if verbose:
            print(f"[2f] head_still filter: {before} → {len(out)}")
    return out


def _finalize_figure_2f_points(
    rows: list[dict],
    *,
    cfg: dict,
    bins: int,
    apply_iqr_clip: bool,
    n_skip_mode: int = 0,
    n_skip_profile: int = 0,
    n_pairs: int = 0,
    n_mono: int = 0,
    pairing_mode: str = "synced_ms",
) -> Figure2fPoints:
    if not rows:
        raise ValueError("No speeds for figure 2f after filters")

    points = pd.DataFrame(rows)
    animals_arr = points["animal"].astype(str).to_numpy()
    u, c = np.unique(animals_arr, return_counts=True)
    wmap = {a: (len(animals_arr) / (len(u) * cnt)) for a, cnt in zip(u, c)}
    points["weight"] = [wmap[str(a)] for a in animals_arr]

    if apply_iqr_clip:
        iqr_mult = float(cfg.get("iqr_multiplier", 60.0))
        right = points["right_peak_v"].to_numpy(dtype=float)
        left = points["left_peak_v"].to_numpy(dtype=float)
        r_lo, r_hi = _iqr_bounds(right, iqr_mult)
        l_lo, l_hi = _iqr_bounds(left, iqr_mult)
        keep = (right >= r_lo) & (right <= r_hi) & (left >= l_lo) & (left <= l_hi)
        points = points.loc[keep].reset_index(drop=True)

    limits = resolve_figure_2f_view_limits(
        points["right_peak_v"].to_numpy(dtype=float),
        points["left_peak_v"].to_numpy(dtype=float),
        cfg,
    )
    cfg = dict(cfg)
    cfg["macro_range"] = list(limits["macro_range"])
    cfg["micro_range"] = list(limits["micro_range"])
    cfg["macro_tick_list"] = list(limits["macro_tick_list"])
    cfg["micro_tick_list"] = list(limits["micro_tick_list"])
    macro_range = tuple(limits["macro_range"])
    micro_range = tuple(limits["micro_range"])

    event_mode = str(cfg.get("event_mode", "monocular")).lower()
    sample_mode = str(cfg.get("sample_mode", "contra_window")).lower()
    print(
        f"[2f] pairing_mode={pairing_mode} event_mode={event_mode} "
        f"sample_mode={sample_mode} kept={len(points)} "
        f"binocular_pairs={n_pairs} monocular={n_mono} "
        f"skip_mode={n_skip_mode} skip_profile={n_skip_profile} "
        f"exclude={sorted(cfg.get('exclude_animals', []))} "
        f"animals={sorted(u.tolist())} "
        f"macro_range={list(macro_range)} micro_range={list(micro_range)}"
    )
    return Figure2fPoints(
        points=points,
        macro_range=(float(macro_range[0]), float(macro_range[1])),
        micro_range=(float(micro_range[0]), float(micro_range[1])),
        bins=bins,
        cfg=cfg,
        n_skip_mode=n_skip_mode,
        n_skip_profile=n_skip_profile,
    )


def collect_figure_2f_points_legacy(
    tables: EventTables,
    *,
    cfg: dict | None = None,
    apply_iqr_clip: bool = True,
) -> Figure2fPoints:
    """
    Row-wise collector matching ``multiple_figures_pipeline_migration.ipynb``.

    Classifies monocular vs binocular via nearest contra-eye onset within
    ``contra_event_window_ms`` (default 100 ms). Both eyes use ipsi event peak
    plus a ±51 ms contra trace window sample (even for classified binocular
    events). Optional ``pair_merge_ms`` collapses duplicate L/R binocular rows.
    """
    cfg = figure_2f_config(tables, cfg)
    event_mode = str(cfg.get("event_mode", "monocular")).lower()
    sample_mode = str(cfg.get("sample_mode", "contra_window")).lower()
    contra_sample = float(cfg.get("contra_sample_ms", 51.0))
    contra_win = float(cfg.get("contra_event_window_ms", 100.0))
    pair_merge_ms = cfg.get("pair_merge_ms", 60.0)
    if pair_merge_ms is not None:
        pair_merge_ms = float(pair_merge_ms)
    bins = int(cfg.get("bins", 60))

    df = _apply_figure_2f_row_filters(tables.all_saccades, cfg)
    block_map = tables.block_dict
    onsets_by_block: dict[tuple[str, object], dict[str, np.ndarray]] = {}
    if not df.empty and {"animal", "block", "eye", "saccade_on_ms"}.issubset(df.columns):
        for (animal, block), g in df.groupby(["animal", "block"], dropna=False):
            onsets_by_block[(str(animal), block)] = {
                "L": g.loc[g["eye"].astype(str) == "L", "saccade_on_ms"].to_numpy(dtype=float),
                "R": g.loc[g["eye"].astype(str) == "R", "saccade_on_ms"].to_numpy(dtype=float),
            }

    include_pairs = event_mode in {"all", "binocular"}
    include_mono = event_mode in {"all", "monocular"}
    rows: list[dict] = []
    n_skip_mode = 0
    n_skip_profile = 0
    n_pairs = 0
    n_mono = 0
    seen_pairs: set[tuple] = set()

    for _, row in df.iterrows():
        animal = str(row["animal"])
        block = row["block"]
        block_key = _row_block_key(animal, block)
        bundle = block_map.get(block_key)
        if bundle is None:
            continue

        eye = str(row["eye"])
        if eye not in {"L", "R"}:
            continue
        t0 = float(row["saccade_on_ms"])
        block_onsets = onsets_by_block.get((animal, block), {"L": np.array([]), "R": np.array([])})
        other_eye = "R" if eye == "L" else "L"
        is_binocular = _contra_has_event(block_onsets[other_eye], t0, contra_win)

        if event_mode == "monocular" and is_binocular:
            n_skip_mode += 1
            continue
        if event_mode == "binocular" and not is_binocular:
            n_skip_mode += 1
            continue

        if is_binocular and pair_merge_ms is not None and include_pairs:
            q = int(round(t0 / pair_merge_ms))
            pair_key = (animal, block, q)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

        if eye == "L":
            ipsi_df, contra_df = bundle.left, bundle.right
        else:
            ipsi_df, contra_df = bundle.right, bundle.left

        frame_ms = _estimate_frame_period_ms(ipsi_df, t0)
        ipsi_peak = _event_peak_deg_per_ms(row, ipsi_df)
        contra_peak = _resolve_contra_peak(
            contra_df,
            row,
            sample_mode=sample_mode,
            contra_sample_ms=contra_sample,
            frame_ms=frame_ms,
        )
        if not (np.isfinite(ipsi_peak) and np.isfinite(contra_peak)):
            n_skip_profile += 1
            continue

        if eye == "L":
            right_peak, left_peak = contra_peak, ipsi_peak
        else:
            right_peak, left_peak = ipsi_peak, contra_peak

        rec = {k: row[k] for k in row.index}
        rec["right_peak_v"] = right_peak
        rec["left_peak_v"] = left_peak
        rec["block_key"] = block_key
        rec["2f_source"] = "legacy_binocular" if is_binocular else "legacy_monocular"
        rows.append(rec)
        if is_binocular:
            n_pairs += 1
        else:
            n_mono += 1

    return _finalize_figure_2f_points(
        rows,
        cfg=cfg,
        bins=bins,
        apply_iqr_clip=apply_iqr_clip,
        n_skip_mode=n_skip_mode,
        n_skip_profile=n_skip_profile,
        n_pairs=n_pairs,
        n_mono=n_mono,
        pairing_mode="legacy_contra_table",
    )


def compare_figure_2f_to_reference(
    collected: Figure2fPoints,
    reference_pickle: Path | str,
    *,
    bins: int | None = None,
) -> dict[str, object]:
    """
    Compare a live Fig 2f collection to an archived ``figure_2f_nodowncast.pickle``.

    Returns counts, speed summaries, histogram deltas, and per-animal breakdown.
    """
    reference_pickle = Path(reference_pickle)
    with open(reference_pickle, "rb") as handle:
        ref = pickle.load(handle)

    ref_r = np.asarray(ref["right_eye_speeds"], dtype=float)
    ref_l = np.asarray(ref["left_eye_speeds"], dtype=float)
    cur_r = collected.right_peak_v
    cur_l = collected.left_peak_v
    cfg = collected.cfg
    hist_bins = int(bins if bins is not None else cfg.get("bins", 60))
    macro_range = tuple(cfg.get("macro_range", collected.macro_range))
    ref_macro = ref.get("macro", {})
    cur_macro = histogram2d_from_points(collected, view="macro")
    ref_norm = np.asarray(ref_macro.get("norm_counts", []), dtype=float)
    cur_norm = cur_macro["norm_counts"]
    hist_l1 = float(np.abs(ref_norm - cur_norm).sum()) if ref_norm.shape == cur_norm.shape else float("nan")

    def _speed_stats(r: np.ndarray, l: np.ndarray) -> dict[str, float]:
        if r.size == 0:
            return {"n": 0}
        return {
            "n": int(r.size),
            "r_min": float(np.nanmin(r)),
            "r_max": float(np.nanmax(r)),
            "l_min": float(np.nanmin(l)),
            "l_max": float(np.nanmax(l)),
            "r_median": float(np.nanmedian(r)),
            "l_median": float(np.nanmedian(l)),
        }

    per_animal: dict[str, dict[str, int]] = {}
    if not collected.points.empty and "animal" in collected.points.columns:
        for animal, g in collected.points.groupby(collected.points["animal"].astype(str)):
            per_animal[str(animal)] = {"n_points": int(len(g))}

    return {
        "reference_pickle": str(reference_pickle),
        "reference": _speed_stats(ref_r, ref_l),
        "current": _speed_stats(cur_r, cur_l),
        "n_delta": int(cur_r.size - ref_r.size),
        "n_ratio": float(cur_r.size / ref_r.size) if ref_r.size else float("nan"),
        "reference_vmax_all": float(ref.get("vmax_all", float("nan"))),
        "current_vmax_all": float(
            max(np.nanmax(cur_macro["norm_counts"]), 1e-12)
        ),
        "macro_hist_l1_delta": hist_l1,
        "macro_range": list(macro_range),
        "filters": {
            "event_mode": cfg.get("event_mode"),
            "pairing_mode": cfg.get("pairing_mode"),
            "require_head_stationary": cfg.get("require_head_stationary"),
            "exclude_animals": sorted(cfg.get("exclude_animals", [])),
            "contra_event_window_ms": cfg.get("contra_event_window_ms"),
        },
        "per_animal_points": per_animal,
    }


def summarize_figure_2f_event_table(
    tables: EventTables,
    cfg: dict | None = None,
) -> dict[str, object]:
    """Event-table counts before/after Fig 2f row filters (for reproduction reports)."""
    cfg = figure_2f_config(tables, cfg)
    raw = tables.all_saccades.copy()
    filtered = _apply_figure_2f_row_filters(raw, cfg, verbose=False)
    animals = sorted(raw["animal"].astype(str).unique()) if "animal" in raw.columns else []
    per_animal: dict[str, dict[str, int]] = {}
    if "animal" in raw.columns:
        for animal in animals:
            sub_raw = raw[raw["animal"].astype(str) == animal]
            sub_filt = filtered[filtered["animal"].astype(str) == animal]
            head_col = sub_raw.get("head_movement")
            per_animal[animal] = {
                "n_all": int(len(sub_raw)),
                "n_after_filters": int(len(sub_filt)),
                "n_still": int((head_col == False).sum()) if head_col is not None else 0,  # noqa: E712
                "n_moving": int((head_col == True).sum()) if head_col is not None else 0,  # noqa: E712
                "n_head_nan": int(head_col.isna().sum()) if head_col is not None else int(len(sub_raw)),
            }
    return {
        "n_events_total": int(len(raw)),
        "n_events_after_filters": int(len(filtered)),
        "animals_in_registry": animals,
        "exclude_animals": sorted(cfg.get("exclude_animals", [])),
        "per_animal": per_animal,
    }


def collect_figure_2f_points(
    tables: EventTables,
    *,
    cfg: dict | None = None,
    apply_iqr_clip: bool = True,
) -> Figure2fPoints:
    """
    Build (right, left) inter-ocular peak speeds for Fig 2f / ROI tools.

    Concurrent L/R onsets (``binocular.sync_diff_ms``, default 34 ms) contribute
    **one** point whose coordinates are the two detected saccade peaks — the
    contra eye is not re-sampled in a time window. Unpaired (monocular) events
    all stay in: ipsilateral peak from the event, contralateral value from the
    ±51 ms trace window so near-threshold contra motion is visible.
    """
    cfg = figure_2f_config(tables, cfg)
    pairing_mode = str(cfg.get("pairing_mode", "synced_ms")).lower()
    if pairing_mode in {"legacy", "legacy_contra_table"}:
        return collect_figure_2f_points_legacy(
            tables, cfg=cfg, apply_iqr_clip=apply_iqr_clip
        )

    event_mode = str(cfg.get("event_mode", "monocular")).lower()
    sample_mode = str(cfg.get("sample_mode", "contra_window")).lower()
    contra_sample = float(cfg.get("contra_sample_ms", 51.0))
    pair_ms = float(
        tables.params.get("binocular", {}).get("sync_diff_ms", 34.0)
    )
    bins = int(cfg.get("bins", 60))

    df = _apply_figure_2f_row_filters(tables.all_saccades, cfg)
    block_map = tables.block_dict
    rows: list[dict] = []
    n_skip_mode = 0
    n_skip_profile = 0
    n_pairs = 0
    n_mono = 0

    include_pairs = event_mode in {"all", "binocular"}
    include_mono = event_mode in {"all", "monocular"}

    grouped = []
    if not df.empty and {"animal", "block"}.issubset(df.columns):
        grouped = list(df.groupby(["animal", "block"], dropna=False))

    for (animal, block), g in grouped:
        bundle = block_map.get(_row_block_key(animal, block))
        if bundle is None:
            continue
        synced, mono = find_synced_saccades_ms(g, sync_diff_ms=pair_ms)

        if include_pairs and synced is not None and not synced.empty and "Main" in synced.columns:
            for _, pg in synced.groupby("Main"):
                l_rows = pg[pg["eye"].astype(str) == "L"]
                r_rows = pg[pg["eye"].astype(str) == "R"]
                if l_rows.empty or r_rows.empty:
                    n_skip_profile += 1
                    continue
                l_row = l_rows.iloc[0]
                r_row = r_rows.iloc[0]
                left_peak = _event_peak_deg_per_ms(l_row, bundle.left)
                right_peak = _event_peak_deg_per_ms(r_row, bundle.right)
                if not (np.isfinite(left_peak) and np.isfinite(right_peak)):
                    n_skip_profile += 1
                    continue
                rec = {k: l_row[k] for k in l_row.index if k not in {"Main", "Sub"}}
                rec["eye"] = "LR"
                rec["right_peak_v"] = right_peak
                rec["left_peak_v"] = left_peak
                rec["block_key"] = bundle.spec.block_key
                rec["2f_source"] = "binocular_peaks"
                rows.append(rec)
                n_pairs += 1
        elif (
            not include_pairs
            and synced is not None
            and not synced.empty
            and "Main" in synced.columns
        ):
            n_skip_mode += int(synced["Main"].nunique())

        if include_mono and mono is not None and not mono.empty:
            for _, row in mono.iterrows():
                eye = str(row["eye"])
                t0 = float(row["saccade_on_ms"])
                if eye == "L":
                    ipsi_df, contra_df = bundle.left, bundle.right
                else:
                    ipsi_df, contra_df = bundle.right, bundle.left
                frame_ms = _estimate_frame_period_ms(ipsi_df, t0)
                ipsi_peak = _event_peak_deg_per_ms(row, ipsi_df)
                contra_peak = _resolve_contra_peak(
                    contra_df,
                    row,
                    sample_mode=sample_mode,
                    contra_sample_ms=contra_sample,
                    frame_ms=frame_ms,
                )
                if not (np.isfinite(ipsi_peak) and np.isfinite(contra_peak)):
                    n_skip_profile += 1
                    continue
                if eye == "L":
                    right_peak, left_peak = contra_peak, ipsi_peak
                else:
                    right_peak, left_peak = ipsi_peak, contra_peak
                rec = {k: row[k] for k in row.index if k not in {"Main", "Sub"}}
                rec["right_peak_v"] = right_peak
                rec["left_peak_v"] = left_peak
                rec["block_key"] = bundle.spec.block_key
                rec["2f_source"] = "monocular_window"
                rows.append(rec)
                n_mono += 1
        elif not include_mono and mono is not None and not mono.empty:
            n_skip_mode += int(len(mono))

    return _finalize_figure_2f_points(
        rows,
        cfg=cfg,
        bins=bins,
        apply_iqr_clip=apply_iqr_clip,
        n_skip_mode=n_skip_mode,
        n_skip_profile=n_skip_profile,
        n_pairs=n_pairs,
        n_mono=n_mono,
        pairing_mode=pairing_mode,
    )


def histogram2d_from_points(
    points: Figure2fPoints,
    *,
    view: str = "macro",
) -> dict[str, np.ndarray]:
    """Normalized 2f histogram for macro or micro range."""
    rng = points.macro_range if view == "macro" else points.micro_range
    right = points.right_peak_v if not points.points.empty else np.array([])
    left = points.left_peak_v if not points.points.empty else np.array([])
    weights = points.weights if not points.points.empty else np.array([])
    return histogram2d_xy(right, left, weights, rng, points.bins)


def select_figure_2f_roi(
    points: Figure2fPoints,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> pd.DataFrame:
    """Return event rows whose (right_peak_v, left_peak_v) fall inside the ROI."""
    xa, xb = (float(min(x0, x1)), float(max(x0, x1)))
    ya, yb = (float(min(y0, y1)), float(max(y0, y1)))
    right = points.points["right_peak_v"].to_numpy(dtype=float)
    left = points.points["left_peak_v"].to_numpy(dtype=float)
    mask = (right >= xa) & (right <= xb) & (left >= ya) & (left <= yb)
    return points.points.loc[mask].reset_index(drop=True)


def export_figure_2f(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    """
    Inter-ocular peak-speed 2D histogram.

    Paper caption path: ``event_mode=all``, head-stationary, equal-animal
    weights. Concurrent pairs use both event peaks (one point); unpaired
    events use the ±51 ms contra-window sample. Macro/micro share ``vmax_all``.
    S3 is a separate still/moving exporter — this function does not write it.
    """
    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)
    collected = collect_figure_2f_points(tables)
    cfg = collected.cfg
    right = collected.right_peak_v
    left = collected.left_peak_v
    weights = collected.weights
    bins = collected.bins
    macro_range = collected.macro_range
    micro_range = collected.micro_range

    macro = histogram2d_from_points(collected, view="macro")
    micro = histogram2d_from_points(collected, view="micro")
    vmax_all = float(
        max(np.nanmax(macro["norm_counts"]), np.nanmax(micro["norm_counts"]), 1e-12)
    )

    data = {
        "figure_name": "figure_2f",
        "right_eye_speeds": right,
        "left_eye_speeds": left,
        "weights": weights,
        "bins": bins,
        "macro_range": macro_range,
        "micro_range": micro_range,
        "macro_n_ticks": 6,
        "micro_n_ticks": 6,
        "macro_tick_list": cfg.get("macro_tick_list", [0, 0.25, 0.5]),
        "micro_tick_list": cfg.get("micro_tick_list", [0, 0.05, 0.1]),
        "iqr_multiplier": cfg.get("iqr_multiplier", 60.0),
        "macro": macro,
        "micro": micro,
        "vmax_all": vmax_all,
        "event_mode": cfg.get("event_mode", "monocular"),
        "sample_mode": cfg.get("sample_mode", "contra_window"),
        "exclude_animals": sorted(cfg.get("exclude_animals", [])),
        "auto_view_limits": bool(cfg.get("auto_view_limits")),
        "macro_pct": float(cfg.get("macro_pct", 99.5)),
        "micro_frac_of_macro": float(cfg.get("micro_frac_of_macro", 0.2)),
        "versions": {"numpy": np.__version__},
    }
    pkl = metadata_dir / "figure_2f_nodowncast.pickle"
    write_pickle_with_meta(
        data,
        pkl,
        meta={
            "csv_choices": tables.csv_meta,
            "params": cfg,
            "figure": "2f",
            "n": int(right.size),
        "event_mode": str(cfg.get("event_mode", "monocular")),
        "sample_mode": str(cfg.get("sample_mode", "contra_window")),
        "exclude_animals": sorted(cfg.get("exclude_animals", [])),
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_2f_2h_2i.export_figure_2f",
    )

    turbo = plt.get_cmap("turbo", 256)
    colors = turbo(np.linspace(0, 1, 256))
    colors[0] = np.array([1, 1, 1, 1])
    cmap = mcolors.ListedColormap(colors)
    fig, axs = plt.subplots(1, 2, figsize=(3.2, 1.85), dpi=300, constrained_layout=True)
    for ax, hist, rng, title, ticks in (
        (axs[0], macro, macro_range, "Macro", data["macro_tick_list"]),
        (axs[1], micro, micro_range, "Micro", data["micro_tick_list"]),
    ):
        _draw_coupling_heatmap(ax, hist, rng, ticks, vmax=vmax_all, cmap=cmap)
        ax.set_title(title, fontsize=8)
    fig.savefig(figures_dir / "figure_2f.pdf", bbox_inches="tight", dpi=300)
    show_and_close(fig, show)
    # Standalone colorbar (matches reproduction figure_2f.py)
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax_all if vmax_all > 0 else 1)
    )
    sm.set_array([])
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Probability", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig_cbar.savefig(figures_dir / "figure_2f_colorbar.pdf", bbox_inches="tight", dpi=150)
    show_and_close(fig_cbar, show)
    return pkl


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


def export_figure_s3(tables: EventTables, out_dir: Path, *, show: bool = False) -> dict[str, Path]:
    """Head-still vs head-moving coupling histograms with a shared colorbar.

    Same collector as Fig 2f (one point per concurrent pair from event peaks;
    unpaired events keep the ±51 ms contra-window sample) and equal-animal
    weights, but **all** saccades (no head-stationary filter). The two PDFs
    share vmax; they do not share a colorbar with Fig 2f.
    """
    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)
    cfg = figure_2f_config(
        tables,
        {
            "event_mode": "all",
            "require_head_stationary": False,
            "exclude_animals": ["PV_62"],
            "pairing_mode": "legacy_contra_table",
            "macro_range": [0.0, 0.2],
            "micro_range": [0.0, 0.1],
            "bins": 100,
            "macro_tick_list": [0.0, 0.1, 0.2],
            "auto_view_limits": False,
        },
    )
    collected = collect_figure_2f_points(tables, cfg=cfg)
    points = collected.points
    rng = collected.macro_range
    ticks = list(cfg.get("macro_tick_list") or [0.0, 0.1, 0.2])
    cmap = _turbo_white0()

    if "head_movement" not in points.columns:
        raise ValueError("S3 needs head_movement labels on events")
    hm = points["head_movement"].fillna(False).astype(bool)
    still = Figure2fPoints(
        points=points.loc[~hm].reset_index(drop=True),
        macro_range=collected.macro_range,
        micro_range=collected.micro_range,
        bins=collected.bins,
        cfg=cfg,
    )
    moving = Figure2fPoints(
        points=points.loc[hm].reset_index(drop=True),
        macro_range=collected.macro_range,
        micro_range=collected.micro_range,
        bins=collected.bins,
        cfg=cfg,
    )
    hist_still = histogram2d_from_points(still, view="macro")
    hist_moving = histogram2d_from_points(moving, view="macro")
    vmax = float(
        max(
            np.nanmax(hist_still["norm_counts"]) if hist_still["norm_counts"].size else 0.0,
            np.nanmax(hist_moving["norm_counts"]) if hist_moving["norm_counts"].size else 0.0,
            1e-12,
        )
    )

    written: dict[str, Path] = {}
    for name, hist, n in (
        ("figure_S3_head_still.pdf", hist_still, int((~hm).sum())),
        ("figure_S3_head_moving.pdf", hist_moving, int(hm.sum())),
    ):
        fig, ax = plt.subplots(figsize=(1.7, 1.7), dpi=300)
        _draw_coupling_heatmap(ax, hist, rng, ticks, vmax=vmax, cmap=cmap)
        ax.set_title(f"n={n}", fontsize=8)
        out = figures_dir / name
        fig.savefig(out, bbox_inches="tight", dpi=300)
        show_and_close(fig, show)
        written[name] = out

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Probability", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    cbar_path = figures_dir / "figure_S3_colorbar.pdf"
    fig_cbar.savefig(cbar_path, bbox_inches="tight", dpi=150)
    show_and_close(fig_cbar, show)
    written["figure_S3_colorbar.pdf"] = cbar_path

    pkl = metadata_dir / "figure_S3.pickle"
    write_pickle_with_meta(
        {
            "figure_name": "figure_S3",
            "vmax": vmax,
            "n_still": int((~hm).sum()),
            "n_moving": int(hm.sum()),
            "macro_range": rng,
            "bins": collected.bins,
            "ticks": list(ticks),
            "pairing_mode": str(cfg.get("pairing_mode") or "legacy_contra_table"),
            "exclude_animals": list(cfg.get("exclude_animals") or []),
            "binocular_onset_window_ms": 60.0,
            "binocular_merge_note": (
                "Published S3 caption: contra onset within ±60 ms. "
                "Does not change the ±51 ms pairing used for the speed axes."
            ),
            "contra_sample_ms": float(cfg.get("contra_sample_ms", 51.0)),
            "still": hist_still,
            "moving": hist_moving,
        },
        pkl,
        meta={"csv_choices": tables.csv_meta, "params": cfg, "figure": "S3", "vmax": vmax},
        entrypoint="eye_tracking_system_tools.analysis.figures_2f_2h_2i.export_figure_s3",
    )
    written["figure_S3.pickle"] = pkl
    from eye_tracking_system_tools.analysis.plot_bundle import write_replot_script

    write_replot_script(Path(out_dir), "figure_s3")
    return written


def export_figure_2h(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    cfg = dict(tables.params.get("figure_2h", {}))
    nbins = int(cfg.get("nbins", 200))
    gmin = float(cfg.get("global_min", -10))
    gmax = float(cfg.get("global_max", 10))
    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)

    def _endpoints(df, eye):
        if df is None or df.empty or "eye" not in df.columns:
            return np.array([]), np.array([])
        sub = df[df["eye"] == eye]
        if sub.empty:
            return np.array([]), np.array([])
        x = (
            sub["delta_phi"].to_numpy(dtype=float)
            if "delta_phi" in sub.columns
            else np.array([])
        )
        y = (
            sub["delta_theta"].to_numpy(dtype=float)
            if "delta_theta" in sub.columns
            else np.array([])
        )
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    synced = tables.synced
    nons = tables.non_synced
    data = {
        "figure_name": "figure_2h",
        "synced": {
            "R": {"x": _endpoints(synced, "R")[0].astype(np.float32), "y": _endpoints(synced, "R")[1].astype(np.float32)},
            "L": {"x": _endpoints(synced, "L")[0].astype(np.float32), "y": _endpoints(synced, "L")[1].astype(np.float32)},
        },
        "non_synced": {
            "R": {"x": _endpoints(nons, "R")[0].astype(np.float32), "y": _endpoints(nons, "R")[1].astype(np.float32)},
            "L": {"x": _endpoints(nons, "L")[0].astype(np.float32), "y": _endpoints(nons, "L")[1].astype(np.float32)},
        },
        "rotation_catalog": None,
        "nbins": nbins,
        "global_min": gmin,
        "global_max": gmax,
    }
    pkl = metadata_dir / "figure_2h.pickle"
    write_pickle_with_meta(
        data,
        pkl,
        meta={"csv_choices": tables.csv_meta, "params": cfg, "figure": "2h"},
        entrypoint="eye_tracking_system_tools.analysis.figures_2f_2h_2i.export_figure_2h",
    )

    def _kde(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 5:
            return None
        try:
            k = gaussian_kde(np.vstack([x, y]))
        except Exception:
            return None
        xi, yi = np.mgrid[gmin:gmax:nbins * 1j, gmin:gmax:nbins * 1j]
        zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        s = float(np.sum(zi))
        if s > 0:
            zi /= s
        return zi

    panels = [
        ("synced", "R"),
        ("synced", "L"),
        ("non_synced", "R"),
        ("non_synced", "L"),
    ]
    zis = []
    for cond, eye in panels:
        x = data[cond][eye]["x"]
        y = data[cond][eye]["y"]
        zis.append(_kde(x, y))
    finite = [z for z in zis if z is not None]
    vmin, vmax = (0, 1) if not finite else (min(z.min() for z in finite), max(z.max() for z in finite))
    fig, axes = plt.subplots(2, 2, figsize=(4, 2.5), dpi=300, sharex=True, sharey=True)
    for ax, zi in zip(axes.ravel(), zis):
        if zi is not None:
            ax.imshow(zi.T, extent=[gmin, gmax, gmin, gmax], origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.plot(0, 0, "k+", ms=7, zorder=5)
        ax.annotate(
            "pre",
            (0, 0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            color="k",
        )
        ax.set_xlabel("Δφ post [deg]", fontsize=7)
        ax.set_ylabel("Δθ post [deg]", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_2h.pdf", format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    return pkl


def export_figure_2i(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    cfg = dict(tables.params.get("figure_2i", {}))
    num_bins = int(cfg.get("num_bins", 36))
    baseline = float(cfg.get("baseline_prob", 0.005))
    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)
    df = tables.all_saccades
    per_eye = {}
    rotation_catalog = {}
    for animal, adf in df.groupby("animal"):
        per_eye[str(animal)] = {}
        for eye in ("L", "R"):
            angles = adf[adf["eye"] == eye]["overall_angle_deg"].to_numpy(dtype=float)
            angles = angles[np.isfinite(angles)] % 360
            if angles.size == 0:
                continue
            counts, edges = np.histogram(angles, bins=num_bins, range=(0, 360), density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            # Dominant axis: bin of max density (directionless 0-180)
            dens180 = counts[: num_bins // 2] + counts[num_bins // 2 :]
            dom = float(centers[int(np.argmax(dens180))] % 180)
            # Shortest rotation to align axis to 0/180
            rot = ((0 - dom + 90) % 180) - 90
            rotated = (angles + rot) % 360
            r_counts, r_edges = np.histogram(rotated, bins=num_bins, range=(0, 360), density=True)
            r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
            per_eye[str(animal)][eye] = {
                "angles": angles.astype(np.float32),
                "counts": counts.astype(np.float32),
                "centers": centers.astype(np.float32),
                "rotated_angles": rotated.astype(np.float32),
                "rotated_counts": r_counts.astype(np.float32),
                "rotated_centers": r_centers.astype(np.float32),
                "dominant_axis": dom,
                "rotation_angle": float(rot),
                "n_saccades": int(angles.size),
            }
            rotation_catalog[(str(animal), eye)] = float(rot)

    data = {
        "figure_name": "figure_2i",
        "per_eye_histograms": per_eye,
        "rotation_catalog": rotation_catalog,
        "num_bins": num_bins,
        "baseline_prob": baseline,
    }
    pkl = metadata_dir / "figure_2i.pickle"
    write_pickle_with_meta(
        data,
        pkl,
        meta={"csv_choices": tables.csv_meta, "params": cfg, "figure": "2i"},
        entrypoint="eye_tracking_system_tools.analysis.figures_2f_2h_2i.export_figure_2i",
    )

    animals = sorted(per_eye.keys())
    color_map = build_color_map(animals, template="okabeito", order=animals)
    fig, axs = plt.subplots(1, 2, figsize=(5, 4), dpi=300, subplot_kw=dict(projection="polar"))
    for ax, eye in zip(axs, ("R", "L")):
        for animal in animals:
            if eye not in per_eye[animal]:
                continue
            ed = per_eye[animal][eye]
            theta = np.deg2rad(np.r_[ed["rotated_centers"], ed["rotated_centers"][0]])
            rho = np.r_[ed["rotated_counts"], ed["rotated_counts"][0]]
            ax.plot(
                theta,
                rho,
                lw=1.2,
                color=color_map[animal],
                label=f"{animal} (n={ed['n_saccades']})",
            )
        ring = np.linspace(0, 2 * np.pi, 512)
        ax.fill_between(ring, 0, baseline, alpha=0.15, color="gray", zorder=0)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_facecolor("white")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_title(f"{'Right' if eye == 'R' else 'Left'} eye", fontsize=9)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_2i.pdf", format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    return pkl


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


def _saccade_threshold_deg_per_frame(tables: EventTables, default: float) -> float:
    thr = (getattr(tables, "params", {}) or {}).get("saccade", {}).get(
        "speed_threshold_deg_per_frame"
    )
    if thr is None or not np.isfinite(float(thr)):
        return float(default)
    return float(thr)


def _colorbar_pdf(cmap, vmax: float, path: Path, *, show: bool = False) -> Path:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=float(vmax)))
    sm.set_array([])
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Probability", fontsize=8)
    fig_cbar.savefig(path, format="pdf", bbox_inches="tight", dpi=150)
    show_and_close(fig_cbar, show)
    return path


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


def export_figure_2f_lizard_mouse_all(
    lizard_tables: EventTables,
    mouse_tables: EventTables,
    out_dir: Path | str,
    *,
    show: bool = False,
    thr_lizard: float | None = None,
    thr_mouse: float | None = None,
) -> dict[str, Path]:
    """Side-by-side all-saccade 2f heatmaps with matched-percentile zoom.

    XY limits are independent. Mouse keeps its auto 99.5th-percentile xmax.
    Lizard uses the same percentile of lizard speeds, then snaps onto a 0.5
    grid so ticks are round and evenly spaced. Histogram uses the Fig 2f grid
    (``bins`` edges → bins-1 cells). Turbo with 0 = white.
    Writes a joint-vmax two-panel PDF and a separate-vmax two-panel PDF.
    """
    from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

    overrides = {
        "require_head_stationary": False,
        "event_mode": "all",
        "auto_view_limits": True,
    }
    liz = collect_figure_2f_points(
        lizard_tables, cfg=figure_2f_config(lizard_tables, overrides)
    )
    mou = collect_figure_2f_points(
        mouse_tables, cfg=figure_2f_config(mouse_tables, overrides)
    )
    thr_l = (
        float(thr_lizard)
        if thr_lizard is not None
        else _saccade_threshold_deg_per_frame(lizard_tables, 0.8)
    )
    thr_m = (
        float(thr_mouse)
        if thr_mouse is not None
        else _saccade_threshold_deg_per_frame(mouse_tables, 3.23)
    )
    pct = float(liz.cfg.get("macro_pct", mou.cfg.get("macro_pct", 99.5)))
    bins = int(max(liz.bins, mou.bins, 60))
    views = build_s8j_views(
        liz.right_peak_v,
        liz.left_peak_v,
        liz.weights,
        mou.right_peak_v,
        mou.left_peak_v,
        mou.weights,
        bins=bins,
        full_pct=pct,
        zoom_keep_pct=50.0,
        xmax_mouse_full=float(mou.macro_range[1]),
    )
    full_l, full_m = views["full_lizard"], views["full_mouse"]
    zoom_l, zoom_m = views["zoom_lizard"], views["zoom_mouse"]
    rng_l, ticks_l, hist_l, vmax_l = full_l["range"], full_l["ticks"], full_l["hist"], full_l["vmax"]
    rng_m, ticks_m, hist_m, vmax_m = full_m["range"], full_m["ticks"], full_m["hist"], full_m["vmax"]
    vmax_joint = views["vmax_joint_full"]
    cmap = _turbo_white0()

    bundle = begin_plot_bundle(
        out_dir,
        "s8_2f_lizard_mouse_all",
        kind="figure_2f_lizard_mouse",
        logic_key="figure_2f_lizard_mouse",
        params={
            "require_head_stationary": False,
            "event_mode": "all",
            "thr_lizard": thr_l,
            "thr_mouse": thr_m,
            "range_lizard": list(rng_l),
            "range_mouse": list(rng_m),
            "range_lizard_zoom": list(zoom_l["range"]),
            "range_mouse_zoom": list(zoom_m["range"]),
            "bins": bins,
            "macro_pct": pct,
            "zoom_keep_pct": 50.0,
            "axis_mode": "matched_percentile_plus_inner50",
        },
    )

    def _two_panel(
        hist_a,
        rng_a,
        ticks_a,
        title_a,
        hist_b,
        rng_b,
        ticks_b,
        title_b,
        vmax_pair: tuple[float, float],
        pdf_name: str,
    ) -> Path:
        fig, axs = plt.subplots(1, 2, figsize=(3.0, 1.7), dpi=300, constrained_layout=True)
        _draw_coupling_heatmap(axs[0], hist_a, rng_a, ticks_a, vmax=vmax_pair[0], cmap=cmap)
        axs[0].set_title(title_a, fontsize=8)
        _draw_coupling_heatmap(axs[1], hist_b, rng_b, ticks_b, vmax=vmax_pair[1], cmap=cmap)
        axs[1].set_title(title_b, fontsize=8)
        heat_pdf = bundle.plots_dir / pdf_name
        fig.savefig(heat_pdf, format="pdf", bbox_inches="tight", dpi=300)
        show_and_close(fig, show)
        return heat_pdf

    n_l = int(liz.right_peak_v.size)
    n_m = int(mou.right_peak_v.size)
    heat_joint = _two_panel(
        hist_l, rng_l, ticks_l, f"lizard all n={n_l}",
        hist_m, rng_m, ticks_m, f"mouse all n={n_m}",
        (vmax_joint, vmax_joint),
        "S8j_lizard_mouse_all_2f.pdf",
    )
    heat_sep = _two_panel(
        hist_l, rng_l, ticks_l, f"lizard all n={n_l}",
        hist_m, rng_m, ticks_m, f"mouse all n={n_m}",
        (vmax_l, vmax_m),
        "S8j_lizard_mouse_all_2f_separate.pdf",
    )
    zoom_joint = _two_panel(
        zoom_l["hist"], zoom_l["range"], zoom_l["ticks"],
        f"lizard inner 50% n={zoom_l['n_in']}",
        zoom_m["hist"], zoom_m["range"], zoom_m["ticks"],
        f"mouse inner 50% n={zoom_m['n_in']}",
        (views["vmax_joint_zoom"], views["vmax_joint_zoom"]),
        "S8j_lizard_mouse_zoom_2f.pdf",
    )
    zoom_sep = _two_panel(
        zoom_l["hist"], zoom_l["range"], zoom_l["ticks"],
        f"lizard inner 50% n={zoom_l['n_in']}",
        zoom_m["hist"], zoom_m["range"], zoom_m["ticks"],
        f"mouse inner 50% n={zoom_m['n_in']}",
        (zoom_l["vmax"], zoom_m["vmax"]),
        "S8j_lizard_mouse_zoom_2f_separate.pdf",
    )
    cbar_joint = _colorbar_pdf(cmap, vmax_joint, bundle.plots_dir / "S8j_colorbar.pdf", show=show)
    cbar_liz = _colorbar_pdf(cmap, vmax_l, bundle.plots_dir / "S8j_colorbar_lizard.pdf", show=show)
    cbar_mou = _colorbar_pdf(cmap, vmax_m, bundle.plots_dir / "S8j_colorbar_mouse.pdf", show=show)
    cbar_zoom = _colorbar_pdf(
        cmap, views["vmax_joint_zoom"], bundle.plots_dir / "S8j_colorbar_zoom.pdf", show=show
    )
    cbar_liz_z = _colorbar_pdf(
        cmap, zoom_l["vmax"], bundle.plots_dir / "S8j_colorbar_lizard_zoom.pdf", show=show
    )
    cbar_mou_z = _colorbar_pdf(
        cmap, zoom_m["vmax"], bundle.plots_dir / "S8j_colorbar_mouse_zoom.pdf", show=show
    )

    pkl = bundle.metadata_dir / "s8_2f_lizard_mouse_all.pkl"
    write_pickle_with_meta(
        {
            "lizard": {
                "right": liz.right_peak_v,
                "left": liz.left_peak_v,
                "weights": liz.weights,
                "range": rng_l,
                "ticks": ticks_l,
                "vmax": vmax_l,
                "threshold": thr_l,
                "hist": hist_l,
            },
            "mouse": {
                "right": mou.right_peak_v,
                "left": mou.left_peak_v,
                "weights": mou.weights,
                "range": rng_m,
                "ticks": ticks_m,
                "vmax": vmax_m,
                "threshold": thr_m,
                "hist": hist_m,
            },
            "bins": bins,
            "vmax_joint": vmax_joint,
            "cmap": "turbo_white0",
            "macro_pct": pct,
            "zoom_keep_pct": 50.0,
            "axis_mode": "matched_percentile_plus_inner50",
            "zoom": {
                "lizard": {k: zoom_l[k] for k in ("range", "ticks", "vmax", "hist", "n_in")},
                "mouse": {k: zoom_m[k] for k in ("range", "ticks", "vmax", "hist", "n_in")},
                "vmax_joint": views["vmax_joint_zoom"],
            },
        },
        pkl,
        meta={
            "n_lizard": int(liz.right_peak_v.size),
            "n_mouse": int(mou.right_peak_v.size),
            "thr_lizard": thr_l,
            "thr_mouse": thr_m,
            "xmax_lizard": rng_l[1],
            "xmax_mouse": rng_m[1],
            "xmax_lizard_zoom": zoom_l["range"][1],
            "xmax_mouse_zoom": zoom_m["range"][1],
            "macro_pct": pct,
            "zoom_keep_pct": 50.0,
            "axis_mode": "matched_percentile_plus_inner50",
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_2f_2h_2i.export_figure_2f_lizard_mouse_all",
    )
    finish_plot_bundle(bundle)
    return {
        "S8j_lizard_mouse_all_2f.pdf": heat_joint,
        "S8j_lizard_mouse_all_2f_separate.pdf": heat_sep,
        "S8j_lizard_mouse_zoom_2f.pdf": zoom_joint,
        "S8j_lizard_mouse_zoom_2f_separate.pdf": zoom_sep,
        "S8j_colorbar.pdf": cbar_joint,
        "S8j_colorbar_lizard.pdf": cbar_liz,
        "S8j_colorbar_mouse.pdf": cbar_mou,
        "S8j_colorbar_zoom.pdf": cbar_zoom,
        "S8j_colorbar_lizard_zoom.pdf": cbar_liz_z,
        "S8j_colorbar_mouse_zoom.pdf": cbar_mou_z,
        "_bundle_dir": bundle.bundle_dir,
        "s8_2f_lizard_mouse_all.pkl": pkl,
    }


def rebin_s8j_payload(
    data: dict,
    *,
    pct: float | None = None,
    zoom_keep_pct: float = 50.0,
) -> dict:
    """Recompute full + inner-50% S8j histograms from stored speeds."""
    data = dict(data)
    liz = dict(data.get("lizard") or {})
    mou = dict(data.get("mouse") or {})
    pct_use = float(pct if pct is not None else data.get("macro_pct") or 99.5)
    bins = int(data.get("bins") or 60)
    xmax_m = float((mou.get("range") or (0.0, 1.25))[1])
    views = build_s8j_views(
        liz.get("right"),
        liz.get("left"),
        liz.get("weights"),
        mou.get("right"),
        mou.get("left"),
        mou.get("weights"),
        bins=bins,
        full_pct=pct_use,
        zoom_keep_pct=float(zoom_keep_pct),
        xmax_mouse_full=xmax_m,
    )
    full_l, full_m = views["full_lizard"], views["full_mouse"]
    zoom_l, zoom_m = views["zoom_lizard"], views["zoom_mouse"]
    liz.update(
        {
            "range": full_l["range"],
            "ticks": full_l["ticks"],
            "vmax": full_l["vmax"],
            "hist": full_l["hist"],
        }
    )
    mou.update(
        {
            "range": full_m["range"],
            "ticks": full_m["ticks"],
            "vmax": full_m["vmax"],
            "hist": full_m["hist"],
        }
    )
    data["lizard"] = liz
    data["mouse"] = mou
    data["bins"] = views["bins"]
    data["vmax_joint"] = views["vmax_joint_full"]
    data["macro_pct"] = pct_use
    data["zoom_keep_pct"] = float(zoom_keep_pct)
    data["axis_mode"] = "matched_percentile_plus_inner50"
    data["zoom"] = {
        "lizard": {k: zoom_l[k] for k in ("range", "ticks", "vmax", "hist", "n_in")},
        "mouse": {k: zoom_m[k] for k in ("range", "ticks", "vmax", "hist", "n_in")},
        "vmax_joint": views["vmax_joint_zoom"],
    }
    return data
