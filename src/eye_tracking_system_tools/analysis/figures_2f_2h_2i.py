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
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [0.0, 0.5]
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
    event_mode = str(cfg.get("event_mode", "monocular")).lower()
    sample_mode = str(cfg.get("sample_mode", "contra_window")).lower()
    contra_sample = float(cfg.get("contra_sample_ms", 51.0))
    pair_ms = float(
        tables.params.get("binocular", {}).get("sync_diff_ms", 34.0)
    )
    exclude = {str(a) for a in cfg.get("exclude_animals", [])}
    bins = int(cfg.get("bins", 60))

    df = tables.all_saccades.copy()
    if exclude:
        df = df[~df["animal"].astype(str).isin(exclude)]
    if cfg.get("require_head_stationary"):
        if "head_movement" not in df.columns or df["head_movement"].notna().sum() == 0:
            print("[2f] WARNING: require_head_stationary but no labels; keeping all rows")
        else:
            before = len(df)
            df = df[df["head_movement"] == False]  # noqa: E712
            print(f"[2f] head_stationary filter: {before} → {len(df)}")

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

    print(
        f"[2f] event_mode={event_mode} sample_mode={sample_mode} kept={len(points)} "
        f"binocular_pairs={n_pairs} monocular={n_mono} "
        f"skip_mode={n_skip_mode} skip_profile={n_skip_profile} "
        f"exclude={sorted(exclude)} "
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
    fig, axs = plt.subplots(1, 2, figsize=(3, 1.7), dpi=300, constrained_layout=True)
    for ax, hist, rng, title, ticks in (
        (axs[0], macro, macro_range, "Macro", data["macro_tick_list"]),
        (axs[1], micro, micro_range, "Micro", data["micro_tick_list"]),
    ):
        ax.pcolormesh(
            hist["xedges"],
            hist["yedges"],
            hist["norm_counts"].T,
            cmap=cmap,
            vmin=0,
            vmax=vmax_all if vmax_all > 0 else 1,
            shading="flat",
        )
        ax.set_xlim(*rng)
        ax.set_ylim(*rng)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.plot([rng[0], rng[1]], [rng[0], rng[1]], ls="--", color="gray", lw=1)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Right max V [deg/ms]", fontsize=9)
        ax.set_ylabel("Left max V [deg/ms]", fontsize=9)
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        mesh = ax.collections[0] if ax.collections else None
        if mesh is not None:
            mesh.set_rasterized(True)
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
    ax.plot([rng[0], rng[1]], [rng[0], rng[1]], ls="--", color="gray", lw=1)
    ax.set_xlabel("Right max V [deg/ms]", fontsize=9)
    ax.set_ylabel("Left max V [deg/ms]", fontsize=9)
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
            "exclude_animals": [],
        },
    )
    collected = collect_figure_2f_points(tables, cfg=cfg)
    points = collected.points
    rng = collected.macro_range
    ticks = cfg.get("macro_tick_list", [0.0, 0.25, 0.5])
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
