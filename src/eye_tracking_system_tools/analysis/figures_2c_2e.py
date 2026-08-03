"""Figure 2c/2d (amp-binned pos/vel) and 2e (amp–peak-speed linear fit).

Fig 2c/2d follows ``plot_pos_and_vel_by_amp_bins_per_animal_kernel`` from
``main_sequence_analysis.ipynb``: true-dt angular velocity from eye traces,
peak alignment, histogram accumulation onto a fine time grid, and Gaussian
smoothing of numerator/denominator (``bandwidth_ms``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from eye_tracking_system_tools.analysis.colors import build_color_map
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.pipeline import EventTables, _row_block_key
from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def _ms_params(tables: EventTables) -> dict[str, Any]:
    return dict(tables.params.get("main_sequence", {}))


def _resolve_plot_animals(
    df: pd.DataFrame, plot_animals: list[str] | None
) -> list[str] | None:
    """
    Keep ``plot_animals`` when it intersects the event table; otherwise fall back.

    Paper YAML defaults to ``[PV_106]``. Flexible / mouse runs with other animals
    used to produce empty 2c/2d PDFs with no error — treat a total miss as “plot
    whoever is present”.
    """
    if plot_animals is None:
        return None
    present = {str(a) for a in df["animal"].dropna().unique()}
    keep = [a for a in plot_animals if a in present]
    if keep:
        return keep
    warnings.warn(
        f"main_sequence.plot_animals={plot_animals} matches none of the "
        f"animals in this run ({sorted(present)}); plotting all present animals.",
        UserWarning,
        stacklevel=2,
    )
    return None


def _peak_velocity_deg_per_ms(df: pd.DataFrame, frame_rate_fps: float) -> pd.Series:
    """
    Peak angular speed in deg/ms for Fig 2e.

    Event tables store ``peak_velocity`` in deg/frame (paper convention). Convert
    with ``peak_dpf * (fps / 1000)`` as in ``main_sequence_analysis.ipynb``.
    """
    frame_duration_ms = 1000.0 / float(frame_rate_fps)
    if "peak_velocity" in df.columns and df["peak_velocity"].notna().any():
        return df["peak_velocity"].astype(float) / frame_duration_ms

    def _one(sp):
        if sp is None:
            return np.nan
        arr = np.asarray(sp, dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return np.nan
        return float(np.nanmax(arr) / frame_duration_ms)

    return df["speed_profile_angular"].map(_one)


def _bin_viridis_colors(n: int, cmap_name: str = "viridis") -> list[tuple]:
    """Paper Fig 2c: viridis sampled at linspace(0, 1, n)."""
    cmap = plt.get_cmap(cmap_name)
    if n <= 0:
        return []
    if n == 1:
        return [tuple(float(x) for x in cmap(0.0))]
    return [tuple(float(x) for x in cmap(i / (n - 1))) for i in range(n)]


def _ensure_eye_traces(tables: EventTables, df: pd.DataFrame) -> EventTables:
    """Reload left/right CSVs for blocks referenced by ``df`` when traces are empty."""
    from eye_tracking_system_tools.analysis.event_cache import ensure_traces_for_blocks

    keys = sorted(
        {
            _row_block_key(a, b)
            for a, b in zip(df["animal"].tolist(), df["block"].tolist())
        }
    )
    if not keys:
        return tables
    return ensure_traces_for_blocks(tables, keys)


def _eye_dataframe(
    tables: EventTables, animal: Any, block: Any, eye: Any
) -> pd.DataFrame | None:
    key = _row_block_key(animal, block)
    bundle = tables.block_dict.get(key)
    if bundle is None:
        return None
    eye_u = str(eye).upper()
    if eye_u.startswith("L"):
        df = bundle.left
    elif eye_u.startswith("R"):
        df = bundle.right
    else:
        return None
    if df is None or df.empty:
        return None
    return df


def _raw_velocity_trace(
    eye_df: pd.DataFrame,
    *,
    framerate_hz: float | None = None,
    use_fixed_dt: bool = False,
    velocity_unit: str = "deg/sec",
) -> tuple[np.ndarray, np.ndarray]:
    """Angular speed from k_phi/k_theta with true Δt (notebook ``_raw_velocity_trace_local``)."""
    t = eye_df["ms_axis"].to_numpy(dtype=float)
    phi = eye_df["k_phi"].to_numpy(dtype=float)
    th = eye_df["k_theta"].to_numpy(dtype=float)
    dphi = np.diff(phi)
    dth = np.diff(th)
    if use_fixed_dt and framerate_hz and np.isfinite(framerate_hz) and framerate_hz > 0:
        dt_s = np.full_like(dphi, 1.0 / float(framerate_hz), dtype=float)
    else:
        dt_s = np.diff(t) / 1000.0
    v = np.divide(
        np.hypot(dphi, dth),
        dt_s,
        out=np.full_like(dphi, np.nan),
        where=dt_s > 0,
    )
    unit = str(velocity_unit).lower()
    if unit in {"deg/ms", "dpms"}:
        v = v / 1000.0
    return t[1:], v


def _peak_time_in_window(
    t_ms: np.ndarray,
    v: np.ndarray,
    on_ms: float,
    off_ms: float,
    *,
    subsample_peak: bool = False,
) -> float:
    m = (t_ms >= on_ms) & (t_ms <= off_ms) & np.isfinite(v)
    if not np.any(m):
        return float("nan")
    vv = v[m]
    tloc = t_ms[m]
    i = int(np.nanargmax(vv))
    if (not subsample_peak) or i == 0 or i == vv.size - 1:
        return float(tloc[i])
    v_1, v0, v1 = vv[i - 1], vv[i], vv[i + 1]
    denom = v_1 - 2 * v0 + v1
    if not np.isfinite(denom) or denom == 0:
        return float(tloc[i])
    d = 0.5 * (v_1 - v1) / denom
    d = float(np.clip(d, -1.0, 1.0))
    dt_local = (tloc[min(i + 1, vv.size - 1)] - tloc[max(i - 1, 0)]) / 2.0
    return float(tloc[i] + d * dt_local)


def export_pos_vel_bundle(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    """
    Export mean vel/pos traces by amplitude bin (Fig 2c/2d).

    Matches the paper notebook kernel path: reload eye traces if needed, compute
    true-dt angular velocity, align to peak (or onset), accumulate onto ``t_grid``,
    and optionally Gaussian-smooth numerator/denominator with ``bandwidth_ms``.
    """
    p = _ms_params(tables)
    amp_col = p.get("amp_col", "net_angular_disp")
    bin_width = float(p.get("bin_width_deg", 5.0))
    min_amp = float(p.get("min_amp_deg", 0.5))
    max_amp_pct = float(p.get("max_amp_pct", 99.5))
    t_window = tuple(p.get("t_window_ms", [-100.0, 100.0]))
    dt_ms = float(p.get("dt_ms", 2.0))
    bandwidth_ms = float(p.get("bandwidth_ms", 10.0))
    min_events = int(p.get("min_events_per_bin_traces", 30))
    max_bins = int(p.get("per_animal_max_bins", 8))
    bin_cmap = str(p.get("bin_cmap", "viridis"))
    smoothing = bool(p.get("smoothing", True))
    subsample_peak = bool(p.get("subsample_peak", False))
    align_to = str(p.get("align_to", "peak")).lower()
    position_calc = str(p.get("position_calc", "mov_axis")).lower()
    velocity_unit = str(p.get("velocity_unit", "deg/sec"))
    framerate_hz = p.get("framerate_hz")
    framerate_hz = float(framerate_hz) if framerate_hz is not None else None
    use_fixed_dt = bool(p.get("use_fixed_dt", False))
    normalize_velocity_to_peak = bool(p.get("normalize_velocity_to_peak", False))
    normalize_position_to_amp = bool(p.get("normalize_position_to_amp", False))
    normalize_final_curve = bool(p.get("normalize_final_curve", False))
    eye_filter = p.get("eye_filter")
    if eye_filter is not None:
        eye_filter = str(eye_filter)
    plot_animals = p.get("plot_animals")
    if plot_animals is not None:
        plot_animals = [str(a) for a in plot_animals]

    if align_to not in {"peak", "onset"}:
        raise ValueError("align_to must be 'peak' or 'onset'")
    if position_calc not in {"mov_axis", "integrate"}:
        raise ValueError("position_calc must be 'mov_axis' or 'integrate'")

    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)
    tmin, tmax = float(t_window[0]), float(t_window[1])
    t_grid = np.arange(tmin, tmax + dt_ms * 0.5, dt_ms, dtype=np.float64)
    nb = len(t_grid)
    sigma_bins = max(1.0, bandwidth_ms / dt_ms) if bandwidth_ms > 0 else 0.0

    df = tables.all_saccades.copy()
    if df.empty or amp_col not in df.columns:
        raise ValueError("No saccades available for pos/vel export")
    req = {"animal", "block", "eye", "saccade_on_ms", "saccade_off_ms"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"all_saccades missing columns for kernel 2c/2d: {sorted(missing)}")

    plot_animals = _resolve_plot_animals(df, plot_animals)
    if eye_filter:
        df = df[df["eye"].astype(str).str.upper() == eye_filter.upper()].copy()

    tables = _ensure_eye_traces(tables, df)

    # Cache per-eye arrays so we do not recompute velocity for every event.
    eye_cache: dict[tuple[str, str, str], dict[str, np.ndarray] | None] = {}

    def _cached_eye(animal: Any, block: Any, eye: Any) -> dict[str, np.ndarray] | None:
        key = (str(animal), str(block), str(eye).upper())
        if key in eye_cache:
            return eye_cache[key]
        eye_df = _eye_dataframe(tables, animal, block, eye)
        if eye_df is None or not {"ms_axis", "k_phi", "k_theta"}.issubset(eye_df.columns):
            eye_cache[key] = None
            return None
        t = eye_df["ms_axis"].to_numpy(dtype=float)
        phi = eye_df["k_phi"].to_numpy(dtype=float)
        th = eye_df["k_theta"].to_numpy(dtype=float)
        t_ms, v_vel = _raw_velocity_trace(
            eye_df,
            framerate_hz=framerate_hz,
            use_fixed_dt=use_fixed_dt,
            velocity_unit=velocity_unit,
        )
        packed = {"t": t, "phi": phi, "th": th, "t_ms": t_ms, "v_vel": v_vel}
        eye_cache[key] = packed
        return packed

    animals_out: dict[str, Any] = {}
    for animal, adf in df.groupby("animal"):
        animal_s = str(animal)
        if plot_animals is not None and animal_s not in plot_animals:
            continue
        sub = adf[pd.to_numeric(adf[amp_col], errors="coerce") >= min_amp].copy()
        if sub.empty:
            continue

        max_amp = float(np.nanpercentile(sub[amp_col], max_amp_pct))
        edges = np.arange(0.0, max_amp + bin_width, bin_width)
        if len(edges) > max_bins + 1:
            edges = edges[: max_bins + 1]
        if len(edges) < 2:
            continue
        labels = [f"{int(edges[i])}-{int(edges[i + 1])}°" for i in range(len(edges) - 1)]
        sub["amp_bin"] = pd.cut(
            sub[amp_col],
            bins=edges,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        series: list[dict[str, Any]] = []
        kept_labels: list[str] = []
        used_edges = [float(edges[0])]

        for lab in labels:
            g = sub[sub["amp_bin"] == lab]
            n_events = int(len(g))
            if n_events < min_events:
                continue

            v_num = np.zeros(nb, dtype=float)
            v_den = np.zeros(nb, dtype=float)
            p_num = np.zeros(nb, dtype=float)
            p_den = np.zeros(nb, dtype=float)

            for _, row in g.iterrows():
                packed = _cached_eye(row["animal"], row["block"], row["eye"])
                if packed is None:
                    continue
                t = packed["t"]
                phi = packed["phi"]
                th = packed["th"]
                t_ms = packed["t_ms"]
                v_vel = packed["v_vel"]
                if t_ms.size < 3:
                    continue

                on_t = float(row["saccade_on_ms"])
                off_t = float(row["saccade_off_ms"])
                on_i = int(np.argmin(np.abs(t - on_t)))
                off_i = int(np.argmin(np.abs(t - off_t)))
                if off_i <= on_i:
                    continue
                dphi = phi[off_i] - phi[on_i]
                dth = th[off_i] - th[on_i]
                amp = float(np.hypot(dphi, dth))
                if not np.isfinite(amp) or amp <= 0:
                    continue
                ux, uy = dphi / amp, dth / amp

                if align_to == "peak":
                    pk_ms = _peak_time_in_window(
                        t_ms, v_vel, on_t, off_t, subsample_peak=subsample_peak
                    )
                    if not np.isfinite(pk_ms):
                        continue
                    t0 = pk_ms
                else:
                    t0 = on_t

                m = (t_ms >= t0 + tmin) & (t_ms <= t0 + tmax) & np.isfinite(v_vel)
                if not np.any(m):
                    continue

                t_rel = t_ms[m] - t0
                v = v_vel[m]
                idx = np.floor((t_rel - tmin) / dt_ms).astype(int)
                ok = (idx >= 0) & (idx < nb)
                if not np.any(ok):
                    continue
                np.add.at(v_num, idx[ok], v[ok])
                np.add.at(v_den, idx[ok], 1.0)

                if position_calc == "mov_axis":
                    phi_seg = phi[1:][m]
                    th_seg = th[1:][m]
                    pos_along = (phi_seg - phi[on_i]) * ux + (th_seg - th[on_i]) * uy
                else:
                    dt_s_true = np.diff(t) / 1000.0
                    v_phi = np.divide(
                        np.diff(phi),
                        dt_s_true,
                        out=np.full_like(dt_s_true, np.nan),
                        where=dt_s_true > 0,
                    )
                    v_th = np.divide(
                        np.diff(th),
                        dt_s_true,
                        out=np.full_like(dt_s_true, np.nan),
                        where=dt_s_true > 0,
                    )
                    v_axis = v_phi * ux + v_th * uy
                    vax = v_axis[m]
                    trel_s = (t_ms[m] - t0) / 1000.0
                    if trel_s.size < 2 or np.all(~np.isfinite(vax)):
                        continue
                    dv = 0.5 * (vax[1:] + vax[:-1]) * np.diff(trel_s)
                    pos_along = np.concatenate(([0.0], np.cumsum(np.nan_to_num(dv))))

                if pos_along.shape[0] != ok.shape[0]:
                    # Guard against rare length mismatches between vel/pos masks.
                    n = min(pos_along.shape[0], ok.shape[0])
                    pos_use = pos_along[:n]
                    ok_pos = ok[:n]
                    idx_pos = idx[:n]
                else:
                    pos_use = pos_along
                    ok_pos = ok
                    idx_pos = idx
                if normalize_position_to_amp:
                    pos_use = pos_use / amp
                np.add.at(p_num, idx_pos[ok_pos], pos_use[ok_pos])
                np.add.at(p_den, idx_pos[ok_pos], 1.0)

            if v_den.sum() == 0 and p_den.sum() == 0:
                continue

            if v_den.sum() > 0:
                if smoothing and sigma_bins > 0:
                    v_num_s = gaussian_filter1d(v_num, sigma=sigma_bins, mode="nearest")
                    v_den_s = gaussian_filter1d(v_den, sigma=sigma_bins, mode="nearest")
                    vel_center = np.divide(
                        v_num_s,
                        v_den_s,
                        out=np.full(nb, np.nan),
                        where=v_den_s > 1e-6,
                    )
                else:
                    vel_center = np.divide(
                        v_num,
                        v_den,
                        out=np.full(nb, np.nan),
                        where=v_den > 1e-6,
                    )
                if normalize_velocity_to_peak:
                    v_peak = np.nanmax(vel_center)
                    if np.isfinite(v_peak) and v_peak > 0:
                        vel_center = vel_center / v_peak
            else:
                vel_center = np.full(nb, np.nan)

            if p_den.sum() > 0:
                if smoothing and sigma_bins > 0:
                    p_num_s = gaussian_filter1d(p_num, sigma=sigma_bins, mode="nearest")
                    p_den_s = gaussian_filter1d(p_den, sigma=sigma_bins, mode="nearest")
                    pos_center = np.divide(
                        p_num_s,
                        p_den_s,
                        out=np.full(nb, np.nan),
                        where=p_den_s > 1e-6,
                    )
                else:
                    pos_center = np.divide(
                        p_num,
                        p_den,
                        out=np.full(nb, np.nan),
                        where=p_den > 1e-6,
                    )
                if normalize_final_curve:
                    p_min = np.nanmin(pos_center)
                    p_max = np.nanmax(pos_center)
                    if np.isfinite(p_min) and np.isfinite(p_max) and p_max > p_min:
                        pos_center = (pos_center - p_min) / (p_max - p_min)
            else:
                pos_center = np.full(nb, np.nan)

            if not np.any(np.isfinite(vel_center)) and not np.any(np.isfinite(pos_center)):
                continue

            lo_i = labels.index(lab)
            lo, hi = float(edges[lo_i]), float(edges[lo_i + 1])
            kept_labels.append(lab)
            used_edges.append(hi)
            series.append(
                {
                    "raw_label": lab,
                    "label": f"{lab} (n={n_events})",
                    "n_events": n_events,
                    "color_rgba": (0.0, 0.0, 0.0, 1.0),
                    "vel_center": vel_center.astype(np.float32),
                    "pos_center": pos_center.astype(np.float32),
                    "time_axis": t_grid.astype(np.float32),
                }
            )

        for s, color in zip(series, _bin_viridis_colors(len(series), bin_cmap)):
            s["color_rgba"] = color
        animals_out[animal_s] = {
            "animal": animal_s,
            "labels": kept_labels,
            "edges": used_edges,
            "series": series,
        }

    if not animals_out or all(not v.get("series") for v in animals_out.values()):
        present = sorted({str(a) for a in df["animal"].dropna().unique()})
        n_trace = sum(
            1
            for b in tables.blocks
            if b.left is not None
            and not b.left.empty
            and b.right is not None
            and not b.right.empty
        )
        raise ValueError(
            "No amp-bin velocity/position traces to plot "
            f"(animals in table={present}, plot_animals={plot_animals}, "
            f"min_events_per_bin_traces={min_events}, amp_col={amp_col}, "
            f"blocks_with_traces={n_trace}/{len(tables.blocks)}). "
            "Ensure eye CSVs with k_phi/k_theta/ms_axis are loadable."
        )

    params = {
        "amp_col": amp_col,
        "bin_width_deg": bin_width,
        "min_amp_deg": min_amp,
        "max_amp_pct": max_amp_pct,
        "t_window_ms": (tmin, tmax),
        "dt_ms": dt_ms,
        "bandwidth_ms": bandwidth_ms,
        "min_events_per_bin": min_events,
        "per_animal_max_bins": max_bins,
        "plot_animals": plot_animals,
        "eye_filter": eye_filter,
        "subsample_peak": subsample_peak,
        "smoothing": smoothing,
        "plot_bins_as_scatter": False,
        "min_bin_count": 3,
        "normalize_velocity_to_peak": normalize_velocity_to_peak,
        "normalize_position_to_amp": normalize_position_to_amp,
        "normalize_final_curve": normalize_final_curve,
        "velocity_threshold": p.get("velocity_threshold", 50.0),
        "legend_ncol": 2,
        "bin_cmap": bin_cmap,
        "show_position": True,
        "show_velocity": True,
        "vel_ylim": None,
        "pos_ylim": None,
        "align_to": align_to,
        "position_calc": position_calc,
        "velocity_unit": velocity_unit,
        "framerate_hz": framerate_hz,
        "use_fixed_dt": use_fixed_dt,
        "fig_size": tuple(p.get("fig_size", (1.8, 1.8))),
        "dpi": int(p.get("dpi", 300)),
        "lw": float(p.get("lw", 1.0)),
    }
    bundle = {"t_grid": t_grid.astype(np.float32), "params": params, "animals": animals_out}
    pkl = metadata_dir / "pos_vel_by_amp_bins_bundle.pkl"
    write_pickle_with_meta(
        bundle,
        pkl,
        meta={"csv_choices": tables.csv_meta, "params": params, "figure": "2c/2d"},
        entrypoint="eye_tracking_system_tools.analysis.figures_2c_2e.export_pos_vel_bundle",
    )

    _plot_pos_vel_pdfs(bundle, figures_dir, show=show)
    return pkl


def _export_amp_bin_legend_pdf(
    series_list: list[dict[str, Any]],
    out_path: Path,
    *,
    ncol: int = 2,
    fontsize: int = 6,
    title: str | None = None,
    show: bool = False,
) -> Path | None:
    """Standalone legend: amp-bin color → label including per-bin event counts."""
    if not series_list:
        return None
    handles = [
        plt.Line2D([0], [0], color=s["color_rgba"], lw=2, label=s["label"])
        for s in series_list
        if s.get("label")
    ]
    if not handles:
        return None
    n = len(handles)
    w = max(2.2, 0.75 * min(ncol, n) * max(1, (n + ncol - 1) // ncol))
    h = max(0.7, 0.28 * ((n + ncol - 1) // ncol) + (0.35 if title else 0.15))
    fig_leg = plt.figure(figsize=(w, h), dpi=300)
    fig_leg.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        frameon=False,
        fontsize=fontsize,
        ncol=ncol,
        loc="center",
        title=title,
        title_fontsize=fontsize + 1,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_leg.savefig(out_path, bbox_inches="tight", dpi=300)
    show_and_close(fig_leg, show)
    return out_path


def _plot_pos_vel_pdfs(
    bundle: dict, figures_dir: Path, *, show: bool = False
) -> dict[str, Path]:
    """
    One velocity (2c) + position (2d) figure **per animal**.

    Events from all blocks of an animal are already pooled in ``bundle["animals"]``.
    Paper-canonical ``figure_2c.pdf`` / ``figure_2d.pdf`` are written for the
    primary animal (sole animal, or first of ``plot_animals``). Per-animal copies
    and standalone amp-bin legend PDFs (with ``n=`` counts) are always written.
    """
    params = bundle["params"]
    t_grid = bundle["t_grid"]
    plot_animals = params.get("plot_animals")
    animals = dict(bundle["animals"])
    if plot_animals is not None:
        animals = {k: v for k, v in animals.items() if k in set(plot_animals)}

    if not animals:
        return {}

    if len(animals) > 1 and plot_animals is None:
        warnings.warn(
            f"Fig 2c/2d: {len(animals)} animals present ({sorted(animals)}); "
            "writing one velocity/position pair per animal (blocks already pooled). "
            "Set main_sequence.plot_animals: [ANIMAL] for a single paper panel.",
            UserWarning,
            stacklevel=2,
        )

    align_to = str(params.get("align_to", "peak"))
    xlabel = f"Time from {align_to} (ms)"
    vel_unit = params.get("velocity_unit", "deg/sec")
    if params.get("normalize_velocity_to_peak"):
        vel_ylabel = "Velocity (norm. to peak)"
    elif str(vel_unit).lower() in {"deg/ms", "dpms"}:
        vel_ylabel = "Velocity (deg/ms)"
    else:
        vel_ylabel = f"Angular speed ({vel_unit})"

    if params.get("normalize_final_curve"):
        pos_ylabel = "Position (final norm 0–1)"
    elif params.get("normalize_position_to_amp"):
        pos_ylabel = "Position (norm. amp)"
    else:
        pos_ylabel = "Position (deg)"

    fig_size = tuple(params.get("fig_size", (1.8, 1.8)))
    dpi = int(params.get("dpi", 300))
    lw = float(params.get("lw", 1.0))
    legend_ncol = int(params.get("legend_ncol", 2))
    # Larger canvas when displaying inline so the amp-bin legend is readable.
    show_size = tuple(params.get("show_fig_size", (3.6, 2.8)))

    primary_animal = next(iter(animals))
    written: dict[str, Path] = {}

    for animal, animal_data in animals.items():
        series_list = list(animal_data.get("series") or [])
        if not series_list:
            continue
        n_total = int(sum(int(s.get("n_events", 0)) for s in series_list))
        print(
            f"[2c/2d] {animal}: {len(series_list)} amp bins, "
            f"{n_total} events (pooled across blocks)"
        )
        for s in series_list:
            print(f"         {s.get('label')}")

        for kind, key, ylabel, canon_name in (
            ("vel", "vel_center", vel_ylabel, "figure_2c.pdf"),
            ("pos", "pos_center", pos_ylabel, "figure_2d.pdf"),
        ):
            fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
            for series in series_list:
                y = series.get(key)
                if y is None:
                    continue
                tax = series.get("time_axis", t_grid)
                ax.plot(
                    tax,
                    y,
                    color=series["color_rgba"],
                    lw=lw,
                    label=series["label"],
                )
            ax.set_title(str(animal), fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(float(t_grid[0]), float(t_grid[-1]))
            fig.tight_layout()

            animal_pdf = figures_dir / f"{Path(canon_name).stem}_{animal}.pdf"
            fig.savefig(animal_pdf, format="pdf", bbox_inches="tight")
            written[animal_pdf.name] = animal_pdf
            if animal == primary_animal:
                canon_pdf = figures_dir / canon_name
                fig.savefig(canon_pdf, format="pdf", bbox_inches="tight")
                written[canon_name] = canon_pdf

            legend_name = f"{Path(canon_name).stem}_{animal}_legend.pdf"
            legend_pdf = _export_amp_bin_legend_pdf(
                series_list,
                figures_dir / legend_name,
                ncol=legend_ncol,
                fontsize=6,
                title=f"{animal} amp bins",
                show=False,
            )
            if legend_pdf is not None:
                written[legend_name] = legend_pdf
                if animal == primary_animal:
                    canon_leg = figures_dir / f"{Path(canon_name).stem}_legend.pdf"
                    legend_pdf_bytes = legend_pdf.read_bytes()
                    canon_leg.write_bytes(legend_pdf_bytes)
                    written[canon_leg.name] = canon_leg

            if show:
                # Notebook view: larger axes + on-figure legend with n= counts.
                fig_show, ax_show = plt.subplots(figsize=show_size, dpi=120)
                for series in series_list:
                    y = series.get(key)
                    if y is None:
                        continue
                    tax = series.get("time_axis", t_grid)
                    ax_show.plot(
                        tax,
                        y,
                        color=series["color_rgba"],
                        lw=1.5,
                        label=series["label"],
                    )
                ax_show.set_title(f"{animal} — {kind}", fontsize=10)
                ax_show.set_xlabel(xlabel, fontsize=9)
                ax_show.set_ylabel(ylabel, fontsize=9)
                ax_show.tick_params(labelsize=8)
                ax_show.spines["top"].set_visible(False)
                ax_show.spines["right"].set_visible(False)
                ax_show.set_xlim(float(t_grid[0]), float(t_grid[-1]))
                ax_show.legend(
                    frameon=False,
                    fontsize=7,
                    ncol=1,
                    loc="best",
                    title="Amplitude bin",
                    title_fontsize=8,
                )
                fig_show.tight_layout()
                show_and_close(fig_show, True)

            plt.close(fig)

    return written


def export_amplitude_velocity_fit(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    p = _ms_params(tables)
    amp_col = p.get("amp_col", "net_angular_disp")
    bin_width = float(p.get("bin_width_deg", 5.0))
    min_amp = float(p.get("min_amp_deg", 0.5))
    max_amp_pct = float(p.get("max_amp_pct", 99.5))
    min_events = int(p.get("min_events_per_bin", 15))
    frame_rate = float(p.get("frame_rate_fps", 60.0))

    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)

    df = tables.all_saccades.copy()
    if df.empty:
        raise ValueError("No saccades for 2e export")
    df["peak_velocity"] = _peak_velocity_deg_per_ms(df, frame_rate)
    df = df[np.isfinite(df[amp_col]) & np.isfinite(df["peak_velocity"]) & (df[amp_col] >= min_amp)]
    amp_hi = float(np.nanpercentile(df[amp_col], max_amp_pct))
    df = df[df[amp_col] <= amp_hi]

    edges = np.arange(0.0, max(25.0, np.ceil(amp_hi / bin_width) * bin_width) + 1e-9, bin_width)
    per_animal_stats: dict[str, pd.DataFrame] = {}
    linear_rows = []
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
        if len(adf) >= 3:
            slope, intercept, r, pval, se = stats.linregress(
                adf[amp_col].to_numpy(float), adf["peak_velocity"].to_numpy(float)
            )
            linear_rows.append(
                {
                    "animal": str(animal),
                    "n": int(len(adf)),
                    "slope": float(slope),
                    "se_slope": float(se),
                    "t_slope": float(slope / se) if se else np.nan,
                    "p_slope": float(pval),
                    "intercept": float(intercept),
                    "se_intercept": np.nan,
                    "t_intercept": np.nan,
                    "p_intercept": np.nan,
                    "R2": float(r**2),
                }
            )

    slope, intercept, r, pval, se = stats.linregress(
        df[amp_col].to_numpy(float), df["peak_velocity"].to_numpy(float)
    )
    global_fit = {
        "n": int(len(df)),
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(r**2),
        "se_slope": float(se),
        "t_slope": float(slope / se) if se else np.nan,
        "p_slope": float(pval),
        "se_intercept": np.nan,
        "t_intercept": np.nan,
        "p_intercept": np.nan,
    }
    params = {
        "amp_col": amp_col,
        "bin_width_deg": bin_width,
        "min_amp_deg": min_amp,
        "max_amp_pct": max_amp_pct,
        "min_events_per_bin": min_events,
        "frame_rate_fps": frame_rate,
        "figsize": (1.5, 1.7),
        "dpi": 300,
        "lw": 1.0,
    }
    bundle = {
        "params": params,
        "edges": edges,
        "per_animal_stats": per_animal_stats,
        "global_fit": global_fit,
        "linear_stats_df": pd.DataFrame(linear_rows),
    }
    pkl = metadata_dir / "amplitude_velocity_linear_fit_bundle.pkl"
    write_pickle_with_meta(
        bundle,
        pkl,
        meta={"csv_choices": tables.csv_meta, "params": params, "figure": "2e", "n": global_fit["n"]},
        entrypoint="eye_tracking_system_tools.analysis.figures_2c_2e.export_amplitude_velocity_fit",
    )

    animals = list(per_animal_stats.keys())
    color_map = build_color_map(animals, template="okabeito", order=animals)
    fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
    for animal, sdf in per_animal_stats.items():
        if sdf.empty:
            continue
        x = (sdf["amp_lo"] + sdf["amp_hi"]) / 2
        ax.plot(
            x,
            sdf["mean_peak_v"],
            "o-",
            color=color_map[animal],
            ms=3,
            lw=params["lw"],
            label=animal,
        )
    x_fit = np.linspace(float(df[amp_col].min()), float(df[amp_col].max()), 50)
    ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=params["lw"])
    ax.set_xlabel("Amplitude [deg]", fontsize=8)
    ax.set_ylabel("Peak V [deg/ms]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_2e.pdf", format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    return pkl
