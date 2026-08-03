"""Figures 2f (peak-speed coupling), 2h (endpoint heatmaps), 2i (polar dirs)."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.stats import gaussian_kde

from eye_tracking_system_tools.analysis.colors import build_color_map
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.pipeline import EventTables
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


def _estimate_frame_period_ms(eye_df: pd.DataFrame, t_ms: float) -> float:
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
    if "angular_speed_r" not in contra_df.columns or "ms_axis" not in contra_df.columns:
        return np.nan
    series = contra_df.query(
        "ms_axis >= @t_ms - @halfwin_ms and ms_axis <= @t_ms + @halfwin_ms"
    )["angular_speed_r"]
    if series.notna().sum() == 0:
        return np.nan
    return float(np.nanmax(series.to_numpy(dtype=float))) / frame_ms


def export_figure_2f(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    """
    Inter-ocular peak-speed 2D histogram.

    Paper / R3-1 path (migration notebook): ``event_mode=monocular``,
    ``head_movement==False``, exclude PV_62/PV_57, contra window 100 ms.
    """
    cfg = dict(tables.params.get("figure_2f", {}))
    event_mode = str(cfg.get("event_mode", "monocular")).lower()
    contra_win = float(cfg.get("contra_event_window_ms", 100.0))
    contra_sample = float(cfg.get("contra_sample_ms", 51.0))
    exclude = {str(a) for a in cfg.get("exclude_animals", [])}
    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)

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

    # Per-block onset caches + eye traces
    block_map = tables.block_dict
    onset_cache: dict[tuple[str, str], np.ndarray] = {}
    for key, bundle in block_map.items():
        for eye, ev in (("L", bundle.l_saccades), ("R", bundle.r_saccades)):
            if ev is None or ev.empty or "saccade_on_ms" not in ev.columns:
                onset_cache[(key, eye)] = np.array([], dtype=float)
            else:
                onset_cache[(key, eye)] = ev["saccade_on_ms"].to_numpy(dtype=float)

    rights, lefts, animals = [], [], []
    n_skip_mode = 0
    n_skip_profile = 0
    for _, row in df.iterrows():
        animal = str(row["animal"])
        block_key = f"{animal}_block_{row['block']}"
        bundle = block_map.get(block_key)
        if bundle is None:
            continue
        eye = str(row["eye"])
        t0 = float(row["saccade_on_ms"])
        other = "R" if eye == "L" else "L"
        is_binocular = _contra_has_event(onset_cache[(block_key, other)], t0, contra_win)
        if event_mode == "monocular" and is_binocular:
            n_skip_mode += 1
            continue
        if event_mode == "binocular" and not is_binocular:
            n_skip_mode += 1
            continue

        if eye == "L":
            ipsi_df, contra_df = bundle.left, bundle.right
        else:
            ipsi_df, contra_df = bundle.right, bundle.left

        frame_ms = _estimate_frame_period_ms(ipsi_df, t0)
        sp = row.get("speed_profile_angular", None)
        if sp is None or len(sp) == 0 or not np.isfinite(np.nanmax(sp)):
            n_skip_profile += 1
            continue
        ipsi_peak = float(np.nanmax(sp)) / frame_ms
        contra_peak = _sample_contra_peak(contra_df, t0, contra_sample, frame_ms)
        if not np.isfinite(contra_peak):
            n_skip_profile += 1
            continue

        if eye == "L":
            lefts.append(ipsi_peak)
            rights.append(contra_peak)
        else:
            rights.append(ipsi_peak)
            lefts.append(contra_peak)
        animals.append(animal)

    right = np.asarray(rights, dtype=float)
    left = np.asarray(lefts, dtype=float)
    animals_arr = np.asarray(animals)
    print(
        f"[2f] event_mode={event_mode} kept={right.size} "
        f"skip_mode={n_skip_mode} skip_profile={n_skip_profile} "
        f"exclude={sorted(exclude)}"
    )
    if right.size == 0:
        raise ValueError("No speeds for figure 2f after filters")

    u, c = np.unique(animals_arr, return_counts=True)
    wmap = {a: (len(animals_arr) / (len(u) * cnt)) for a, cnt in zip(u, c)}
    weights = np.array([wmap[a] for a in animals_arr], dtype=float)

    iqr_mult = float(cfg.get("iqr_multiplier", 60.0))
    r_lo, r_hi = _iqr_bounds(right, iqr_mult)
    l_lo, l_hi = _iqr_bounds(left, iqr_mult)
    keep = (right >= r_lo) & (right <= r_hi) & (left >= l_lo) & (left <= l_hi)
    right, left, weights = right[keep], left[keep], weights[keep]

    bins = int(cfg.get("bins", 60))
    macro_range = tuple(cfg.get("macro_range", [0.0, 0.5]))
    micro_range = tuple(cfg.get("micro_range", [0.0, 0.1]))

    def _hist(rng):
        xbins = np.linspace(rng[0], rng[1], bins)
        ybins = np.linspace(rng[0], rng[1], bins)
        counts, xedges, yedges = np.histogram2d(
            right, left, bins=[xbins, ybins], weights=weights
        )
        norm = counts / counts.sum() if counts.sum() > 0 else counts
        return {
            "xedges": xedges.astype(float),
            "yedges": yedges.astype(float),
            "norm_counts": norm.astype(float),
        }

    macro = _hist(macro_range)
    micro = _hist(micro_range)
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
        "iqr_multiplier": iqr_mult,
        "macro": macro,
        "micro": micro,
        "vmax_all": vmax_all,
        "event_mode": event_mode,
        "exclude_animals": sorted(exclude),
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
            "event_mode": event_mode,
            "exclude_animals": sorted(exclude),
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
            vmax=float(np.nanmax(hist["norm_counts"]) or 1),
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
    if cfg.get("also_export_s3", True):
        fig.savefig(figures_dir / "figure_S3.pdf", bbox_inches="tight", dpi=300)
    if show:
        from IPython.display import display

        display(fig)
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
    plt.close(fig)
    return pkl


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
