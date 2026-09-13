"""Mouse vs lizard monocular rate and fellow-eye asymmetry (exploratory).

A saccade is monocular if it is unpaired under ``find_synced_saccades_ms``
(default 34 ms). Unique-gaze % treats one concurrent L/R pair as one event.
Asymmetry uses Fig 2f fellow-eye peak sampling (±51 ms) on unpaired events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.event_cache import reload_traces
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.analysis.saccade_robustness import (
    N_SHUFFLE,
    WINDOW_SWEEP_MS,
    pair_counts,
    shuffle_monocular_fractions,
    threshold_sweep,
    window_sweep,
)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PLOT_ID = "monocular_species_compare"
LIZARD_COLOR = "#0072B2"
MOUSE_COLOR = "#009E73"
DEFAULT_FRAME_MS = 17.0
RELATIVE_THRESHOLD_MULT = (0.5, 0.75, 1.0, 1.25, 1.5)
N_BOOT = 10_000
AI_EVENT_COLS = (
    "species",
    "animal",
    "block",
    "block_key",
    "eye",
    "saccade_on_ms",
    "saccade_off_ms",
    "net_angular_disp",
    "overall_angle_deg",
    "peak_velocity",
    "head_movement",
    "right_peak_v",
    "left_peak_v",
    "2f_source",
    "v_ipsi",
    "v_contra",
    "ai",
    "contra_ipsi_ratio",
    "strict_silent",
    "weight",
)

_2F_OVERRIDES = {
    "require_head_stationary": False,
    "exclude_animals": [],
    "event_mode": "monocular",
    "auto_view_limits": True,
}


def _sync_diff_ms(tables: EventTables, default: float = 34.0) -> float:
    return float(tables.params.get("binocular", {}).get("sync_diff_ms", default))


def _paper_threshold(tables: EventTables, default: float) -> float:
    thr = (tables.params.get("saccade") or {}).get("speed_threshold_deg_per_frame")
    if thr is None or not np.isfinite(float(thr)):
        return float(default)
    return float(thr)


def detector_floor_deg_per_ms(
    tables: EventTables,
    *,
    default_thr: float,
    frame_ms: float = DEFAULT_FRAME_MS,
) -> float:
    """Paper detector floor in Fig 2f units (°/ms)."""
    return _paper_threshold(tables, default_thr) / float(frame_ms)


def concurrency_count_row(events: pd.DataFrame, *, sync_diff_ms: float) -> dict[str, float]:
    """Pair / monocular counts for one events table (already one group)."""
    if events is None or events.empty or "eye" not in events.columns:
        return {
            "n_binocular": 0,
            "n_monocular": 0,
            "n_eye_events": 0,
            "n_unique_gaze": 0,
            "pct_monocular_unique_gaze": float("nan"),
            "pct_monocular_per_eye": float("nan"),
        }
    t_l = events.loc[events["eye"].astype(str) == "L", "saccade_on_ms"].to_numpy(float)
    t_r = events.loc[events["eye"].astype(str) == "R", "saccade_on_ms"].to_numpy(float)
    n_pairs, n_mono = pair_counts(t_l, t_r, sync_diff_ms=float(sync_diff_ms))
    n_eye = int(t_l.size + t_r.size)
    n_gaze = int(n_pairs + n_mono)
    return {
        "n_binocular": int(n_pairs),
        "n_monocular": int(n_mono),
        "n_eye_events": n_eye,
        "n_unique_gaze": n_gaze,
        "pct_monocular_unique_gaze": (
            100.0 * n_mono / n_gaze if n_gaze else float("nan")
        ),
        "pct_monocular_per_eye": (
            100.0 * n_mono / n_eye if n_eye else float("nan")
        ),
    }


def concurrency_counts(
    events: pd.DataFrame,
    *,
    sync_diff_ms: float,
    group_cols: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Per-group unique-gaze and per-eye monocular percentages."""
    cols = list(group_cols)
    if events is None or events.empty:
        return pd.DataFrame(
            columns=cols
            + [
                "n_binocular",
                "n_monocular",
                "n_eye_events",
                "n_unique_gaze",
                "pct_monocular_unique_gaze",
                "pct_monocular_per_eye",
            ]
        )
    present = [c for c in cols if c in events.columns]
    if not present:
        row = concurrency_count_row(events, sync_diff_ms=sync_diff_ms)
        return pd.DataFrame([row])
    rows = []
    for keys, g in events.groupby(present, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {c: v for c, v in zip(present, keys)}
        rec.update(concurrency_count_row(g, sync_diff_ms=sync_diff_ms))
        rows.append(rec)
    return pd.DataFrame(rows)


def monocular_counts_by_group(
    tables: EventTables,
    *,
    level: str = "animal",
    species: str | None = None,
) -> pd.DataFrame:
    """``level`` is ``animal``, ``block``, or ``cohort`` (one pooled row)."""
    sync = _sync_diff_ms(tables)
    events = tables.all_saccades
    if level == "cohort":
        out = pd.DataFrame([concurrency_count_row(events, sync_diff_ms=sync)])
    elif level == "block":
        out = concurrency_counts(events, sync_diff_ms=sync, group_cols=("animal", "block"))
    elif level == "animal":
        out = concurrency_counts(events, sync_diff_ms=sync, group_cols=("animal",))
    else:
        raise ValueError(f"level must be animal, block, or cohort; got {level!r}")
    out.insert(0, "level", level)
    if species is not None:
        out.insert(0, "species", species)
    return out


def asymmetry_index(v_ipsi: np.ndarray | float, v_contra: np.ndarray | float) -> np.ndarray:
    """``(ipsi − contra) / (ipsi + contra)``. 1 = fellow eye silent."""
    ipsi, contra = np.broadcast_arrays(
        np.asarray(v_ipsi, dtype=float),
        np.asarray(v_contra, dtype=float),
    )
    denom = ipsi + contra
    out = np.full(ipsi.shape, np.nan, dtype=float)
    ok = np.isfinite(ipsi) & np.isfinite(contra) & (denom != 0)
    out[ok] = (ipsi[ok] - contra[ok]) / denom[ok]
    return out


def ipsi_contra_from_2f_row(row: pd.Series) -> tuple[float, float]:
    left = float(row["left_peak_v"])
    right = float(row["right_peak_v"])
    eye = str(row.get("eye", ""))
    if eye == "L":
        return left, right
    if eye == "R":
        return right, left
    return max(left, right), min(left, right)


def annotate_monocular_asymmetry(
    points: pd.DataFrame,
    *,
    detector_deg_per_ms: float | None = None,
) -> pd.DataFrame:
    """Add AI / residual columns to Fig 2f monocular-window rows."""
    extra = ["v_ipsi", "v_contra", "ai", "contra_ipsi_ratio", "strict_silent"]
    if points is None or points.empty:
        cols = list(points.columns) if points is not None else []
        return pd.DataFrame(columns=cols + extra)
    work = points.copy()
    if "2f_source" in work.columns:
        work = work.loc[work["2f_source"].astype(str) == "monocular_window"].copy()
    if work.empty:
        work["v_ipsi"] = pd.Series(dtype=float)
        work["v_contra"] = pd.Series(dtype=float)
        work["ai"] = pd.Series(dtype=float)
        work["contra_ipsi_ratio"] = pd.Series(dtype=float)
        work["strict_silent"] = pd.Series(dtype=bool)
        return work
    ipsi = np.empty(len(work), dtype=float)
    contra = np.empty(len(work), dtype=float)
    for i, (_, row) in enumerate(work.iterrows()):
        ipsi[i], contra[i] = ipsi_contra_from_2f_row(row)
    work["v_ipsi"] = ipsi
    work["v_contra"] = contra
    work["ai"] = asymmetry_index(ipsi, contra)
    ratio = np.full(len(work), np.nan)
    ok = np.isfinite(ipsi) & (ipsi != 0)
    ratio[ok] = contra[ok] / ipsi[ok]
    work["contra_ipsi_ratio"] = ratio
    if detector_deg_per_ms is None or not np.isfinite(float(detector_deg_per_ms)):
        work["strict_silent"] = np.nan
    else:
        work["strict_silent"] = np.isfinite(contra) & (contra < float(detector_deg_per_ms))
    return work


def asymmetry_summary_by_animal(df: pd.DataFrame, *, species: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        cols = [
            "animal",
            "n_monocular_2f",
            "median_ai",
            "median_contra_ipsi_ratio",
            "pct_strict_silent",
        ]
        if species is not None:
            cols = ["species"] + cols
        return pd.DataFrame(columns=cols)
    rows = []
    for animal, g in df.groupby("animal", dropna=False):
        silent = pd.to_numeric(g.get("strict_silent"), errors="coerce")
        rec = {
            "animal": animal,
            "n_monocular_2f": int(len(g)),
            "median_ai": float(np.nanmedian(g["ai"])) if "ai" in g else float("nan"),
            "median_contra_ipsi_ratio": (
                float(np.nanmedian(g["contra_ipsi_ratio"]))
                if "contra_ipsi_ratio" in g
                else float("nan")
            ),
            "pct_strict_silent": (
                100.0 * float(silent.mean()) if silent.notna().any() else float("nan")
            ),
        }
        if species is not None:
            rec["species"] = species
        rows.append(rec)
    out = pd.DataFrame(rows)
    if species is not None:
        out = out[["species"] + [c for c in out.columns if c != "species"]]
    return out


def shuffle_result(
    tables: EventTables,
    *,
    n_shuffle: int = N_SHUFFLE,
    threshold: float | None = None,
    sync_diff_ms: float | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Observed unique-gaze monocular fraction vs circular-shift null."""
    thr = float(threshold) if threshold is not None else _paper_threshold(tables, 0.8)
    win = float(sync_diff_ms) if sync_diff_ms is not None else _sync_diff_ms(tables)
    observed, null = shuffle_monocular_fractions(
        tables,
        n_shuffle=int(n_shuffle),
        sync_diff_ms=win,
        threshold=thr,
        rng=rng,
    )
    null = np.asarray(null, dtype=float)
    null_mean = float(np.nanmean(null)) if null.size else float("nan")
    finite = null[np.isfinite(null)]
    # Pairing tighter than chance → observed monocular fraction below the null.
    p_more_paired = (
        float(np.mean(finite <= observed)) if finite.size and np.isfinite(observed) else float("nan")
    )
    return {
        "observed": float(observed) if np.isfinite(observed) else float("nan"),
        "null": null,
        "null_mean": null_mean,
        "mono_minus_null": (
            float(observed - null_mean)
            if np.isfinite(observed) and np.isfinite(null_mean)
            else float("nan")
        ),
        "p_more_paired_than_chance": p_more_paired,
        "n_shuffle": int(n_shuffle),
        "threshold_deg_per_frame": thr,
        "sync_diff_ms": win,
    }


def rank_against(value: float, reference: np.ndarray | pd.Series) -> dict[str, float | bool]:
    """Where ``value`` sits among animal-level reference values (lizard)."""
    refs = np.asarray(reference, dtype=float)
    refs = refs[np.isfinite(refs)]
    if not np.isfinite(value) or refs.size == 0:
        return {
            "value": float(value) if np.isfinite(value) else float("nan"),
            "n_reference": int(refs.size),
            "n_below": float("nan"),
            "percentile": float("nan"),
            "ref_min": float("nan"),
            "ref_max": float("nan"),
            "ref_mean": float("nan"),
            "outside_range": False,
        }
    n_below = int(np.sum(refs < value))
    return {
        "value": float(value),
        "n_reference": int(refs.size),
        "n_below": n_below,
        "percentile": 100.0 * n_below / refs.size,
        "ref_min": float(np.min(refs)),
        "ref_max": float(np.max(refs)),
        "ref_mean": float(np.mean(refs)),
        "outside_range": bool(value < np.min(refs) or value > np.max(refs)),
    }


def hierarchical_bootstrap_difference(
    lizard_animal: np.ndarray | pd.Series,
    mouse_block: np.ndarray | pd.Series,
    *,
    n_boot: int = N_BOOT,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """CI for (mouse-block mean − lizard-animal mean). Not a species test.

    Lizard units are animals; mouse units are blocks of the single animal.
    """
    rng = rng or np.random.default_rng(0)
    liz = np.asarray(lizard_animal, dtype=float)
    mou = np.asarray(mouse_block, dtype=float)
    liz = liz[np.isfinite(liz)]
    mou = mou[np.isfinite(mou)]
    if liz.size == 0 or mou.size == 0:
        return {
            "observed_diff": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_boot": int(n_boot),
            "n_lizard_animals": int(liz.size),
            "n_mouse_blocks": int(mou.size),
        }
    observed = float(np.mean(mou) - np.mean(liz))
    diffs = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        liz_s = rng.choice(liz, size=liz.size, replace=True)
        mou_s = rng.choice(mou, size=mou.size, replace=True)
        diffs[i] = float(np.mean(mou_s) - np.mean(liz_s))
    return {
        "observed_diff": observed,
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "n_boot": int(n_boot),
        "n_lizard_animals": int(liz.size),
        "n_mouse_blocks": int(mou.size),
    }


def relative_threshold_sweep(
    tables: EventTables,
    *,
    paper_threshold: float,
    multipliers: tuple[float, ...] = RELATIVE_THRESHOLD_MULT,
    sync_diff_ms: float | None = None,
) -> pd.DataFrame:
    win = float(sync_diff_ms) if sync_diff_ms is not None else _sync_diff_ms(tables)
    thresholds = tuple(float(paper_threshold) * float(m) for m in multipliers)
    out = threshold_sweep(tables, thresholds=thresholds, sync_diff_ms=win)
    out["paper_threshold_deg_per_frame"] = float(paper_threshold)
    out["threshold_over_paper"] = out["speed_threshold_deg_per_frame"] / float(paper_threshold)
    out["pct_monocular_unique_gaze"] = 100.0 * out["monocular_fraction"]
    return out


def _slim_ai_events(df: pd.DataFrame, *, species: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(AI_EVENT_COLS))
    out = df.copy()
    out["species"] = species
    keep = [c for c in AI_EVENT_COLS if c in out.columns]
    return out[keep]


def _sem(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(np.std(v, ddof=1) / np.sqrt(v.size))


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def _collect_monocular_2f(tables: EventTables):
    from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
        collect_figure_2f_points,
        figure_2f_config,
    )

    tables = reload_traces(tables)
    cfg = figure_2f_config(tables, _2F_OVERRIDES)
    return collect_figure_2f_points(tables, cfg=cfg, apply_iqr_clip=False)


def _plot_pct_by_animal(
    lizard_animal: pd.DataFrame,
    mouse_animal: pd.DataFrame,
    mouse_block: pd.DataFrame,
    *,
    ylabel: str,
    col: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
    liz = lizard_animal[col].to_numpy(float)
    mou_a = mouse_animal[col].to_numpy(float)
    mou_b = mouse_block[col].to_numpy(float)
    rng = np.random.default_rng(1)
    if liz.size:
        jitter = rng.uniform(-0.08, 0.08, size=liz.size)
        ax.scatter(
            np.full(liz.size, 0.0) + jitter,
            liz,
            s=22,
            color=LIZARD_COLOR,
            zorder=3,
            label="lizard animal",
        )
        mean = float(np.nanmean(liz))
        sem = _sem(liz)
        ax.errorbar(
            [0.0],
            [mean],
            yerr=sem if np.isfinite(sem) else None,
            fmt="o",
            color="0.15",
            ms=5,
            capsize=3,
            zorder=4,
            label="lizard mean±SEM",
        )
    if mou_b.size:
        jitter = rng.uniform(-0.06, 0.06, size=mou_b.size)
        ax.scatter(
            np.full(mou_b.size, 1.12) + jitter,
            mou_b,
            s=16,
            facecolors="none",
            edgecolors=MOUSE_COLOR,
            zorder=3,
            label="M_002 blocks",
        )
    if mou_a.size:
        ax.scatter(
            np.full(mou_a.size, 1.0),
            mou_a,
            s=36,
            color=MOUSE_COLOR,
            zorder=4,
            marker="D",
            label="M_002 pooled",
        )
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["lizard", "mouse"])
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(-0.45, 1.55)
    _style(ax)
    ax.legend(fontsize=5.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig


def _plot_shuffle(liz: dict[str, Any], mou: dict[str, Any]) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.9), dpi=300, sharey=True)
    for ax, body, title, color in (
        (axs[0], liz, "lizard", LIZARD_COLOR),
        (axs[1], mou, "mouse (M_002)", MOUSE_COLOR),
    ):
        null = np.asarray(body.get("null", []), dtype=float)
        obs = body.get("observed", float("nan"))
        if null.size:
            ax.hist(null, bins=25, color="0.75", edgecolor="0.3", density=False)
        if np.isfinite(obs):
            ax.axvline(obs, color=color, lw=1.5, label=f"observed {obs:.3f}")
        nm = body.get("null_mean", float("nan"))
        if np.isfinite(nm):
            ax.axvline(nm, color="0.2", lw=1.0, ls="--", label=f"null mean {nm:.3f}")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Monocular fraction", fontsize=8)
        _style(ax)
        ax.legend(fontsize=5.5, frameon=False)
    axs[0].set_ylabel("Shuffle count", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_ai_hist(
    lizard_events: pd.DataFrame,
    mouse_events: pd.DataFrame,
    lizard_animal: pd.DataFrame,
    mouse_animal: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.2, 2.3), dpi=300)
    for df, color, label in (
        (lizard_events, LIZARD_COLOR, "lizard events"),
        (mouse_events, MOUSE_COLOR, "mouse events"),
    ):
        if df is None or df.empty or "ai" not in df.columns:
            continue
        vals = pd.to_numeric(df["ai"], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        ax.hist(
            vals,
            bins=np.linspace(-1.0, 1.0, 41),
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color=color,
            label=label,
            edgecolor=color,
        )
    for _, row in lizard_animal.iterrows():
        v = row.get("median_ai")
        if v is not None and np.isfinite(float(v)):
            ax.axvline(float(v), color=LIZARD_COLOR, lw=0.8, alpha=0.8)
    for _, row in mouse_animal.iterrows():
        v = row.get("median_ai")
        if v is not None and np.isfinite(float(v)):
            ax.axvline(float(v), color=MOUSE_COLOR, lw=1.4, ls="--")
    ax.set_xlabel("Asymmetry index (ipsi−contra)/(ipsi+contra)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlim(-1.02, 1.02)
    _style(ax)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def _plot_sweep(df: pd.DataFrame, *, x: str, xlabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2.8, 2.0), dpi=300)
    for species, color in (("lizard", LIZARD_COLOR), ("mouse", MOUSE_COLOR)):
        g = df.loc[df["species"] == species]
        if g.empty:
            continue
        ax.plot(
            g[x],
            g["pct_monocular_unique_gaze"],
            "o-",
            color=color,
            lw=1.2,
            ms=4,
            label=species,
        )
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("% monocular (unique gaze)", fontsize=8)
    _style(ax)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def _plot_2f_heatmaps(liz_pts, mou_pts) -> plt.Figure | None:
    from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
        _draw_coupling_heatmap,
        _turbo_white0,
        histogram2d_from_points,
    )

    if liz_pts is None or mou_pts is None:
        return None
    if liz_pts.points.empty or mou_pts.points.empty:
        return None
    hist_l = histogram2d_from_points(liz_pts, view="macro")
    hist_m = histogram2d_from_points(mou_pts, view="macro")
    cmap = _turbo_white0()
    vmax = float(
        max(np.nanmax(hist_l["norm_counts"]), np.nanmax(hist_m["norm_counts"]), 1e-12)
    )
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.8), dpi=300, constrained_layout=True)
    _draw_coupling_heatmap(
        axs[0],
        hist_l,
        liz_pts.macro_range,
        list(liz_pts.cfg.get("macro_tick_list", [0.0, 0.25, 0.5])),
        vmax=vmax,
        cmap=cmap,
    )
    axs[0].set_title("lizard monocular", fontsize=8)
    _draw_coupling_heatmap(
        axs[1],
        hist_m,
        mou_pts.macro_range,
        list(mou_pts.cfg.get("macro_tick_list", [0.0, 0.25, 0.5])),
        vmax=vmax,
        cmap=cmap,
    )
    axs[1].set_title("mouse monocular", fontsize=8)
    return fig


def _py(obj: Any) -> Any:
    """Make numpy scalars YAML-safe."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _cohort_meta(lizard_tables: EventTables, mouse_tables: EventTables) -> dict[str, Any]:
    from eye_tracking_system_tools.analysis.plot_bundle import (
        animals_from_tables,
        block_keys_from_tables,
    )

    return {
        "cohort": "lizard_mouse",
        "animals": animals_from_tables(lizard_tables) + animals_from_tables(mouse_tables),
        "block_keys": block_keys_from_tables(lizard_tables)
        + block_keys_from_tables(mouse_tables),
        "rule": "explicit_mixed",
    }


def export_monocular_species_compare(
    lizard_tables: EventTables,
    mouse_tables: EventTables,
    out_dir: Path | str,
    *,
    n_shuffle: int = N_SHUFFLE,
    n_boot: int = N_BOOT,
    show: bool = False,
    collect_2f: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, Path]:
    """Write unique-gaze % monocular, shuffle, and monocular AI plots."""
    rng = rng or np.random.default_rng(0)
    liz_thr = _paper_threshold(lizard_tables, 0.8)
    mou_thr = _paper_threshold(mouse_tables, 3.23)
    liz_sync = _sync_diff_ms(lizard_tables)
    mou_sync = _sync_diff_ms(mouse_tables)

    lizard_animal = monocular_counts_by_group(lizard_tables, level="animal", species="lizard")
    lizard_block = monocular_counts_by_group(lizard_tables, level="block", species="lizard")
    lizard_cohort = monocular_counts_by_group(lizard_tables, level="cohort", species="lizard")
    mouse_animal = monocular_counts_by_group(mouse_tables, level="animal", species="mouse")
    mouse_block = monocular_counts_by_group(mouse_tables, level="block", species="mouse")
    mouse_cohort = monocular_counts_by_group(mouse_tables, level="cohort", species="mouse")
    counts = pd.concat(
        [lizard_animal, lizard_block, lizard_cohort, mouse_animal, mouse_block, mouse_cohort],
        ignore_index=True,
    )

    liz_vals = lizard_animal["pct_monocular_unique_gaze"]
    mou_animal_val = (
        float(mouse_animal["pct_monocular_unique_gaze"].iloc[0])
        if not mouse_animal.empty
        else float("nan")
    )
    mou_block_vals = mouse_block["pct_monocular_unique_gaze"]
    rank_pct = rank_against(mou_animal_val, liz_vals)
    boot_pct = hierarchical_bootstrap_difference(liz_vals, mou_block_vals, n_boot=n_boot, rng=rng)

    liz_shuffle = shuffle_result(
        lizard_tables, n_shuffle=n_shuffle, threshold=liz_thr, sync_diff_ms=liz_sync, rng=rng
    )
    mou_shuffle = shuffle_result(
        mouse_tables, n_shuffle=n_shuffle, threshold=mou_thr, sync_diff_ms=mou_sync, rng=rng
    )

    win_liz = window_sweep(lizard_tables, windows_ms=WINDOW_SWEEP_MS, threshold=liz_thr)
    win_liz["species"] = "lizard"
    win_liz["pct_monocular_unique_gaze"] = 100.0 * win_liz["monocular_fraction"]
    win_mou = window_sweep(mouse_tables, windows_ms=WINDOW_SWEEP_MS, threshold=mou_thr)
    win_mou["species"] = "mouse"
    win_mou["pct_monocular_unique_gaze"] = 100.0 * win_mou["monocular_fraction"]
    windows = pd.concat([win_liz, win_mou], ignore_index=True)

    thr_liz = relative_threshold_sweep(lizard_tables, paper_threshold=liz_thr, sync_diff_ms=liz_sync)
    thr_liz["species"] = "lizard"
    thr_mou = relative_threshold_sweep(mouse_tables, paper_threshold=mou_thr, sync_diff_ms=mou_sync)
    thr_mou["species"] = "mouse"
    thresholds = pd.concat([thr_liz, thr_mou], ignore_index=True)

    liz_ai = pd.DataFrame()
    mou_ai = pd.DataFrame()
    liz_ai_animal = pd.DataFrame()
    mou_ai_animal = pd.DataFrame()
    liz_pts = mou_pts = None
    rank_ai: dict[str, Any] = {}
    boot_ai: dict[str, Any] = {}
    if collect_2f:
        liz_pts = _collect_monocular_2f(lizard_tables)
        mou_pts = _collect_monocular_2f(mouse_tables)
        liz_floor = detector_floor_deg_per_ms(lizard_tables, default_thr=0.8)
        mou_floor = detector_floor_deg_per_ms(mouse_tables, default_thr=3.23)
        liz_ai = annotate_monocular_asymmetry(liz_pts.points, detector_deg_per_ms=liz_floor)
        mou_ai = annotate_monocular_asymmetry(mou_pts.points, detector_deg_per_ms=mou_floor)
        liz_ai_animal = asymmetry_summary_by_animal(liz_ai, species="lizard")
        mou_ai_animal = asymmetry_summary_by_animal(mou_ai, species="mouse")
        mou_ai_val = (
            float(mou_ai_animal["median_ai"].iloc[0]) if not mou_ai_animal.empty else float("nan")
        )
        rank_ai = rank_against(mou_ai_val, liz_ai_animal["median_ai"] if not liz_ai_animal.empty else [])
        boot_ai = hierarchical_bootstrap_difference(
            liz_ai_animal["median_ai"] if not liz_ai_animal.empty else [],
            # mouse n=1 animal: bootstrap its event medians per block if present
            mou_ai.groupby("block")["ai"].median() if not mou_ai.empty and "block" in mou_ai.columns else [],
            n_boot=n_boot,
            rng=rng,
        )

    bundle = begin_plot_bundle(
        out_dir,
        PLOT_ID,
        kind=PLOT_ID,
        cohort=_cohort_meta(lizard_tables, mouse_tables),
        logic_key=PLOT_ID,
        params={
            "lizard_threshold_deg_per_frame": liz_thr,
            "mouse_threshold_deg_per_frame": mou_thr,
            "lizard_sync_diff_ms": liz_sync,
            "mouse_sync_diff_ms": mou_sync,
            "n_shuffle": int(n_shuffle),
            "n_boot": int(n_boot),
            "require_head_stationary": False,
            "event_mode": "monocular",
            "frame_ms_for_detector_floor": DEFAULT_FRAME_MS,
        },
        extra=_py(
            {
                "claim": (
                    "M_002 vs lizard paper cohort under the 34 ms unpaired definition; "
                    "not a well-powered Mus vs Pogona species test."
                ),
                "rank_pct_monocular": rank_pct,
                "bootstrap_pct_monocular": boot_pct,
                "rank_median_ai": rank_ai,
                "bootstrap_median_ai": boot_ai,
                "lizard_shuffle": {k: v for k, v in liz_shuffle.items() if k != "null"},
                "mouse_shuffle": {k: v for k, v in mou_shuffle.items() if k != "null"},
            }
        ),
    )
    figures_dir, metadata_dir = bundle.plots_dir, bundle.metadata_dir
    written: dict[str, Path] = {}

    def _save(fig: plt.Figure | None, name: str) -> None:
        if fig is None:
            return
        path = figures_dir / name
        fig.savefig(path, format="pdf", bbox_inches="tight")
        show_and_close(fig, show)
        written[name] = path

    _save(
        _plot_pct_by_animal(
            lizard_animal,
            mouse_animal,
            mouse_block,
            ylabel="% monocular (unique gaze)",
            col="pct_monocular_unique_gaze",
        ),
        "monocular_pct_by_animal.pdf",
    )
    _save(_plot_shuffle(liz_shuffle, mou_shuffle), "monocular_shuffle.pdf")
    if collect_2f:
        _save(
            _plot_ai_hist(liz_ai, mou_ai, liz_ai_animal, mou_ai_animal),
            "monocular_ai_hist.pdf",
        )
        _save(_plot_2f_heatmaps(liz_pts, mou_pts), "monocular_2f_heatmaps.pdf")
    _save(_plot_sweep(windows, x="pairing_window_ms", xlabel="Pairing window [ms]"), "monocular_window_sweep.pdf")
    _save(
        _plot_sweep(
            thresholds,
            x="threshold_over_paper",
            xlabel="Detector threshold / paper threshold",
        ),
        "monocular_threshold_sweep.pdf",
    )

    counts.to_csv(metadata_dir / "monocular_counts.csv", index=False)
    windows.to_csv(metadata_dir / "monocular_window_sweep.csv", index=False)
    thresholds.to_csv(metadata_dir / "monocular_threshold_sweep.csv", index=False)
    if not liz_ai.empty:
        _slim_ai_events(liz_ai, species="lizard").to_csv(
            metadata_dir / "lizard_monocular_ai_events.csv", index=False
        )
    if not mou_ai.empty:
        _slim_ai_events(mou_ai, species="mouse").to_csv(
            metadata_dir / "mouse_monocular_ai_events.csv", index=False
        )
    ai_animal = pd.concat([liz_ai_animal, mou_ai_animal], ignore_index=True)
    if not ai_animal.empty:
        ai_animal.to_csv(metadata_dir / "monocular_ai_by_animal.csv", index=False)
    pd.DataFrame({"null_monocular_fraction": liz_shuffle["null"]}).to_csv(
        metadata_dir / "lizard_shuffle_null.csv", index=False
    )
    pd.DataFrame({"null_monocular_fraction": mou_shuffle["null"]}).to_csv(
        metadata_dir / "mouse_shuffle_null.csv", index=False
    )

    summary = pd.DataFrame(
        [
            {
                "species": "lizard",
                "n_animals": int(len(lizard_animal)),
                "n_blocks": int(len(lizard_block)),
                "pct_monocular_unique_gaze_mean": float(np.nanmean(liz_vals)) if len(liz_vals) else float("nan"),
                "pct_monocular_unique_gaze_sem": _sem(liz_vals.to_numpy(float)),
                "pct_monocular_unique_gaze_pooled": (
                    float(lizard_cohort["pct_monocular_unique_gaze"].iloc[0])
                    if not lizard_cohort.empty
                    else float("nan")
                ),
                "shuffle_observed": liz_shuffle["observed"],
                "shuffle_null_mean": liz_shuffle["null_mean"],
                "mono_minus_null": liz_shuffle["mono_minus_null"],
                "p_more_paired_than_chance": liz_shuffle["p_more_paired_than_chance"],
                "median_ai_mean_across_animals": (
                    float(np.nanmean(liz_ai_animal["median_ai"])) if not liz_ai_animal.empty else float("nan")
                ),
            },
            {
                "species": "mouse",
                "n_animals": int(len(mouse_animal)),
                "n_blocks": int(len(mouse_block)),
                "pct_monocular_unique_gaze_mean": mou_animal_val,
                "pct_monocular_unique_gaze_sem": _sem(mou_block_vals.to_numpy(float)),
                "pct_monocular_unique_gaze_pooled": (
                    float(mouse_cohort["pct_monocular_unique_gaze"].iloc[0])
                    if not mouse_cohort.empty
                    else float("nan")
                ),
                "shuffle_observed": mou_shuffle["observed"],
                "shuffle_null_mean": mou_shuffle["null_mean"],
                "mono_minus_null": mou_shuffle["mono_minus_null"],
                "p_more_paired_than_chance": mou_shuffle["p_more_paired_than_chance"],
                "median_ai_mean_across_animals": (
                    float(np.nanmean(mou_ai_animal["median_ai"])) if not mou_ai_animal.empty else float("nan")
                ),
            },
        ]
    )
    summary.to_csv(metadata_dir / "species_summary.csv", index=False)

    payload = {
        "counts": counts,
        "summary": summary,
        "rank_pct_monocular": rank_pct,
        "bootstrap_pct_monocular": boot_pct,
        "rank_median_ai": rank_ai,
        "bootstrap_median_ai": boot_ai,
        "lizard_shuffle": {
            k: v for k, v in liz_shuffle.items() if k != "null"
        },
        "mouse_shuffle": {k: v for k, v in mou_shuffle.items() if k != "null"},
    }
    write_pickle_with_meta(
        payload,
        metadata_dir / "monocular_species_compare.pkl",
        meta={"plot_id": PLOT_ID},
        entrypoint="eye_tracking_system_tools.analysis.monocular_species.export_monocular_species_compare",
    )
    finish_plot_bundle(bundle)
    return written
