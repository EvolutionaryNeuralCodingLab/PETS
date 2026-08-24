"""Saccade-detection robustness and tracking-noise floor (R3-2)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.event_cache import reload_traces
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.analysis.rayleigh_core import (
    RAYLEIGH_EXCESS_KURTOSIS,
    RAYLEIGH_SKEW,
)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

THRESHOLD_SWEEP = (0.4, 0.6, 0.8, 1.0, 1.2)
WINDOW_SWEEP_MS = (17.0, 34.0, 51.0, 68.0, 100.0)
DEFAULT_THRESHOLD = 0.8
DEFAULT_WINDOW_MS = 34.0
# 200 circular shifts is enough for a null histogram; 1000 was minutes of DataFrames.
N_SHUFFLE = 200

UNIFIED_NOISE_PLOT_ID = "unified_noise_measure"
UNIFIED_NOISE_PDF = "unified_noise_measure.pdf"
UNIFIED_NOISE_PANELS: tuple[tuple[str, str, str], ...] = (
    ("rigid", "rigid lizard", "#D55E00"),
    ("modular", "modular lizard", "#0072B2"),
    ("turtle", "turtle", "#CC79A7"),
    ("mouse", "mouse", "#009E73"),
)
RAW_INTERFRAME_PLOT_ID = "raw_interframe_delta"
RAW_INTERFRAME_PDF = "raw_interframe_delta.pdf"
RAW_INTERFRAME_LOGX_PDF = "raw_interframe_delta_logx.pdf"
RAW_INTERFRAME_PICKLE = "raw_interframe_delta.pkl"


def _events_at_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "peak_velocity" not in df.columns:
        return df.copy()
    return df.loc[pd.to_numeric(df["peak_velocity"], errors="coerce") >= float(threshold)].copy()


def pair_counts(t_l: np.ndarray, t_r: np.ndarray, *, sync_diff_ms: float) -> tuple[int, int]:
    """Greedy nearest-R matching (same rule as ``find_synced_saccades_ms``)."""
    t_l = np.asarray(t_l, dtype=float)
    t_r = np.asarray(t_r, dtype=float)
    t_l = t_l[np.isfinite(t_l)]
    t_r = t_r[np.isfinite(t_r)]
    if t_l.size == 0:
        return 0, int(t_r.size)
    if t_r.size == 0:
        return 0, int(t_l.size)
    used = np.zeros(t_r.size, dtype=bool)
    n_pairs = 0
    for t in t_l:
        dt = np.abs(t_r - t)
        j = int(np.argmin(dt))
        if (not used[j]) and dt[j] < sync_diff_ms:
            used[j] = True
            n_pairs += 1
    n_mono = int(t_l.size - n_pairs) + int(t_r.size - n_pairs)
    return n_pairs, n_mono


def monocular_fraction(events: pd.DataFrame, *, sync_diff_ms: float) -> float:
    if events is None or events.empty:
        return float("nan")
    n_pairs = 0
    n_mono = 0
    for _, g in events.groupby(["animal", "block"], dropna=False):
        t_l = g.loc[g["eye"].astype(str) == "L", "saccade_on_ms"].to_numpy(float)
        t_r = g.loc[g["eye"].astype(str) == "R", "saccade_on_ms"].to_numpy(float)
        p, m = pair_counts(t_l, t_r, sync_diff_ms=float(sync_diff_ms))
        n_pairs += p
        n_mono += m
    denom = n_pairs + n_mono
    if denom <= 0:
        return float("nan")
    return n_mono / denom


def threshold_sweep(
    tables: EventTables,
    *,
    thresholds: tuple[float, ...] = THRESHOLD_SWEEP,
    sync_diff_ms: float = DEFAULT_WINDOW_MS,
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        ev = _events_at_threshold(tables.all_saccades, thr)
        rows.append(
            {
                "speed_threshold_deg_per_frame": float(thr),
                "monocular_fraction": monocular_fraction(ev, sync_diff_ms=sync_diff_ms),
                "n_events": int(len(ev)),
            }
        )
    return pd.DataFrame(rows)


def window_sweep(
    tables: EventTables,
    *,
    windows_ms: tuple[float, ...] = WINDOW_SWEEP_MS,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    ev = _events_at_threshold(tables.all_saccades, threshold)
    rows = []
    for win in windows_ms:
        rows.append(
            {
                "pairing_window_ms": float(win),
                "monocular_fraction": monocular_fraction(ev, sync_diff_ms=win),
                "n_events": int(len(ev)),
            }
        )
    return pd.DataFrame(rows)


def shuffle_monocular_fractions(
    tables: EventTables,
    *,
    n_shuffle: int = N_SHUFFLE,
    sync_diff_ms: float = DEFAULT_WINDOW_MS,
    threshold: float = DEFAULT_THRESHOLD,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray]:
    """Circular-shift right-eye onsets within each block; return (observed, null).

    Tests whether L/R coincidence exceeds chance given the same rates. This is
    **not** a measure of tracking noise (see ``noise_floor``).
    """
    rng = rng or np.random.default_rng(0)
    ev = _events_at_threshold(tables.all_saccades, threshold)
    observed = monocular_fraction(ev, sync_diff_ms=sync_diff_ms)
    n_shuffle = int(n_shuffle)
    if ev.empty or n_shuffle <= 0:
        return observed, np.array([], dtype=float)

    groups = []
    for _, g in ev.groupby(["animal", "block"], dropna=False):
        t_l = g.loc[g["eye"].astype(str) == "L", "saccade_on_ms"].to_numpy(float)
        t_r = g.loc[g["eye"].astype(str) == "R", "saccade_on_ms"].to_numpy(float)
        t_all = g["saccade_on_ms"].to_numpy(float)
        tmin = float(np.nanmin(t_all)) if t_all.size else 0.0
        tmax = float(np.nanmax(t_all)) if t_all.size else 1.0
        groups.append((t_l, t_r, tmin, max(tmax - tmin, 1.0)))

    null = np.empty(n_shuffle, dtype=float)
    win = float(sync_diff_ms)
    for i in range(n_shuffle):
        n_pairs = n_mono = 0
        for t_l, t_r, tmin, span in groups:
            if t_r.size:
                shift = float(rng.uniform(0.0, span))
                t_r_s = tmin + np.mod(t_r - tmin + shift, span)
            else:
                t_r_s = t_r
            p, m = pair_counts(t_l, t_r_s, sync_diff_ms=win)
            n_pairs += p
            n_mono += m
        denom = n_pairs + n_mono
        null[i] = n_mono / denom if denom else np.nan
    return observed, null


def interframe_steps(
    df: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every consecutive Kerr step: ``(dangle_deg, dt_ms, t_mid_ms)``.

    ``dangle`` is ``hypot(Δφ, Δθ)`` between adjacent samples (deg/frame).
    No saccade mask. Empty arrays if Kerr angles are missing.
    """
    empty = np.array([], dtype=float)
    if df is None or df.empty or "k_phi" not in df.columns:
        return empty, empty, empty
    if "ms_axis" in df.columns:
        t = pd.to_numeric(df["ms_axis"], errors="coerce").to_numpy()
    else:
        t = np.arange(len(df), dtype=float)
    phi = pd.to_numeric(df["k_phi"], errors="coerce").to_numpy()
    th = (
        pd.to_numeric(df["k_theta"], errors="coerce").to_numpy()
        if "k_theta" in df.columns
        else np.zeros_like(phi)
    )
    if phi.size < 2:
        return empty, empty, empty
    dang = np.hypot(np.diff(phi), np.diff(th))
    dt = np.diff(t)
    mid = 0.5 * (t[1:] + t[:-1])
    return dang.astype(float), dt.astype(float), mid.astype(float)


def interframe_axis_deltas(
    df: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consecutive Kerr steps: ``(Δφ, Δθ, D)`` in deg/frame. No saccade mask."""
    empty = np.array([], dtype=float)
    if df is None or df.empty or "k_phi" not in df.columns:
        return empty, empty, empty
    phi = pd.to_numeric(df["k_phi"], errors="coerce").to_numpy()
    th = (
        pd.to_numeric(df["k_theta"], errors="coerce").to_numpy()
        if "k_theta" in df.columns
        else np.zeros_like(phi)
    )
    if phi.size < 2:
        return empty, empty, empty
    dphi = np.diff(phi).astype(float)
    dth = np.diff(th).astype(float)
    ok = np.isfinite(dphi) & np.isfinite(dth)
    return dphi[ok], dth[ok], np.hypot(dphi[ok], dth[ok])


def interframe_dangle(df: pd.DataFrame | None) -> np.ndarray:
    """All finite inter-frame Δangle values (deg/frame). No saccade exclusion."""
    dang, _dt, _mid = interframe_steps(df)
    return dang[np.isfinite(dang)]


def non_saccade_interframe_dangle(
    df: pd.DataFrame | None,
    events: pd.DataFrame | None = None,
    *,
    pad_ms: float = 40.0,
) -> np.ndarray:
    """Inter-frame Δangle (deg/frame) outside saccade windows.

    If ``events`` is empty/missing (turtle; blocks without detection), every
    finite inter-frame step is kept.
    """
    dang, _dt, mid = interframe_steps(df)
    if dang.size == 0:
        return dang
    sacc = np.zeros(dang.shape, dtype=bool)
    if events is not None and not events.empty and "saccade_on_ms" in events.columns:
        on = pd.to_numeric(events["saccade_on_ms"], errors="coerce").to_numpy()
        off = (
            pd.to_numeric(events["saccade_off_ms"], errors="coerce").to_numpy()
            if "saccade_off_ms" in events.columns
            else on
        )
        for a, b in zip(on, off):
            if np.isfinite(a) and np.isfinite(b):
                sacc |= (mid >= a - pad_ms) & (mid <= b + pad_ms)
    return dang[np.isfinite(dang) & ~sacc]


def _engbert_sigma(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan")
    med = float(np.median(samples))
    mad = float(np.median(np.abs(samples - med)))
    return mad / 0.6745 if mad > 0 else float("nan")


def noise_floor(
    tables: EventTables,
    *,
    threshold_deg_per_frame: float = DEFAULT_THRESHOLD,
    pad_ms: float = 40.0,
) -> pd.DataFrame:
    """Inter-frame angular step (deg/frame) on samples outside saccade windows.

    Follows the Engbert–Kliegl / Nyström–Holmqvist idea: estimate velocity noise
    with a robust scale (MAD) from non-saccade samples, then compare the
    detector threshold to that floor.
    """
    tables = reload_traces(tables)
    rows = []
    sample_chunks: list[np.ndarray] = []
    for bundle in tables.blocks:
        for eye, ev, df in (
            ("L", bundle.l_saccades, bundle.left),
            ("R", bundle.r_saccades, bundle.right),
        ):
            keep = non_saccade_interframe_dangle(df, ev, pad_ms=pad_ms)
            if keep.size == 0:
                continue
            med = float(np.median(keep))
            mad = float(np.median(np.abs(keep - med)))
            sigma = mad / 0.6745 if mad > 0 else float("nan")
            rows.append(
                {
                    "animal": bundle.spec.animal,
                    "block": bundle.spec.block_num,
                    "eye": eye,
                    "n_samples": int(keep.size),
                    "median_deg_per_frame": med,
                    "mad_deg_per_frame": mad,
                    "engbert_sigma_deg_per_frame": sigma,
                    "p95_deg_per_frame": float(np.nanpercentile(keep, 95)),
                    "frac_above_threshold": float(np.mean(keep >= threshold_deg_per_frame)),
                    "threshold_deg_per_frame": float(threshold_deg_per_frame),
                }
            )
            sample_chunks.append(keep)
    out = pd.DataFrame(rows)
    out.attrs["samples"] = (
        np.concatenate(sample_chunks) if sample_chunks else np.array([], dtype=float)
    )
    return out


def collect_gui_speed_thresholds(tables: EventTables) -> pd.DataFrame:
    """Per-block GUI finalize thresholds from ``analysis/saccades/detection_params.yaml``."""
    rows: list[dict[str, Any]] = []
    for bundle in tables.blocks:
        spec = bundle.spec
        path = Path(spec.block_path) / "analysis" / "saccades" / "detection_params.yaml"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        if not isinstance(params, dict):
            continue
        thr = (params.get("saccade") or {}).get("speed_threshold_deg_per_frame")
        if thr is None:
            continue
        rows.append(
            {
                "animal": spec.animal,
                "block": spec.block_num,
                "speed_threshold_deg_per_frame": float(thr),
                "speed_threshold_deg_per_ms": params.get("speed_threshold_deg_per_ms"),
                "frame_ms": params.get("frame_ms"),
            }
        )
    return pd.DataFrame(rows)


def resolve_detector_thresholds(
    tables: EventTables,
    *,
    detector_thresholds: Sequence[float] | None = None,
) -> tuple[np.ndarray, str, pd.DataFrame]:
    """Return ``(values, source, per_block_table)``.

    Prefers explicit values, then on-disk GUI finalize params, then the run YAML.
    """
    if detector_thresholds is not None:
        vals = np.asarray(list(detector_thresholds), dtype=float)
        vals = vals[np.isfinite(vals)]
        return vals, "explicit", pd.DataFrame(
            {"speed_threshold_deg_per_frame": vals}
        )
    gui = collect_gui_speed_thresholds(tables)
    if not gui.empty:
        vals = pd.to_numeric(gui["speed_threshold_deg_per_frame"], errors="coerce")
        vals = vals.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return vals, "gui_finalized", gui
    yaml_thr = float(
        (getattr(tables, "params", {}) or {})
        .get("saccade", {})
        .get("speed_threshold_deg_per_frame", DEFAULT_THRESHOLD)
    )
    return np.array([yaml_thr], dtype=float), "yaml", pd.DataFrame(
        {"speed_threshold_deg_per_frame": [yaml_thr]}
    )


def _noise_floor_figure(
    samples: np.ndarray,
    *,
    detector_mean: float,
    detector_lo: float | None = None,
    detector_hi: float | None = None,
    sigma: float | None = None,
    detector_label: str | None = None,
) -> plt.Figure:
    samples = np.asarray(samples, dtype=float)
    fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
    xmax = float(detector_mean) * 1.5
    if detector_hi is not None and np.isfinite(detector_hi):
        xmax = max(xmax, float(detector_hi) * 1.15)
    if samples.size:
        hi = float(np.nanpercentile(samples, 99.5))
        xmax = max(xmax, hi)
        ax.hist(samples, bins=60, range=(0, xmax), color="#0072B2", density=True)
    if (
        detector_lo is not None
        and detector_hi is not None
        and np.isfinite(detector_lo)
        and np.isfinite(detector_hi)
        and float(detector_hi) > float(detector_lo)
    ):
        ax.axvspan(
            float(detector_lo),
            float(detector_hi),
            color="0.75",
            alpha=0.45,
            lw=0,
            label=f"GUI range {detector_lo:.2f}–{detector_hi:.2f}",
            zorder=0,
        )
    label = detector_label or f"mean GUI {detector_mean:.2f}°/frame"
    ax.axvline(float(detector_mean), color="#D55E00", lw=1.2, ls="--", label=label)
    if sigma is not None and np.isfinite(sigma):
        ax.axvline(float(sigma), color="0.2", lw=1.0, ls=":", label=f"MAD σ={sigma:.3f}")
    ax.set_xlabel("Inter-frame Δangle [deg/frame]", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _line_plot(x, y, *, xlabel: str, ylabel: str, figsize=(2.4, 1.8)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.plot(x, y, "o-", color="#0072B2", lw=1.2, ms=4)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def export_robustness(
    tables: EventTables,
    out_dir: Path,
    *,
    n_shuffle: int = N_SHUFFLE,
    show: bool = False,
    rng: np.random.Generator | None = None,
    plot_id: str = "robustness",
) -> dict[str, Path]:
    """Write lizard threshold/window sweeps, noise-floor hist, and optional shuffle.

    ``plot_id`` is the bundle folder name (default ``robustness``). Mouse
    noise-floor comparison uses :func:`export_noise_floor` instead.
    """
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="robustness",
        tables=tables,
        logic_key="robustness",
        params=dict(getattr(tables, "params", {}) or {}),
    )
    figures_dir, metadata_dir = bundle.plots_dir, bundle.metadata_dir
    written: dict[str, Path] = {}

    thr = threshold_sweep(tables)
    fig = _line_plot(
        thr["speed_threshold_deg_per_frame"],
        thr["monocular_fraction"],
        xlabel="Speed threshold [deg/frame]",
        ylabel="Monocular fraction",
    )
    p = figures_dir / "robustness_threshold_sweep.pdf"
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p
    thr.to_csv(metadata_dir / "robustness_threshold_sweep.csv", index=False)

    win = window_sweep(tables)
    fig = _line_plot(
        win["pairing_window_ms"],
        win["monocular_fraction"],
        xlabel="Pairing window [ms]",
        ylabel="Monocular fraction",
    )
    p = figures_dir / "robustness_window_sweep.pdf"
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p
    win.to_csv(metadata_dir / "robustness_window_sweep.csv", index=False)

    nf = noise_floor(tables, threshold_deg_per_frame=DEFAULT_THRESHOLD)
    nf.to_csv(metadata_dir / "robustness_noise_floor.csv", index=False)
    samples = np.asarray(nf.attrs.get("samples", []), dtype=float)
    sigma = None
    if not nf.empty and nf["engbert_sigma_deg_per_frame"].notna().any():
        sigma = float(np.nanmedian(nf["engbert_sigma_deg_per_frame"]))
    fig = _noise_floor_figure(
        samples,
        detector_mean=DEFAULT_THRESHOLD,
        sigma=sigma,
        detector_label="detector 0.8°/frame",
    )
    p = figures_dir / "robustness_noise_floor.pdf"
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p

    if n_shuffle > 0:
        observed, null = shuffle_monocular_fractions(
            tables, n_shuffle=n_shuffle, rng=rng
        )
        fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
        if null.size:
            ax.hist(null, bins=30, color="0.7", edgecolor="black")
        ax.axvline(observed, color="#D55E00", lw=1.5, label=f"observed {observed:.3f}")
        ax.set_xlabel("Monocular fraction (circular-shift null)", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        p = figures_dir / "robustness_shuffle.pdf"
        fig.savefig(p, format="pdf", bbox_inches="tight")
        show_and_close(fig, show)
        written[p.name] = p
        pd.DataFrame({"null_monocular_fraction": null}).to_csv(
            metadata_dir / "robustness_shuffle.csv", index=False
        )
    finish_plot_bundle(bundle)
    return written


_SWEEP_ARTIFACTS = (
    "robustness_threshold_sweep.pdf",
    "robustness_window_sweep.pdf",
    "robustness_shuffle.pdf",
    "robustness_threshold_sweep.csv",
    "robustness_window_sweep.csv",
    "robustness_shuffle.csv",
)


def export_noise_floor(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
    plot_id: str = "mouse_robustness",
    detector_thresholds: Sequence[float] | None = None,
    drop_sweeps: bool = True,
) -> dict[str, Path]:
    """Noise-floor histogram vs mean GUI detector (gray band = per-block range).

    Default folder is ``mouse_robustness`` so the lizard ``robustness`` bundle
    is not overwritten. No threshold/window sweep is written.
    """
    vals, source, per_block = resolve_detector_thresholds(
        tables, detector_thresholds=detector_thresholds
    )
    if vals.size == 0:
        raise ValueError("No detector thresholds available for noise-floor export")
    detector_mean = float(np.mean(vals))
    detector_lo = float(np.min(vals))
    detector_hi = float(np.max(vals))

    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="robustness",
        tables=tables,
        logic_key="robustness_noise_floor",
        params=dict(getattr(tables, "params", {}) or {}),
        extra={
            "detector_source": source,
            "detector_mean_deg_per_frame": detector_mean,
            "detector_min_deg_per_frame": detector_lo,
            "detector_max_deg_per_frame": detector_hi,
            "detector_n_blocks": int(vals.size),
        },
    )
    figures_dir, metadata_dir = bundle.plots_dir, bundle.metadata_dir
    written: dict[str, Path] = {}

    per_block.to_csv(metadata_dir / "gui_detector_thresholds.csv", index=False)
    nf = noise_floor(tables, threshold_deg_per_frame=detector_mean)
    nf.to_csv(metadata_dir / "robustness_noise_floor.csv", index=False)
    samples = np.asarray(nf.attrs.get("samples", []), dtype=float)
    sigma = None
    if not nf.empty and nf["engbert_sigma_deg_per_frame"].notna().any():
        sigma = float(np.nanmedian(nf["engbert_sigma_deg_per_frame"]))
    span = (detector_lo, detector_hi) if vals.size > 1 else (None, None)
    fig = _noise_floor_figure(
        samples,
        detector_mean=detector_mean,
        detector_lo=span[0],
        detector_hi=span[1],
        sigma=sigma,
    )
    p = figures_dir / "robustness_noise_floor.pdf"
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p

    if drop_sweeps:
        for name in _SWEEP_ARTIFACTS:
            for folder in (figures_dir, metadata_dir):
                stale = folder / name
                if stale.is_file():
                    stale.unlink()

    finish_plot_bundle(bundle)
    return written


def _hist_percent(samples: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return bins[:-1], np.zeros(len(bins) - 1)
    hist, edges = np.histogram(samples, bins=bins)
    return edges[:-1], (hist / samples.size) * 100.0


def _block_spec_from_jitter(spec) -> Any:
    from eye_tracking_system_tools.analysis.block_registry import BlockSpec, _infer_block_num

    path = Path(spec.block_path)
    return BlockSpec(animal=str(spec.animal), block_path=path, block_num=_infer_block_num(path))


def collect_noise_pools_from_jitter_specs(
    specs,
    *,
    pad_ms: float = 40.0,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Non-saccade inter-frame Δangle pooled by jitter ``mount_type``.

    Blocks without Kerr angles are skipped. Turtle (and any block without
    finalized saccades) keeps every finite inter-frame step.
    """
    from eye_tracking_system_tools.analysis.eye_trace_io import load_block_eyes
    from eye_tracking_system_tools.analysis.saccade_export import (
        has_finalized_saccades,
        read_finalized_saccades,
    )

    chunks: dict[str, list[np.ndarray]] = {k: [] for k, _l, _c in UNIFIED_NOISE_PANELS}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        mount = str(getattr(spec, "mount_type", "") or "")
        if mount not in chunks:
            continue
        block = _block_spec_from_jitter(spec)
        try:
            loaded = load_block_eyes(block, log=verbose)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[skip] {block.block_key}: no Kerr traces ({exc})")
            continue
        l_ev = r_ev = None
        n_ev = 0
        if has_finalized_saccades(block.block_path):
            try:
                fin = read_finalized_saccades(block.block_path)
                all_ev = fin.all_saccades
                if all_ev is not None and not all_ev.empty and "eye" in all_ev.columns:
                    l_ev = all_ev.loc[all_ev["eye"].astype(str).str.upper() == "L"]
                    r_ev = all_ev.loc[all_ev["eye"].astype(str).str.upper() == "R"]
                    n_ev = int(len(all_ev))
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[warn] {block.block_key}: finalized saccades unreadable ({exc})")
        for eye, ev, df in (("L", l_ev, loaded.left), ("R", r_ev, loaded.right)):
            keep = non_saccade_interframe_dangle(df, ev, pad_ms=pad_ms)
            if keep.size == 0:
                continue
            chunks[mount].append(keep)
            rows.append(
                {
                    "animal": block.animal,
                    "block": block.block_num,
                    "block_key": block.block_key,
                    "mount_type": mount,
                    "eye": eye,
                    "n_samples": int(keep.size),
                    "n_saccade_events": n_ev,
                    "median_deg_per_frame": float(np.median(keep)),
                    "engbert_sigma_deg_per_frame": _engbert_sigma(keep),
                    "p95_deg_per_frame": float(np.nanpercentile(keep, 95)),
                    "p99_5_deg_per_frame": float(np.nanpercentile(keep, 99.5)),
                }
            )
    pools = {
        k: (np.concatenate(v) if v else np.array([], dtype=float)) for k, v in chunks.items()
    }
    return pools, pd.DataFrame(rows)


def split_tables_by_jitter_mount(
    lizard: EventTables | None,
    mouse: EventTables | None,
    specs,
) -> dict[str, EventTables]:
    """Map paper/mouse event tables onto jitter mount types."""
    from eye_tracking_system_tools.analysis.pipeline import filter_event_tables

    by_animal: dict[str, str] = {}
    for spec in specs or []:
        by_animal[str(spec.animal)] = str(spec.mount_type)
    out: dict[str, EventTables] = {}
    if lizard is not None:
        lizard_animals = {b.spec.animal for b in lizard.blocks}
        rigid = {a for a, m in by_animal.items() if m == "rigid"}
        modular = {a for a, m in by_animal.items() if m == "modular"}
        for animal in lizard_animals:
            if animal not in by_animal:
                rigid.add(animal)
        if rigid:
            out["rigid"] = filter_event_tables(lizard, animals=rigid)
        if modular:
            out["modular"] = filter_event_tables(lizard, animals=modular)
    if mouse is not None:
        out["mouse"] = mouse
    return out


def figure_unified_noise(
    pools: dict[str, np.ndarray],
    *,
    xmax: float | None = None,
    n_bins: int = 60,
):
    """2×2 noise histograms with a shared x-limit from the noisiest condition."""
    from eye_tracking_system_tools.analysis.jitter_epochs import shared_jitter_xmax

    if xmax is None:
        xmax, _ = shared_jitter_xmax(pools)
    xmax = max(float(xmax), 1e-6)
    bins = np.linspace(0, xmax, int(n_bins) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.8), dpi=150, sharex=True)
    for ax, (mount, label, color) in zip(axes.ravel(), UNIFIED_NOISE_PANELS):
        values = np.asarray(pools.get(mount, np.array([])), dtype=float)
        n = int(np.isfinite(values).sum()) if values.size else 0
        sigma = _engbert_sigma(values)
        x, y = _hist_percent(values, bins)
        ax.bar(
            x,
            y,
            width=np.diff(bins),
            align="edge",
            color=color,
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_xlim(0, xmax)
        sig = f", σ={sigma:.3f}" if np.isfinite(sigma) else ""
        ax.set_title(f"{label} (n={n}{sig})", fontsize=8)
        ax.set_ylabel("% samples", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Inter-frame Δangle [deg/frame]", fontsize=8)
    fig.tight_layout()
    return fig


def export_unified_noise(
    out_dir: Path,
    *,
    specs=None,
    tables_by_mount: dict[str, EventTables] | None = None,
    pools: dict[str, np.ndarray] | None = None,
    n_bins: int = 60,
    show: bool = False,
    plot_id: str = UNIFIED_NOISE_PLOT_ID,
    pad_ms: float = 40.0,
) -> dict[str, Path]:
    """Four-panel non-saccade noise comparison (no detector threshold line).

    Prefer ``tables_by_mount`` (cached events, same exclusion as the earlier
    noise-floor plots). Empty mounts are filled from ``specs`` (turtle).
    """
    from eye_tracking_system_tools.analysis.jitter_epochs import shared_jitter_xmax

    per_block = pd.DataFrame()
    frames: list[pd.DataFrame] = []
    if pools is None:
        pools = {k: np.array([], dtype=float) for k, _l, _c in UNIFIED_NOISE_PANELS}
        if tables_by_mount:
            for mount, tables in tables_by_mount.items():
                nf = noise_floor(tables, pad_ms=pad_ms)
                pools[str(mount)] = np.asarray(nf.attrs.get("samples", []), dtype=float)
                if not nf.empty:
                    extra = nf.copy()
                    extra["mount_type"] = str(mount)
                    extra["block_key"] = (
                        extra["animal"].astype(str)
                        + "_block_"
                        + extra["block"].astype(str).str.zfill(3)
                    )
                    extra["n_saccade_events"] = np.nan
                    frames.append(extra)
        if specs is None and not tables_by_mount:
            from eye_tracking_system_tools.analysis.jitter_epochs import load_jitter_registry

            repo = Path(__file__).resolve().parents[3]
            specs = load_jitter_registry(repo / "configs" / "jitter_mount_blocks.yaml")
        if specs is not None:
            need = [s for s in specs if np.asarray(pools.get(s.mount_type, []), dtype=float).size == 0]
            if need:
                extra_pools, extra_df = collect_noise_pools_from_jitter_specs(
                    need, pad_ms=pad_ms
                )
                for mount, arr in extra_pools.items():
                    if np.asarray(pools.get(mount, []), dtype=float).size == 0 and arr.size:
                        pools[mount] = arr
                if extra_df is not None and not extra_df.empty:
                    frames.append(extra_df)
        per_block = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    xmax, xmax_source = shared_jitter_xmax(pools)

    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="unified_noise",
        logic_key="unified_noise",
        cohort={
            "cohort": "multi",
            "animals": sorted({str(a) for a in per_block["animal"].unique()}) if not per_block.empty else [],
            "block_keys": per_block["block_key"].tolist() if not per_block.empty else [],
            "rule": "registry_mount_type",
            "mount_type": "all",
        },
        extra={
            "n_bins": int(n_bins),
            "xmax": float(xmax),
            "xmax_source": xmax_source,
            "xmax_percentile": 99.5,
            "pad_ms": float(pad_ms),
            "exclude_saccades": True,
        },
    )
    written: dict[str, Path] = {}
    if not per_block.empty:
        per_block.to_csv(bundle.metadata_dir / "unified_noise_blocks.csv", index=False)
    fig = figure_unified_noise(pools, xmax=xmax, n_bins=n_bins)
    p = bundle.plots_dir / UNIFIED_NOISE_PDF
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p

    payload = {
        "pools": {k: np.asarray(v, dtype=float) for k, v in pools.items()},
        "n_bins": int(n_bins),
        "xmax": float(xmax),
        "xmax_source": xmax_source,
        "pdf_name": UNIFIED_NOISE_PDF,
        "xlabel": "Inter-frame Δangle [deg/frame]",
        "ylabel": "% samples",
        "panels": [list(row) for row in UNIFIED_NOISE_PANELS],
    }
    write_pickle_with_meta(
        payload,
        bundle.metadata_dir / "unified_noise.pkl",
        meta={"xmax": float(xmax), "xmax_source": xmax_source, "n_bins": int(n_bins)},
        entrypoint="eye_tracking_system_tools.analysis.saccade_robustness.export_unified_noise",
    )
    summary = {
        "n_bins": int(n_bins),
        "xmax": float(xmax),
        "xmax_source": xmax_source,
        "xmax_percentile": 99.5,
        "pad_ms": float(pad_ms),
        "mounts": {},
    }
    for mount, label, _color in UNIFIED_NOISE_PANELS:
        arr = np.asarray(pools.get(mount, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        summary["mounts"][mount] = {
            "label": label,
            "n_samples": int(arr.size),
            "median": float(np.median(arr)) if arr.size else None,
            "engbert_sigma": float(_engbert_sigma(arr)) if arr.size else None,
            "p95": float(np.percentile(arr, 95)) if arr.size else None,
            "p99_5": float(np.percentile(arr, 99.5)) if arr.size else None,
        }
    with open(bundle.metadata_dir / "unified_noise_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    finish_plot_bundle(bundle)
    written["summary"] = bundle.metadata_dir / "unified_noise_summary.yaml"
    return written


def _pool_percentiles(arr: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n_samples": 0,
            "min": None,
            "median": None,
            "p95": None,
            "p99": None,
            "p99_5": None,
            "p99_9": None,
            "max": None,
        }
    qs = np.percentile(arr, [50, 95, 99, 99.5, 99.9])
    return {
        "n_samples": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(qs[0]),
        "p95": float(qs[1]),
        "p99": float(qs[2]),
        "p99_5": float(qs[3]),
        "p99_9": float(qs[4]),
        "max": float(np.max(arr)),
    }


def rayleigh_pdf(x: np.ndarray, scale: float) -> np.ndarray:
    """Rayleigh PDF with loc=0. ``scale`` B is σ of each Gaussian axis."""
    b = max(float(scale), 1e-15)
    x = np.asarray(x, dtype=float)
    return (x / (b * b)) * np.exp(-0.5 * (x / b) ** 2)


def fit_rayleigh_distance(arr: np.ndarray) -> dict[str, float | int | None]:
    """MLE Rayleigh scale on distances, plus shape checks (skew / excess kurtosis).

    If Δφ and Δθ are i.i.d. Gaussian with std σ, D=hypot(Δφ, Δθ) is Rayleigh
    with B=σ. MLE: B = sqrt(mean(D²)/2). No percentile trim.
    """
    from scipy import stats

    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a) & (a >= 0)]
    if a.size < 8:
        return {
            "n": int(a.size),
            "B": None,
            "sigma_axis": None,
            "mean": None,
            "rayleigh_mean": None,
            "skew": None,
            "excess_kurtosis": None,
            "skew_target": RAYLEIGH_SKEW,
            "kurtosis_target": RAYLEIGH_EXCESS_KURTOSIS,
        }
    scale = float(np.sqrt(np.mean(a * a) / 2.0))
    return {
        "n": int(a.size),
        "B": scale,
        "sigma_axis": scale,
        "mean": float(np.mean(a)),
        "rayleigh_mean": float(scale * np.sqrt(np.pi / 2.0)),
        "skew": float(stats.skew(a, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(a, fisher=True, bias=False)),
        "skew_target": RAYLEIGH_SKEW,
        "kurtosis_target": RAYLEIGH_EXCESS_KURTOSIS,
    }


def collect_raw_interframe_pools(
    specs,
    *,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """All inter-frame Δangle values, pooled by jitter ``mount_type``.

    Loads Kerr traces only. Does not read saccade tables or apply a detector
    threshold. Blocks without Kerr angles are skipped.
    """
    from eye_tracking_system_tools.analysis.eye_trace_io import load_block_eyes

    chunks: dict[str, list[np.ndarray]] = {k: [] for k, _l, _c in UNIFIED_NOISE_PANELS}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        mount = str(getattr(spec, "mount_type", "") or "")
        if mount not in chunks:
            continue
        block = _block_spec_from_jitter(spec)
        try:
            loaded = load_block_eyes(block, log=verbose)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[skip] {block.block_key}: no Kerr traces ({exc})")
            continue
        for eye, df in (("L", loaded.left), ("R", loaded.right)):
            dang, dt, _mid = interframe_steps(df)
            keep = dang[np.isfinite(dang)]
            if keep.size == 0:
                continue
            dt_ok = dt[np.isfinite(dang)]
            dt_ok = dt_ok[np.isfinite(dt_ok)]
            chunks[mount].append(keep)
            rows.append(
                {
                    "animal": block.animal,
                    "block": block.block_num,
                    "block_key": block.block_key,
                    "mount_type": mount,
                    "eye": eye,
                    "n_samples": int(keep.size),
                    "median_dt_ms": float(np.median(dt_ok)) if dt_ok.size else float("nan"),
                    "median_deg_per_frame": float(np.median(keep)),
                    "p95_deg_per_frame": float(np.nanpercentile(keep, 95)),
                    "p99_5_deg_per_frame": float(np.nanpercentile(keep, 99.5)),
                    "max_deg_per_frame": float(np.nanmax(keep)),
                }
            )
            if verbose:
                print(
                    f"[raw Δ] {block.block_key} {eye} n={keep.size} "
                    f"median={np.median(keep):.4f} p99.5={np.nanpercentile(keep, 99.5):.3f}"
                )
    pools = {
        k: (np.concatenate(v) if v else np.array([], dtype=float)) for k, v in chunks.items()
    }
    return pools, pd.DataFrame(rows)


def figure_raw_interframe_delta(
    pools: dict[str, np.ndarray],
    *,
    xmax: float | None = None,
    n_bins: int = 50,
    log_x: bool = False,
    fits: dict[str, dict] | None = None,
):
    """2×2 histograms of every inter-frame step (no trim). Linear panels add a Rayleigh PDF."""
    if fits is None:
        fits = {m: fit_rayleigh_distance(pools.get(m, [])) for m, _l, _c in UNIFIED_NOISE_PANELS}
    if log_x:
        if xmax is None:
            xmax = 1e-6
            for arr in pools.values():
                vals = np.asarray(arr, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    xmax = max(xmax, float(np.nanmax(vals)))
        xmax = max(float(xmax), 1e-6)
        lo = 1e-3
        for arr in pools.values():
            vals = np.asarray(arr, dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                lo = min(lo, float(np.nanpercentile(vals, 0.1)))
        lo = max(lo, 1e-4)
        bins = np.logspace(np.log10(lo), np.log10(xmax), int(n_bins) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0), dpi=150, sharex=True)
        for ax, (mount, label, color) in zip(axes.ravel(), UNIFIED_NOISE_PANELS):
            values = np.asarray(pools.get(mount, np.array([])), dtype=float)
            values = values[np.isfinite(values)]
            n = int(values.size)
            x, y = _hist_percent(values, bins)
            ax.bar(
                x,
                y,
                width=np.diff(bins),
                align="edge",
                color=color,
                edgecolor="black",
                alpha=0.7,
            )
            ax.set_xscale("log")
            ax.set_xlim(bins[0], xmax)
            ax.set_title(f"{label} (n={n})", fontsize=8)
            ax.set_ylabel("% samples", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for ax in axes[1]:
            ax.set_xlabel("Inter-frame Δangle [deg/frame]", fontsize=8)
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0), dpi=150)
    for ax, (mount, label, color) in zip(axes.ravel(), UNIFIED_NOISE_PANELS):
        values = np.asarray(pools.get(mount, np.array([])), dtype=float)
        values = values[np.isfinite(values) & (values >= 0)]
        n = int(values.size)
        hi = float(xmax) if xmax is not None else (float(np.nanmax(values)) if n else 1.0)
        hi = max(hi, 1e-6)
        if n:
            ax.hist(
                values,
                bins=int(n_bins),
                range=(0.0, hi),
                density=True,
                color=color,
                edgecolor="0.3",
                alpha=0.7,
                label="data",
            )
            fit = fits.get(mount) or {}
            scale = fit.get("B")
            if scale is not None and np.isfinite(float(scale)) and float(scale) > 0:
                xs = np.linspace(0.0, hi, 400)
                ax.plot(
                    xs,
                    rayleigh_pdf(xs, float(scale)),
                    color="0.15",
                    lw=1.5,
                    label=f"Rayleigh B={float(scale):.3f}",
                )
                skew = fit.get("skew")
                kurt = fit.get("excess_kurtosis")
                extra = ""
                if skew is not None and kurt is not None:
                    extra = f"\nskew={float(skew):.3f} (0.631)  kurt={float(kurt):.3f} (0.245)"
                ax.set_title(f"{label}  n={n}{extra}", fontsize=7)
            else:
                ax.set_title(f"{label} (n={n})", fontsize=8)
            ax.legend(fontsize=6, frameon=False)
        else:
            ax.set_title(f"{label} (n=0)", fontsize=8)
        ax.set_xlim(0.0, hi)
        ax.set_ylabel("Probability density", fontsize=8)
        ax.set_xlabel("Inter-frame Δangle [deg/frame]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def export_raw_interframe_delta(
    out_dir: Path,
    *,
    specs=None,
    pools: dict[str, np.ndarray] | None = None,
    per_block: pd.DataFrame | None = None,
    n_bins: int = 50,
    show: bool = False,
    plot_id: str = RAW_INTERFRAME_PLOT_ID,
    verbose: bool = True,
) -> dict[str, Path]:
    """Four-panel Δangle histograms with **no** saccade exclusion or trim.

    Each linear panel is a PDF histogram of every finite sample plus an MLE
    Rayleigh overlay (scale B = σ of each Gaussian axis if the model holds).
    """
    from eye_tracking_system_tools.analysis.jitter_epochs import load_jitter_registry

    if pools is None:
        if specs is None:
            repo = Path(__file__).resolve().parents[3]
            specs = load_jitter_registry(repo / "configs" / "jitter_mount_blocks.yaml")
        pools, per_block = collect_raw_interframe_pools(specs, verbose=verbose)
    if per_block is None:
        per_block = pd.DataFrame()

    fits = {
        mount: fit_rayleigh_distance(pools.get(mount, []))
        for mount, _label, _color in UNIFIED_NOISE_PANELS
    }
    log_xmax = 1e-6
    log_source: str | None = None
    for mount, arr in pools.items():
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        hi = float(np.nanmax(vals))
        if hi >= log_xmax:
            log_xmax = hi
            log_source = str(mount)

    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind="raw_interframe_delta",
        logic_key="raw_interframe_delta",
        cohort={
            "cohort": "multi",
            "animals": sorted({str(a) for a in per_block["animal"].unique()})
            if not per_block.empty
            else [],
            "block_keys": per_block["block_key"].tolist() if not per_block.empty else [],
            "rule": "registry_mount_type",
            "mount_type": "all",
        },
        extra={
            "n_bins": int(n_bins),
            "exclude_saccades": False,
            "trimmed": False,
            "metric": "hypot(d_k_phi, d_k_theta) deg/frame",
            "noise_measure": "Rayleigh scale B (deg/frame) = axis σ if Δφ,Δθ iid Gaussian",
            "rayleigh_skew_target": RAYLEIGH_SKEW,
            "rayleigh_excess_kurtosis_target": RAYLEIGH_EXCESS_KURTOSIS,
        },
    )
    written: dict[str, Path] = {}
    if not per_block.empty:
        csv_path = bundle.metadata_dir / "raw_interframe_delta_blocks.csv"
        per_block.to_csv(csv_path, index=False)
        written[csv_path.name] = csv_path

    fig = figure_raw_interframe_delta(
        pools, n_bins=n_bins, log_x=False, fits=fits
    )
    p = bundle.plots_dir / RAW_INTERFRAME_PDF
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p

    fig_log = figure_raw_interframe_delta(
        pools, xmax=log_xmax, n_bins=n_bins, log_x=True, fits=fits
    )
    p_log = bundle.plots_dir / RAW_INTERFRAME_LOGX_PDF
    fig_log.savefig(p_log, format="pdf", bbox_inches="tight")
    show_and_close(fig_log, show)
    written[p_log.name] = p_log

    payload = {
        "pools": {k: np.asarray(v, dtype=np.float64) for k, v in pools.items()},
        "n_bins": int(n_bins),
        "logx_xmax": float(log_xmax),
        "logx_xmax_source": log_source,
        "pdf_name": RAW_INTERFRAME_PDF,
        "logx_pdf_name": RAW_INTERFRAME_LOGX_PDF,
        "xlabel": "Inter-frame Δangle [deg/frame]",
        "ylabel": "Probability density",
        "panels": [list(row) for row in UNIFIED_NOISE_PANELS],
        "exclude_saccades": False,
        "trimmed": False,
        "metric": "hypot(d_k_phi, d_k_theta) deg/frame",
        "fits": fits,
        "rayleigh_skew_target": RAYLEIGH_SKEW,
        "rayleigh_excess_kurtosis_target": RAYLEIGH_EXCESS_KURTOSIS,
        "per_block": per_block.to_dict(orient="list") if not per_block.empty else {},
    }
    pkl = bundle.metadata_dir / RAW_INTERFRAME_PICKLE
    write_pickle_with_meta(
        payload,
        pkl,
        meta={
            "n_bins": int(n_bins),
            "exclude_saccades": False,
            "trimmed": False,
            "fits": {
                m: {k: v for k, v in body.items() if k in {"n", "B", "skew", "excess_kurtosis"}}
                for m, body in fits.items()
            },
            "n_per_mount": {
                k: int(np.isfinite(np.asarray(v, dtype=float)).sum())
                for k, v in pools.items()
            },
        },
        entrypoint="eye_tracking_system_tools.analysis.saccade_robustness.export_raw_interframe_delta",
    )
    written[pkl.name] = pkl

    summary = {
        "exclude_saccades": False,
        "trimmed": False,
        "metric": "hypot(d_k_phi, d_k_theta) deg/frame",
        "noise_measure": "Rayleigh B (deg/frame)",
        "n_bins": int(n_bins),
        "rayleigh_skew_target": RAYLEIGH_SKEW,
        "rayleigh_excess_kurtosis_target": RAYLEIGH_EXCESS_KURTOSIS,
        "logx_xmax": float(log_xmax),
        "mounts": {},
    }
    for mount, label, _color in UNIFIED_NOISE_PANELS:
        stats = _pool_percentiles(pools.get(mount, []))
        stats["label"] = label
        stats.update({k: fits[mount].get(k) for k in ("B", "sigma_axis", "skew", "excess_kurtosis")})
        summary["mounts"][mount] = stats
    with open(bundle.metadata_dir / "raw_interframe_delta_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    finish_plot_bundle(bundle)
    written["summary"] = bundle.metadata_dir / "raw_interframe_delta_summary.yaml"
    return written

