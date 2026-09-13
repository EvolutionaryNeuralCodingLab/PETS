"""Secondary detected eye movements after large, head-stationary-onset saccades.

Reviewer question (video S1): after a large saccade, how often do smaller
follow-up / reversing eye movements occur, and are they tied to later head
motion? This module reuses existing saccade tables and ``lizMov.mat`` bout
onsets. It does not re-detect saccades and does not label the follow-ups as
nystagmus, VOR, or corrective saccades.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

from eye_tracking_system_tools.analysis.binocular import find_synced_saccades_ms
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.head_labels import (
    find_lizmov_mat,
    load_lizmov_bout_onsets_ms,
    load_lizmov_samples,
    load_lizmov_times_ms,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PLOT_ID = "Rev_3_Q6_nystagmus_like"
KIND = "secondary_after_saccade"

LOOKBACK_MS = 50.0
POST_WINDOW_MS = 250.0
BOUT_GAP_MS = 40.0
REVERSE_MIN_DEG = 90.0
AMP_PERCENTILE = 75.0
AMP_FLOOR_DEG = 5.0
PRE_PLOT_MS = 80.0
POST_PLOT_MS = 400.0
TRACE_DT_MS = 2.0
MAX_POP_TRACES = 250
N_BROWSER_PER_CELL = 4
CONSTITUENT_TOL_MS = 1.0
DEFAULT_SYNC_DIFF_MS = 34.0

L_COLOR = "#0072B2"
R_COLOR = "#D55E00"
SAME_COLOR = "#009E73"
REVERSE_COLOR = "#CC79A7"
NONE_COLOR = "0.55"
HEAD_COLOR = "#E69F00"
PRIMARY_COLOR = "0.25"

CLASS_COLORS = {"none": NONE_COLOR, "same": SAME_COLOR, "reverse": REVERSE_COLOR, "unknown": "0.4"}

PICKLE_NAME = "secondary_after_saccade.pkl"
SUMMARY_YAML = "summary.yaml"
CAPTIONS_MD = "captions.md"
EVENTS_CSV = "qualifying_events.csv"
STATIONARY_CSV = "stationary_onset_events.csv"
SECONDARY_CSV = "secondary_events.csv"
DIAG_CSV = "block_diagnostics.csv"

PDF_AMP = "amplitude_distribution.pdf"
PDF_BROWSER = "event_browser.pdf"
PDF_GALLERY = "example_gallery.pdf"
PDF_FRACTIONS = "secondary_fractions.pdf"
PDF_LATENCY = "secondary_latency_amp.pdf"
PDF_HEAD = "head_association.pdf"
PDF_FINAL = "final_position.pdf"
PDF_POP = "population_traces.pdf"
PDF_POP_HEAD = "population_traces_by_head.pdf"


def _sync_diff_ms(tables: EventTables) -> float:
    return float(tables.params.get("binocular", {}).get("sync_diff_ms", DEFAULT_SYNC_DIFF_MS))


def circular_diff_deg(a: float, b: float) -> float:
    return ((float(a) - float(b) + 180.0) % 360.0) - 180.0


def direction_class(primary_angle: float, secondary_angle: float, *, reverse_min_deg: float = REVERSE_MIN_DEG) -> str:
    if not (np.isfinite(primary_angle) and np.isfinite(secondary_angle)):
        return "unknown"
    d = abs(circular_diff_deg(secondary_angle, primary_angle))
    if d > float(reverse_min_deg):
        return "reverse"
    return "same"


def head_stationary_at_onset(
    t_on: float,
    mov_times: np.ndarray,
    *,
    lookback_ms: float = LOOKBACK_MS,
) -> bool:
    """True when no movement sample falls in ``[t_on - lookback, t_on)``."""
    if not np.isfinite(t_on):
        return False
    times = np.asarray(mov_times, dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return True
    i0 = int(np.searchsorted(times, t_on - float(lookback_ms), side="left"))
    i1 = int(np.searchsorted(times, t_on, side="left"))
    return i1 <= i0


def propose_amplitude_threshold(
    amps: np.ndarray,
    *,
    percentile: float = AMP_PERCENTILE,
    floor_deg: float = AMP_FLOOR_DEG,
) -> tuple[float, dict[str, float]]:
    """Upper-quartile amplitude, rounded down to 0.5°, not below ``floor_deg``.

    The floor is one Fig 2e / main-sequence bin (5°). The percentile is taken
    on head-stationary-at-onset unique events so the cutoff tracks this cohort
    rather than a round number chosen in advance.
    """
    x = np.asarray(amps, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    stats = {
        "n": float(x.size),
        "p50": float("nan"),
        "p75": float("nan"),
        "p90": float("nan"),
        "percentile_used": float(percentile),
        "floor_deg": float(floor_deg),
    }
    if x.size == 0:
        stats["proposed_deg"] = float(floor_deg)
        return float(floor_deg), stats
    for p, key in ((50.0, "p50"), (75.0, "p75"), (90.0, "p90"), (float(percentile), "p_used")):
        stats[key] = float(np.percentile(x, p))
    raw = float(stats.get("p_used", stats["p75"]))
    proposed = max(float(floor_deg), np.floor(raw * 2.0) / 2.0)
    stats["proposed_deg"] = float(proposed)
    return float(proposed), stats


def _amp_of(row: pd.Series) -> float:
    if "net_angular_disp" in row.index:
        return float(pd.to_numeric(row["net_angular_disp"], errors="coerce"))
    return float("nan")


def _angle_of(row: pd.Series) -> float:
    if "overall_angle_deg" in row.index:
        return float(pd.to_numeric(row["overall_angle_deg"], errors="coerce"))
    return float("nan")


def unique_gaze_events(
    events: pd.DataFrame,
    *,
    sync_diff_ms: float = DEFAULT_SYNC_DIFF_MS,
    animal: str | None = None,
    block: str | None = None,
    block_key: str | None = None,
) -> pd.DataFrame:
    """One unique gaze event per concurrent L/R pair (paper pairing window)."""
    cols = [
        "animal",
        "block",
        "block_key",
        "event_id",
        "saccade_on_ms",
        "saccade_off_ms",
        "amp_deg",
        "overall_angle_deg",
        "primary_eye",
        "concurrency",
        "eyes",
        "constituent_onsets",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    work = events.copy()
    if "saccade_on_ms" not in work.columns:
        return pd.DataFrame(columns=cols)
    if "animal" not in work.columns:
        work["animal"] = animal
    if "block" not in work.columns:
        work["block"] = block
    rows: list[dict[str, Any]] = []

    def _cell(part: pd.DataFrame, col: str, default: Any) -> Any:
        if col not in part.columns or part.empty:
            return default
        v = part[col].iloc[0]
        return default if pd.isna(v) else v

    def _emit(part: pd.DataFrame, concurrency: str) -> None:
        if part is None or part.empty:
            return
        on = pd.to_numeric(part["saccade_on_ms"], errors="coerce")
        off = (
            pd.to_numeric(part["saccade_off_ms"], errors="coerce")
            if "saccade_off_ms" in part.columns
            else pd.Series(np.nan, index=part.index)
        )
        amp = (
            pd.to_numeric(part["net_angular_disp"], errors="coerce")
            if "net_angular_disp" in part.columns
            else pd.Series(np.nan, index=part.index)
        )
        t_on = float(np.nanmin(on.to_numpy(float)))
        t_off = float(np.nanmax(off.to_numpy(float))) if off.notna().any() else float("nan")
        if amp.notna().any():
            primary = part.loc[amp.idxmax()]
        else:
            primary = part.iloc[0]
        constituents = []
        for _, rec in part.iterrows():
            constituents.append(
                {
                    "eye": str(rec.get("eye", "")),
                    "saccade_on_ms": float(pd.to_numeric(rec.get("saccade_on_ms"), errors="coerce")),
                    "saccade_off_ms": float(pd.to_numeric(rec.get("saccade_off_ms"), errors="coerce")),
                    "amp_deg": _amp_of(rec),
                    "overall_angle_deg": _angle_of(rec),
                }
            )
        animal_i = str(_cell(part, "animal", animal))
        block_i = str(_cell(part, "block", block))
        key = str(_cell(part, "block_key", block_key or ""))
        eid = f"{animal_i}_{block_i}_{int(round(t_on))}_{str(primary.get('eye', 'X'))}"
        rows.append(
            {
                "animal": animal_i,
                "block": block_i,
                "block_key": key,
                "event_id": eid,
                "saccade_on_ms": t_on,
                "saccade_off_ms": t_off,
                "amp_deg": _amp_of(primary),
                "overall_angle_deg": _angle_of(primary),
                "primary_eye": str(primary.get("eye", "")),
                "concurrency": concurrency,
                "eyes": int(part["eye"].nunique()) if "eye" in part.columns else int(len(part)),
                "constituent_onsets": constituents,
            }
        )

    group_cols = [c for c in ("animal", "block") if c in work.columns]
    grouped = work.groupby(group_cols, dropna=False) if group_cols else [(None, work)]
    for _, g in grouped:
        if "eye" not in g.columns:
            for _, rec in g.iterrows():
                _emit(pd.DataFrame([rec.to_dict()]), "unknown")
            continue
        synced, mono = find_synced_saccades_ms(g, sync_diff_ms=float(sync_diff_ms))
        if synced is not None and not synced.empty and "Main" in synced.columns:
            for _, pair in synced.groupby("Main"):
                _emit(pair, "binocular")
        if mono is not None and not mono.empty:
            for _, rec in mono.iterrows():
                _emit(pd.DataFrame([rec.to_dict()]), "monocular")

    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    dup = out.groupby("event_id").cumcount()
    out.loc[dup > 0, "event_id"] = out.loc[dup > 0, "event_id"].astype(str) + "_" + dup[dup > 0].astype(str)
    return out


def _is_constituent(eye: str, t_on: float, constituents: list[dict[str, Any]], *, tol_ms: float = CONSTITUENT_TOL_MS) -> bool:
    for rec in constituents or []:
        if str(rec.get("eye", "")) != str(eye):
            continue
        t = rec.get("saccade_on_ms")
        if t is not None and np.isfinite(t) and abs(float(t) - float(t_on)) <= tol_ms:
            return True
    return False


def _interp_col(df: pd.DataFrame | None, t_ms: np.ndarray, col: str) -> np.ndarray:
    t_ms = np.asarray(t_ms, dtype=float)
    out = np.full(t_ms.shape, np.nan)
    if df is None or df.empty or col not in df.columns or "ms_axis" not in df.columns:
        return out
    t = pd.to_numeric(df["ms_axis"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    m = np.isfinite(t) & np.isfinite(y)
    if int(m.sum()) < 2:
        return out
    order = np.argsort(t[m])
    tt, yy = t[m][order], y[m][order]
    inside = (t_ms >= tt[0]) & (t_ms <= tt[-1])
    out[inside] = np.interp(t_ms[inside], tt, yy)
    return out


def _axis_unit(angle_deg: float) -> tuple[float, float]:
    rad = np.radians(float(angle_deg))
    return float(np.cos(rad)), float(np.sin(rad))


def _signed_along_axis(
    phi: np.ndarray,
    theta: np.ndarray,
    *,
    phi0: float,
    th0: float,
    angle_deg: float,
) -> np.ndarray:
    ux, uy = _axis_unit(angle_deg)
    return (np.asarray(phi, dtype=float) - phi0) * ux + (np.asarray(theta, dtype=float) - th0) * uy


def _first_onset_after(t_on: float, onsets: np.ndarray, *, window_ms: float) -> tuple[float, float]:
    onsets = np.asarray(onsets, dtype=float)
    onsets = np.sort(onsets[np.isfinite(onsets)])
    if onsets.size == 0 or not np.isfinite(t_on):
        return float("nan"), float("nan")
    i = int(np.searchsorted(onsets, t_on, side="right"))
    if i >= onsets.size:
        return float("nan"), float("nan")
    t_h = float(onsets[i])
    dt = t_h - float(t_on)
    if dt <= float(window_ms):
        return t_h, float(dt)
    return float("nan"), float("nan")


def collect_secondary_after_saccade(
    tables: EventTables,
    *,
    lookback_ms: float = LOOKBACK_MS,
    post_window_ms: float = POST_WINDOW_MS,
    bout_gap_ms: float = BOUT_GAP_MS,
    reverse_min_deg: float = REVERSE_MIN_DEG,
    amp_percentile: float = AMP_PERCENTILE,
    amp_floor_deg: float = AMP_FLOOR_DEG,
    amp_threshold_deg: float | None = None,
    sync_diff_ms: float | None = None,
    pre_plot_ms: float = PRE_PLOT_MS,
    post_plot_ms: float = POST_PLOT_MS,
    trace_dt_ms: float = TRACE_DT_MS,
    reload_missing_traces: bool = True,
) -> dict[str, Any]:
    """Build unique events, apply still-head + large-amp gates, attach follow-ups."""
    if sync_diff_ms is None:
        sync_diff_ms = _sync_diff_ms(tables)
    if reload_missing_traces:
        missing = any(b.left is None or b.left.empty or b.right is None or b.right.empty for b in tables.blocks)
        if missing:
            from eye_tracking_system_tools.analysis.event_cache import reload_traces

            tables = reload_traces(tables)

    diag_rows: list[dict[str, Any]] = []
    stationary_parts: list[pd.DataFrame] = []
    secondary_rows: list[dict[str, Any]] = []
    block_ctx: dict[str, dict[str, Any]] = {}

    for bundle in tables.blocks:
        spec = bundle.spec
        ev = bundle.all_saccades
        n_saccades = 0 if ev is None or ev.empty else int(len(ev))
        mat = find_lizmov_mat(spec)
        ctx: dict[str, Any] = {
            "spec": spec,
            "left": bundle.left,
            "right": bundle.right,
            "events": ev if ev is not None else pd.DataFrame(),
            "lizmov_path": str(mat) if mat is not None else "",
            "mov_times": np.asarray([], dtype=float),
            "mov_all": np.asarray([], dtype=float),
            "bout_onsets": np.asarray([], dtype=float),
        }
        if mat is None:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "block_path": str(spec.block_path),
                    "lizmov_path": "",
                    "n_eye_saccades": n_saccades,
                    "n_unique_events": 0,
                    "n_stationary_onset": 0,
                    "n_head_onsets": 0,
                    "note": "no lizMov.mat",
                }
            )
            block_ctx[spec.block_key] = ctx
            continue
        try:
            t_raw, mov_raw = load_lizmov_samples(mat)
            mov_times = np.sort(load_lizmov_times_ms(mat))
            mov_times = mov_times[np.isfinite(mov_times)]
            onsets = load_lizmov_bout_onsets_ms(mat, gap_ms=bout_gap_ms)
        except Exception as exc:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "block_path": str(spec.block_path),
                    "lizmov_path": str(mat),
                    "n_eye_saccades": n_saccades,
                    "n_unique_events": 0,
                    "n_stationary_onset": 0,
                    "n_head_onsets": 0,
                    "note": f"lizMov load failed: {exc}",
                }
            )
            block_ctx[spec.block_key] = ctx
            continue
        ctx["mov_times"] = mov_times
        ctx["mov_all"] = np.asarray(mov_raw, dtype=float)
        ctx["mov_t_raw"] = np.asarray(t_raw, dtype=float)
        ctx["bout_onsets"] = np.sort(np.asarray(onsets, dtype=float))
        ctx["bout_onsets"] = ctx["bout_onsets"][np.isfinite(ctx["bout_onsets"])]
        if ev is None or ev.empty:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "block_path": str(spec.block_path),
                    "lizmov_path": str(mat),
                    "n_eye_saccades": 0,
                    "n_unique_events": 0,
                    "n_stationary_onset": 0,
                    "n_head_onsets": int(ctx["bout_onsets"].size),
                    "note": "no saccades",
                }
            )
            block_ctx[spec.block_key] = ctx
            continue
        unique = unique_gaze_events(
            ev,
            sync_diff_ms=float(sync_diff_ms),
            animal=spec.animal,
            block=spec.block_num,
            block_key=spec.block_key,
        )
        unique["block_key"] = spec.block_key
        unique["animal"] = spec.animal
        unique["block"] = spec.block_num
        still = unique["saccade_on_ms"].map(
            lambda t: head_stationary_at_onset(float(t), mov_times, lookback_ms=lookback_ms)
        )
        kept = unique.loc[still].copy()
        stationary_parts.append(kept)
        diag_rows.append(
            {
                "block_key": spec.block_key,
                "animal": spec.animal,
                "block": spec.block_num,
                "block_path": str(spec.block_path),
                "lizmov_path": str(mat),
                "n_eye_saccades": n_saccades,
                "n_unique_events": int(len(unique)),
                "n_stationary_onset": int(len(kept)),
                "n_head_onsets": int(ctx["bout_onsets"].size),
                "note": "",
            }
        )
        block_ctx[spec.block_key] = ctx

    stationary = (
        pd.concat(stationary_parts, ignore_index=True)
        if stationary_parts
        else pd.DataFrame(
            columns=[
                "animal",
                "block",
                "block_key",
                "event_id",
                "saccade_on_ms",
                "saccade_off_ms",
                "amp_deg",
                "overall_angle_deg",
                "primary_eye",
                "concurrency",
                "eyes",
                "constituent_onsets",
            ]
        )
    )
    amps = pd.to_numeric(stationary["amp_deg"], errors="coerce").to_numpy(float) if not stationary.empty else np.asarray([], dtype=float)
    proposed, amp_stats = propose_amplitude_threshold(
        amps, percentile=amp_percentile, floor_deg=amp_floor_deg
    )
    threshold = float(amp_threshold_deg) if amp_threshold_deg is not None else proposed
    amp_stats["threshold_deg"] = float(threshold)
    amp_stats["threshold_source"] = "user" if amp_threshold_deg is not None else "proposed"
    print(
        f"[secondary_after_saccade] still-head unique events={int(len(stationary))}  "
        f"threshold={threshold:.1f}°  ({amp_stats.get('threshold_source')})"
    )

    if stationary.empty:
        qualifying = stationary.copy()
    else:
        qualifying = stationary.loc[pd.to_numeric(stationary["amp_deg"], errors="coerce") >= threshold].copy()
        qualifying = qualifying.reset_index(drop=True)

    grid = np.arange(-float(pre_plot_ms), float(post_plot_ms) + 0.5 * float(trace_dt_ms), float(trace_dt_ms))
    pos_rows: list[np.ndarray] = []
    head_rows: list[np.ndarray] = []
    example_traces: list[dict[str, Any]] = []

    event_rows: list[dict[str, Any]] = []
    for rec in qualifying.itertuples(index=False):
        ctx = block_ctx.get(str(rec.block_key), {})
        ev = ctx.get("events", pd.DataFrame())
        constituents = list(rec.constituent_onsets or [])
        t0 = float(rec.saccade_on_ms)
        t_off = float(rec.saccade_off_ms)
        seconds: list[dict[str, Any]] = []
        if ev is not None and not ev.empty and "saccade_on_ms" in ev.columns:
            on = pd.to_numeric(ev["saccade_on_ms"], errors="coerce")
            in_win = (on > t0) & (on <= t0 + float(post_window_ms))
            cand = ev.loc[in_win]
            for _, srow in cand.iterrows():
                eye = str(srow.get("eye", ""))
                t_s = float(pd.to_numeric(srow.get("saccade_on_ms"), errors="coerce"))
                if _is_constituent(eye, t_s, constituents):
                    continue
                s_amp = _amp_of(srow)
                s_ang = _angle_of(srow)
                dclass = direction_class(float(rec.overall_angle_deg), s_ang, reverse_min_deg=reverse_min_deg)
                item = {
                    "event_id": rec.event_id,
                    "animal": rec.animal,
                    "block": rec.block,
                    "block_key": rec.block_key,
                    "primary_on_ms": t0,
                    "eye": eye,
                    "saccade_on_ms": t_s,
                    "saccade_off_ms": float(pd.to_numeric(srow.get("saccade_off_ms"), errors="coerce")),
                    "amp_deg": s_amp,
                    "overall_angle_deg": s_ang,
                    "latency_ms": float(t_s - t0),
                    "angle_diff_deg": abs(circular_diff_deg(s_ang, float(rec.overall_angle_deg)))
                    if np.isfinite(s_ang) and np.isfinite(float(rec.overall_angle_deg))
                    else float("nan"),
                    "direction_class": dclass,
                    "smaller_than_primary": bool(np.isfinite(s_amp) and np.isfinite(rec.amp_deg) and s_amp < float(rec.amp_deg)),
                }
                seconds.append(item)
                secondary_rows.append(item)
        seconds.sort(key=lambda d: d["latency_ms"])
        n_sec = len(seconds)
        n_same = int(sum(s["direction_class"] == "same" for s in seconds))
        n_rev = int(sum(s["direction_class"] == "reverse" for s in seconds))
        first = seconds[0] if seconds else None
        sec_class = first["direction_class"] if first else "none"
        t_head, dt_head = _first_onset_after(t0, ctx.get("bout_onsets", np.asarray([])), window_ms=post_window_ms)

        left = ctx.get("left")
        right = ctx.get("right")
        primary_df = left if str(rec.primary_eye).upper() != "R" else right
        t_end = t0 + float(post_window_ms)
        phi0 = _interp_col(primary_df, np.array([t0]), "k_phi")[0]
        th0 = _interp_col(primary_df, np.array([t0]), "k_theta")[0]
        phi_end = _interp_col(primary_df, np.array([t_end]), "k_phi")[0]
        th_end = _interp_col(primary_df, np.array([t_end]), "k_theta")[0]
        if np.isfinite(phi0) and np.isfinite(th0) and np.isfinite(float(rec.overall_angle_deg)):
            final_along = float(
                _signed_along_axis(
                    np.array([phi_end]),
                    np.array([th_end]),
                    phi0=phi0,
                    th0=th0,
                    angle_deg=float(rec.overall_angle_deg),
                )[0]
            )
        else:
            final_along = float("nan")
        amp = float(rec.amp_deg) if rec.amp_deg is not None else float("nan")
        remaining_frac = float(final_along / amp) if np.isfinite(final_along) and np.isfinite(amp) and amp != 0 else float("nan")

        t_grid = t0 + grid
        if np.isfinite(phi0) and np.isfinite(th0) and np.isfinite(float(rec.overall_angle_deg)):
            phi_g = _interp_col(primary_df, t_grid, "k_phi")
            th_g = _interp_col(primary_df, t_grid, "k_theta")
            pos = _signed_along_axis(phi_g, th_g, phi0=phi0, th0=th0, angle_deg=float(rec.overall_angle_deg))
        else:
            pos = np.full(grid.shape, np.nan)
        mov_times = ctx.get("mov_times", np.asarray([], dtype=float))
        head_occ = np.zeros(grid.shape, dtype=float)
        if mov_times.size:
            half = 0.5 * float(trace_dt_ms)
            for i, t in enumerate(t_grid):
                j0 = int(np.searchsorted(mov_times, t - half, side="left"))
                j1 = int(np.searchsorted(mov_times, t + half, side="right"))
                head_occ[i] = 1.0 if j1 > j0 else 0.0
        pos_rows.append(pos.astype(np.float32))
        head_rows.append(head_occ.astype(np.float32))

        event_rows.append(
            {
                "event_id": rec.event_id,
                "animal": rec.animal,
                "block": rec.block,
                "block_key": rec.block_key,
                "saccade_on_ms": t0,
                "saccade_off_ms": t_off,
                "amp_deg": amp,
                "overall_angle_deg": float(rec.overall_angle_deg) if rec.overall_angle_deg is not None else float("nan"),
                "primary_eye": rec.primary_eye,
                "concurrency": rec.concurrency,
                "eyes": int(rec.eyes),
                "n_secondary": n_sec,
                "n_same": n_same,
                "n_reverse": n_rev,
                "has_secondary": n_sec > 0,
                "has_multi_secondary": n_sec >= 2,
                "has_both_directions": n_same > 0 and n_rev > 0,
                "secondary_class": sec_class,
                "first_secondary_eye": first["eye"] if first else "",
                "first_secondary_latency_ms": first["latency_ms"] if first else float("nan"),
                "first_secondary_amp_deg": first["amp_deg"] if first else float("nan"),
                "first_secondary_angle_diff_deg": first["angle_diff_deg"] if first else float("nan"),
                "first_secondary_smaller": first["smaller_than_primary"] if first else False,
                "has_subsequent_head": bool(np.isfinite(dt_head)),
                "head_onset_ms": t_head,
                "head_latency_ms": dt_head,
                "final_along_deg": final_along,
                "remaining_frac": remaining_frac,
                "lizmov_path": ctx.get("lizmov_path", ""),
            }
        )

    events_df = pd.DataFrame(event_rows)
    secondary_df = pd.DataFrame(secondary_rows)
    pos_mat = np.vstack(pos_rows) if pos_rows else np.zeros((0, grid.size), dtype=np.float32)
    head_mat = np.vstack(head_rows) if head_rows else np.zeros((0, grid.size), dtype=np.float32)
    print(f"[secondary_after_saccade] qualifying large events={int(len(event_rows))}")

    return {
        "tables": tables,
        "block_ctx": block_ctx,
        "stationary": stationary,
        "qualifying": events_df,
        "secondaries": secondary_df,
        "diag": pd.DataFrame(diag_rows),
        "amp_stats": amp_stats,
        "threshold_deg": float(threshold),
        "grid_ms": grid.astype(np.float64),
        "pos_along": pos_mat,
        "head_occ": head_mat,
        "example_traces": example_traces,
        "params": {
            "lookback_ms": float(lookback_ms),
            "post_window_ms": float(post_window_ms),
            "bout_gap_ms": float(bout_gap_ms),
            "reverse_min_deg": float(reverse_min_deg),
            "amp_percentile": float(amp_percentile),
            "amp_floor_deg": float(amp_floor_deg),
            "sync_diff_ms": float(sync_diff_ms),
            "pre_plot_ms": float(pre_plot_ms),
            "post_plot_ms": float(post_plot_ms),
            "trace_dt_ms": float(trace_dt_ms),
        },
    }


def summarize_secondary(payload: dict[str, Any]) -> dict[str, Any]:
    ev = payload.get("qualifying")
    if ev is None or ev.empty:
        n = 0
        ev = pd.DataFrame()
    else:
        n = int(len(ev))
    stationary = payload.get("stationary")
    n_stat = int(len(stationary)) if stationary is not None else 0
    amp_stats = dict(payload.get("amp_stats") or {})
    n_stat_amp = int(amp_stats.get("n") or n_stat)
    params = dict(payload.get("params") or {})

    def _pct(k: int) -> float:
        return float(100.0 * k / n) if n else float("nan")

    n_sec = int(ev["has_secondary"].sum()) if n and "has_secondary" in ev.columns else 0
    n_same = int((ev["secondary_class"] == "same").sum()) if n else 0
    n_rev = int((ev["secondary_class"] == "reverse").sum()) if n else 0
    n_unk = int((ev["secondary_class"] == "unknown").sum()) if n else 0
    n_multi = int(ev["has_multi_secondary"].sum()) if n and "has_multi_secondary" in ev.columns else 0
    n_both = int(ev["has_both_directions"].sum()) if n and "has_both_directions" in ev.columns else 0
    n_head = int(ev["has_subsequent_head"].sum()) if n and "has_subsequent_head" in ev.columns else 0
    n_head_and_sec = (
        int((ev["has_subsequent_head"] & ev["has_secondary"]).sum()) if n else 0
    )

    def _median(col: str, mask: pd.Series | None = None) -> float:
        if n == 0 or col not in ev.columns:
            return float("nan")
        s = ev[col] if mask is None else ev.loc[mask, col]
        x = pd.to_numeric(s, errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else float("nan")

    remaining = pd.to_numeric(ev["remaining_frac"], errors="coerce").to_numpy(float) if n else np.asarray([])
    remaining = remaining[np.isfinite(remaining)]
    stayed = int(np.sum(remaining >= 0.5)) if remaining.size else 0
    returned = int(np.sum(remaining < 0.5)) if remaining.size else 0

    per_animal: dict[str, Any] = {}
    if n:
        for animal, g in ev.groupby("animal", dropna=False):
            gn = int(len(g))
            per_animal[str(animal)] = {
                "n_qualifying": gn,
                "pct_any_secondary": float(100.0 * g["has_secondary"].mean()) if gn else float("nan"),
                "pct_reverse_first": float(100.0 * (g["secondary_class"] == "reverse").mean()) if gn else float("nan"),
                "pct_same_first": float(100.0 * (g["secondary_class"] == "same").mean()) if gn else float("nan"),
                "pct_subsequent_head": float(100.0 * g["has_subsequent_head"].mean()) if gn else float("nan"),
                "median_remaining_frac": float(np.nanmedian(pd.to_numeric(g["remaining_frac"], errors="coerce"))),
            }

    head_by_class: dict[str, Any] = {}
    for cls in ("none", "same", "reverse"):
        g = ev.loc[ev["secondary_class"] == cls] if n else ev
        gn = int(len(g))
        head_by_class[cls] = {
            "n": gn,
            "n_subsequent_head": int(g["has_subsequent_head"].sum()) if gn else 0,
            "pct_subsequent_head": float(100.0 * g["has_subsequent_head"].mean()) if gn else float("nan"),
        }

    return {
        "question": (
            "Among large saccades that start while the head is still, how often is "
            "a subsequent detected eye movement present, is it same-direction or "
            "counter-directed, and is it associated with later head-bout onset?"
        ),
        "n_stationary_onset": n_stat,
        "n_stationary_with_amp": n_stat_amp,
        "n_qualifying": n,
        "amp_threshold_deg": float(payload.get("threshold_deg", amp_stats.get("threshold_deg", float("nan")))),
        "amp_stats": amp_stats,
        "params": params,
        "n_with_secondary": n_sec,
        "pct_with_secondary": _pct(n_sec),
        "n_first_same": n_same,
        "pct_first_same": _pct(n_same),
        "n_first_reverse": n_rev,
        "pct_first_reverse": _pct(n_rev),
        "n_first_unknown": n_unk,
        "pct_first_unknown": _pct(n_unk),
        "n_no_secondary": n - n_sec,
        "pct_no_secondary": _pct(n - n_sec),
        "n_multi_secondary": n_multi,
        "pct_multi_secondary": _pct(n_multi),
        "n_both_directions": n_both,
        "pct_both_directions": _pct(n_both),
        "n_subsequent_head": n_head,
        "pct_subsequent_head": _pct(n_head),
        "n_secondary_and_head": n_head_and_sec,
        "pct_secondary_given_head": (
            float(100.0 * n_head_and_sec / n_head) if n_head else float("nan")
        ),
        "pct_head_given_secondary": (
            float(100.0 * n_head_and_sec / n_sec) if n_sec else float("nan")
        ),
        "median_first_latency_ms": _median("first_secondary_latency_ms", ev["has_secondary"] if n else None),
        "median_first_secondary_amp_deg": _median("first_secondary_amp_deg", ev["has_secondary"] if n else None),
        "median_head_latency_ms": _median("head_latency_ms", ev["has_subsequent_head"] if n else None),
        "median_remaining_frac": float(np.median(remaining)) if remaining.size else float("nan"),
        "n_remaining_ge_0p5": stayed,
        "pct_remaining_ge_0p5": float(100.0 * stayed / remaining.size) if remaining.size else float("nan"),
        "n_remaining_lt_0p5": returned,
        "pct_remaining_lt_0p5": float(100.0 * returned / remaining.size) if remaining.size else float("nan"),
        "head_by_secondary_class": head_by_class,
        "per_animal": per_animal,
    }


def _event_window_traces(
    payload: dict[str, Any],
    row: pd.Series,
    *,
    pre_ms: float | None = None,
    post_ms: float | None = None,
) -> dict[str, Any]:
    params = payload.get("params") or {}
    pre_ms = float(params.get("pre_plot_ms", PRE_PLOT_MS) if pre_ms is None else pre_ms)
    post_ms = float(params.get("post_plot_ms", POST_PLOT_MS) if post_ms is None else post_ms)
    ctx = payload["block_ctx"].get(str(row["block_key"]), {})
    t0 = float(row["saccade_on_ms"])
    t_abs = np.arange(t0 - pre_ms, t0 + post_ms + 1.0, 2.0)
    t_rel = t_abs - t0
    left, right = ctx.get("left"), ctx.get("right")
    out = {
        "t_rel_ms": t_rel,
        "L_phi": _interp_col(left, t_abs, "k_phi"),
        "L_theta": _interp_col(left, t_abs, "k_theta"),
        "R_phi": _interp_col(right, t_abs, "k_phi"),
        "R_theta": _interp_col(right, t_abs, "k_theta"),
        "head_t_rel": np.asarray([], dtype=float),
        "head_mov": np.asarray([], dtype=float),
    }
    t_raw = ctx.get("mov_t_raw", np.asarray([], dtype=float))
    mov = ctx.get("mov_all", np.asarray([], dtype=float))
    if t_raw.size and mov.size:
        m = (t_raw >= t_abs[0]) & (t_raw <= t_abs[-1])
        out["head_t_rel"] = t_raw[m] - t0
        out["head_mov"] = mov[m] if mov.size == t_raw.size else np.ones(int(m.sum()), dtype=float)
    else:
        times = ctx.get("mov_times", np.asarray([], dtype=float))
        m = (times >= t_abs[0]) & (times <= t_abs[-1])
        out["head_t_rel"] = times[m] - t0
        out["head_mov"] = np.ones(int(m.sum()), dtype=float)
    return out


def _recenter(y: np.ndarray, t_rel: np.ndarray, *, pre_ms: float = 20.0) -> np.ndarray:
    m = (t_rel >= -float(pre_ms)) & (t_rel <= 0) & np.isfinite(y)
    if not np.any(m):
        return y
    return y - float(np.nanmedian(y[m]))


def draw_event_browser_page(
    fig: plt.Figure,
    payload: dict[str, Any],
    row: pd.Series,
    secondaries: pd.DataFrame,
) -> None:
    traces = _event_window_traces(payload, row)
    t = traces["t_rel_ms"]
    axes = fig.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1.15, 1.15, 0.7]})
    t0 = float(row["saccade_on_ms"])
    t_off = float(row["saccade_off_ms"]) - t0 if np.isfinite(row["saccade_off_ms"]) else float("nan")
    for ax, eye, phi, th, color in (
        (axes[0], "L", traces["L_phi"], traces["L_theta"], L_COLOR),
        (axes[1], "R", traces["R_phi"], traces["R_theta"], R_COLOR),
    ):
        ax.plot(t, _recenter(phi, t), color=color, lw=1.15, label=rf"{eye} $\phi$")
        ax.plot(t, _recenter(th, t), color=color, lw=1.15, ls="--", label=rf"{eye} $\theta$")
        ax.axvline(0.0, color=PRIMARY_COLOR, ls="--", lw=0.9)
        if np.isfinite(t_off):
            ax.axvspan(0.0, t_off, color=PRIMARY_COLOR, alpha=0.12, lw=0)
        ax.set_ylabel(f"{eye} angle [deg]")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    axh = axes[2]
    ht, hm = traces["head_t_rel"], traces["head_mov"]
    if ht.size:
        axh.vlines(ht, 0.0, np.clip(hm, 0, None), color=HEAD_COLOR, lw=0.9)
        axh.plot(ht, np.clip(hm, 0, None), color=HEAD_COLOR, ls="none", marker="|", ms=4)
    if bool(row.get("has_subsequent_head")) and np.isfinite(row.get("head_latency_ms", np.nan)):
        axh.axvline(float(row["head_latency_ms"]), color=HEAD_COLOR, ls="-", lw=1.2, label="head bout onset")
    axh.axvline(0.0, color=PRIMARY_COLOR, ls="--", lw=0.9)
    axh.set_ylabel("Head (lizMov)")
    axh.set_xlabel("Time from large-saccade onset [ms]")
    axh.spines["top"].set_visible(False)
    axh.spines["right"].set_visible(False)
    eid = str(row["event_id"])
    if secondaries is not None and not secondaries.empty:
        sub = secondaries.loc[secondaries["event_id"].astype(str) == eid]
        for _, s in sub.iterrows():
            col = CLASS_COLORS.get(str(s["direction_class"]), "k")
            for ax in axes[:2]:
                ax.axvline(float(s["latency_ms"]), color=col, ls=":", lw=0.9)
            axes[0].axvline(float(s["latency_ms"]), color=col, ls=":", lw=0.9)
    post = float((payload.get("params") or {}).get("post_window_ms", POST_WINDOW_MS))
    for ax in axes:
        ax.axvline(post, color="0.75", ls=":", lw=0.7)
        ax.set_xlim(float(t[0]), float(t[-1]))
    title = (
        f"{row['animal']}  block {row['block']}  {row['event_id']}   "
        f"amp={float(row['amp_deg']):.1f}°  class={row['secondary_class']}  "
        f"head={'yes' if row['has_subsequent_head'] else 'no'}  "
        f"t_on={float(row['saccade_on_ms']):.0f} ms"
    )
    axes[0].set_title(title, fontsize=9)


def _stratified_event_ids(events: pd.DataFrame, *, n_per_cell: int, rng: np.random.Generator) -> list[str]:
    if events is None or events.empty:
        return []
    picked: list[str] = []
    for cls in ("reverse", "same", "none"):
        for head in (True, False):
            g = events.loc[(events["secondary_class"] == cls) & (events["has_subsequent_head"] == head)]
            if g.empty:
                continue
            n = min(int(n_per_cell), int(len(g)))
            idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
            picked.extend(g.loc[idx, "event_id"].astype(str).tolist())
    return picked


def figure_amplitude_distribution(
    stationary: pd.DataFrame,
    *,
    threshold_deg: float,
    amp_stats: dict[str, Any],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=150)
    amps = (
        pd.to_numeric(stationary["amp_deg"], errors="coerce").to_numpy(float)
        if stationary is not None and not stationary.empty
        else np.asarray([], dtype=float)
    )
    amps = amps[np.isfinite(amps)]
    xmax = max(float(np.percentile(amps, 99.5)) if amps.size else 10.0, float(threshold_deg) * 1.15, 10.0)
    bins = np.linspace(0.0, xmax, 40)
    ax.hist(amps, bins=bins, color="#0072B2", edgecolor="0.25", linewidth=0.4, alpha=0.85)
    ax.axvline(float(threshold_deg), color="#D55E00", ls="--", lw=1.4, label=f"threshold {threshold_deg:.1f}°")
    p75 = amp_stats.get("p75")
    p50 = amp_stats.get("p50")
    if p50 is not None and np.isfinite(p50):
        ax.axvline(float(p50), color="0.35", ls=":", lw=1.0, label=f"median {float(p50):.1f}°")
    if p75 is not None and np.isfinite(p75):
        ax.axvline(float(p75), color="#009E73", ls=":", lw=1.0, label=f"P75 {float(p75):.1f}°")
    n_large = int(np.sum(amps >= float(threshold_deg))) if amps.size else 0
    ax.set_xlabel("Unique-event amplitude (net angular displacement) [deg]")
    ax.set_ylabel("Count")
    ax.set_title(f"Head-stationary-at-onset amplitudes  n={amps.size}  ≥thr n={n_large}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_secondary_fractions(summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=150)
    labels = ["no follow-up", "first: same dir.", "first: reverse"]
    vals = [
        float(summary.get("pct_no_secondary") or 0.0),
        float(summary.get("pct_first_same") or 0.0),
        float(summary.get("pct_first_reverse") or 0.0),
    ]
    colors = [NONE_COLOR, SAME_COLOR, REVERSE_COLOR]
    ax.bar(np.arange(3), vals, color=colors, edgecolor="0.2", width=0.72)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of large, still-head saccades")
    ax.set_ylim(0, 100)
    n = int(summary.get("n_qualifying") or 0)
    ax.set_title(f"Follow-up detected eye movements  n={n}")
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    extra = (
        f"≥2 follow-ups: {summary.get('pct_multi_secondary', float('nan')):.1f}%\n"
        f"both directions: {summary.get('pct_both_directions', float('nan')):.1f}%"
    )
    ax.text(0.98, 0.98, extra, transform=ax.transAxes, ha="right", va="top", fontsize=8, family="monospace")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_latency_amp(events: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=150)
    if events is None or events.empty:
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        return fig
    sub = events.loc[events["has_secondary"] == True]  # noqa: E712
    for cls, color in (("same", SAME_COLOR), ("reverse", REVERSE_COLOR)):
        g = sub.loc[sub["secondary_class"] == cls]
        lat = pd.to_numeric(g["first_secondary_latency_ms"], errors="coerce").to_numpy(float)
        amp = pd.to_numeric(g["first_secondary_amp_deg"], errors="coerce").to_numpy(float)
        lat = lat[np.isfinite(lat)]
        amp = amp[np.isfinite(amp)]
        if lat.size:
            axes[0].hist(lat, bins=20, range=(0, 250), color=color, alpha=0.55, edgecolor="0.25", label=cls)
        if amp.size:
            axes[1].hist(amp, bins=20, color=color, alpha=0.55, edgecolor="0.25", label=cls)
    axes[0].set_xlabel("First follow-up latency [ms]")
    axes[1].set_xlabel("First follow-up amplitude [deg]")
    for ax in axes:
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title("Latency from large-saccade onset")
    axes[1].set_title("Follow-up amplitude")
    fig.tight_layout()
    return fig


def figure_head_association(summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    by = summary.get("head_by_secondary_class") or {}
    labels = ["no follow-up", "first: same", "first: reverse"]
    keys = ["none", "same", "reverse"]
    vals = [float((by.get(k) or {}).get("pct_subsequent_head") or 0.0) for k in keys]
    ns = [int((by.get(k) or {}).get("n") or 0) for k in keys]
    colors = [NONE_COLOR, SAME_COLOR, REVERSE_COLOR]
    ax.bar(np.arange(3), vals, color=colors, edgecolor="0.2", width=0.72)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("% with head-bout onset in post window")
    ax.set_ylim(0, 100)
    ax.set_title("Subsequent head movement vs follow-up class")
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + 1.5, f"{v:.1f}%\nn={n}", ha="center", va="bottom", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_final_position(events: pd.DataFrame, summary: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=150)
    if events is None or events.empty:
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        return fig
    rem = pd.to_numeric(events["remaining_frac"], errors="coerce").to_numpy(float)
    along = pd.to_numeric(events["final_along_deg"], errors="coerce").to_numpy(float)
    rem = rem[np.isfinite(rem)]
    along = along[np.isfinite(along)]
    if rem.size:
        axes[0].hist(rem, bins=30, range=(-0.5, 1.5), color="#0072B2", edgecolor="0.25", alpha=0.85)
    axes[0].axvline(1.0, color="0.3", ls="--", lw=0.9, label="stay at new pos.")
    axes[0].axvline(0.0, color="#D55E00", ls="--", lw=0.9, label="return to start")
    axes[0].axvline(0.5, color="0.5", ls=":", lw=0.8)
    med = summary.get("median_remaining_frac")
    axes[0].set_xlabel("Final displacement / primary amplitude")
    axes[0].set_ylabel("Count")
    med_s = f"{med:.2f}" if med is not None and np.isfinite(med) else "n/a"
    axes[0].set_title(f"Remaining fraction  median={med_s}")
    axes[0].legend(frameon=False, fontsize=7)
    if along.size:
        lo, hi = np.percentile(along, [1, 99])
        axes[1].hist(along, bins=30, range=(min(lo, -1), max(hi, 1)), color="#0072B2", edgecolor="0.25", alpha=0.85)
    axes[1].axvline(0.0, color="#D55E00", ls="--", lw=0.9)
    axes[1].set_xlabel("Final position along primary axis [deg]")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Signed final displacement vs pre-saccade")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _plot_trace_stack(
    ax: plt.Axes,
    grid: np.ndarray,
    mat: np.ndarray,
    *,
    color: str,
    max_n: int = MAX_POP_TRACES,
    rng: np.random.Generator | None = None,
    ylabel: str = "",
    title: str = "",
) -> None:
    if mat.size == 0:
        ax.set_title(title)
        return
    n = int(mat.shape[0])
    use = np.arange(n)
    if n > max_n:
        rng = rng or np.random.default_rng(0)
        use = rng.choice(n, size=max_n, replace=False)
    segs = []
    for i in use:
        y = mat[i]
        m = np.isfinite(y)
        if int(m.sum()) < 2:
            continue
        segs.append(np.column_stack([grid[m], y[m]]))
    if segs:
        lc = LineCollection(segs, colors=color, linewidths=0.45, alpha=0.22)
        ax.add_collection(lc)
        med = np.nanmedian(mat[use], axis=0)
        ax.plot(grid, med, color="k", lw=1.4, label="median")
    ax.axvline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlim(float(grid[0]), float(grid[-1]))
    if segs:
        ys = np.concatenate([s[:, 1] for s in segs])
        lo, hi = np.nanpercentile(ys, [2, 98])
        pad = 0.1 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{title}  n={n}", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_population_traces(
    payload: dict[str, Any],
    *,
    split: str = "secondary_class",
    rng_seed: int = 0,
) -> plt.Figure:
    events = payload.get("qualifying")
    grid = np.asarray(payload.get("grid_ms"), dtype=float)
    pos = np.asarray(payload.get("pos_along"))
    head = np.asarray(payload.get("head_occ"))
    rng = np.random.default_rng(rng_seed)
    if events is None or events.empty or pos.size == 0:
        fig, ax = plt.subplots(figsize=(4, 2.4), dpi=150)
        ax.set_title("No qualifying events")
        ax.axis("off")
        return fig
    if split == "secondary_class":
        groups = [("none", NONE_COLOR), ("same", SAME_COLOR), ("reverse", REVERSE_COLOR)]
        key = "secondary_class"
    else:
        groups = [(False, NONE_COLOR), (True, HEAD_COLOR)]
        key = "has_subsequent_head"
    n_g = len(groups)
    fig, axes = plt.subplots(2, n_g, figsize=(4.1 * n_g, 5.6), dpi=150, sharex=True, squeeze=False)
    labels = events[key].to_numpy()
    for j, (lab, color) in enumerate(groups):
        mask = labels == lab
        title = {False: "no later head", True: "later head onset"}.get(lab, str(lab))
        _plot_trace_stack(
            axes[0, j],
            grid,
            pos[mask],
            color=color,
            rng=rng,
            ylabel="Pos. along primary axis [deg]" if j == 0 else "",
            title=title,
        )
        _plot_trace_stack(
            axes[1, j],
            grid,
            head[mask],
            color=HEAD_COLOR,
            rng=rng,
            ylabel="Head occupancy" if j == 0 else "",
            title="",
        )
        axes[1, j].set_xlabel("Time from large-saccade onset [ms]")
        axes[1, j].set_ylim(-0.05, 1.15)
    fig.suptitle("Individual peri-saccade traces (median in black)", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def figure_example_gallery(
    payload: dict[str, Any],
    event_ids: list[str],
    *,
    ncols: int = 2,
) -> plt.Figure:
    events = payload.get("qualifying")
    secondaries = payload.get("secondaries")
    if events is None or events.empty or not event_ids:
        fig, ax = plt.subplots(figsize=(4, 2.4), dpi=150)
        ax.set_title("No examples")
        ax.axis("off")
        return fig
    n = len(event_ids)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(7.6 * ncols / 2, 3.4 * nrows), dpi=120)
    outer = fig.add_gridspec(nrows, ncols, hspace=0.38, wspace=0.22)
    for i, eid in enumerate(event_ids):
        sub = events.loc[events["event_id"].astype(str) == str(eid)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        gs = outer[i // ncols, i % ncols].subgridspec(3, 1, height_ratios=[1.1, 1.1, 0.65], hspace=0.08)
        axes = [fig.add_subplot(gs[k]) for k in range(3)]
        traces = _event_window_traces(payload, row)
        t = traces["t_rel_ms"]
        t_off = float(row["saccade_off_ms"]) - float(row["saccade_on_ms"]) if np.isfinite(row["saccade_off_ms"]) else float("nan")
        for ax, eye, phi, th, color in (
            (axes[0], "L", traces["L_phi"], traces["L_theta"], L_COLOR),
            (axes[1], "R", traces["R_phi"], traces["R_theta"], R_COLOR),
        ):
            ax.plot(t, _recenter(phi, t), color=color, lw=0.9)
            ax.plot(t, _recenter(th, t), color=color, lw=0.9, ls="--")
            ax.axvline(0.0, color=PRIMARY_COLOR, ls="--", lw=0.7)
            if np.isfinite(t_off):
                ax.axvspan(0.0, t_off, color=PRIMARY_COLOR, alpha=0.1, lw=0)
            ax.set_ylabel(eye, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ax is not axes[1]:
                ax.set_xticklabels([])
        axh = axes[2]
        ht, hm = traces["head_t_rel"], traces["head_mov"]
        if ht.size:
            axh.vlines(ht, 0.0, np.clip(hm, 0, None), color=HEAD_COLOR, lw=0.7)
        if bool(row.get("has_subsequent_head")) and np.isfinite(row.get("head_latency_ms", np.nan)):
            axh.axvline(float(row["head_latency_ms"]), color=HEAD_COLOR, lw=1.0)
        axh.axvline(0.0, color=PRIMARY_COLOR, ls="--", lw=0.7)
        axh.tick_params(labelsize=6)
        axh.spines["top"].set_visible(False)
        axh.spines["right"].set_visible(False)
        axh.set_xlabel("ms", fontsize=7)
        if secondaries is not None and not secondaries.empty:
            sub_s = secondaries.loc[secondaries["event_id"].astype(str) == str(eid)]
            for _, s in sub_s.iterrows():
                col = CLASS_COLORS.get(str(s["direction_class"]), "k")
                for ax in axes:
                    ax.axvline(float(s["latency_ms"]), color=col, ls=":", lw=0.7)
        axes[0].set_title(
            f"{row['animal']} {row['block']} {float(row['saccade_on_ms']):.0f}ms  "
            f"{float(row['amp_deg']):.1f}° {row['secondary_class']}",
            fontsize=8,
        )
        for ax in axes:
            ax.set_xlim(float(t[0]), float(t[-1]))
    return fig


def write_event_browser_pdf(
    path: Path,
    payload: dict[str, Any],
    event_ids: list[str],
) -> Path:
    events = payload.get("qualifying")
    secondaries = payload.get("secondaries")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        if events is None or events.empty or not event_ids:
            fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=130)
            ax.set_title("No events to browse")
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            return path
        for eid in event_ids:
            sub = events.loc[events["event_id"].astype(str) == str(eid)]
            if sub.empty:
                continue
            fig = plt.figure(figsize=(8.5, 6.5), dpi=130)
            draw_event_browser_page(fig, payload, sub.iloc[0], secondaries)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return path


def captions_markdown(summary: dict[str, Any], *, plot_files: dict[str, str]) -> str:
    n = int(summary.get("n_qualifying") or 0)
    n_stat = int(summary.get("n_stationary_with_amp") or summary.get("n_stationary_onset") or 0)
    thr = summary.get("amp_threshold_deg")
    thr_s = f"{float(thr):.1f}" if thr is not None and np.isfinite(float(thr)) else "n/a"
    lines = [
        "# Captions — secondary eye movements after large still-head saccades",
        "",
        "These panels quantify follow-up *detected* saccades after large unique gaze events ",
        "that start while the head is still. Labels such as nystagmus, VOR, or corrective ",
        "saccade are not applied.",
        "",
        f"**Cohort filter:** unique L/R events (paper pairing window) with no `lizMov` movement ",
        f"sample in the {summary.get('params', {}).get('lookback_ms', LOOKBACK_MS):.0f} ms before onset; ",
        f"amplitude ≥ {thr_s}° (P{summary.get('amp_stats', {}).get('percentile_used', AMP_PERCENTILE):.0f} ",
        f"of {n_stat} still-head events, floored at {summary.get('amp_stats', {}).get('floor_deg', AMP_FLOOR_DEG):.0f}°). ",
        f"Follow-ups are existing detections with onset in (t0, t0+{summary.get('params', {}).get('post_window_ms', POST_WINDOW_MS):.0f} ms], ",
        "excluding the L/R constituents of the primary event. Reverse = circular direction difference > 90°.",
        "",
    ]
    captions = {
        PDF_AMP: (
            f"**{PDF_AMP}.** Amplitude distribution of unique gaze events whose head is still at onset "
            f"(n={n_stat}). The dashed orange line is the large-event threshold ({thr_s}°), set to the "
            "cohort 75th percentile rounded down to 0.5° and not below one main-sequence bin (5°)."
        ),
        PDF_BROWSER: (
            f"**{PDF_BROWSER}.** Multi-page event browser. Each page is one qualifying large saccade. "
            r"Top two traces: left and right eye φ (solid) and θ (dashed), recentered to the pre-onset "
            "median. Shading marks primary on–off. Dotted vertical lines mark subsequent detected saccades "
            "(green = same-direction, magenta = reverse). Bottom: `lizMov` movement samples and the next "
            "head-bout onset (orange). Title carries animal, block, event id, and t_on so the source video "
            "can be recovered."
        ),
        PDF_GALLERY: (
            f"**{PDF_GALLERY}.** Stratified examples of the same peri-event layout as the browser "
            "(reverse / same / none × with / without later head onset)."
        ),
        PDF_FRACTIONS: (
            f"**{PDF_FRACTIONS}.** Fraction of large still-head saccades (n={n}) with no subsequent "
            "detected eye movement, with a first follow-up in the same direction as the primary, or with "
            f"a first follow-up that is counter-directed. Inset: ≥2 follow-ups "
            f"({summary.get('pct_multi_secondary', float('nan')):.1f}%) and both directions in the window "
            f"({summary.get('pct_both_directions', float('nan')):.1f}%)."
        ),
        PDF_LATENCY: (
            f"**{PDF_LATENCY}.** Latency from primary onset and amplitude of the first subsequent detected "
            "saccade, split by same-direction vs reverse."
        ),
        PDF_HEAD: (
            f"**{PDF_HEAD}.** Fraction of events in each follow-up class that have a head-bout onset in the "
            "same post-saccadic window. This is an association, not a mechanistic claim."
        ),
        PDF_FINAL: (
            f"**{PDF_FINAL}.** Final primary-eye position at the end of the post window, projected on the "
            "primary movement axis and divided by primary amplitude (left). 1 = remains at the new position; "
            "0 = back at the pre-saccadic position. Right: signed displacement vs pre-saccade (degrees). "
            f"Median remaining fraction = {summary.get('median_remaining_frac', float('nan')):.2f}; "
            f"{summary.get('pct_remaining_ge_0p5', float('nan')):.1f}% remain ≥ 0.5 of the primary displacement."
        ),
        PDF_POP: (
            f"**{PDF_POP}.** Individual peri-event traces aligned to primary onset (t=0), split by first "
            "follow-up class. Top: signed eye position along the primary axis. Bottom: head-movement occupancy "
            "from `lizMov` samples. Black = median. Traces are not averaged away; at most "
            f"{MAX_POP_TRACES} events per panel are drawn if the class is larger."
        ),
        PDF_POP_HEAD: (
            f"**{PDF_POP_HEAD}.** Same individual-trace layout, split by whether a head-bout onset occurs "
            "in the post-saccadic window."
        ),
    }
    for name in (
        PDF_AMP,
        PDF_BROWSER,
        PDF_GALLERY,
        PDF_FRACTIONS,
        PDF_LATENCY,
        PDF_HEAD,
        PDF_FINAL,
        PDF_POP,
        PDF_POP_HEAD,
    ):
        if name in plot_files or True:
            lines.append(captions[name])
            lines.append("")
    lines.append("## Quantitative summary")
    lines.append("")
    lines.append(f"- Still-head unique events: **{n_stat}**")
    lines.append(f"- Qualifying large events: **{n}** (threshold {thr_s}°)")
    lines.append(
        f"- Any subsequent detected eye movement: **{summary.get('pct_with_secondary', float('nan')):.1f}%** "
        f"({summary.get('n_with_secondary', 0)}/{n})"
    )
    lines.append(
        f"- First follow-up same-direction: **{summary.get('pct_first_same', float('nan')):.1f}%**; "
        f"reverse: **{summary.get('pct_first_reverse', float('nan')):.1f}%**"
        + (
            f"; unclassified direction: **{summary.get('pct_first_unknown', 0):.1f}%**"
            if (summary.get("n_first_unknown") or 0)
            else ""
        )
    )
    lines.append(
        f"- ≥2 follow-ups: **{summary.get('pct_multi_secondary', float('nan')):.1f}%**; "
        f"both directions in window: **{summary.get('pct_both_directions', float('nan')):.1f}%**"
    )
    lines.append(
        f"- Subsequent head-bout onset: **{summary.get('pct_subsequent_head', float('nan')):.1f}%**; "
        f"among events with a follow-up, **{summary.get('pct_head_given_secondary', float('nan')):.1f}%** also have later head onset"
    )
    lines.append(
        f"- Median first-follow-up latency: **{summary.get('median_first_latency_ms', float('nan')):.0f} ms**; "
        f"amplitude **{summary.get('median_first_secondary_amp_deg', float('nan')):.1f}°**"
    )
    lines.append(
        f"- Median remaining fraction of primary displacement at window end: "
        f"**{summary.get('median_remaining_frac', float('nan')):.2f}** "
        f"({summary.get('pct_remaining_ge_0p5', float('nan')):.1f}% stay ≥ 0.5)"
    )
    return "\n".join(lines).rstrip() + "\n"


def _drop_constituents_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "constituent_onsets" in out.columns:
        out["n_constituents"] = out["constituent_onsets"].map(lambda x: len(x) if isinstance(x, list) else 0)
        out = out.drop(columns=["constituent_onsets"])
    return out


def export_secondary_after_saccade(
    tables: EventTables,
    out_dir: Path,
    *,
    lookback_ms: float = LOOKBACK_MS,
    post_window_ms: float = POST_WINDOW_MS,
    bout_gap_ms: float = BOUT_GAP_MS,
    reverse_min_deg: float = REVERSE_MIN_DEG,
    amp_percentile: float = AMP_PERCENTILE,
    amp_floor_deg: float = AMP_FLOOR_DEG,
    amp_threshold_deg: float | None = None,
    show: bool = False,
    rng_seed: int = 0,
    n_browser_per_cell: int = N_BROWSER_PER_CELL,
    plot_id: str = PLOT_ID,
) -> dict[str, Path]:
    params = {
        "lookback_ms": float(lookback_ms),
        "post_window_ms": float(post_window_ms),
        "bout_gap_ms": float(bout_gap_ms),
        "reverse_min_deg": float(reverse_min_deg),
        "amp_percentile": float(amp_percentile),
        "amp_floor_deg": float(amp_floor_deg),
        "amp_threshold_deg": amp_threshold_deg,
        "sync_diff_ms": _sync_diff_ms(tables),
        "rng_seed": int(rng_seed),
    }
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind=KIND,
        tables=tables,
        logic_key=KIND,
        params={**dict(getattr(tables, "params", {}) or {}), "secondary_after_saccade": params},
        extra=params,
    )
    payload = collect_secondary_after_saccade(
        tables,
        lookback_ms=lookback_ms,
        post_window_ms=post_window_ms,
        bout_gap_ms=bout_gap_ms,
        reverse_min_deg=reverse_min_deg,
        amp_percentile=amp_percentile,
        amp_floor_deg=amp_floor_deg,
        amp_threshold_deg=amp_threshold_deg,
    )
    summary = summarize_secondary(payload)
    rng = np.random.default_rng(rng_seed)
    events = payload["qualifying"]
    browser_ids = _stratified_event_ids(events, n_per_cell=n_browser_per_cell, rng=rng)
    gallery_ids = _stratified_event_ids(events, n_per_cell=1, rng=rng)

    _drop_constituents_column(payload["stationary"]).to_csv(bundle.metadata_dir / STATIONARY_CSV, index=False)
    events.to_csv(bundle.metadata_dir / EVENTS_CSV, index=False)
    payload["secondaries"].to_csv(bundle.metadata_dir / SECONDARY_CSV, index=False)
    payload["diag"].to_csv(bundle.metadata_dir / DIAG_CSV, index=False)
    with open(bundle.metadata_dir / SUMMARY_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    pickle_payload = {
        "grid_ms": payload["grid_ms"],
        "pos_along": payload["pos_along"],
        "head_occ": payload["head_occ"],
        "qualifying": events.to_dict(orient="list") if events is not None else {},
        "secondary_class": events["secondary_class"].astype(str).to_numpy() if events is not None and not events.empty else np.array([], dtype=object),
        "has_subsequent_head": events["has_subsequent_head"].to_numpy(bool) if events is not None and not events.empty else np.array([], dtype=bool),
        "amp_stats": payload["amp_stats"],
        "threshold_deg": payload["threshold_deg"],
        "summary": summary,
        "params": payload["params"],
        "stationary_amp_deg": pd.to_numeric(payload["stationary"]["amp_deg"], errors="coerce").to_numpy(float)
        if payload["stationary"] is not None and not payload["stationary"].empty
        else np.asarray([], dtype=float),
        "browser_event_ids": browser_ids,
        "gallery_event_ids": gallery_ids,
        "pdf_amp": PDF_AMP,
        "pdf_fractions": PDF_FRACTIONS,
        "pdf_latency": PDF_LATENCY,
        "pdf_head": PDF_HEAD,
        "pdf_final": PDF_FINAL,
        "pdf_pop": PDF_POP,
        "pdf_pop_head": PDF_POP_HEAD,
    }
    pkl = bundle.metadata_dir / PICKLE_NAME
    write_pickle_with_meta(
        pickle_payload,
        pkl,
        meta={
            "n_qualifying": summary["n_qualifying"],
            "amp_threshold_deg": summary["amp_threshold_deg"],
            "pct_with_secondary": summary["pct_with_secondary"],
            "pct_first_reverse": summary["pct_first_reverse"],
        },
        entrypoint="eye_tracking_system_tools.analysis.secondary_after_saccade.export_secondary_after_saccade",
    )

    written: dict[str, Path] = {
        STATIONARY_CSV: bundle.metadata_dir / STATIONARY_CSV,
        EVENTS_CSV: bundle.metadata_dir / EVENTS_CSV,
        SECONDARY_CSV: bundle.metadata_dir / SECONDARY_CSV,
        DIAG_CSV: bundle.metadata_dir / DIAG_CSV,
        SUMMARY_YAML: bundle.metadata_dir / SUMMARY_YAML,
        PICKLE_NAME: pkl,
    }

    def _save(fig: plt.Figure, name: str) -> None:
        p = bundle.plots_dir / name
        fig.savefig(p, format="pdf", bbox_inches="tight")
        fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
        show_and_close(fig, show)
        written[name] = p

    _save(
        figure_amplitude_distribution(
            payload["stationary"], threshold_deg=payload["threshold_deg"], amp_stats=payload["amp_stats"]
        ),
        PDF_AMP,
    )
    _save(figure_secondary_fractions(summary), PDF_FRACTIONS)
    _save(figure_latency_amp(events), PDF_LATENCY)
    _save(figure_head_association(summary), PDF_HEAD)
    _save(figure_final_position(events, summary), PDF_FINAL)
    _save(figure_population_traces(payload, split="secondary_class", rng_seed=rng_seed), PDF_POP)
    _save(figure_population_traces(payload, split="head", rng_seed=rng_seed), PDF_POP_HEAD)
    _save(figure_example_gallery(payload, gallery_ids, ncols=2), PDF_GALLERY)

    browser_path = bundle.plots_dir / PDF_BROWSER
    write_event_browser_pdf(browser_path, payload, browser_ids)
    written[PDF_BROWSER] = browser_path

    captions = captions_markdown(summary, plot_files={k: str(v) for k, v in written.items()})
    cap_meta = bundle.metadata_dir / CAPTIONS_MD
    cap_root = bundle.bundle_dir / CAPTIONS_MD
    cap_meta.write_text(captions, encoding="utf-8")
    cap_root.write_text(captions, encoding="utf-8")
    written[CAPTIONS_MD] = cap_root

    finish_plot_bundle(bundle)
    written["params.yaml"] = bundle.metadata_dir / "params.yaml"
    written["LOGIC.md"] = bundle.metadata_dir / "LOGIC.md"
    written["replot.py"] = bundle.bundle_dir / "replot.py"
    return written


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Quantify subsequent detected eye movements after large saccades that "
            "start while the head is still (reviewer Q6)."
        )
    )
    parser.add_argument("--registry", type=Path, default=repo / "configs" / "paper_blocks.yaml")
    parser.add_argument("--params", type=Path, default=repo / "configs" / "analysis_params.yaml")
    parser.add_argument(
        "--out",
        type=Path,
        default=repo / "development" / PLOT_ID,
        help="Plot-bundle directory (default: development/Rev_3_Q6_nystagmus_like)",
    )
    parser.add_argument("--lookback-ms", type=float, default=LOOKBACK_MS)
    parser.add_argument("--post-window-ms", type=float, default=POST_WINDOW_MS)
    parser.add_argument("--bout-gap-ms", type=float, default=BOUT_GAP_MS)
    parser.add_argument("--reverse-min-deg", type=float, default=REVERSE_MIN_DEG)
    parser.add_argument("--amp-percentile", type=float, default=AMP_PERCENTILE)
    parser.add_argument("--amp-floor-deg", type=float, default=AMP_FLOOR_DEG)
    parser.add_argument("--amp-threshold-deg", type=float, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args(argv)

    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.event_cache import build_or_load_event_tables
    from eye_tracking_system_tools.analysis.export_meta import load_params_yaml

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = out_dir / "metadata"
    cache_meta.mkdir(parents=True, exist_ok=True)

    params = load_params_yaml(args.params)
    specs = load_registry(args.registry)
    tables, cache_path, from_cache = build_or_load_event_tables(
        specs,
        params,
        cache_meta,
        keep_traces=False,
        prefer_finalized=True,
        force=bool(args.force_cache),
    )
    print(f"registry: {args.registry}")
    print(f"blocks={len(tables.blocks)}  events={len(tables.all_saccades)}  cache={'hit' if from_cache else 'miss'}")
    print(f"cache:    {cache_path}")
    print(f"out:      {out_dir}")

    written = export_secondary_after_saccade(
        tables,
        out_dir,
        lookback_ms=float(args.lookback_ms),
        post_window_ms=float(args.post_window_ms),
        bout_gap_ms=float(args.bout_gap_ms),
        reverse_min_deg=float(args.reverse_min_deg),
        amp_percentile=float(args.amp_percentile),
        amp_floor_deg=float(args.amp_floor_deg),
        amp_threshold_deg=args.amp_threshold_deg,
        show=False,
        rng_seed=int(args.rng_seed),
        plot_id=PLOT_ID,
    )
    summary_path = written.get(SUMMARY_YAML)
    if summary_path is not None and summary_path.is_file():
        summary = yaml.safe_load(summary_path.read_text()) or {}
        print(f"threshold: {summary.get('amp_threshold_deg')} deg")
        print(
            f"n_qualifying={summary.get('n_qualifying')}  "
            f"pct_secondary={summary.get('pct_with_secondary')}  "
            f"pct_reverse={summary.get('pct_first_reverse')}  "
            f"pct_head={summary.get('pct_subsequent_head')}"
        )
        print(f"remaining_frac median={summary.get('median_remaining_frac')}")
    print("\nWrote:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
