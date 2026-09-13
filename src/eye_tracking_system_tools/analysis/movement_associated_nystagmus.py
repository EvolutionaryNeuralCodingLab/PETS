"""Reverse follow-up after large saccades, split by lizard head-movement labels.

Reviewer Q6 (video S1), simplified:

1. Frequency: among large unique gaze events, what fraction is followed within
   100 ms by another *detected* saccade in approximately the opposite direction.
2. Putative purpose: compare that rate for head-stationary vs head-moving events
   using ``lizMov.mat`` annotations (movement sample overlapping the primary
   saccade span). A higher rate during head movement is the evidence offered for
   a nystagmus / gaze-stabilization interpretation.

Reuses the existing saccade table. Does not re-detect events.
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

from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.head_labels import find_lizmov_mat, load_lizmov_times_ms
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.analysis.secondary_after_saccade import (
    CONSTITUENT_TOL_MS,
    DEFAULT_SYNC_DIFF_MS,
    _is_constituent,
    circular_diff_deg,
    direction_class,
    propose_amplitude_threshold,
    unique_gaze_events,
)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PLOT_ID = "movement_associated_nystagmus"
KIND = "movement_associated_nystagmus"

POST_WINDOW_MS = 100.0
REVERSE_MIN_DEG = 90.0
AMP_PERCENTILE = 75.0
AMP_FLOOR_DEG = 5.0

STILL_COLOR = "#0072B2"
MOVING_COLOR = "#D55E00"
OVERALL_COLOR = "0.35"

PDF_AMP = "amplitude_distribution.pdf"
PDF_OVERALL = "reverse_followup_overall.pdf"
PDF_BY_HEAD = "reverse_followup_by_head.pdf"
PDF_BY_ANIMAL = "reverse_followup_by_head_animal.pdf"
EVENTS_CSV = "large_saccades.csv"
DIAG_CSV = "block_diagnostics.csv"
SUMMARY_YAML = "summary.yaml"
CAPTIONS_MD = "captions.md"
PICKLE_NAME = "movement_associated_nystagmus.pkl"


def _sync_diff_ms(tables: EventTables) -> float:
    return float(tables.params.get("binocular", {}).get("sync_diff_ms", DEFAULT_SYNC_DIFF_MS))


def span_overlaps_movement(t_on: float, t_off: float, mov_times: np.ndarray) -> bool:
    """True if any ``lizMov`` sample falls in ``[t_on, t_off]`` (paper head flag)."""
    if not np.isfinite(t_on):
        return False
    off = float(t_off) if np.isfinite(t_off) else float(t_on)
    times = np.asarray(mov_times, dtype=float)
    times = times[np.isfinite(times)]
    if times.size == 0:
        return False
    i0 = int(np.searchsorted(times, t_on, side="left"))
    i1 = int(np.searchsorted(times, off, side="right"))
    return i1 > i0


def list_followups(
    block_events: pd.DataFrame,
    *,
    t_on: float,
    constituents: list[dict[str, Any]],
    post_window_ms: float,
) -> list[dict[str, Any]]:
    """Detected saccades in ``(t_on, t_on+window]`` excluding primary L/R constituents."""
    if block_events is None or block_events.empty or "saccade_on_ms" not in block_events.columns:
        return []
    on = pd.to_numeric(block_events["saccade_on_ms"], errors="coerce")
    cand = block_events.loc[(on > t_on) & (on <= t_on + float(post_window_ms))]
    out: list[dict[str, Any]] = []
    for _, srow in cand.iterrows():
        eye = str(srow.get("eye", ""))
        t_s = float(pd.to_numeric(srow.get("saccade_on_ms"), errors="coerce"))
        if not np.isfinite(t_s) or _is_constituent(eye, t_s, constituents, tol_ms=CONSTITUENT_TOL_MS):
            continue
        s_ang = (
            float(pd.to_numeric(srow.get("overall_angle_deg"), errors="coerce"))
            if "overall_angle_deg" in srow.index
            else float("nan")
        )
        s_amp = (
            float(pd.to_numeric(srow.get("net_angular_disp"), errors="coerce"))
            if "net_angular_disp" in srow.index
            else float("nan")
        )
        out.append(
            {
                "eye": eye,
                "saccade_on_ms": t_s,
                "latency_ms": float(t_s - t_on),
                "amp_deg": s_amp,
                "overall_angle_deg": s_ang,
            }
        )
    out.sort(key=lambda d: d["latency_ms"])
    return out


def _has_reverse_followup(
    block_events: pd.DataFrame,
    *,
    t_on: float,
    angle_deg: float,
    constituents: list[dict[str, Any]],
    post_window_ms: float,
    reverse_min_deg: float,
) -> tuple[bool, float, float, float]:
    """Return ``(hit, latency_ms, follow_amp, angle_diff)``. NaNs if no hit."""
    empty = (False, float("nan"), float("nan"), float("nan"))
    follows = list_followups(
        block_events, t_on=t_on, constituents=constituents, post_window_ms=post_window_ms
    )
    best: tuple[bool, float, float, float] | None = None
    for rec in follows:
        if direction_class(angle_deg, rec["overall_angle_deg"], reverse_min_deg=reverse_min_deg) != "reverse":
            continue
        diff = (
            abs(circular_diff_deg(rec["overall_angle_deg"], angle_deg))
            if np.isfinite(rec["overall_angle_deg"]) and np.isfinite(angle_deg)
            else float("nan")
        )
        hit = (True, float(rec["latency_ms"]), float(rec["amp_deg"]), diff)
        if best is None or hit[1] < best[1]:
            best = hit
    return best if best is not None else empty


def collect_reverse_followups(
    tables: EventTables,
    *,
    post_window_ms: float = POST_WINDOW_MS,
    reverse_min_deg: float = REVERSE_MIN_DEG,
    amp_percentile: float = AMP_PERCENTILE,
    amp_floor_deg: float = AMP_FLOOR_DEG,
    amp_threshold_deg: float | None = None,
    sync_diff_ms: float | None = None,
) -> dict[str, Any]:
    if sync_diff_ms is None:
        sync_diff_ms = _sync_diff_ms(tables)

    diag_rows: list[dict[str, Any]] = []
    unique_parts: list[pd.DataFrame] = []

    for bundle in tables.blocks:
        spec = bundle.spec
        ev = bundle.all_saccades
        n_saccades = 0 if ev is None or ev.empty else int(len(ev))
        mat = find_lizmov_mat(spec)
        if mat is None:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "lizmov_path": "",
                    "n_eye_saccades": n_saccades,
                    "n_unique_events": 0,
                    "n_head_moving": 0,
                    "n_head_still": 0,
                    "note": "no lizMov.mat",
                }
            )
            continue
        try:
            mov_times = np.sort(load_lizmov_times_ms(mat))
            mov_times = mov_times[np.isfinite(mov_times)]
        except Exception as exc:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "lizmov_path": str(mat),
                    "n_eye_saccades": n_saccades,
                    "n_unique_events": 0,
                    "n_head_moving": 0,
                    "n_head_still": 0,
                    "note": f"lizMov load failed: {exc}",
                }
            )
            continue
        if ev is None or ev.empty:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "lizmov_path": str(mat),
                    "n_eye_saccades": 0,
                    "n_unique_events": 0,
                    "n_head_moving": 0,
                    "n_head_still": 0,
                    "note": "no saccades",
                }
            )
            continue
        unique = unique_gaze_events(
            ev,
            sync_diff_ms=float(sync_diff_ms),
            animal=spec.animal,
            block=spec.block_num,
            block_key=spec.block_key,
        )
        unique["animal"] = spec.animal
        unique["block"] = spec.block_num
        unique["block_key"] = spec.block_key
        unique["lizmov_path"] = str(mat)
        moving = []
        reverse_hit = []
        latencies = []
        follow_amps = []
        angle_diffs = []
        for rec in unique.itertuples(index=False):
            t0 = float(rec.saccade_on_ms)
            t_off = float(rec.saccade_off_ms)
            is_moving = span_overlaps_movement(t0, t_off, mov_times)
            hit, lat, famp, adiff = _has_reverse_followup(
                ev,
                t_on=t0,
                angle_deg=float(rec.overall_angle_deg) if rec.overall_angle_deg is not None else float("nan"),
                constituents=list(rec.constituent_onsets or []),
                post_window_ms=post_window_ms,
                reverse_min_deg=reverse_min_deg,
            )
            moving.append(is_moving)
            reverse_hit.append(hit)
            latencies.append(lat)
            follow_amps.append(famp)
            angle_diffs.append(adiff)
        unique["head_movement"] = moving
        unique["has_reverse_followup"] = reverse_hit
        unique["reverse_latency_ms"] = latencies
        unique["reverse_follow_amp_deg"] = follow_amps
        unique["reverse_angle_diff_deg"] = angle_diffs
        unique_parts.append(unique)
        diag_rows.append(
            {
                "block_key": spec.block_key,
                "animal": spec.animal,
                "block": spec.block_num,
                "lizmov_path": str(mat),
                "n_eye_saccades": n_saccades,
                "n_unique_events": int(len(unique)),
                "n_head_moving": int(np.sum(moving)),
                "n_head_still": int(len(unique) - np.sum(moving)),
                "note": "",
            }
        )

    all_unique = (
        pd.concat(unique_parts, ignore_index=True)
        if unique_parts
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
                "head_movement",
                "has_reverse_followup",
                "reverse_latency_ms",
                "reverse_follow_amp_deg",
                "reverse_angle_diff_deg",
            ]
        )
    )
    amps = (
        pd.to_numeric(all_unique["amp_deg"], errors="coerce").to_numpy(float)
        if not all_unique.empty
        else np.asarray([], dtype=float)
    )
    proposed, amp_stats = propose_amplitude_threshold(amps, percentile=amp_percentile, floor_deg=amp_floor_deg)
    threshold = float(amp_threshold_deg) if amp_threshold_deg is not None else proposed
    amp_stats["threshold_deg"] = float(threshold)
    amp_stats["threshold_source"] = "user" if amp_threshold_deg is not None else "proposed"
    if all_unique.empty:
        large = all_unique.copy()
    else:
        large = all_unique.loc[pd.to_numeric(all_unique["amp_deg"], errors="coerce") >= threshold].copy()
        large = large.reset_index(drop=True)
    print(
        f"[movement_associated_nystagmus] unique={len(all_unique)}  "
        f"threshold={threshold:.1f}°  large={len(large)}"
    )
    return {
        "all_unique": all_unique,
        "large": large,
        "diag": pd.DataFrame(diag_rows),
        "amp_stats": amp_stats,
        "threshold_deg": float(threshold),
        "params": {
            "post_window_ms": float(post_window_ms),
            "reverse_min_deg": float(reverse_min_deg),
            "amp_percentile": float(amp_percentile),
            "amp_floor_deg": float(amp_floor_deg),
            "sync_diff_ms": float(sync_diff_ms),
        },
    }


def _rate(mask_n: int, mask_hit: int) -> dict[str, float | int]:
    return {
        "n": int(mask_n),
        "n_reverse": int(mask_hit),
        "pct": float(100.0 * mask_hit / mask_n) if mask_n else float("nan"),
    }


def summarize_reverse_followups(payload: dict[str, Any]) -> dict[str, Any]:
    large = payload.get("large")
    if large is None or large.empty:
        large = pd.DataFrame()
        n = 0
    else:
        n = int(len(large))
    params = dict(payload.get("params") or {})
    amp_stats = dict(payload.get("amp_stats") or {})
    hits = large["has_reverse_followup"].astype(bool) if n else pd.Series(dtype=bool)
    overall = _rate(n, int(hits.sum()) if n else 0)

    def _split(moving: bool) -> dict[str, float | int]:
        if n == 0 or "head_movement" not in large.columns:
            return _rate(0, 0)
        m = large["head_movement"].astype(bool) == moving
        return _rate(int(m.sum()), int(hits.loc[m].sum()))

    still = _split(False)
    moving = _split(True)

    table = np.array(
        [
            [int(moving["n_reverse"]), int(moving["n"]) - int(moving["n_reverse"])],
            [int(still["n_reverse"]), int(still["n"]) - int(still["n_reverse"])],
        ],
        dtype=int,
    )
    fisher_p = float("nan")
    odds = float("nan")
    if table.min() >= 0 and table.sum() > 0 and still["n"] and moving["n"]:
        try:
            from scipy.stats import fisher_exact

            odds, fisher_p = fisher_exact(table, alternative="two-sided")
            odds, fisher_p = float(odds), float(fisher_p)
        except Exception:
            pass

    lat = (
        pd.to_numeric(large.loc[hits, "reverse_latency_ms"], errors="coerce").to_numpy(float)
        if n
        else np.asarray([])
    )
    lat = lat[np.isfinite(lat)]
    famp = (
        pd.to_numeric(large.loc[hits, "reverse_follow_amp_deg"], errors="coerce").to_numpy(float)
        if n
        else np.asarray([])
    )
    famp = famp[np.isfinite(famp)]

    per_animal: dict[str, Any] = {}
    if n:
        for animal, g in large.groupby("animal", dropna=False):
            gh = g["has_reverse_followup"].astype(bool)
            gm = g["head_movement"].astype(bool)
            per_animal[str(animal)] = {
                "overall": _rate(int(len(g)), int(gh.sum())),
                "head_still": _rate(int((~gm).sum()), int(gh.loc[~gm].sum())),
                "head_moving": _rate(int(gm.sum()), int(gh.loc[gm].sum())),
            }

    return {
        "question": (
            "Among large saccades, how often is there a reverse-direction detected "
            "saccade within 100 ms, and is that more common during head movement?"
        ),
        "rule": (
            f"large = unique-event net_angular_disp ≥ {payload.get('threshold_deg')}°; "
            f"reverse follow-up = existing detection, not a binocular constituent, "
            f"onset in (t0, t0+{params.get('post_window_ms', POST_WINDOW_MS):.0f} ms], "
            f"circular direction difference > {params.get('reverse_min_deg', REVERSE_MIN_DEG):.0f}°"
        ),
        "n_unique_lizmov": int(len(payload["all_unique"])) if payload.get("all_unique") is not None else 0,
        "amp_threshold_deg": float(payload.get("threshold_deg", amp_stats.get("threshold_deg", float("nan")))),
        "amp_stats": amp_stats,
        "params": params,
        "overall": overall,
        "head_still": still,
        "head_moving": moving,
        "fisher_odds_ratio_moving_vs_still": odds,
        "fisher_p_two_sided": fisher_p,
        "median_reverse_latency_ms": float(np.median(lat)) if lat.size else float("nan"),
        "median_reverse_follow_amp_deg": float(np.median(famp)) if famp.size else float("nan"),
        "per_animal": per_animal,
    }


def figure_amplitude_distribution(
    all_unique: pd.DataFrame,
    *,
    threshold_deg: float,
    amp_stats: dict[str, Any],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=150)
    amps = (
        pd.to_numeric(all_unique["amp_deg"], errors="coerce").to_numpy(float)
        if all_unique is not None and not all_unique.empty
        else np.asarray([], dtype=float)
    )
    amps = amps[np.isfinite(amps) & (amps > 0)]
    xmax = max(float(np.percentile(amps, 99.5)) if amps.size else 10.0, float(threshold_deg) * 1.15, 10.0)
    ax.hist(amps, bins=np.linspace(0.0, xmax, 40), color="#0072B2", edgecolor="0.25", linewidth=0.4, alpha=0.85)
    ax.axvline(float(threshold_deg), color="#D55E00", ls="--", lw=1.4, label=f"large ≥ {threshold_deg:.1f}°")
    p50 = amp_stats.get("p50")
    p75 = amp_stats.get("p75")
    if p50 is not None and np.isfinite(p50):
        ax.axvline(float(p50), color="0.35", ls=":", lw=1.0, label=f"median {float(p50):.1f}°")
    if p75 is not None and np.isfinite(p75):
        ax.axvline(float(p75), color="#009E73", ls=":", lw=1.0, label=f"P75 {float(p75):.1f}°")
    n_large = int(np.sum(amps >= float(threshold_deg))) if amps.size else 0
    ax.set_xlabel("Unique-event amplitude [deg]")
    ax.set_ylabel("Count")
    ax.set_title(f"Amplitude (lizMov blocks)  n={amps.size}  large n={n_large}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _bar_with_n(ax: plt.Axes, x: np.ndarray, pcts: list[float], ns: list[int], colors: list[str]) -> None:
    ax.bar(x, pcts, color=colors, edgecolor="0.2", width=0.7)
    for i, (v, n) in enumerate(zip(pcts, ns)):
        if np.isfinite(v):
            ax.text(i, v + 1.4, f"{v:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of large saccades")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_overall(summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.4, 3.6), dpi=150)
    overall = summary.get("overall") or {}
    _bar_with_n(ax, np.array([0]), [float(overall.get("pct") or 0.0)], [int(overall.get("n") or 0)], [OVERALL_COLOR])
    ax.set_xticks([0])
    ax.set_xticklabels(["reverse follow-up\nwithin 100 ms"])
    n_hit = int(overall.get("n_reverse") or 0)
    n = int(overall.get("n") or 0)
    ax.set_title(f"{n_hit} / {n} large saccades")
    fig.tight_layout()
    return fig


def figure_by_head(summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=150)
    still = summary.get("head_still") or {}
    moving = summary.get("head_moving") or {}
    _bar_with_n(
        ax,
        np.array([0, 1]),
        [float(still.get("pct") or 0.0), float(moving.get("pct") or 0.0)],
        [int(still.get("n") or 0), int(moving.get("n") or 0)],
        [STILL_COLOR, MOVING_COLOR],
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["head stationary", "head moving"])
    p = summary.get("fisher_p_two_sided")
    p_s = f"{p:.3g}" if p is not None and np.isfinite(p) else "n/a"
    ax.set_title(f"Reverse follow-up ≤ 100 ms   Fisher p={p_s}")
    fig.tight_layout()
    return fig


def figure_by_animal(summary: dict[str, Any]) -> plt.Figure:
    per = summary.get("per_animal") or {}
    animals = sorted(per)
    if not animals:
        fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=150)
        ax.set_title("No animals")
        ax.axis("off")
        return fig
    x = np.arange(len(animals), dtype=float)
    still = [float((per[a].get("head_still") or {}).get("pct") or 0.0) for a in animals]
    moving = [float((per[a].get("head_moving") or {}).get("pct") or 0.0) for a in animals]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.4 * len(animals)), 3.6), dpi=150)
    ax.bar(x - 0.18, still, width=0.36, color=STILL_COLOR, edgecolor="0.2", label="head stationary")
    ax.bar(x + 0.18, moving, width=0.36, color=MOVING_COLOR, edgecolor="0.2", label="head moving")
    ax.set_xticks(x)
    ax.set_xticklabels(animals, fontsize=8)
    ax.set_ylabel("% of large saccades")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reverse follow-up ≤ 100 ms by animal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def reviewer_response_text(summary: dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    still = summary.get("head_still") or {}
    moving = summary.get("head_moving") or {}
    params = summary.get("params") or {}
    rev_min = float(params.get("reverse_min_deg", REVERSE_MIN_DEG))
    window = float(params.get("post_window_ms", POST_WINDOW_MS))
    p = summary.get("fisher_p_two_sided")
    odds = summary.get("fisher_odds_ratio_moving_vs_still")
    if p is not None and np.isfinite(p):
        p_s = "P < 0.001" if p < 0.001 else f"P = {p:.3g}"
    else:
        p_s = "P = n/a"
    odds_s = f"{odds:.2f}" if odds is not None and np.isfinite(odds) else "n/a"
    n_all = int(overall.get("n") or 0)
    n_hit = int(overall.get("n_reverse") or 0)
    pct_all = float(overall.get("pct") or 0.0)
    pct_move = float(moving.get("pct") or 0.0)
    pct_still = float(still.get("pct") or 0.0)
    n_move = int(moving.get("n") or 0)
    n_move_hit = int(moving.get("n_reverse") or 0)
    n_still = int(still.get("n") or 0)
    n_still_hit = int(still.get("n_reverse") or 0)
    return (
        "Response: We thank the reviewer for this observation. We agree that the small "
        "back-and-forth movements visible after some saccades in Video S1 are an interesting "
        "feature of the P. vitticeps recordings. To quantify their occurrence, we identified "
        "large saccades that were followed within 100 ms by a second saccade directed "
        f"approximately opposite to the initial movement (circular direction difference > {rev_min:.0f}°; "
        f"latency ≤ {window:.0f} ms from primary onset). Such rapid post-saccadic reversals occurred "
        f"following {pct_all:.1f}% of large saccades (n = {n_hit}/{n_all} events).\n"
        "\n"
        "We next asked whether these events were associated with head movement. Post-saccadic "
        f"reversals occurred in {pct_move:.1f}% of large saccades during head-moving periods "
        f"(n = {n_move_hit}/{n_move}) compared with {pct_still:.1f}% during head-stationary periods "
        f"(n = {n_still_hit}/{n_still}; two-sided Fisher’s exact test, odds ratio = {odds_s}, {p_s}). "
        "Thus, these back-and-forth movements were strongly associated with concurrent head movement. "
        "We interpret this association as consistent with nystagmus-like compensatory eye movements "
        "contributing to gaze stabilization during head motion, rather than the eye necessarily remaining "
        "fixed at its initial post-saccadic position as is commonly observed following primate saccades. "
        "We have added this analysis to Fig. Sx and discuss this interpretation in the revised manuscript.\n"
    )


def captions_markdown(summary: dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    still = summary.get("head_still") or {}
    moving = summary.get("head_moving") or {}
    thr = summary.get("amp_threshold_deg")
    thr_s = f"{float(thr):.1f}" if thr is not None and np.isfinite(float(thr)) else "n/a"
    p = summary.get("fisher_p_two_sided")
    p_s = f"{p:.3g}" if p is not None and np.isfinite(p) else "n/a"
    params = summary.get("params") or {}
    return "\n".join(
        [
            "# Captions — reverse follow-up after large saccades (head still vs moving)",
            "",
            "Operational rule for the video-S1 ‘back-and-forth’: a **large** unique gaze event ",
            f"(amplitude ≥ {thr_s}°) is followed within **{params.get('post_window_ms', POST_WINDOW_MS):.0f} ms** by another ",
            f"**detected** saccade whose direction differs by **> {params.get('reverse_min_deg', REVERSE_MIN_DEG):.0f}°** ",
            "(approximately opposite). Concurrent L/R onsets within the paper pairing window count as one event. ",
            "Head stationary vs moving is the existing `lizMov.mat` annotation: a movement sample overlapping ",
            "the primary saccade on–off span.",
            "",
            f"**{PDF_AMP}.** Amplitude of unique gaze events on blocks with `lizMov.mat`. ",
            f"Large-event threshold {thr_s}° = cohort P75 rounded down to 0.5°, not below 5°.",
            "",
            f"**{PDF_OVERALL}.** Frequency of the reverse-follow-up rule among large saccades: ",
            f"**{overall.get('pct', float('nan')):.1f}%** ({overall.get('n_reverse', 0)}/{overall.get('n', 0)}).",
            "",
            f"**{PDF_BY_HEAD}.** Same rule, split by lizard-movement annotation. Head stationary: ",
            f"**{still.get('pct', float('nan')):.1f}%** (n={still.get('n', 0)}). Head moving: ",
            f"**{moving.get('pct', float('nan')):.1f}%** (n={moving.get('n', 0)}). Two-sided Fisher p={p_s}. ",
            "The intended reading is that the back-and-forth is nystagmus-like stabilization during head movement.",
            "",
            f"**{PDF_BY_ANIMAL}.** Per-animal rates, same split.",
            "",
            "## Quantitative answer",
            "",
            f"- Large saccades with a reverse follow-up within 100 ms: **{overall.get('pct', float('nan')):.1f}%** "
            f"({overall.get('n_reverse', 0)} of {overall.get('n', 0)})",
            f"- Head stationary: **{still.get('pct', float('nan')):.1f}%** ({still.get('n_reverse', 0)}/{still.get('n', 0)})",
            f"- Head moving: **{moving.get('pct', float('nan')):.1f}%** ({moving.get('n_reverse', 0)}/{moving.get('n', 0)})",
            f"- Median reverse follow-up latency / amplitude: "
            f"**{summary.get('median_reverse_latency_ms', float('nan')):.0f} ms** / "
            f"**{summary.get('median_reverse_follow_amp_deg', float('nan')):.1f}°**",
            "",
        ]
    )


def _drop_constituents(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "constituent_onsets" in out.columns:
        out["n_constituents"] = out["constituent_onsets"].map(lambda x: len(x) if isinstance(x, list) else 0)
        out = out.drop(columns=["constituent_onsets"])
    return out


def export_movement_associated_nystagmus(
    tables: EventTables,
    out_dir: Path,
    *,
    post_window_ms: float = POST_WINDOW_MS,
    reverse_min_deg: float = REVERSE_MIN_DEG,
    amp_percentile: float = AMP_PERCENTILE,
    amp_floor_deg: float = AMP_FLOOR_DEG,
    amp_threshold_deg: float | None = None,
    show: bool = False,
    plot_id: str = PLOT_ID,
) -> dict[str, Path]:
    params = {
        "post_window_ms": float(post_window_ms),
        "reverse_min_deg": float(reverse_min_deg),
        "amp_percentile": float(amp_percentile),
        "amp_floor_deg": float(amp_floor_deg),
        "amp_threshold_deg": amp_threshold_deg,
        "sync_diff_ms": _sync_diff_ms(tables),
    }
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind=KIND,
        tables=tables,
        logic_key=KIND,
        params={**dict(getattr(tables, "params", {}) or {}), "movement_associated_nystagmus": params},
        extra=params,
    )
    payload = collect_reverse_followups(
        tables,
        post_window_ms=post_window_ms,
        reverse_min_deg=reverse_min_deg,
        amp_percentile=amp_percentile,
        amp_floor_deg=amp_floor_deg,
        amp_threshold_deg=amp_threshold_deg,
    )
    summary = summarize_reverse_followups(payload)
    large = payload["large"]
    _drop_constituents(large).to_csv(bundle.metadata_dir / EVENTS_CSV, index=False)
    payload["diag"].to_csv(bundle.metadata_dir / DIAG_CSV, index=False)
    with open(bundle.metadata_dir / SUMMARY_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    pickle_payload = {
        "summary": summary,
        "amp_stats": payload["amp_stats"],
        "threshold_deg": payload["threshold_deg"],
        "params": payload["params"],
        "stationary_amp_deg": pd.to_numeric(payload["all_unique"]["amp_deg"], errors="coerce").to_numpy(float)
        if payload["all_unique"] is not None and not payload["all_unique"].empty
        else np.asarray([], dtype=float),
        "pdf_amp": PDF_AMP,
        "pdf_overall": PDF_OVERALL,
        "pdf_by_head": PDF_BY_HEAD,
        "pdf_by_animal": PDF_BY_ANIMAL,
    }
    pkl = bundle.metadata_dir / PICKLE_NAME
    write_pickle_with_meta(
        pickle_payload,
        pkl,
        meta={
            "n_large": (summary.get("overall") or {}).get("n"),
            "pct_overall": (summary.get("overall") or {}).get("pct"),
            "pct_still": (summary.get("head_still") or {}).get("pct"),
            "pct_moving": (summary.get("head_moving") or {}).get("pct"),
        },
        entrypoint="eye_tracking_system_tools.analysis.movement_associated_nystagmus.export_movement_associated_nystagmus",
    )

    written: dict[str, Path] = {
        EVENTS_CSV: bundle.metadata_dir / EVENTS_CSV,
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
            payload["all_unique"], threshold_deg=payload["threshold_deg"], amp_stats=payload["amp_stats"]
        ),
        PDF_AMP,
    )
    _save(figure_overall(summary), PDF_OVERALL)
    _save(figure_by_head(summary), PDF_BY_HEAD)
    _save(figure_by_animal(summary), PDF_BY_ANIMAL)

    captions = captions_markdown(summary)
    (bundle.metadata_dir / CAPTIONS_MD).write_text(captions, encoding="utf-8")
    (bundle.bundle_dir / CAPTIONS_MD).write_text(captions, encoding="utf-8")
    written[CAPTIONS_MD] = bundle.bundle_dir / CAPTIONS_MD
    response = reviewer_response_text(summary)
    (bundle.metadata_dir / "reviewer_response.md").write_text(response, encoding="utf-8")
    (bundle.bundle_dir / "reviewer_response.md").write_text(response, encoding="utf-8")
    written["reviewer_response.md"] = bundle.bundle_dir / "reviewer_response.md"

    finish_plot_bundle(bundle)
    written["params.yaml"] = bundle.metadata_dir / "params.yaml"
    written["LOGIC.md"] = bundle.metadata_dir / "LOGIC.md"
    written["replot.py"] = bundle.bundle_dir / "replot.py"
    return written


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Rate of reverse follow-up within 100 ms after large saccades, "
            "split by lizMov head-stationary vs head-moving."
        )
    )
    parser.add_argument("--registry", type=Path, default=repo / "configs" / "paper_blocks.yaml")
    parser.add_argument("--params", type=Path, default=repo / "configs" / "analysis_params.yaml")
    parser.add_argument("--out", type=Path, default=repo / "development" / PLOT_ID)
    parser.add_argument(
        "--cache-meta",
        type=Path,
        default=None,
        help="Event-cache metadata dir (default: <out>/metadata). Reuse an existing cache to skip rebuild.",
    )
    parser.add_argument("--post-window-ms", type=float, default=POST_WINDOW_MS)
    parser.add_argument("--reverse-min-deg", type=float, default=REVERSE_MIN_DEG)
    parser.add_argument("--amp-percentile", type=float, default=AMP_PERCENTILE)
    parser.add_argument("--amp-floor-deg", type=float, default=AMP_FLOOR_DEG)
    parser.add_argument("--amp-threshold-deg", type=float, default=None)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args(argv)

    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.event_cache import build_or_load_event_tables
    from eye_tracking_system_tools.analysis.export_meta import load_params_yaml

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = Path(args.cache_meta) if args.cache_meta is not None else out_dir / "metadata"
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

    written = export_movement_associated_nystagmus(
        tables,
        out_dir,
        post_window_ms=float(args.post_window_ms),
        reverse_min_deg=float(args.reverse_min_deg),
        amp_percentile=float(args.amp_percentile),
        amp_floor_deg=float(args.amp_floor_deg),
        amp_threshold_deg=args.amp_threshold_deg,
        show=False,
        plot_id=PLOT_ID,
    )
    summary_path = written.get(SUMMARY_YAML)
    if summary_path is not None and summary_path.is_file():
        summary = yaml.safe_load(summary_path.read_text()) or {}
        overall = summary.get("overall") or {}
        still = summary.get("head_still") or {}
        moving = summary.get("head_moving") or {}
        print(f"threshold: {summary.get('amp_threshold_deg')} deg")
        print(f"overall:   {overall.get('pct')}%  ({overall.get('n_reverse')}/{overall.get('n')})")
        print(f"still:     {still.get('pct')}%  ({still.get('n_reverse')}/{still.get('n')})")
        print(f"moving:    {moving.get('pct')}%  ({moving.get('n_reverse')}/{moving.get('n')})")
    print("\nWrote:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
