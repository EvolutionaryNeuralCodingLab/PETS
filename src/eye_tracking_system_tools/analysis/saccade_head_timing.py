"""Timing of head-bout onsets relative to unique saccade events.

Question
--------
When head motion is available (``lizMov.mat``), do saccades start before or
after the head begins to move?

Method
------
1. Keep only blocks that have Mark's ``lizMov.mat``.  ``t_mov_ms`` are sparse
   accel-envelope threshold crossings (~4 ms).  Head bouts are runs of those
   samples split wherever the gap exceeds ``bout_gap_ms`` (default 40 ms);
   each bout's onset is its first sample.
2. Collapse concurrent left/right saccades within the binocular pairing window
   into one gaze event (onset = earlier eye) so a binocular pair is not counted
   twice against the same head bout.
3. For each unique saccade, take the nearest head-bout onset.  The signed lag
   is ``dt = t_head_onset − t_saccade_onset``:
     * ``dt < 0`` — head started first (head leads)
     * ``dt > 0`` — saccade started first (saccade leads)
4. The onset histogram uses events whose nearest onset falls inside ``±window_ms``.
   A companion occupancy PETH shows the fraction of saccades with a head-movement
   sample in each time bin.

This is *not* the same as the review-answer peri-head plot, which only keeps
saccades whose duration overlaps a movement sample (that overlap constraint
biases lags toward zero).
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
from eye_tracking_system_tools.analysis.head_labels import (
    find_lizmov_mat,
    load_lizmov_bout_onsets_ms,
    load_lizmov_times_ms,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

PLOT_ID = "saccade_head_mov_temporal_dist"
KIND = "saccade_head_timing"

WINDOW_MS = 500.0
BIN_MS = 10.0
# One video frame at 60 Hz (1000/60 ≈ 16.67 ms), rounded up to 17 ms.
SIMULTANEOUS_MS = 17.0
BOUT_GAP_MS = 40.0
DEFAULT_SYNC_DIFF_MS = 34.0

HEAD_LEAD_COLOR = "#0072B2"
SACC_LEAD_COLOR = "#D55E00"
SIMUL_COLOR = "0.55"

PDF_POOLED = "head_onset_relative_to_saccade.pdf"
PDF_BY_ANIMAL = "head_onset_relative_to_saccade_by_animal.pdf"
PDF_LEAD_LAG = "lead_lag_summary.pdf"
PDF_PETH = "head_motion_peth.pdf"
OFFSETS_CSV = "saccade_head_offsets.csv"
DIAG_CSV = "block_diagnostics.csv"
SUMMARY_YAML = "summary.yaml"
PICKLE_NAME = "saccade_head_timing.pkl"


def _sync_diff_ms(tables: EventTables) -> float:
    return float(tables.params.get("binocular", {}).get("sync_diff_ms", DEFAULT_SYNC_DIFF_MS))


def unique_saccade_onsets(
    events: pd.DataFrame,
    *,
    sync_diff_ms: float = DEFAULT_SYNC_DIFF_MS,
    animal: str | None = None,
    block: str | None = None,
) -> pd.DataFrame:
    """One row per unique gaze event after merging concurrent L/R onsets."""
    cols = ["animal", "block", "saccade_on_ms", "concurrency", "eyes"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    work = events.copy()
    if "saccade_on_ms" not in work.columns:
        return pd.DataFrame(columns=cols)
    if "animal" not in work.columns:
        work["animal"] = animal
    if "block" not in work.columns:
        work["block"] = block
    work = work.dropna(subset=["saccade_on_ms"])
    if work.empty:
        return pd.DataFrame(columns=cols)

    parts: list[pd.DataFrame] = []
    group_cols = [c for c in ("animal", "block") if c in work.columns]
    grouped = work.groupby(group_cols, dropna=False) if group_cols else [(("", ""), work)]
    for key, g in grouped:
        if group_cols == ["animal", "block"]:
            animal_i, block_i = key
        elif group_cols == ["animal"]:
            animal_i, block_i = key, block
        elif group_cols == ["block"]:
            animal_i, block_i = animal, key
        else:
            animal_i, block_i = animal, block
        if "eye" not in g.columns:
            on = pd.to_numeric(g["saccade_on_ms"], errors="coerce").to_numpy(float)
            on = on[np.isfinite(on)]
            parts.append(
                pd.DataFrame(
                    {
                        "animal": animal_i,
                        "block": block_i,
                        "saccade_on_ms": on,
                        "concurrency": "unknown",
                        "eyes": 1,
                    }
                )
            )
            continue
        left = np.sort(
            pd.to_numeric(g.loc[g["eye"].astype(str) == "L", "saccade_on_ms"], errors="coerce")
            .to_numpy(float)
        )
        right = np.sort(
            pd.to_numeric(g.loc[g["eye"].astype(str) == "R", "saccade_on_ms"], errors="coerce")
            .to_numpy(float)
        )
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        i = j = 0
        rows: list[dict[str, Any]] = []
        while i < left.size and j < right.size:
            dt = left[i] - right[j]
            if abs(dt) <= sync_diff_ms:
                rows.append(
                    {
                        "animal": animal_i,
                        "block": block_i,
                        "saccade_on_ms": float(min(left[i], right[j])),
                        "concurrency": "binocular",
                        "eyes": 2,
                    }
                )
                i += 1
                j += 1
            elif left[i] < right[j]:
                rows.append(
                    {
                        "animal": animal_i,
                        "block": block_i,
                        "saccade_on_ms": float(left[i]),
                        "concurrency": "monocular",
                        "eyes": 1,
                    }
                )
                i += 1
            else:
                rows.append(
                    {
                        "animal": animal_i,
                        "block": block_i,
                        "saccade_on_ms": float(right[j]),
                        "concurrency": "monocular",
                        "eyes": 1,
                    }
                )
                j += 1
        while i < left.size:
            rows.append(
                {
                    "animal": animal_i,
                    "block": block_i,
                    "saccade_on_ms": float(left[i]),
                    "concurrency": "monocular",
                    "eyes": 1,
                }
            )
            i += 1
        while j < right.size:
            rows.append(
                {
                    "animal": animal_i,
                    "block": block_i,
                    "saccade_on_ms": float(right[j]),
                    "concurrency": "monocular",
                    "eyes": 1,
                }
            )
            j += 1
        other = g.loc[~g["eye"].astype(str).isin(["L", "R"])]
        if not other.empty:
            on = pd.to_numeric(other["saccade_on_ms"], errors="coerce").to_numpy(float)
            for t in on[np.isfinite(on)]:
                rows.append(
                    {
                        "animal": animal_i,
                        "block": block_i,
                        "saccade_on_ms": float(t),
                        "concurrency": "unknown",
                        "eyes": 1,
                    }
                )
        if rows:
            parts.append(pd.DataFrame(rows))
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def nearest_onset(t: float, onsets: np.ndarray) -> tuple[float, float]:
    """Return ``(nearest_onset, onset - t)``.  NaNs if ``onsets`` is empty."""
    onsets = np.asarray(onsets, dtype=float)
    onsets = onsets[np.isfinite(onsets)]
    if onsets.size == 0 or not np.isfinite(t):
        return float("nan"), float("nan")
    i = int(np.searchsorted(onsets, t))
    cands: list[float] = []
    if i < onsets.size:
        cands.append(float(onsets[i]))
    if i > 0:
        cands.append(float(onsets[i - 1]))
    nearest = min(cands, key=lambda c: abs(c - t))
    return nearest, float(nearest - t)


def collect_saccade_head_timing(
    tables: EventTables,
    *,
    window_ms: float = WINDOW_MS,
    bin_ms: float = BIN_MS,
    bout_gap_ms: float = BOUT_GAP_MS,
    sync_diff_ms: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Per-saccade nearest head-bout lags, diagnostics, and occupancy PETH.

    ``dt_ms`` is head-onset minus saccade-onset.  Rows with no ``lizMov.mat``
    are omitted from the event table and recorded in diagnostics only.
    Head bouts are gap-clustered ``t_mov_ms`` samples (see ``head_labels``).
    """
    if sync_diff_ms is None:
        sync_diff_ms = _sync_diff_ms(tables)
    edges = _bin_edges(window_ms, bin_ms)
    occ = np.zeros(edges.size - 1, dtype=np.int64)
    n_peth = 0
    event_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for bundle in tables.blocks:
        spec = bundle.spec
        mat = find_lizmov_mat(spec)
        ev = bundle.all_saccades
        n_saccades = 0 if ev is None or ev.empty else int(len(ev))
        if mat is None:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "block_path": str(spec.block_path),
                    "lizmov_path": "",
                    "n_head_onsets": 0,
                    "n_eye_saccades": n_saccades,
                    "n_unique_events": 0,
                    "n_in_window": 0,
                    "note": "no lizMov.mat",
                }
            )
            continue
        try:
            times = np.sort(load_lizmov_times_ms(mat))
            times = times[np.isfinite(times)]
            onsets = load_lizmov_bout_onsets_ms(mat, gap_ms=bout_gap_ms)
        except Exception as exc:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "block_path": str(spec.block_path),
                    "lizmov_path": str(mat),
                    "n_head_onsets": 0,
                    "n_eye_saccades": n_saccades,
                    "n_unique_events": 0,
                    "n_in_window": 0,
                    "note": f"lizMov load failed: {exc}",
                }
            )
            continue
        onsets = np.sort(np.asarray(onsets, dtype=float))
        onsets = onsets[np.isfinite(onsets)]
        if ev is None or ev.empty:
            diag_rows.append(
                {
                    "block_key": spec.block_key,
                    "animal": spec.animal,
                    "block": spec.block_num,
                    "block_path": str(spec.block_path),
                    "lizmov_path": str(mat),
                    "n_head_onsets": int(onsets.size),
                    "n_eye_saccades": 0,
                    "n_unique_events": 0,
                    "n_in_window": 0,
                    "note": "no saccades",
                }
            )
            continue
        unique = unique_saccade_onsets(
            ev,
            sync_diff_ms=float(sync_diff_ms),
            animal=spec.animal,
            block=spec.block_num,
        )
        n_in_window = 0
        for rec in unique.itertuples(index=False):
            t = float(rec.saccade_on_ms)
            nearest, dt = nearest_onset(t, onsets)
            in_window = bool(np.isfinite(dt) and abs(dt) <= window_ms)
            if in_window:
                n_in_window += 1
            if np.isfinite(t) and times.size:
                i0 = int(np.searchsorted(times, t - window_ms, side="left"))
                i1 = int(np.searchsorted(times, t + window_ms, side="right"))
                if i1 > i0:
                    counts, _ = np.histogram(times[i0:i1] - t, bins=edges)
                    occ += (counts > 0).astype(np.int64)
                n_peth += 1
            event_rows.append(
                {
                    "animal": rec.animal,
                    "block": rec.block,
                    "block_key": spec.block_key,
                    "saccade_on_ms": t,
                    "concurrency": rec.concurrency,
                    "eyes": int(rec.eyes),
                    "nearest_head_onset_ms": nearest,
                    "dt_ms": dt,
                    "in_window": in_window,
                    "lizmov_path": str(mat),
                }
            )
        note = ""
        if onsets.size == 0:
            note = "no head onsets"
        diag_rows.append(
            {
                "block_key": spec.block_key,
                "animal": spec.animal,
                "block": spec.block_num,
                "block_path": str(spec.block_path),
                "lizmov_path": str(mat),
                "n_head_onsets": int(onsets.size),
                "n_eye_saccades": n_saccades,
                "n_unique_events": int(len(unique)),
                "n_in_window": n_in_window,
                "note": note,
            }
        )
    events_df = pd.DataFrame(event_rows)
    if events_df.empty:
        events_df = pd.DataFrame(
            columns=[
                "animal",
                "block",
                "block_key",
                "saccade_on_ms",
                "concurrency",
                "eyes",
                "nearest_head_onset_ms",
                "dt_ms",
                "in_window",
                "lizmov_path",
            ]
        )
    occupancy = (occ.astype(float) / n_peth) if n_peth else np.zeros_like(occ, dtype=float)
    peth = {
        "edges_ms": edges,
        "occupancy": occupancy,
        "n_saccades": int(n_peth),
    }
    return events_df, pd.DataFrame(diag_rows), peth


def in_window_offsets(events: pd.DataFrame) -> np.ndarray:
    if events is None or events.empty or "dt_ms" not in events.columns:
        return np.asarray([], dtype=float)
    mask = events["in_window"].astype(bool) if "in_window" in events.columns else np.ones(len(events), dtype=bool)
    dt = pd.to_numeric(events.loc[mask, "dt_ms"], errors="coerce").to_numpy(float)
    return dt[np.isfinite(dt)]


def summarize_timing(
    events: pd.DataFrame,
    diag: pd.DataFrame,
    *,
    window_ms: float = WINDOW_MS,
    bin_ms: float = BIN_MS,
    simultaneous_ms: float = SIMULTANEOUS_MS,
) -> dict[str, Any]:
    dt = in_window_offsets(events)
    n = int(dt.size)
    n_unique = int(len(events)) if events is not None else 0
    n_blocks_with_lizmov = int((diag["lizmov_path"].astype(str) != "").sum()) if not diag.empty else 0
    n_blocks = int(len(diag))
    n_head_leads = int(np.sum(dt < -simultaneous_ms)) if n else 0
    n_sacc_leads = int(np.sum(dt > simultaneous_ms)) if n else 0
    n_simul = int(n - n_head_leads - n_sacc_leads) if n else 0
    median = float(np.median(dt)) if n else float("nan")
    mean = float(np.mean(dt)) if n else float("nan")
    pct = lambda k: (100.0 * k / n) if n else float("nan")
    if n:
        if median < -simultaneous_ms:
            answer = "Head movement typically starts before the saccade."
        elif median > simultaneous_ms:
            answer = "Saccades typically start before the head movement."
        else:
            answer = "Head-bout onset and saccade onset are typically near-simultaneous."
        if n_head_leads > n_sacc_leads:
            majority = "head leads"
        elif n_sacc_leads > n_head_leads:
            majority = "saccade leads"
        else:
            majority = "tie"
    else:
        answer = "No saccades with a head-bout onset inside the analysis window."
        majority = "n/a"
    per_animal: dict[str, Any] = {}
    if events is not None and not events.empty:
        for animal, g in events.groupby("animal", dropna=False):
            a_dt = in_window_offsets(g)
            a_n = int(a_dt.size)
            per_animal[str(animal)] = {
                "n_unique_events": int(len(g)),
                "n_in_window": a_n,
                "median_dt_ms": float(np.median(a_dt)) if a_n else float("nan"),
                "pct_head_leads": (100.0 * float(np.sum(a_dt < -simultaneous_ms)) / a_n) if a_n else float("nan"),
                "pct_saccade_leads": (100.0 * float(np.sum(a_dt > simultaneous_ms)) / a_n) if a_n else float("nan"),
            }
    return {
        "question": "Do saccades occur before or after the head movement?",
        "answer": answer,
        "majority": majority,
        "sign_convention": "dt_ms = t_head_onset - t_saccade_onset; negative = head leads",
        "window_ms": float(window_ms),
        "bin_ms": float(bin_ms),
        "simultaneous_ms": float(simultaneous_ms),
        "n_blocks": n_blocks,
        "n_blocks_with_lizmov": n_blocks_with_lizmov,
        "n_unique_saccade_events": n_unique,
        "n_in_window": n,
        "pct_saccades_with_nearby_head": (100.0 * n / n_unique) if n_unique else float("nan"),
        "median_dt_ms": median,
        "mean_dt_ms": mean,
        "n_head_leads": n_head_leads,
        "n_saccade_leads": n_sacc_leads,
        "n_simultaneous": n_simul,
        "pct_head_leads": pct(n_head_leads),
        "pct_saccade_leads": pct(n_sacc_leads),
        "pct_simultaneous": pct(n_simul),
        "per_animal": per_animal,
    }


def _bin_edges(window_ms: float, bin_ms: float) -> np.ndarray:
    w = float(window_ms)
    b = float(bin_ms)
    n = int(np.round(2.0 * w / b))
    return np.linspace(-w, w, n + 1)


def _bar_colors(edges: np.ndarray) -> list[str]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors: list[str] = []
    for c in centers:
        if c < -1e-12:
            colors.append(HEAD_LEAD_COLOR)
        elif c > 1e-12:
            colors.append(SACC_LEAD_COLOR)
        else:
            colors.append(SIMUL_COLOR)
    return colors


def draw_timing_histogram(
    ax: plt.Axes,
    offsets: np.ndarray,
    *,
    window_ms: float = WINDOW_MS,
    bin_ms: float = BIN_MS,
    annotate: bool = True,
    summary: dict[str, Any] | None = None,
) -> None:
    offsets = np.asarray(offsets, dtype=float)
    offsets = offsets[np.isfinite(offsets)]
    edges = _bin_edges(window_ms, bin_ms)
    counts, _ = np.histogram(offsets, bins=edges)
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color=_bar_colors(edges),
        edgecolor="0.25",
        linewidth=0.3,
    )
    ax.axvline(0.0, color="k", lw=0.9, ls="--")
    ax.set_xlim(-float(window_ms), float(window_ms))
    ax.set_xlabel("Head onset − saccade onset [ms]")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymin, ymax = ax.get_ylim()
    ytxt = ymax * 0.04 if ymax > 0 else 0.04
    ax.text(-0.98 * float(window_ms), ytxt, "← head leads", ha="left", va="bottom", fontsize=8, color=HEAD_LEAD_COLOR)
    ax.text(0.98 * float(window_ms), ytxt, "saccade leads →", ha="right", va="bottom", fontsize=8, color=SACC_LEAD_COLOR)
    if annotate and summary:
        n = int(summary.get("n_in_window") or 0)
        med = summary.get("median_dt_ms")
        med_s = f"{med:.1f}" if med is not None and np.isfinite(med) else "n/a"
        txt = (
            f"n = {n}\n"
            f"median = {med_s} ms\n"
            f"head leads {summary.get('pct_head_leads', float('nan')):.1f}%\n"
            f"saccade leads {summary.get('pct_saccade_leads', float('nan')):.1f}%\n"
            f"|dt| ≤ {summary.get('simultaneous_ms', SIMULTANEOUS_MS):.0f} ms: "
            f"{summary.get('pct_simultaneous', float('nan')):.1f}%"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
        )


def figure_pooled_histogram(
    offsets: np.ndarray,
    summary: dict[str, Any],
    *,
    window_ms: float = WINDOW_MS,
    bin_ms: float = BIN_MS,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=150)
    draw_timing_histogram(
        ax,
        offsets,
        window_ms=window_ms,
        bin_ms=bin_ms,
        annotate=True,
        summary=summary,
    )
    ax.set_title("Head-bout onset relative to saccade onset")
    fig.tight_layout()
    return fig


def figure_by_animal(
    events: pd.DataFrame,
    *,
    window_ms: float = WINDOW_MS,
    bin_ms: float = BIN_MS,
) -> plt.Figure:
    animals = []
    if events is not None and not events.empty:
        animals = [str(a) for a in events["animal"].dropna().unique().tolist()]
        animals = sorted(animals)
    if not animals:
        fig, ax = plt.subplots(figsize=(4.0, 2.4), dpi=150)
        ax.set_title("No animals with lizMov.mat")
        ax.axis("off")
        fig.tight_layout()
        return fig
    n = len(animals)
    ncols = 3 if n > 2 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 2.6 * nrows),
        dpi=150,
        sharex=True,
        squeeze=False,
    )
    for i, animal in enumerate(animals):
        ax = axes[i // ncols][i % ncols]
        g = events.loc[events["animal"].astype(str) == animal]
        dt = in_window_offsets(g)
        draw_timing_histogram(ax, dt, window_ms=window_ms, bin_ms=bin_ms, annotate=False)
        med = float(np.median(dt)) if dt.size else float("nan")
        med_s = f"{med:.0f} ms" if np.isfinite(med) else "n/a"
        ax.set_title(f"{animal}  n={dt.size}  median={med_s}", fontsize=9)
        if i // ncols < nrows - 1:
            ax.set_xlabel("")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("Per-animal head onset relative to saccade onset", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def figure_lead_lag(summary: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=150)
    head = float(summary.get("pct_head_leads") or 0.0)
    sacc = float(summary.get("pct_saccade_leads") or 0.0)
    simul = float(summary.get("pct_simultaneous") or 0.0)
    vals = [head, sacc, simul]
    labels = ["head leads", "saccade leads", "simultaneous"]
    colors = [HEAD_LEAD_COLOR, SACC_LEAD_COLOR, SIMUL_COLOR]
    ax.bar([0, 1, 2], vals, color=colors, width=0.7, edgecolor="0.2")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% of paired events")
    ax.set_ylim(0, 100)
    n = int(summary.get("n_in_window") or 0)
    ax.set_title(f"Who starts first?  n={n}", fontsize=10)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def figure_peth(peth: dict[str, Any]) -> plt.Figure:
    edges = np.asarray(peth.get("edges_ms", []), dtype=float)
    occ = np.asarray(peth.get("occupancy", []), dtype=float)
    n = int(peth.get("n_saccades") or 0)
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=150)
    if edges.size >= 2 and occ.size == edges.size - 1:
        ax.bar(
            edges[:-1],
            100.0 * occ,
            width=np.diff(edges),
            align="edge",
            color=HEAD_LEAD_COLOR,
            edgecolor="0.25",
            linewidth=0.3,
        )
    ax.axvline(0.0, color="k", lw=0.9, ls="--")
    if edges.size >= 2:
        ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel("Time relative to saccade onset [ms]")
    ax.set_ylabel("% of saccades with head motion")
    ax.set_title(f"Head-movement occupancy around saccades  n={n}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymin, ymax = ax.get_ylim()
    ytxt = ymax * 0.04 if ymax > 0 else 0.04
    lo = float(edges[0]) if edges.size else -500.0
    hi = float(edges[-1]) if edges.size else 500.0
    ax.text(lo * 0.98, ytxt, "← before saccade", ha="left", va="bottom", fontsize=8, color=HEAD_LEAD_COLOR)
    ax.text(hi * 0.98, ytxt, "after saccade →", ha="right", va="bottom", fontsize=8, color=SACC_LEAD_COLOR)
    fig.tight_layout()
    return fig


def export_saccade_head_timing(
    tables: EventTables,
    out_dir: Path,
    *,
    window_ms: float = WINDOW_MS,
    bin_ms: float = BIN_MS,
    simultaneous_ms: float = SIMULTANEOUS_MS,
    bout_gap_ms: float = BOUT_GAP_MS,
    show: bool = False,
    plot_id: str = PLOT_ID,
) -> dict[str, Path]:
    params = {
        "window_ms": float(window_ms),
        "bin_ms": float(bin_ms),
        "simultaneous_ms": float(simultaneous_ms),
        "bout_gap_ms": float(bout_gap_ms),
        "sync_diff_ms": _sync_diff_ms(tables),
        "sign_convention": "dt_ms = t_head_onset - t_saccade_onset; negative = head leads",
    }
    bundle = begin_plot_bundle(
        out_dir,
        plot_id,
        kind=KIND,
        tables=tables,
        logic_key=KIND,
        params={**dict(getattr(tables, "params", {}) or {}), "saccade_head_timing": params},
        extra=params,
    )
    events, diag, peth = collect_saccade_head_timing(
        tables,
        window_ms=window_ms,
        bin_ms=bin_ms,
        bout_gap_ms=bout_gap_ms,
    )
    summary = summarize_timing(
        events,
        diag,
        window_ms=window_ms,
        bin_ms=bin_ms,
        simultaneous_ms=simultaneous_ms,
    )
    offsets = in_window_offsets(events)

    events.to_csv(bundle.metadata_dir / OFFSETS_CSV, index=False)
    diag.to_csv(bundle.metadata_dir / DIAG_CSV, index=False)
    with open(bundle.metadata_dir / SUMMARY_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    per_animal_offsets = {
        str(animal): in_window_offsets(g).astype(np.float64)
        for animal, g in (events.groupby("animal", dropna=False) if not events.empty else [])
    }
    payload = {
        "offsets_ms": offsets.astype(np.float64),
        "per_animal": per_animal_offsets,
        "window_ms": float(window_ms),
        "bin_ms": float(bin_ms),
        "simultaneous_ms": float(simultaneous_ms),
        "bout_gap_ms": float(bout_gap_ms),
        "summary": summary,
        "peth_edges_ms": np.asarray(peth["edges_ms"], dtype=np.float64),
        "peth_occupancy": np.asarray(peth["occupancy"], dtype=np.float64),
        "peth_n_saccades": int(peth["n_saccades"]),
        "pdf_pooled": PDF_POOLED,
        "pdf_by_animal": PDF_BY_ANIMAL,
        "pdf_lead_lag": PDF_LEAD_LAG,
        "pdf_peth": PDF_PETH,
    }
    pkl = bundle.metadata_dir / PICKLE_NAME
    write_pickle_with_meta(
        payload,
        pkl,
        meta={
            "n_in_window": summary["n_in_window"],
            "median_dt_ms": summary["median_dt_ms"],
            "majority": summary["majority"],
            "answer": summary["answer"],
        },
        entrypoint="eye_tracking_system_tools.analysis.saccade_head_timing.export_saccade_head_timing",
    )

    written: dict[str, Path] = {
        OFFSETS_CSV: bundle.metadata_dir / OFFSETS_CSV,
        DIAG_CSV: bundle.metadata_dir / DIAG_CSV,
        SUMMARY_YAML: bundle.metadata_dir / SUMMARY_YAML,
        PICKLE_NAME: pkl,
    }
    fig = figure_pooled_histogram(offsets, summary, window_ms=window_ms, bin_ms=bin_ms)
    p = bundle.plots_dir / PDF_POOLED
    fig.savefig(p, format="pdf", bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    show_and_close(fig, show)
    written[PDF_POOLED] = p

    fig = figure_by_animal(events, window_ms=window_ms, bin_ms=bin_ms)
    p = bundle.plots_dir / PDF_BY_ANIMAL
    fig.savefig(p, format="pdf", bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    show_and_close(fig, show)
    written[PDF_BY_ANIMAL] = p

    fig = figure_lead_lag(summary)
    p = bundle.plots_dir / PDF_LEAD_LAG
    fig.savefig(p, format="pdf", bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    show_and_close(fig, show)
    written[PDF_LEAD_LAG] = p

    fig = figure_peth(peth)
    p = bundle.plots_dir / PDF_PETH
    fig.savefig(p, format="pdf", bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    show_and_close(fig, show)
    written[PDF_PETH] = p

    finish_plot_bundle(bundle)
    written["params.yaml"] = bundle.metadata_dir / "params.yaml"
    written["LOGIC.md"] = bundle.metadata_dir / "LOGIC.md"
    written["replot.py"] = bundle.bundle_dir / "replot.py"
    return written


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Histogram of head-bout onset relative to unique saccade events "
            "(blocks with lizMov.mat only)."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo / "configs" / "paper_blocks.yaml",
        help="YAML registry of animal → block paths",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=repo / "configs" / "analysis_params.yaml",
        help="YAML analysis parameters",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo / "outputs",
        help="Parent of the run folder (default: outputs/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Explicit run directory (overrides --out-root / --tag)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional snapshot suffix: saccade_head_mov_temporal_dist_<tag>",
    )
    parser.add_argument("--window-ms", type=float, default=WINDOW_MS)
    parser.add_argument("--bin-ms", type=float, default=BIN_MS)
    parser.add_argument("--simultaneous-ms", type=float, default=SIMULTANEOUS_MS)
    parser.add_argument("--bout-gap-ms", type=float, default=BOUT_GAP_MS)
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Rebuild the event cache even if one exists",
    )
    args = parser.parse_args(argv)

    from eye_tracking_system_tools.analysis.block_registry import load_registry
    from eye_tracking_system_tools.analysis.event_cache import build_or_load_event_tables
    from eye_tracking_system_tools.analysis.export_meta import load_params_yaml

    if args.out is not None:
        out_dir = Path(args.out)
    else:
        name = PLOT_ID if not str(args.tag).strip() else f"{PLOT_ID}_{str(args.tag).strip()}"
        out_dir = Path(args.out_root) / name
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

    written = export_saccade_head_timing(
        tables,
        out_dir,
        window_ms=float(args.window_ms),
        bin_ms=float(args.bin_ms),
        simultaneous_ms=float(args.simultaneous_ms),
        bout_gap_ms=float(args.bout_gap_ms),
        show=False,
        plot_id=PLOT_ID,
    )
    summary_path = written.get(SUMMARY_YAML)
    if summary_path is not None and summary_path.is_file():
        summary = yaml.safe_load(summary_path.read_text()) or {}
        print(f"answer:   {summary.get('answer')}")
        print(
            f"n_in_window={summary.get('n_in_window')}  "
            f"median={summary.get('median_dt_ms')} ms  "
            f"head_leads={summary.get('pct_head_leads')}%  "
            f"saccade_leads={summary.get('pct_saccade_leads')}%"
        )
    print("\nWrote:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
