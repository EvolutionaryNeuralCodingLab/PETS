"""Corrective mono→contra partners and peri-head-onset timing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.binocular import find_synced_saccades_ms
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.head_labels import (
    find_lizmov_mat,
    label_saccades_head_movement,
    load_lizmov_bout_onsets_ms,
)
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

CORRECTIVE_WINDOW_MS = 80.0
CORRECTIVE_ANGLE_DEG = 45.0
HEAD_WINDOW_MS = 250.0
HEAD_BIN_MS = 10.0


def _angle_diff_deg(a: float, b: float) -> float:
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def corrective_partners(
    tables: EventTables,
    *,
    window_ms: float = CORRECTIVE_WINDOW_MS,
    angle_deg: float = CORRECTIVE_ANGLE_DEG,
    sync_diff_ms: float | None = None,
) -> pd.DataFrame:
    """One row per monocular event; ``has_corrective`` if a same-direction contra partner exists."""
    if sync_diff_ms is None:
        sync_diff_ms = float(tables.params.get("binocular", {}).get("sync_diff_ms", 34.0))
    _synced, mono = find_synced_saccades_ms(
        tables.all_saccades, sync_diff_ms=float(sync_diff_ms)
    )
    if mono is None or mono.empty:
        return pd.DataFrame(
            columns=["animal", "block", "eye", "saccade_on_ms", "has_corrective", "latency_ms"]
        )

    rows = []
    for (animal, block), g in mono.groupby(["animal", "block"], dropna=False):
        g = g.sort_values("saccade_on_ms")
        on = g["saccade_on_ms"].to_numpy(float)
        eyes = g["eye"].astype(str).to_numpy()
        angles = (
            g["overall_angle_deg"].to_numpy(float)
            if "overall_angle_deg" in g.columns
            else np.full(len(g), np.nan)
        )
        for i in range(len(g)):
            t0 = on[i]
            eye0 = eyes[i]
            ang0 = angles[i]
            hit = False
            lat = np.nan
            for j in range(i + 1, len(g)):
                dt = on[j] - t0
                if dt > window_ms:
                    break
                if dt <= 0:
                    continue
                if eyes[j] == eye0:
                    continue
                if np.isfinite(ang0) and np.isfinite(angles[j]):
                    if _angle_diff_deg(ang0, angles[j]) > angle_deg:
                        continue
                hit = True
                lat = float(dt)
                break
            rec = g.iloc[i]
            rows.append(
                {
                    "animal": animal,
                    "block": block,
                    "eye": eye0,
                    "saccade_on_ms": t0,
                    "has_corrective": hit,
                    "latency_ms": lat,
                }
            )
    return pd.DataFrame(rows)


def head_peri_saccade_offsets(
    tables: EventTables,
    *,
    window_ms: float = HEAD_WINDOW_MS,
) -> tuple[np.ndarray, pd.DataFrame]:
    offsets: list[float] = []
    diag_rows: list[dict] = []
    for bundle in tables.blocks:
        mat = find_lizmov_mat(bundle.spec)
        n_onsets = 0
        n_tagged = 0
        n_in_window = 0
        note = ""
        if mat is None:
            note = "no lizMov.mat"
            diag_rows.append(
                {
                    "block_key": bundle.spec.block_key,
                    "block_path": str(bundle.spec.block_path),
                    "n_head_onsets": n_onsets,
                    "n_tagged": n_tagged,
                    "n_in_window": n_in_window,
                    "note": note,
                }
            )
            continue
        try:
            onsets = load_lizmov_bout_onsets_ms(mat)
        except Exception as exc:
            diag_rows.append(
                {
                    "block_key": bundle.spec.block_key,
                    "block_path": str(bundle.spec.block_path),
                    "n_head_onsets": 0,
                    "n_tagged": 0,
                    "n_in_window": 0,
                    "note": f"lizMov load failed: {exc}",
                }
            )
            continue
        onsets = np.sort(onsets[np.isfinite(onsets)])
        n_onsets = int(onsets.size)
        ev = bundle.all_saccades
        if ev is None or ev.empty:
            note = "no saccades"
            diag_rows.append(
                {
                    "block_key": bundle.spec.block_key,
                    "block_path": str(bundle.spec.block_path),
                    "n_head_onsets": n_onsets,
                    "n_tagged": 0,
                    "n_in_window": 0,
                    "note": note,
                }
            )
            continue
        ev = label_saccades_head_movement(ev, bundle.spec)
        tagged = ev[ev["head_movement"] == True]  # noqa: E712
        n_tagged = int(len(tagged))
        if tagged.empty or onsets.size == 0:
            note = "no tagged events" if tagged.empty else "no head onsets"
            diag_rows.append(
                {
                    "block_key": bundle.spec.block_key,
                    "block_path": str(bundle.spec.block_path),
                    "n_head_onsets": n_onsets,
                    "n_tagged": n_tagged,
                    "n_in_window": 0,
                    "note": note,
                }
            )
            continue
        t_sacc = tagged["saccade_on_ms"].to_numpy(float)
        idx = np.searchsorted(onsets, t_sacc)
        for t, i in zip(t_sacc, idx):
            cands = []
            if i < onsets.size:
                cands.append(onsets[i])
            if i > 0:
                cands.append(onsets[i - 1])
            if not cands:
                continue
            nearest = cands[int(np.argmin([abs(t - c) for c in cands]))]
            dt = float(t - nearest)
            if abs(dt) <= window_ms:
                offsets.append(dt)
                n_in_window += 1
        diag_rows.append(
            {
                "block_key": bundle.spec.block_key,
                "block_path": str(bundle.spec.block_path),
                "n_head_onsets": n_onsets,
                "n_tagged": n_tagged,
                "n_in_window": n_in_window,
                "note": "",
            }
        )
    return np.asarray(offsets, dtype=float), pd.DataFrame(diag_rows)


def export_corrective_head(
    tables: EventTables,
    out_dir: Path,
    *,
    show: bool = False,
) -> dict[str, Path]:
    bundle = begin_plot_bundle(
        out_dir,
        "corrective_head",
        kind="corrective_head",
        tables=tables,
        logic_key="corrective_head",
        params=dict(getattr(tables, "params", {}) or {}),
    )
    figures_dir, metadata_dir = bundle.plots_dir, bundle.metadata_dir
    written: dict[str, Path] = {}

    partners = corrective_partners(tables)
    partners.to_csv(metadata_dir / "corrective_partners.csv", index=False)
    n = int(len(partners))
    n_hit = int(partners["has_corrective"].sum()) if n else 0
    pct_hit = 100.0 * n_hit / n if n else 0.0
    pct_miss = 100.0 - pct_hit if n else 0.0
    fig, ax = plt.subplots(figsize=(2.0, 1.8), dpi=300)
    ax.bar(
        [0, 1],
        [pct_miss, pct_hit],
        color=["#0072B2", "#D55E00"],
        width=0.7,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non-corrective", "corrective"], fontsize=7)
    ax.set_ylabel("% of monocular events", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title(f"n={n}", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    p = figures_dir / "corrective_percent.pdf"
    fig.tight_layout()
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p

    lats = partners.loc[partners["has_corrective"], "latency_ms"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(2.2, 1.8), dpi=300)
    bins = np.arange(0.0, CORRECTIVE_WINDOW_MS + HEAD_BIN_MS, HEAD_BIN_MS)
    if lats.size:
        ax.hist(lats, bins=bins, color="#0072B2", edgecolor="black")
    ax.set_xlabel("Latency [ms]", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_xlim(0, CORRECTIVE_WINDOW_MS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    p = figures_dir / "corrective_latency_hist.pdf"
    fig.tight_layout()
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p

    offsets, diag = head_peri_saccade_offsets(tables)
    np.save(metadata_dir / "head_peri_offsets.npy", offsets)
    diag.to_csv(metadata_dir / "head_peri_diagnostics.csv", index=False)
    fig, ax = plt.subplots(figsize=(2.4, 1.8), dpi=300)
    bins = np.arange(-HEAD_WINDOW_MS, HEAD_WINDOW_MS + HEAD_BIN_MS, HEAD_BIN_MS)
    if offsets.size:
        ax.hist(offsets, bins=bins, color="#D55E00", edgecolor="black")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Saccade − head onset [ms]", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    p = figures_dir / "head_peri_saccade.pdf"
    fig.tight_layout()
    fig.savefig(p, format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    written[p.name] = p
    finish_plot_bundle(bundle)
    return written
