"""Figures 2g (amplitude distributions) and 2j (orientation tuning)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.colors import build_color_map
from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.pipeline import EventTables
from eye_tracking_system_tools.analysis.figure_display import show_and_close
from eye_tracking_system_tools.analysis.run_layout import resolve_figure_dirs

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def export_figure_2g(tables: EventTables, out_dir: Path, *, show: bool = False) -> tuple[Path, Path]:
    cfg = dict(tables.params.get("figure_2g", {}))
    n_bins = int(cfg.get("amp_bins", 40))
    amp_max = float(cfg.get("amp_max_deg", 25.0))
    bins = np.linspace(0, amp_max, n_bins + 1)
    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)

    synced = tables.synced
    nons = tables.non_synced
    # Per-animal normalized histograms, then mean±SEM across animals.
    animals = sorted(set(tables.all_saccades["animal"].astype(str).unique()))
    synced_mats = []
    nons_mats = []
    for animal in animals:
        s = (
            synced.query("animal == @animal")["net_angular_disp"].to_numpy(float)
            if not synced.empty and "net_angular_disp" in synced.columns
            else np.array([])
        )
        n = (
            nons.query("animal == @animal")["net_angular_disp"].to_numpy(float)
            if not nons.empty and "net_angular_disp" in nons.columns
            else np.array([])
        )
        s = s[np.isfinite(s)]
        n = n[np.isfinite(n)]
        hs, _ = np.histogram(s, bins=bins, density=True) if s.size else (np.zeros(n_bins), bins)
        hn, _ = np.histogram(n, bins=bins, density=True) if n.size else (np.zeros(n_bins), bins)
        # density can be zero-sum; use probability mass
        if s.size:
            hs, _ = np.histogram(s, bins=bins, density=False)
            hs = hs / hs.sum()
        if n.size:
            hn, _ = np.histogram(n, bins=bins, density=False)
            hn = hn / hn.sum()
        synced_mats.append(hs)
        nons_mats.append(hn)

    synced_mats = np.vstack(synced_mats) if synced_mats else np.zeros((1, n_bins))
    nons_mats = np.vstack(nons_mats) if nons_mats else np.zeros((1, n_bins))
    top = {
        "synced_mean": synced_mats.mean(axis=0),
        "synced_sem": synced_mats.std(axis=0, ddof=1) / np.sqrt(max(len(animals), 1))
        if len(animals) > 1
        else np.zeros(n_bins),
        "non_synced_mean": nons_mats.mean(axis=0),
        "non_synced_sem": nons_mats.std(axis=0, ddof=1) / np.sqrt(max(len(animals), 1))
        if len(animals) > 1
        else np.zeros(n_bins),
        "bins": bins,
    }
    # Coarser bins for difference traces
    bins_bot = np.linspace(0, amp_max, 20)
    animal_diff = {}
    for animal in animals:
        s = (
            synced.query("animal == @animal")["net_angular_disp"].to_numpy(float)
            if not synced.empty
            else np.array([])
        )
        n = (
            nons.query("animal == @animal")["net_angular_disp"].to_numpy(float)
            if not nons.empty
            else np.array([])
        )
        s = s[np.isfinite(s)]
        n = n[np.isfinite(n)]
        hs, _ = np.histogram(s, bins=bins_bot, density=False)
        hn, _ = np.histogram(n, bins=bins_bot, density=False)
        hs = hs / hs.sum() if hs.sum() else hs
        hn = hn / hn.sum() if hn.sum() else hn
        animal_diff[animal] = hs - hn
    bot = {"animal_diff_traces": animal_diff, "bins": bins_bot}

    top_pkl = metadata_dir / "averaged_saccade_amplitude_angle_data.pkl"
    bot_pkl = metadata_dir / "saccade_amplitude_difference_all_animals_data.pkl"
    write_pickle_with_meta(
        top,
        top_pkl,
        meta={"csv_choices": tables.csv_meta, "params": cfg, "figure": "2g_top"},
        entrypoint="eye_tracking_system_tools.analysis.figures_2g_2j.export_figure_2g",
    )
    write_pickle_with_meta(
        bot,
        bot_pkl,
        meta={"csv_choices": tables.csv_meta, "params": cfg, "figure": "2g_bot"},
        entrypoint="eye_tracking_system_tools.analysis.figures_2g_2j.export_figure_2g",
    )

    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(1.7, 1.2), dpi=300)
    ax.plot(centers, top["synced_mean"], color="green", lw=1.5, label="Synchronized")
    ax.fill_between(
        centers,
        top["synced_mean"] - top["synced_sem"],
        top["synced_mean"] + top["synced_sem"],
        color="green",
        alpha=0.3,
    )
    ax.plot(centers, top["non_synced_mean"], color="blue", lw=1.5, label="Monocular")
    ax.fill_between(
        centers,
        top["non_synced_mean"] - top["non_synced_sem"],
        top["non_synced_mean"] + top["non_synced_sem"],
        color="blue",
        alpha=0.3,
    )
    ax.set_xlabel("Saccade Amplitude [deg]", fontsize=8)
    ax.set_ylabel("Probability", fontsize=8)
    ax.legend(fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.savefig(figures_dir / "fig_2g_top.pdf", format="pdf", bbox_inches="tight")
    show_and_close(fig, show)

    centers_b = 0.5 * (bins_bot[:-1] + bins_bot[1:])
    color_map = build_color_map(animals, template="okabeito", order=animals)
    fig, ax = plt.subplots(figsize=(1.7, 1.0), dpi=300)
    for animal, trace in animal_diff.items():
        ax.plot(centers_b, trace, color=color_map[animal], lw=1.5, label=animal)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Amplitude [deg]", fontsize=8)
    ax.set_ylabel("Diff [probability]", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=0)
    fig.savefig(figures_dir / "fig_2g_bot.pdf", format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    return top_pkl, bot_pkl


def calculate_orientation_tuning(saccade_angles) -> float:
    saccade_angles = np.asarray(saccade_angles) % 360
    if saccade_angles.size == 0:
        return float("nan")
    horizontal_ranges = [(315, 360), (0, 45), (135, 225)]
    is_horizontal = np.logical_or.reduce(
        [
            (saccade_angles >= low) & (saccade_angles <= high)
            if low < high
            else (saccade_angles >= low) | (saccade_angles <= high)
            for low, high in horizontal_ranges
        ]
    )
    p_horizontal = float(np.sum(is_horizontal) / len(saccade_angles))
    p_vertical = 1.0 - p_horizontal
    return float((p_horizontal - p_vertical) / (p_horizontal + p_vertical))


def export_figure_2j(tables: EventTables, out_dir: Path, *, show: bool = False) -> Path:
    figures_dir, metadata_dir = resolve_figure_dirs(out_dir)
    synced = tables.synced
    nons = tables.non_synced
    payload = {"synced_df": synced, "non_synced_df": nons}
    pkl = metadata_dir / "saccade_angles_data.pkl"
    write_pickle_with_meta(
        payload,
        pkl,
        meta={
            "csv_choices": tables.csv_meta,
            "figure": "2j",
            "n_synced_rows": int(len(synced)),
            "n_non_synced": int(len(nons)),
        },
        entrypoint="eye_tracking_system_tools.analysis.figures_2g_2j.export_figure_2j",
    )

    animals = sorted(set(tables.all_saccades["animal"].astype(str).unique()))
    color_map = build_color_map(animals, template="okabeito", order=animals)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
    xs, ys = [], []
    for animal in animals:
        sa = (
            synced.query("animal == @animal")["overall_angle_deg"].to_numpy(float)
            if not synced.empty
            else np.array([])
        )
        na = (
            nons.query("animal == @animal")["overall_angle_deg"].to_numpy(float)
            if not nons.empty
            else np.array([])
        )
        sx = calculate_orientation_tuning(sa)
        sy = calculate_orientation_tuning(na)
        xs.append(sx)
        ys.append(sy)
        ax.scatter(sx, sy, color=color_map[animal], s=30, label=animal)
    ax.axhline(0, color="gray", ls="--", lw=0.7)
    ax.axvline(0, color="gray", ls="--", lw=0.7)
    finite = [v for v in xs + ys if np.isfinite(v)]
    lim = (max(abs(min(finite)), abs(max(finite))) * 1.1) if finite else 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Concurrent [A.U]", fontsize=8)
    ax.set_ylabel("Monocular [A.U]", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_2j.pdf", format="pdf", bbox_inches="tight")
    show_and_close(fig, show)
    return pkl
