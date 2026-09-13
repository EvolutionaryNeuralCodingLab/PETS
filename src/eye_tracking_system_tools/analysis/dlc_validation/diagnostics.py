"""Part E: diagnostic analyses and publication-quality plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_landmark_error_distribution(
    frame_errors: pd.DataFrame,
    output_dir: Path,
    *,
    split: str = "test",
) -> Path:
    """Histogram/ECDF of held-out landmark errors."""
    sub = frame_errors[frame_errors["split"] == split] if split else frame_errors
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    errs = sub["error_px"].dropna().to_numpy()
    axes[0].hist(errs, bins=40, color="#0072B2", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Landmark error (px)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Landmark error histogram ({split})")
    if errs.size:
        sorted_e = np.sort(errs)
        axes[1].plot(sorted_e, np.linspace(0, 1, len(sorted_e)), color="#D55E00")
    axes[1].set_xlabel("Landmark error (px)")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title(f"Landmark error ECDF ({split})")
    fig.tight_layout()
    out = output_dir / f"landmark_error_dist_{split}.pdf"
    _save_fig(fig, out)
    return out


def plot_per_landmark_errors(frame_errors: pd.DataFrame, output_dir: Path, *, split: str = "test") -> Path:
    sub = frame_errors[frame_errors["split"] == split] if split else frame_errors
    order = sorted(sub["bodypart"].unique())
    data = [sub.loc[sub["bodypart"] == bp, "error_px"].dropna().to_numpy() for bp in order]
    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.6), 4))
    ax.boxplot(data, labels=order, vert=True)
    ax.set_ylabel("Error (px)")
    ax.set_title(f"Per-landmark error ({split})")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out = output_dir / f"per_landmark_error_{split}.pdf"
    _save_fig(fig, out)
    return out


def plot_ellipse_rmse_distribution(frame_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Histogram and ECDF of filtered-mode ellipse RMSE."""
    paths = []
    valid = frame_df[frame_df["filt_fit_valid"]]
    rmse = valid["filt_residual_rmse"].dropna().to_numpy()
    norm = valid["filt_residual_rmse_norm"].dropna().to_numpy()

    for values, label, fname in (
        (rmse, "Ellipse RMSE (px)", "ellipse_rmse"),
        (norm, "Normalized ellipse RMSE (fraction of diameter)", "ellipse_rmse_norm"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(values, bins=50, color="#009E73", edgecolor="white", alpha=0.85)
        axes[0].set_xlabel(label)
        axes[0].set_ylabel("Count")
        if values.size:
            s = np.sort(values)
            axes[1].plot(s, np.linspace(0, 1, len(s)), color="#CC79A7")
        axes[1].set_xlabel(label)
        axes[1].set_ylabel("ECDF")
        fig.tight_layout()
        out = output_dir / f"{fname}_dist.pdf"
        _save_fig(fig, out)
        paths.append(out)
    return paths


def plot_scatter_diagnostics(frame_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Scatter plots of ellipse RMSE vs diameter, likelihood, landmark count."""
    valid = frame_df[frame_df["filt_fit_valid"]].copy()
    paths = []
    specs = [
        ("filt_diameter", "filt_residual_rmse", "Ellipse RMSE vs pupil diameter", "diameter_px", "rmse_px", "rmse_vs_diameter"),
        ("filt_likelihood_mean", "filt_residual_rmse", "Ellipse RMSE vs mean DLC likelihood", "mean likelihood", "rmse_px", "rmse_vs_likelihood"),
        ("filt_likelihood_min", "filt_residual_rmse", "Ellipse RMSE vs min DLC likelihood", "min likelihood", "rmse_px", "rmse_vs_min_likelihood"),
        ("filt_n_landmarks_used", "filt_residual_rmse", "Ellipse RMSE vs landmarks used", "n landmarks", "rmse_px", "rmse_vs_n_landmarks"),
        ("filt_eccentricity_ratio", "filt_residual_rmse", "Ellipse RMSE vs eccentricity", "major/minor", "rmse_px", "rmse_vs_eccentricity"),
    ]
    for xcol, ycol, title, xlab, ylab, fname in specs:
        if xcol not in valid.columns or ycol not in valid.columns:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        x = valid[xcol].to_numpy(dtype=float)
        y = valid[ycol].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=4, alpha=0.25, c="#0072B2", edgecolors="none")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        fig.tight_layout()
        out = output_dir / f"{fname}.pdf"
        _save_fig(fig, out)
        paths.append(out)
    return paths


def plot_spatial_error_heatmap(frame_df: pd.DataFrame, output_dir: Path, *, bins: int = 20) -> Path | None:
    valid = frame_df[frame_df["filt_fit_valid"]]
    if valid.empty:
        return None
    cx = valid["filt_center_x"].to_numpy(dtype=float)
    cy = valid["filt_center_y"].to_numpy(dtype=float)
    rmse = valid["filt_residual_rmse"].to_numpy(dtype=float)
    mask = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(rmse)
    if mask.sum() < 10:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(cx[mask], cy[mask], C=rmse[mask], gridsize=bins, reduce_C_function=np.median, cmap="turbo")
    ax.set_xlabel("Pupil center x (px)")
    ax.set_ylabel("Pupil center y (px)")
    ax.set_title("Median ellipse RMSE by pupil position")
    ax.invert_yaxis()
    fig.colorbar(hb, ax=ax, label="Median RMSE (px)")
    fig.tight_layout()
    out = output_dir / "spatial_error_heatmap.pdf"
    _save_fig(fig, out)
    return out


def plot_landmark_count_distribution(frame_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = frame_df["filt_n_landmarks_used"].dropna().astype(int)
    ax.hist(counts, bins=range(int(counts.min()), int(counts.max()) + 2), color="#56B4E9", edgecolor="white")
    ax.set_xlabel("Accepted landmarks per frame (filtered)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of valid landmark counts")
    fig.tight_layout()
    out = output_dir / "landmark_count_dist.pdf"
    _save_fig(fig, out)
    return out


def plot_filtering_comparison(frame_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    delta = frame_df["delta_residual_rmse"].dropna()
    ax.hist(delta, bins=50, color="#E69F00", edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Δ RMSE (filtered − all) px")
    ax.set_ylabel("Count")
    ax.set_title("Effect of likelihood filtering on ellipse RMSE")
    fig.tight_layout()
    out = output_dir / "filtering_delta_rmse.pdf"
    _save_fig(fig, out)
    return out


def run_diagnostic_plots(
    frame_df: pd.DataFrame,
    frame_errors: pd.DataFrame | None,
    output_dir: Path,
    *,
    run_name: str = "dlc_qc",
) -> dict[str, Any]:
    """Generate all Part E diagnostic figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    if frame_errors is not None and not frame_errors.empty:
        for split in ("test", "train"):
            if split in frame_errors["split"].values:
                paths.append(str(plot_landmark_error_distribution(frame_errors, fig_dir, split=split)))
                paths.append(str(plot_per_landmark_errors(frame_errors, fig_dir, split=split)))

    if not frame_df.empty:
        paths.extend(str(p) for p in plot_ellipse_rmse_distribution(frame_df, fig_dir))
        paths.extend(str(p) for p in plot_scatter_diagnostics(frame_df, fig_dir))
        sp = plot_spatial_error_heatmap(frame_df, fig_dir)
        if sp:
            paths.append(str(sp))
        paths.append(str(plot_landmark_count_distribution(frame_df, fig_dir)))
        paths.append(str(plot_filtering_comparison(frame_df, fig_dir)))

    bundle = begin_plot_bundle(
        output_dir,
        run_name,
        kind="dlc_pupil_qc",
        logic_key="dlc_pupil_qc",
        params={"run_name": run_name},
    )
    finish_plot_bundle(bundle)
    return {"figure_paths": paths, "bundle": bundle}
