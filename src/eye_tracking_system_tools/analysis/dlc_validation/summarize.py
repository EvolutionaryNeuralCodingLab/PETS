"""Part D: hierarchical dataset-level QC summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _robust_stats(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {
            "median": np.nan,
            "iqr_low": np.nan,
            "iqr_high": np.nan,
            "p95": np.nan,
            "mean": np.nan,
            "n": 0,
        }
    q25, q75 = np.percentile(v, [25, 75])
    return {
        "median": float(np.median(v)),
        "iqr_low": float(q25),
        "iqr_high": float(q75),
        "p95": float(np.percentile(v, 95)),
        "mean": float(np.mean(v)),
        "n": int(v.size),
    }


def summarize_group(
    df: pd.DataFrame,
    *,
    residual_col: str = "filt_residual_rmse",
    valid_col: str = "filt_fit_valid",
    diameter_col: str = "filt_diameter",
    norm_col: str = "filt_residual_rmse_norm",
    likelihood_col: str = "filt_likelihood_mean",
) -> dict[str, Any]:
    """Summarize one group of frames."""
    n_frames = len(df)
    valid = df[valid_col].astype(bool) if valid_col in df.columns else pd.Series([False] * n_frames)
    n_valid = int(valid.sum())
    pct_valid = 100.0 * n_valid / n_frames if n_frames else 0.0

    sub = df.loc[valid]
    res_stats = _robust_stats(sub[residual_col].to_numpy(dtype=float) if residual_col in sub.columns else np.array([]))
    norm_stats = _robust_stats(sub[norm_col].to_numpy(dtype=float) if norm_col in sub.columns else np.array([]))
    diam_stats = _robust_stats(sub[diameter_col].to_numpy(dtype=float) if diameter_col in sub.columns else np.array([]))
    lik_stats = _robust_stats(sub[likelihood_col].to_numpy(dtype=float) if likelihood_col in sub.columns else np.array([]))

    return {
        "n_frames": n_frames,
        "n_valid_ellipse": n_valid,
        "pct_valid": pct_valid,
        "residual_rmse_median": res_stats["median"],
        "residual_rmse_iqr_low": res_stats["iqr_low"],
        "residual_rmse_iqr_high": res_stats["iqr_high"],
        "residual_rmse_p95": res_stats["p95"],
        "residual_rmse_norm_median": norm_stats["median"],
        "residual_rmse_norm_p95": norm_stats["p95"],
        "diameter_median": diam_stats["median"],
        "diameter_iqr_low": diam_stats["iqr_low"],
        "diameter_iqr_high": diam_stats["iqr_high"],
        "likelihood_median": lik_stats["median"],
        "likelihood_iqr_low": lik_stats["iqr_low"],
        "likelihood_iqr_high": lik_stats["iqr_high"],
    }


def summarize_hierarchy(
    frame_df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    residual_col: str = "filt_residual_rmse",
) -> dict[str, pd.DataFrame]:
    """
    Build summary tables at frame-grouping levels.

    Default hierarchy: video → eye → animal → species.
  When rolling up to species, uses per-video medians to avoid treating frames as i.i.d.
    """
    if frame_df.empty:
        return {}

    levels = group_cols or ["video", "eye", "animal", "species"]
    out: dict[str, pd.DataFrame] = {}

    for level in levels:
        if level not in frame_df.columns:
            continue
        rows = []
        for name, grp in frame_df.groupby(level, sort=True):
            row = summarize_group(grp, residual_col=residual_col)
            row[level] = name
            rows.append(row)
        out[level] = pd.DataFrame(rows)

    # Species-level rollup via per-video medians (non-i.i.d. safe)
    if "video" in frame_df.columns and "species" in frame_df.columns:
        video_summary = out.get("video")
        if video_summary is not None and not video_summary.empty:
            # attach species to each video row
            vid_species = frame_df.groupby("video")["species"].first()
            vs = video_summary.copy()
            vs["species"] = vs["video"].map(vid_species)
            species_rows = []
            for sp, grp in vs.groupby("species"):
                med = _robust_stats(grp["residual_rmse_median"].to_numpy(dtype=float))
                row = {
                    "species": sp,
                    "n_videos": len(grp),
                    "n_frames_total": int(grp["n_frames"].sum()),
                    "pct_valid_weighted": float(grp["n_valid_ellipse"].sum() / grp["n_frames"].sum() * 100) if grp["n_frames"].sum() else 0,
                    "residual_rmse_median_of_video_medians": med["median"],
                    "residual_rmse_iqr_low": med["iqr_low"],
                    "residual_rmse_iqr_high": med["iqr_high"],
                    "residual_rmse_p95": med["p95"],
                }
                species_rows.append(row)
            out["species_rollup"] = pd.DataFrame(species_rows)

    return out
