"""Parts B–C: frame-by-frame ellipse fitting and confidence filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from eye_tracking_system_tools.analysis.dlc_validation.dlc_h5_io import (
    load_analyzed_video_long,
    pivot_video_wide,
)
from eye_tracking_system_tools.analysis.dlc_validation.geometry import (
    fit_ellipse,
    point_to_ellipse_distances,
    residual_stats,
)
from eye_tracking_system_tools.analysis.dlc_validation.project_io import (
    DlcProject,
    find_analyzed_h5_files,
    find_source_video_for_h5,
    parse_video_metadata,
)


def _fit_frame_mode(
    xs: np.ndarray,
    ys: np.ndarray,
    ls: np.ndarray,
    *,
    mode: str,
    likelihood_p_cutoff: float,
    min_points: int,
) -> dict[str, Any]:
    """Fit ellipse for one frame in ``all`` or ``filtered`` mode."""
    n_total = int(np.sum(np.isfinite(xs) & np.isfinite(ys)))
    if mode == "filtered":
        mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ls) & (ls >= likelihood_p_cutoff)
    else:
        mask = np.isfinite(xs) & np.isfinite(ys)

    x_use = xs[mask]
    y_use = ys[mask]
    l_use = ls[mask] if ls is not None else np.array([])
    n_used = int(x_use.size)

    base: dict[str, Any] = {
        "n_landmarks_total": n_total,
        "n_landmarks_used": n_used,
        "likelihood_mean": float(np.nanmean(l_use)) if l_use.size else np.nan,
        "likelihood_min": float(np.nanmin(l_use)) if l_use.size else np.nan,
        "likelihood_max": float(np.nanmax(l_use)) if l_use.size else np.nan,
        "fit_valid": False,
    }

    if n_used < min_points:
        base.update(
            {
                "center_x": np.nan,
                "center_y": np.nan,
                "major_ax": np.nan,
                "minor_ax": np.nan,
                "phi": np.nan,
                "area": np.nan,
                "diameter": np.nan,
                "eccentricity_ratio": np.nan,
                "residual_mae": np.nan,
                "residual_rmse": np.nan,
                "residual_max": np.nan,
                "residual_rmse_norm": np.nan,
                "residual_rmse_pct": np.nan,
            }
        )
        return base

    ell = fit_ellipse(x_use, y_use, min_points=min_points)
    if ell is None:
        base.update(
            {
                "center_x": np.nan,
                "center_y": np.nan,
                "major_ax": np.nan,
                "minor_ax": np.nan,
                "phi": np.nan,
                "area": np.nan,
                "diameter": np.nan,
                "eccentricity_ratio": np.nan,
                "residual_mae": np.nan,
                "residual_rmse": np.nan,
                "residual_max": np.nan,
                "residual_rmse_norm": np.nan,
                "residual_rmse_pct": np.nan,
            }
        )
        return base

    dists = point_to_ellipse_distances(x_use, y_use, ell)
    stats = residual_stats(dists)
    ecc = ell["major_ax"] / ell["minor_ax"] if ell["minor_ax"] > 0 else np.nan
    norm = stats["residual_rmse"] / ell["diameter"] if ell["diameter"] > 0 else np.nan

    base.update(
        {
            "center_x": ell["center_x"],
            "center_y": ell["center_y"],
            "major_ax": ell["major_ax"],
            "minor_ax": ell["minor_ax"],
            "phi": ell["phi"],
            "area": ell["area"],
            "diameter": ell["diameter"],
            "eccentricity_ratio": ecc,
            "fit_valid": True,
            **stats,
            "residual_rmse_norm": norm,
            "residual_rmse_pct": norm * 100.0 if np.isfinite(norm) else np.nan,
        }
    )
    return base


def process_video_h5(
    h5_path: Path,
    pupil_bodyparts: list[str],
    *,
    metadata: dict[str, str] | None = None,
    likelihood_p_cutoff: float = 0.6,
    min_points: int = 6,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Process one analyzed-video H5; return frame-level QC for both fit modes."""
    long_df = load_analyzed_video_long(h5_path, pupil_bodyparts)
    if long_df.empty:
        return pd.DataFrame()

    meta = metadata or parse_video_metadata(h5_path)
    wide = pivot_video_wide(long_df)
    if wide.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    iterator: Any = wide.itertuples(index=False)
    if show_progress:
        iterator = tqdm(wide.itertuples(index=False), total=len(wide), desc=h5_path.stem[:40], leave=False)

    for row in iterator:
        fi = int(row.frame_idx)
        xs = np.array([getattr(row, f"{bp}_x", np.nan) for bp in pupil_bodyparts], dtype=float)
        ys = np.array([getattr(row, f"{bp}_y", np.nan) for bp in pupil_bodyparts], dtype=float)
        ls = np.array([getattr(row, f"{bp}_likelihood", np.nan) for bp in pupil_bodyparts], dtype=float)

        rec: dict[str, Any] = {
            "frame_idx": fi,
            "video": meta.get("video", h5_path.stem),
            "eye": meta.get("eye", "unknown"),
            "animal": meta.get("animal", "unknown"),
            "species": meta.get("species", "unknown"),
            "h5_path": str(h5_path),
            "n_pupil_landmarks_defined": len(pupil_bodyparts),
            "all_landmarks_above_p_cutoff": bool(np.all(np.isfinite(ls) & (ls >= likelihood_p_cutoff))),
        }

        for mode in ("all", "filtered"):
            fit = _fit_frame_mode(
                xs, ys, ls,
                mode=mode,
                likelihood_p_cutoff=likelihood_p_cutoff,
                min_points=min_points,
            )
            prefix = "all_" if mode == "all" else "filt_"
            for k, v in fit.items():
                rec[f"{prefix}{k}"] = v

        rec["delta_residual_rmse"] = (
            rec["filt_residual_rmse"] - rec["all_residual_rmse"]
            if np.isfinite(rec.get("filt_residual_rmse", np.nan)) and np.isfinite(rec.get("all_residual_rmse", np.nan))
            else np.nan
        )
        records.append(rec)

    return pd.DataFrame(records)


def characterize_filtering(frame_df: pd.DataFrame) -> dict[str, Any]:
    """Part C summary statistics comparing all vs filtered fits."""
    if frame_df.empty:
        return {}
    n = len(frame_df)
    return {
        "n_frames": n,
        "frac_all_landmarks_above_p_cutoff": float(frame_df["all_landmarks_above_p_cutoff"].mean()),
        "all_fit_valid_frac": float(frame_df["all_fit_valid"].mean()),
        "filt_fit_valid_frac": float(frame_df["filt_fit_valid"].mean()),
        "filt_n_used_median": float(frame_df["filt_n_landmarks_used"].median()),
        "delta_rmse_median": float(frame_df["delta_residual_rmse"].median()),
        "delta_rmse_mean": float(frame_df["delta_residual_rmse"].mean()),
    }


def run_ellipse_qc(
    project: DlcProject,
    *,
    videos_dir: Path | None = None,
    species: str = "",
    animal: str = "",
    likelihood_p_cutoff: float | None = None,
    min_points: int = 6,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run Parts B–C on all discovered analyzed H5 files."""
    p_cut = project.pcutoff if likelihood_p_cutoff is None else likelihood_p_cutoff
    h5_files = find_analyzed_h5_files(project, videos_dir)
    if not h5_files:
        raise FileNotFoundError(f"No analyzed DLC H5 files under {videos_dir or project.root / 'videos'}")

    all_frames: list[pd.DataFrame] = []
    video_meta: list[dict[str, Any]] = []
    for h5 in h5_files:
        meta = parse_video_metadata(h5, species=species or project.config.get("_qc_species", ""), animal=animal)
        meta["source_video"] = str(find_source_video_for_h5(h5, videos_dir))
        video_meta.append(meta)
        df = process_video_h5(
            h5,
            project.pupil_bodyparts,
            metadata=meta,
            likelihood_p_cutoff=p_cut,
            min_points=min_points,
            show_progress=show_progress,
        )
        if not df.empty:
            all_frames.append(df)

    frame_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    filter_stats = characterize_filtering(frame_df)

    return {
        "frame_df": frame_df,
        "video_meta": video_meta,
        "filter_stats": filter_stats,
        "likelihood_p_cutoff": p_cut,
        "min_points": min_points,
        "h5_files": h5_files,
    }
