"""Part A: DLC landmark prediction accuracy vs ground truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.dlc_validation.dlc_h5_io import (
    load_eval_predictions,
    load_labeled_ground_truth,
)
from eye_tracking_system_tools.analysis.dlc_validation.geometry import fit_ellipse, representative_diameter
from eye_tracking_system_tools.analysis.dlc_validation.project_io import (
    DlcProject,
    find_collected_data_h5,
    find_evaluation_h5,
    find_evaluation_results_csv,
    load_native_eval_summary,
    load_train_test_split,
)


def _error_stats(errors: np.ndarray) -> dict[str, float]:
    e = errors[np.isfinite(errors)]
    if e.size == 0:
        return {
            "rmse": np.nan,
            "median": np.nan,
            "iqr_low": np.nan,
            "iqr_high": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "n": 0,
        }
    q25, q75 = np.percentile(e, [25, 75])
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "median": float(np.median(e)),
        "iqr_low": float(q25),
        "iqr_high": float(q75),
        "p95": float(np.percentile(e, 95)),
        "max": float(np.max(e)),
        "n": int(e.size),
    }


def _gt_diameter_for_frame(gt_sub: pd.DataFrame, diameter_method: str) -> float:
    xs = gt_sub["x"].to_numpy(dtype=float)
    ys = gt_sub["y"].to_numpy(dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 5:
        # fallback: mean pairwise distance among landmarks
        if xs.size < 2:
            return float("nan")
        dists = []
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                dists.append(np.hypot(xs[i] - xs[j], ys[i] - ys[j]))
        return float(2.0 * np.mean(dists))
    ell = fit_ellipse(xs, ys, min_points=5)
    if ell is None:
        return float(2.0 * np.mean([np.hypot(xs[i] - xs[j], ys[i] - ys[j]) for i in range(len(xs)) for j in range(i + 1, len(xs))]))
    return representative_diameter(ell["major_ax"], ell["minor_ax"], diameter_method)


def compute_landmark_errors(
    gt_long: pd.DataFrame,
    pred_long: pd.DataFrame,
    split_labels: list[str],
    frame_keys: list[str],
    *,
    likelihood_p_cutoff: float = 0.6,
    diameter_method: str = "geometric_mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join GT and predictions; return frame×landmark errors and summary table.

    ``split_labels`` aligns with ``frame_keys`` order from CollectedData.
    """
    key_to_split = dict(zip(frame_keys, split_labels))
    merged = pred_long.merge(
        gt_long,
        on=["frame_key", "bodypart"],
        how="inner",
        suffixes=("_pred", "_gt"),
    )
    merged["error_px"] = np.hypot(
        merged["x_pred"] - merged["x_gt"],
        merged["y_pred"] - merged["y_gt"],
    )
    merged["split"] = merged["frame_key"].map(key_to_split).fillna("unknown")
    merged["above_p_cutoff"] = merged["likelihood_pred"] >= likelihood_p_cutoff

    # Per-frame GT diameter for normalization
    diameters: dict[str, float] = {}
    for fk, grp in gt_long.groupby("frame_key"):
        diameters[fk] = _gt_diameter_for_frame(grp, diameter_method)
    merged["gt_diameter_px"] = merged["frame_key"].map(diameters)
    merged["error_norm"] = merged["error_px"] / merged["gt_diameter_px"]

    frame_df = merged[
        [
            "frame_key",
            "bodypart",
            "split",
            "x_gt",
            "y_gt",
            "x_pred",
            "y_pred",
            "likelihood_pred",
            "above_p_cutoff",
            "error_px",
            "error_norm",
            "gt_diameter_px",
        ]
    ].copy()

    summary_rows: list[dict[str, Any]] = []
    for split_name in ("train", "test", "all"):
        if split_name == "all":
            sub = frame_df
        else:
            sub = frame_df[frame_df["split"] == split_name]
        for mask_name, mask in (
            ("all", np.ones(len(sub), dtype=bool)),
            ("p_cutoff", sub["above_p_cutoff"].to_numpy()),
        ):
            errs = sub.loc[mask, "error_px"].to_numpy()
            stats = _error_stats(errs)
            norm_errs = sub.loc[mask, "error_norm"].to_numpy()
            norm_stats = _error_stats(norm_errs)
            summary_rows.append(
                {
                    "split": split_name,
                    "filter": mask_name,
                    "rmse_px": stats["rmse"],
                    "median_px": stats["median"],
                    "iqr_low_px": stats["iqr_low"],
                    "iqr_high_px": stats["iqr_high"],
                    "p95_px": stats["p95"],
                    "max_px": stats["max"],
                    "n_landmarks": stats["n"],
                    "rmse_norm": norm_stats["rmse"],
                    "median_norm": norm_stats["median"],
                    "p95_norm": norm_stats["p95"],
                }
            )

    for bp in sorted(frame_df["bodypart"].unique()):
        sub = frame_df[frame_df["bodypart"] == bp]
        for split_name in ("train", "test", "all"):
            if split_name == "all":
                ssub = sub
            else:
                ssub = sub[sub["split"] == split_name]
            stats = _error_stats(ssub["error_px"].to_numpy())
            summary_rows.append(
                {
                    "split": split_name,
                    "filter": "per_bodypart",
                    "bodypart": bp,
                    "rmse_px": stats["rmse"],
                    "median_px": stats["median"],
                    "p95_px": stats["p95"],
                    "n_landmarks": stats["n"],
                }
            )

    return frame_df, pd.DataFrame(summary_rows)


def run_landmark_validation(
    project: DlcProject,
    *,
    iteration: int | None = None,
    likelihood_p_cutoff: float | None = None,
    diameter_method: str = "geometric_mean",
) -> dict[str, Any]:
    """
    Run Part A landmark validation for a DLC project.

    Returns dict with ``native_summary``, ``frame_errors``, ``summary``, ``notes``.
    """
    it = iteration if iteration is not None else project.iteration
    p_cut = project.pcutoff if likelihood_p_cutoff is None else likelihood_p_cutoff
    notes: list[str] = []

    native_path = find_evaluation_results_csv(project, it)
    native_summary = None
    if native_path is not None:
        native_summary = load_native_eval_summary(native_path)
    else:
        notes.append(f"No native evaluation results CSV for iteration {it}")

    eval_h5 = find_evaluation_h5(project, it)
    if eval_h5 is None:
        raise FileNotFoundError(f"No evaluation H5 for iteration {it}")

    gt_path = find_collected_data_h5(project, it)
    split_info = load_train_test_split(project, it)
    gt_long = load_labeled_ground_truth(gt_path, project.pupil_bodyparts)
    pred_long = load_eval_predictions(eval_h5, project.pupil_bodyparts)

    frame_errors, summary = compute_landmark_errors(
        gt_long,
        pred_long,
        split_info["split_labels"],
        split_info["frame_keys"],
        likelihood_p_cutoff=p_cut,
        diameter_method=diameter_method,
    )

    if project.is_pytorch:
        notes.append("DLC3/PyTorch project: check for mAP/mAR metrics in evaluation output")
    else:
        notes.append("DLC2 project: mAP/mAR not available; using pixel-error metrics from evaluate_network")

    return {
        "native_summary": native_summary,
        "native_summary_path": native_path,
        "eval_h5_path": eval_h5,
        "gt_path": gt_path,
        "frame_errors": frame_errors,
        "summary": summary,
        "split_info": split_info,
        "likelihood_p_cutoff": p_cut,
        "notes": notes,
    }
