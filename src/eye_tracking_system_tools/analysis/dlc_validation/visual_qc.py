"""Part F: visual QC frame extraction and overlays."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.dlc_validation.dlc_h5_io import load_analyzed_video_long
from eye_tracking_system_tools.analysis.dlc_validation.geometry import (
    fit_ellipse,
    point_to_ellipse_distances,
)


def _select_representative_frames(
    video_df: pd.DataFrame,
    *,
    n_each: int = 1,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Pick frame indices for each QC category."""
    valid = video_df[video_df["filt_fit_valid"]].copy()
    failed = video_df[~video_df["filt_fit_valid"]]

    picks: dict[str, list[int]] = {}

    def _pct_frames(df: pd.DataFrame, pct: float, n: int) -> list[int]:
        if df.empty:
            return []
        col = "filt_residual_rmse"
        target = np.percentile(df[col].dropna(), pct)
        near = df.iloc[(df[col] - target).abs().argsort()[:n]]
        return near["frame_idx"].astype(int).tolist()

    if not valid.empty:
        picks["excellent"] = _pct_frames(valid, 5, n_each)
        picks["median"] = _pct_frames(valid, 50, n_each)
        picks["p90"] = _pct_frames(valid, 90, n_each)
        picks["p95"] = _pct_frames(valid, 95, n_each)
        worst_idx = valid["filt_residual_rmse"].idxmax()
        picks["worst"] = [int(valid.loc[worst_idx, "frame_idx"])]
        low_conf = valid.nsmallest(n_each, "filt_likelihood_mean")
        picks["low_confidence"] = low_conf["frame_idx"].astype(int).tolist()

    if not failed.empty:
        sample = failed.sample(min(n_each, len(failed)), random_state=seed)
        picks["failed"] = sample["frame_idx"].astype(int).tolist()

    return picks


def _draw_ellipse_cv(
    img: np.ndarray,
    ellipse: dict[str, float],
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
) -> None:
    center = (int(round(ellipse["center_x"])), int(round(ellipse["center_y"])))
    axes = (int(round(ellipse["major_ax"])), int(round(ellipse["minor_ax"])))
    angle_deg = float(np.degrees(ellipse["phi"]))
    cv2.ellipse(img, center, axes, angle_deg, 0, 360, color, thickness, lineType=cv2.LINE_AA)


def render_qc_frame(
    frame_bgr: np.ndarray,
    landmarks: pd.DataFrame,
    *,
    likelihood_p_cutoff: float,
    min_points: int = 6,
    show_residuals: bool = True,
    title: str = "",
) -> np.ndarray:
    """Render one QC overlay frame with OpenCV; returns BGR image."""
    out = frame_bgr.copy()

    xs = landmarks["x"].to_numpy(dtype=float)
    ys = landmarks["y"].to_numpy(dtype=float)
    ls = landmarks["likelihood"].to_numpy(dtype=float)

    accepted = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ls) & (ls >= likelihood_p_cutoff)
    rejected = np.isfinite(xs) & np.isfinite(ys) & ~accepted

    for x, y in zip(xs[accepted], ys[accepted]):
        cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (int(round(x)), int(round(y))), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    for x, y in zip(xs[rejected], ys[rejected]):
        xi, yi = int(round(x)), int(round(y))
        sz = 6
        cv2.line(out, (xi - sz, yi - sz), (xi + sz, yi + sz), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(out, (xi - sz, yi + sz), (xi + sz, yi - sz), (0, 0, 255), 2, lineType=cv2.LINE_AA)

    x_fit = xs[accepted]
    y_fit = ys[accepted]
    ell = fit_ellipse(x_fit, y_fit, min_points=min_points) if accepted.sum() >= min_points else None
    metrics_line = title
    if ell is not None:
        _draw_ellipse_cv(out, ell)
        if show_residuals:
            dists = point_to_ellipse_distances(x_fit, y_fit, ell)
            for x, y, d in zip(x_fit, y_fit, dists):
                if np.isfinite(d) and d > 0.5:
                    cv2.line(
                        out,
                        (int(round(x)), int(round(y))),
                        (int(round(x)), int(round(y - d))),
                        (0, 255, 255),
                        1,
                        lineType=cv2.LINE_AA,
                    )
        rmse = float(np.sqrt(np.mean(dists ** 2))) if dists.size else float("nan")
        metrics_line = (
            f"{title} | RMSE={rmse:.2f}px D={ell['diameter']:.1f}px "
            f"n={int(accepted.sum())} L={np.nanmean(ls[accepted]):.2f}"
        )

    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, metrics_line[:120], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def export_visual_qc(
    frame_df: pd.DataFrame,
    h5_files: list[Path],
    pupil_bodyparts: list[str],
    output_dir: Path,
    *,
    likelihood_p_cutoff: float = 0.6,
    min_points: int = 6,
    n_each: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Export representative QC frame images per video."""
    out_root = output_dir / "figures" / "qc_frames"
    out_root.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    skipped: list[str] = []

    video_to_h5: dict[str, Path] = {}
    for h5 in h5_files:
        stem = h5.name.split("DLC")[0].rstrip("_")
        video_to_h5[stem] = h5

    for video, vdf in frame_df.groupby("video"):
        picks = _select_representative_frames(vdf, n_each=n_each, seed=seed)
        h5_path = video_to_h5.get(video)
        if h5_path is None:
            for stem, path in video_to_h5.items():
                if video == stem or video.startswith(stem) or stem.startswith(video):
                    h5_path = path
                    break
        if h5_path is None:
            skipped.append(f"{video}: no H5 match")
            continue

        mp4 = None
        for ext in (".mp4", ".avi", ".mov"):
            cand = h5_path.parent / f"{video}{ext}"
            if cand.is_file():
                mp4 = cand
                break
        if mp4 is None:
            skipped.append(f"{video}: no source video")
            continue

        long_df = load_analyzed_video_long(h5_path, pupil_bodyparts)
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            skipped.append(f"{video}: cannot open {mp4}")
            continue

        vid_out = out_root / video
        vid_out.mkdir(parents=True, exist_ok=True)

        for category, frame_indices in picks.items():
            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                lm = long_df[long_df["frame_idx"] == fi]
                row = vdf[vdf["frame_idx"] == fi]
                title = f"{video} f{fi} [{category}]"
                if not row.empty:
                    r = row.iloc[0]
                    title = f"{video} f{fi} [{category}] RMSE={r.get('filt_residual_rmse', float('nan')):.2f}"
                img = render_qc_frame(
                    frame_bgr,
                    lm,
                    likelihood_p_cutoff=likelihood_p_cutoff,
                    min_points=min_points,
                    title=title,
                )
                out_path = vid_out / f"{category}_frame{fi:06d}.png"
                cv2.imwrite(str(out_path), img)
                exported.append(str(out_path))
        cap.release()

    return {"exported": exported, "skipped": skipped, "output_dir": str(out_root)}
