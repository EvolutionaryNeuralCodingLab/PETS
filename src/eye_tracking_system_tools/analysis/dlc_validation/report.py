"""Part G: exports, publication summary, and full pipeline orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from eye_tracking_system_tools.analysis.dlc_validation.diagnostics import run_diagnostic_plots
from eye_tracking_system_tools.analysis.dlc_validation.ellipse_qc import run_ellipse_qc
from eye_tracking_system_tools.analysis.dlc_validation.landmark_validation import run_landmark_validation
from eye_tracking_system_tools.analysis.dlc_validation.project_io import DlcProject, load_dlc_project
from eye_tracking_system_tools.analysis.dlc_validation.summarize import summarize_hierarchy
from eye_tracking_system_tools.analysis.dlc_validation.visual_qc import export_visual_qc


def write_run_config(
    output_dir: Path,
    *,
    project: DlcProject,
    params: dict[str, Any],
    notes: list[str],
) -> Path:
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dlc_project_root": str(project.root),
        "species": project.config.get("_qc_species", ""),
        "iteration": project.iteration,
        "shuffle": project.shuffle,
        "scorer": project.scorer,
        "pupil_bodyparts": project.pupil_bodyparts,
        "bodyparts_all": project.bodyparts,
        "pcutoff": project.pcutoff,
        "is_pytorch": project.is_pytorch,
        "params": params,
        "notes": notes + project.notes,
    }
    path = meta_dir / "run_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def export_tables(
    output_dir: Path,
    *,
    landmark_result: dict[str, Any] | None,
    ellipse_result: dict[str, Any] | None,
    summaries: dict[str, pd.DataFrame] | None,
) -> dict[str, Path]:
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if landmark_result is not None:
        fe = landmark_result.get("frame_errors")
        if fe is not None and not fe.empty:
            p = meta_dir / "landmark_errors_frame.csv"
            fe.to_csv(p, index=False)
            paths["landmark_errors_frame"] = p
        sm = landmark_result.get("summary")
        if sm is not None and not sm.empty:
            p = meta_dir / "landmark_summary.csv"
            sm.to_csv(p, index=False)
            paths["landmark_summary"] = p
        native = landmark_result.get("native_summary")
        if native is not None and not native.empty:
            p = meta_dir / "landmark_native_eval_summary.csv"
            native.to_csv(p, index=False)
            paths["landmark_native_eval"] = p

    if ellipse_result is not None:
        fd = ellipse_result.get("frame_df")
        if fd is not None and not fd.empty:
            p = meta_dir / "ellipse_qc_frame.csv"
            fd.to_csv(p, index=False)
            paths["ellipse_qc_frame"] = p
        fs = ellipse_result.get("filter_stats")
        if fs:
            p = meta_dir / "ellipse_filter_stats.yaml"
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(fs, f)
            paths["ellipse_filter_stats"] = p

    if summaries:
        for level, df in summaries.items():
            if df is not None and not df.empty:
                p = meta_dir / f"ellipse_qc_summary_{level}.csv"
                df.to_csv(p, index=False)
                paths[f"summary_{level}"] = p

    return paths


def build_publication_summary(
    landmark_result: dict[str, Any] | None,
    ellipse_result: dict[str, Any] | None,
    summaries: dict[str, pd.DataFrame] | None,
) -> str:
    """Generate adaptive publication-oriented summary text."""
    parts: list[str] = []

    if landmark_result is not None:
        summary = landmark_result.get("summary")
        fe = landmark_result.get("frame_errors")
        if summary is not None and not summary.empty:
            test_row = summary[(summary["split"] == "test") & (summary["filter"] == "all")]
            test_pc = summary[(summary["split"] == "test") & (summary["filter"] == "p_cutoff")]
            if not test_row.empty:
                rmse = test_row.iloc[0]["rmse_px"]
                rmse_norm = test_row.iloc[0].get("rmse_norm", float("nan"))
                n_frames = fe[fe["split"] == "test"]["frame_key"].nunique() if fe is not None else "?"
                norm_pct = rmse_norm * 100 if pd.notna(rmse_norm) else float("nan")
                if pd.notna(norm_pct):
                    parts.append(
                        f"DLC achieved a held-out landmark RMSE of {rmse:.2f} px "
                        f"({norm_pct:.1f}% of pupil diameter) on {n_frames} test frames."
                    )
                else:
                    parts.append(
                        f"DLC achieved a held-out landmark RMSE of {rmse:.2f} px on {n_frames} test frames."
                    )
            if not test_pc.empty:
                rmse_pc = test_pc.iloc[0]["rmse_px"]
                parts.append(
                    f"With likelihood p-cutoff ({landmark_result.get('likelihood_p_cutoff', '?')}), "
                    f"held-out RMSE was {rmse_pc:.2f} px."
                )
        native = landmark_result.get("native_summary")
        if native is not None and not native.empty:
            row = native.iloc[0]
            if " Test error(px) " in native.columns or "Test error(px)" in native.columns:
                col = " Test error(px) " if " Test error(px) " in native.columns else "Test error(px)"
                parts.append(f"(DLC native evaluate_network test error: {row[col]:.2f} px.)")

    if ellipse_result is not None:
        fd = ellipse_result.get("frame_df")
        if fd is not None and not fd.empty:
            n_frames = len(fd)
            n_videos = fd["video"].nunique()
            pct_valid = 100.0 * fd["filt_fit_valid"].mean()
            valid = fd[fd["filt_fit_valid"]]
            med = valid["filt_residual_rmse"].median()
            q25, q75 = valid["filt_residual_rmse"].quantile([0.25, 0.75])
            p95 = valid["filt_residual_rmse"].quantile(0.95)
            med_norm = valid["filt_residual_rmse_norm"].median() * 100
            parts.append(
                f"Across {n_frames:,} analyzed frames in {n_videos} video(s), a valid pupil ellipse "
                f"was obtained in {pct_valid:.1f}% of frames (filtered mode). "
                f"Median geometric ellipse-fitting RMSE was {med:.2f} px "
                f"(IQR {q25:.2f}–{q75:.2f}; 95th percentile {p95:.2f}), "
                f"corresponding to {med_norm:.1f}% of pupil diameter."
            )

    if summaries and "species_rollup" in summaries:
        sr = summaries["species_rollup"]
        if not sr.empty:
            row = sr.iloc[0]
            parts.append(
                f"Species-level rollup (median of per-video medians): "
                f"{row.get('residual_rmse_median_of_video_medians', float('nan')):.2f} px "
                f"across {row.get('n_videos', '?')} videos."
            )

    if not parts:
        return "Insufficient data to generate QC summary."
    return " ".join(parts)


def run_full_qc_pipeline(
    project_root: Path | str,
    output_dir: Path | str,
    *,
    species: str = "",
    iteration: int | None = None,
    shuffle: int = 1,
    likelihood_p_cutoff: float | None = None,
    min_points_for_ellipse: int = 6,
    diameter_method: str = "geometric_mean",
    videos_dir: Path | None = None,
    run_landmarks: bool = True,
    run_ellipse: bool = True,
    run_plots: bool = True,
    run_visual_qc: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Execute the full Parts A–G QC pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project = load_dlc_project(project_root, iteration=iteration, shuffle=shuffle, species=species)
    p_cut = project.pcutoff if likelihood_p_cutoff is None else likelihood_p_cutoff
    params = {
        "likelihood_p_cutoff": p_cut,
        "min_points_for_ellipse": min_points_for_ellipse,
        "diameter_method": diameter_method,
        "iteration": project.iteration,
        "shuffle": shuffle,
    }
    all_notes: list[str] = list(project.notes)

    landmark_result = None
    if run_landmarks:
        try:
            landmark_result = run_landmark_validation(
                project,
                iteration=iteration,
                likelihood_p_cutoff=p_cut,
                diameter_method=diameter_method,
            )
            all_notes.extend(landmark_result.get("notes", []))
        except FileNotFoundError as exc:
            all_notes.append(f"Landmark validation skipped: {exc}")

    ellipse_result = None
    if run_ellipse:
        ellipse_result = run_ellipse_qc(
            project,
            videos_dir=videos_dir,
            species=species,
            likelihood_p_cutoff=p_cut,
            min_points=min_points_for_ellipse,
            show_progress=show_progress,
        )

    summaries = None
    if ellipse_result and not ellipse_result["frame_df"].empty:
        summaries = summarize_hierarchy(ellipse_result["frame_df"])

    export_paths = export_tables(
        output_dir,
        landmark_result=landmark_result,
        ellipse_result=ellipse_result,
        summaries=summaries,
    )

    plot_info = None
    if run_plots:
        fe = landmark_result.get("frame_errors") if landmark_result else None
        fd = ellipse_result.get("frame_df") if ellipse_result else pd.DataFrame()
        plot_info = run_diagnostic_plots(fd, fe, output_dir, run_name=output_dir.name)

    visual_info = None
    if run_visual_qc and ellipse_result:
        visual_info = export_visual_qc(
            ellipse_result["frame_df"],
            ellipse_result["h5_files"],
            project.pupil_bodyparts,
            output_dir,
            likelihood_p_cutoff=p_cut,
            min_points=min_points_for_ellipse,
        )
        if visual_info.get("skipped"):
            all_notes.extend(f"Visual QC: {s}" for s in visual_info["skipped"])

    config_path = write_run_config(output_dir, project=project, params=params, notes=all_notes)
    summary_text = build_publication_summary(landmark_result, ellipse_result, summaries)

    summary_path = output_dir / "metadata" / "publication_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    return {
        "project": project,
        "landmark_result": landmark_result,
        "ellipse_result": ellipse_result,
        "summaries": summaries,
        "export_paths": export_paths,
        "plot_info": plot_info,
        "visual_info": visual_info,
        "summary_text": summary_text,
        "config_path": config_path,
        "output_dir": output_dir,
    }
