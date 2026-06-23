"""
Sync-free eye pipeline helpers: DLC → ellipses → Kerr → optional join to final_sync_df.

Artifacts are written next to the eye video folder (LE/RE) as documented in
sync_free_eye_ellipse_pipeline.ipynb. Open Ephys timestamps are only applied
after an explicit join to ``final_sync_df``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np
import pandas as pd

from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync


def find_dlc_csv(eye_folder: Path) -> Path:
    """Pick DeepLabCut CSV in ``eye_folder`` (same rules as ``BlockSync.read_dlc_data``)."""
    pl = [eye_folder / i for i in os.listdir(eye_folder) if "DLC" in i and ".csv" in i]
    if not pl:
        raise FileNotFoundError(f"No DLC csv under {eye_folder}")
    if len(pl) > 1:
        filtered = [p for p in pl if "filtered" in p.name.lower()]
        if not filtered:
            raise FileNotFoundError(f"Multiple DLC csv files, none marked filtered: {pl}")
        return filtered[0]
    return pl[0]


def read_dlc_for_ellipse(eye_folder: Path) -> pd.DataFrame:
    """Read DLC with ``header=1`` (same as ``read_dlc_data``)."""
    path = find_dlc_csv(eye_folder)
    return pd.read_csv(path, header=1), path


def video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def build_data_for_eye_tracking(le_csv: pd.DataFrame) -> pd.DataFrame:
    """Replicate ``eye_tracking_analysis`` trimming: ``data = csv.iloc[1:].apply(pd.to_numeric)``."""
    return le_csv.iloc[1:].apply(pd.to_numeric)


def align_ellipse_df_to_video_frames(
    ellipse_df: pd.DataFrame,
    *,
    n_video_frames: int,
    n_data_rows: int,
) -> pd.DataFrame:
    """
    ``eye_tracking_analysis`` fits rows ``data`` indices ``1 .. len(data)-2`` (inclusive).

    So the first ellipse row corresponds to **video / DLC frame index 1**, and the last to
    ``len(data) - 2``. Frame ``0`` and frame ``len(data)-1`` have **no** ellipse from that
    routine (NaN placeholders).

    We require ``n_video_frames == n_data_rows`` (one DLC body row per video frame after trim).
    """
    n_ell = len(ellipse_df)
    expected_ell = max(0, n_data_rows - 2)
    if n_ell != expected_ell:
        raise ValueError(
            f"Ellipse row count mismatch: got {n_ell} ellipse rows, expected {expected_ell} "
            f"(len(data)={n_data_rows})."
        )
    if n_video_frames != n_data_rows:
        raise ValueError(
            f"Video frame count ({n_video_frames}) must equal DLC ``data`` row count ({n_data_rows}) "
            f"after ``iloc[1:]`` trim. If your DLC has a different row count than the video, "
            f"inspect trimming or re-export DLC for this video."
        )

    rows: list[dict[str, Any]] = []
    for ef in range(n_video_frames):
        if ef == 0 or ef == n_data_rows - 1:
            row = {"eye_frame": ef}
            for c in ellipse_df.columns:
                row[c] = np.nan
        else:
            row = {"eye_frame": ef}
            src = ellipse_df.iloc[ef - 1]
            for c in ellipse_df.columns:
                row[c] = src[c]
        rows.append(row)
    return pd.DataFrame(rows)


def run_syncfree_ellipses_for_eye(
    block,
    eye: Literal["left", "right"],
    uncertainty_thr: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run DLC → ellipse for one eye; return wide dataframe (``eye_frame`` 0..N-1) + metadata dict.
    """
    eye_lc = eye.lower()
    if eye_lc == "left":
        folder = Path(block.l_e_path)
        if not block.le_videos:
            block.handle_eye_videos()
        video = Path(block.le_videos[0])
    elif eye_lc == "right":
        folder = Path(block.r_e_path)
        if not block.re_videos:
            block.handle_eye_videos()
        video = Path(block.re_videos[0])
    else:
        raise ValueError("eye must be 'left' or 'right'")

    dlc_csv, dlc_path = read_dlc_for_ellipse(folder)
    data = build_data_for_eye_tracking(dlc_csv)
    ellipse_raw = BlockSync.eye_tracking_analysis(dlc_csv, uncertainty_thr)
    n_vid = video_frame_count(video)
    full = align_ellipse_df_to_video_frames(ellipse_raw, n_video_frames=n_vid, n_data_rows=len(data))

    meta: dict[str, Any] = {
        "animal_call": getattr(block, "animal_call", None),
        "experiment_date": getattr(block, "experiment_date", None),
        "block_num": getattr(block, "block_num", None),
        "eye": eye_lc,
        "dlc_csv": str(dlc_path),
        "video_path": str(video),
        "uncertainty_thr": uncertainty_thr,
        "n_video_frames": int(n_vid),
        "n_ellipse_rows_raw": int(len(ellipse_raw)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return full, meta


def write_meta_json(meta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def append_kerr_columns_syncfree(
    df: pd.DataFrame,
    kerr_ref_x: int,
    kerr_ref_y: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Compute Kerr angles using ``BlockSync.kerr`` with NaN OE placeholders.

    Returns ``(df_with_axes, angles_df, f_z)``. Angles merged on ``eye_frame`` only.
    """
    work = df.copy()
    if "major_ax" not in work.columns or "minor_ax" not in work.columns:
        work = BlockSync.get_maj_min_axes(work)
    work["OE_timestamp"] = np.nan
    work["ms_axis"] = np.nan
    work["ratio2"] = work["minor_ax"] / work["major_ax"]
    work["phi_ellipse"] = work["phi"]

    f_z, kerr_out = BlockSync.kerr(work, aEC=float(kerr_ref_x), bEC=float(kerr_ref_y))
    merged = kerr_out.rename(columns={"phi": "k_phi", "theta": "k_theta", "r": "k_r"})
    angles = merged[["eye_frame", "k_r", "k_theta", "k_phi"]].copy()
    return merged, angles, float(f_z)


def map_syncfree_degrees_to_final_sync(
    block,
    degrees_df: pd.DataFrame,
    eye: Literal["left", "right"],
    *,
    final_sync_filename: str | None = None,
) -> pd.DataFrame:
    """
    Join per-frame degree/ellipse data onto the current ``final_sync_df`` timeline.

    Output rows match ``final_sync_df`` length for that eye (one row per grid row).
    ``OE_timestamp`` ← ``Arena_TTL``; ``ms_axis`` ← ``Arena_TTL / (fs/1000)``.
    """
    from eye_tracking_system_tools.preprocessing.block_sync_core import load_final_sync_df  # noqa: PLC0415

    fs_hz = float(getattr(block, "sample_rate", None) or block.get_sample_rate())
    block.sample_rate = fs_hz
    fs_df = load_final_sync_df(block, filename=final_sync_filename, verbose=False).copy()
    col = "L_eye_frame" if eye.lower() == "left" else "R_eye_frame"
    if col not in fs_df.columns:
        raise ValueError(f"final_sync_df missing {col}")

    timeline = fs_df[["Arena_TTL", col]].copy()
    timeline = timeline.rename(columns={col: "eye_frame", "Arena_TTL": "OE_timestamp"})
    timeline["eye_frame"] = pd.to_numeric(timeline["eye_frame"], errors="coerce")
    timeline = timeline.dropna(subset=["eye_frame"]).copy()
    timeline["eye_frame"] = timeline["eye_frame"].astype("int64")

    payload = degrees_df.copy()
    for drop_c in ("OE_timestamp", "ms_axis"):
        if drop_c in payload.columns:
            payload = payload.drop(columns=[drop_c])
    payload["eye_frame"] = pd.to_numeric(payload["eye_frame"], errors="coerce").astype("int64")

    out = timeline.merge(payload, on="eye_frame", how="left", validate="many_to_one")
    out["ms_axis"] = pd.to_numeric(out["OE_timestamp"], errors="coerce") / (fs_hz / 1000.0)
    return out


def write_self_kerr_refs_csv(
    block,
    *,
    kerr_ref_l_x: Optional[int] = None,
    kerr_ref_l_y: Optional[int] = None,
    kerr_ref_r_x: Optional[int] = None,
    kerr_ref_r_y: Optional[int] = None,
) -> Path:
    """Write ``self_kerr_refs.csv`` compatible with ``calculate_kerr_angles.load_self_kerr_refs``."""
    ap = Path(block.analysis_path)
    ap.mkdir(parents=True, exist_ok=True)
    row = {
        "kerr_ref_l_x": kerr_ref_l_x,
        "kerr_ref_l_y": kerr_ref_l_y,
        "kerr_ref_r_x": kerr_ref_r_x,
        "kerr_ref_r_y": kerr_ref_r_y,
    }
    p = ap / "self_kerr_refs.csv"
    pd.DataFrame([row]).to_csv(p, index=False)
    return p


def video_path_for_eye(block, eye: Literal["left", "right"]) -> Path:
    """Return the primary eye video path, calling ``handle_eye_videos`` if needed."""
    if not getattr(block, "le_videos", None) or not getattr(block, "re_videos", None):
        block.handle_eye_videos()
    if eye.lower() == "left":
        if not block.le_videos:
            raise FileNotFoundError("Left eye video not found.")
        return Path(block.le_videos[0])
    if not block.re_videos:
        raise FileNotFoundError("Right eye video not found.")
    return Path(block.re_videos[0])


def merge_self_kerr_refs(
    block,
    eye: Literal["left", "right"],
    rx: int,
    ry: int,
) -> Path:
    """Merge one eye's Kerr ref into ``analysis/self_kerr_refs.csv`` (notebook helper)."""
    ap = Path(block.analysis_path)
    p = ap / "self_kerr_refs.csv"
    row = {
        "kerr_ref_l_x": getattr(block, "kerr_ref_l_x", None),
        "kerr_ref_l_y": getattr(block, "kerr_ref_l_y", None),
        "kerr_ref_r_x": getattr(block, "kerr_ref_r_x", None),
        "kerr_ref_r_y": getattr(block, "kerr_ref_r_y", None),
    }
    if p.is_file():
        prev = pd.read_csv(p).iloc[0].to_dict()
        for key in row:
            if key in prev and pd.notna(prev[key]):
                row[key] = prev[key]
    eye_lc = eye.lower()
    if eye_lc == "left":
        row["kerr_ref_l_x"], row["kerr_ref_l_y"] = rx, ry
        block.kerr_ref_l_x, block.kerr_ref_l_y = rx, ry
    else:
        row["kerr_ref_r_x"], row["kerr_ref_r_y"] = rx, ry
        block.kerr_ref_r_x, block.kerr_ref_r_y = rx, ry
    return write_self_kerr_refs_csv(
        block,
        kerr_ref_l_x=row["kerr_ref_l_x"],
        kerr_ref_l_y=row["kerr_ref_l_y"],
        kerr_ref_r_x=row["kerr_ref_r_x"],
        kerr_ref_r_y=row["kerr_ref_r_y"],
    )


def maybe_load_kerr_refs(
    block,
    eye: Literal["left", "right"],
) -> tuple[int, int]:
    """Load Kerr reference pixels for one eye from the block or ``self_kerr_refs.csv``."""
    eye_lc = eye.lower()
    if eye_lc == "left":
        if getattr(block, "kerr_ref_l_x", None) is not None and getattr(
            block, "kerr_ref_l_y", None
        ) is not None:
            return int(block.kerr_ref_l_x), int(block.kerr_ref_l_y)
    elif getattr(block, "kerr_ref_r_x", None) is not None and getattr(
        block, "kerr_ref_r_y", None
    ) is not None:
        return int(block.kerr_ref_r_x), int(block.kerr_ref_r_y)

    p = Path(block.analysis_path) / "self_kerr_refs.csv"
    if p.is_file():
        row = pd.read_csv(p).iloc[0]
        if eye_lc == "left" and pd.notna(row.get("kerr_ref_l_x")):
            return int(row["kerr_ref_l_x"]), int(row["kerr_ref_l_y"])
        if eye_lc == "right" and pd.notna(row.get("kerr_ref_r_x")):
            return int(row["kerr_ref_r_x"]), int(row["kerr_ref_r_y"])
    raise FileNotFoundError(
        f"No Kerr ref for {eye}. Run verification first or create {p}"
    )


def syncfree_mapped_is_stale(
    block,
    eye: Literal["left", "right"],
    tag: str,
) -> bool:
    """True when ``final_sync_df.csv`` is newer than the mapped timeline CSV."""
    ap = Path(block.analysis_path)
    final_sync = ap / "final_sync_df.csv"
    mapped = analysis_syncfree_timeline_path(block, eye, tag)
    if not final_sync.is_file() or not mapped.is_file():
        return False
    return final_sync.stat().st_mtime > mapped.stat().st_mtime


def analysis_syncfree_timeline_path(
    block,
    eye: Literal["left", "right"],
    tag: str,
) -> Path:
    """OE-timeline mapping of finalized sync-free eye data (optional post-finalize)."""
    return Path(block.analysis_path) / f"{eye.lower()}_eye_syncfree_{tag}_timeline.csv"


def default_syncfree_paths(video_path: Path, eye: str, tag: str = "v1") -> dict[str, Path]:
    """Canonical sync-free artifacts next to the eye ``.mp4``.

    Final outputs (after **Finalize**):

    - ``{side}_syncfree_{tag}_kerr_refs.csv`` — Kerr reference pixel for this eye
    - ``{side}_syncfree_{tag}_eye_data.csv`` — one row per video frame with ellipse
      geometry and Kerr angles
    - ``{side}_syncfree_{tag}_meta.json`` — provenance and pipeline metadata

    Working file (removed on finalize):

    - ``{side}_syncfree_{tag}_draft.csv`` — ellipse fit awaiting verification
    """
    d = video_path.parent
    side = "left" if eye.lower() == "left" else "right"
    base = f"{side}_syncfree_{tag}"
    return {
        "draft": d / f"{base}_draft.csv",
        "eye_data": d / f"{base}_eye_data.csv",
        "kerr_refs": d / f"{base}_kerr_refs.csv",
        "meta": d / f"{base}_meta.json",
    }


def legacy_syncfree_paths(video_path: Path, eye: str, tag: str = "v1") -> dict[str, Path]:
    """Pre-unification artifact names (read-only migration)."""
    d = video_path.parent
    side = "left" if eye.lower() == "left" else "right"
    base = f"{side}_syncfree_{tag}"
    return {
        "ellipses": d / f"{base}_ellipses.csv",
        "verified": d / f"{base}_verified.csv",
        "kerr_raw": d / f"{base}_kerr_angles.csv",
        "degrees": d / f"{base}_degrees.csv",
        "kerr_refs_sidecar": d / f"{base}_kerr_refs.csv",
        "timeline_legacy": Path("unused"),
    }


def resolve_syncfree_working_csv(video_path: Path, eye: str, tag: str) -> Path | None:
    """Return the best on-disk kinematic CSV for verification (draft or legacy)."""
    paths = default_syncfree_paths(video_path, eye, tag)
    for key in ("draft", "eye_data"):
        if paths[key].is_file():
            return paths[key]
    legacy = legacy_syncfree_paths(video_path, eye, tag)
    for key in ("verified", "ellipses"):
        if legacy[key].is_file():
            return legacy[key]
    return None


def write_syncfree_draft(
    df: pd.DataFrame,
    meta: dict[str, Any],
    video_path: Path,
    eye: str,
    tag: str,
) -> dict[str, Path]:
    """Persist ellipse fit as a draft awaiting interactive verification."""
    paths = default_syncfree_paths(video_path, eye, tag)
    df.to_csv(paths["draft"], index=False)
    meta = dict(meta)
    meta["stage"] = "draft"
    meta["artifact_paths"] = {k: str(v) for k, v in paths.items()}
    write_meta_json(meta, paths["meta"])
    return paths


def finalize_syncfree_eye(
    block,
    eye: Literal["left", "right"],
    kinematic_df: pd.DataFrame,
    kerr_ref_xy: tuple[int, int],
    *,
    tag: str,
    map_to_timeline: bool = False,
    meta_patch: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write finalized per-frame eye data + Kerr refs; optionally map to OE timeline.

    This is the clean exit point for the sync-free pipeline: one CSV per eye with
    all frame-level ellipse and Kerr columns, plus sidecar Kerr refs and meta.
    """
    video = video_path_for_eye(block, eye)
    paths = default_syncfree_paths(video, eye, tag)
    kx, ky = int(kerr_ref_xy[0]), int(kerr_ref_xy[1])

    pd.DataFrame(
        [{"eye": eye.lower(), "kerr_ref_x": kx, "kerr_ref_y": ky}]
    ).to_csv(paths["kerr_refs"], index=False)
    merge_self_kerr_refs(block, eye, kx, ky)

    merged_deg, angles, fz = append_kerr_columns_syncfree(kinematic_df, kx, ky)
    unified = kinematic_df.merge(angles, on="eye_frame", how="left")
    for col in ("major_ax", "minor_ax", "ratio2", "phi_ellipse"):
        if col in merged_deg.columns and col not in unified.columns:
            unified[col] = merged_deg[col]
    unified.to_csv(paths["eye_data"], index=False)
    if paths["draft"].is_file():
        paths["draft"].unlink()

    meta: dict[str, Any] = {}
    if paths["meta"].is_file():
        try:
            with open(paths["meta"], encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
    if meta_patch:
        meta.update(meta_patch)
    meta.update(
        {
            "stage": "finalized",
            "finalized": True,
            "finalized_utc": datetime.now(timezone.utc).isoformat(),
            "kerr_f_z": float(fz),
            "artifact_paths": {k: str(v) for k, v in paths.items()},
            "eye_data_columns": list(unified.columns),
        }
    )
    write_meta_json(meta, paths["meta"])

    written = dict(paths)
    if map_to_timeline:
        timeline = analysis_syncfree_timeline_path(block, eye, tag)
        mapped = map_syncfree_degrees_to_final_sync(block, unified, eye)
        timeline.parent.mkdir(parents=True, exist_ok=True)
        mapped.to_csv(timeline, index=False)
        written["timeline"] = timeline
    return written
