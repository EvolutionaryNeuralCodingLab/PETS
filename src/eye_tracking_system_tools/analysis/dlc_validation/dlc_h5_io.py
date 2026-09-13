"""Load DeepLabCut HDF5 pose outputs into normalized tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.dlc_validation.project_io import (
    _normalize_frame_key,
    is_pupil_bodypart,
    sort_pupil_bodyparts,
)


def load_dlc_h5_wide(path: Path | str) -> pd.DataFrame:
    """Load a DLC H5 file preserving the native multi-index column layout."""
    return pd.read_hdf(path)


def wide_to_long(
    df: pd.DataFrame,
    *,
    source: str = "",
    pupil_only: bool = False,
) -> pd.DataFrame:
    """
    Convert DLC wide multi-index H5 to long format.

    Columns: ``frame_key``, ``scorer``, ``bodypart``, ``x``, ``y``, ``likelihood``, ``source``.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC multi-index columns (scorer, bodyparts, coords)")

    scorer_level = df.columns.names[0] if df.columns.names[0] else "scorer"
    bodypart_level = "bodyparts"
    if bodypart_level not in df.columns.names:
        bodypart_level = df.columns.names[1]

    rows: list[dict] = []
    scorers = df.columns.get_level_values(0).unique()
    bodyparts = df.columns.get_level_values(bodypart_level).unique()

    for frame_key, row in df.iterrows():
        norm_key = _normalize_frame_key(frame_key)
        for scorer in scorers:
            for bp in bodyparts:
                if pupil_only and not is_pupil_bodypart(str(bp)):
                    continue
                try:
                    x = float(row[(scorer, bp, "x")])
                    y = float(row[(scorer, bp, "y")])
                except (KeyError, TypeError):
                    continue
                likelihood = np.nan
                try:
                    likelihood = float(row[(scorer, bp, "likelihood")])
                except (KeyError, TypeError):
                    pass
                rows.append(
                    {
                        "frame_key": norm_key,
                        "scorer": str(scorer),
                        "bodypart": str(bp),
                        "x": x,
                        "y": y,
                        "likelihood": likelihood,
                        "source": source,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["frame_key", "bodypart"]).reset_index(drop=True)


def load_labeled_ground_truth(
    path: Path | str,
    pupil_bodyparts: list[str] | None = None,
) -> pd.DataFrame:
    """Load CollectedData H5 ground truth as long format (no likelihood)."""
    wide = load_dlc_h5_wide(path)
    long = wide_to_long(wide, source="ground_truth", pupil_only=False)
    if pupil_bodyparts:
        allowed = set(pupil_bodyparts)
        long = long[long["bodypart"].isin(allowed)].copy()
    else:
        long = long[long["bodypart"].map(is_pupil_bodypart)].copy()
    long["likelihood"] = 1.0
    return long.reset_index(drop=True)


def load_eval_predictions(
    path: Path | str,
    pupil_bodyparts: list[str] | None = None,
) -> pd.DataFrame:
    """Load evaluation-network H5 predictions on labeled frames."""
    wide = load_dlc_h5_wide(path)
    long = wide_to_long(wide, source="eval_prediction", pupil_only=True)
    if pupil_bodyparts:
        long = long[long["bodypart"].isin(pupil_bodyparts)].copy()
    return long.reset_index(drop=True)


def load_analyzed_video_long(
    path: Path | str,
    pupil_bodyparts: list[str] | None = None,
) -> pd.DataFrame:
    """Load analyzed-video H5; ``frame_key`` is the integer frame index as string."""
    wide = load_dlc_h5_wide(path)
    long = wide_to_long(wide, source=str(path), pupil_only=True)
    if pupil_bodyparts:
        long = long[long["bodypart"].isin(pupil_bodyparts)].copy()
    long["frame_idx"] = long["frame_key"].astype(int)
    return long.reset_index(drop=True)


def pivot_video_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot analyzed-video long data to one row per frame with landmark columns."""
    if long_df.empty:
        return pd.DataFrame()
    frames = sorted(long_df["frame_idx"].unique())
    bodyparts = sort_pupil_bodyparts(long_df["bodypart"].unique().tolist())
    records = []
    for fi in frames:
        sub = long_df[long_df["frame_idx"] == fi]
        rec: dict = {"frame_idx": int(fi)}
        for bp in bodyparts:
            row = sub[sub["bodypart"] == bp]
            if row.empty:
                rec[f"{bp}_x"] = np.nan
                rec[f"{bp}_y"] = np.nan
                rec[f"{bp}_likelihood"] = np.nan
            else:
                r = row.iloc[0]
                rec[f"{bp}_x"] = float(r["x"])
                rec[f"{bp}_y"] = float(r["y"])
                rec[f"{bp}_likelihood"] = float(r["likelihood"])
        records.append(rec)
    return pd.DataFrame(records)
