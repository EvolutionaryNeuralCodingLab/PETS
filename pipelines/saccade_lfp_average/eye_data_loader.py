"""Resolve and load per-eye tracking CSVs for the saccade LFP pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from eye_tracking_system_tools.annotation.event_explorer.eye_csv_resolver import resolve_eye_csv
from eye_tracking_system_tools.preprocessing.block_sync_core import (
    drop_pandas_index_artifact_columns,
    load_eye_tracking_df_csv,
)

EyeDataSource = Literal["auto", "eye_data_csv", "le_re_df"]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return drop_pandas_index_artifact_columns(df)


def normalize_eye_tracking_df(
    df: pd.DataFrame,
    *,
    sample_rate: float | None = None,
) -> pd.DataFrame:
    """Ensure columns expected by velocity / legacy saccade detectors."""
    out = drop_pandas_index_artifact_columns(df.copy())
    if "OE_timestamp" not in out.columns and "Arena_TTL" in out.columns:
        out["OE_timestamp"] = out["Arena_TTL"]
    if "ms_axis" not in out.columns and "OE_timestamp" in out.columns and sample_rate:
        out["ms_axis"] = out["OE_timestamp"] / (sample_rate / 1000.0)
    if "center_x" not in out.columns and "center_x_corrected" in out.columns:
        out["center_x"] = out["center_x_corrected"]
    if "center_y" not in out.columns and "center_y_corrected" in out.columns:
        out["center_y"] = out["center_y_corrected"]
    if "pupil_diameter" not in out.columns and "major_ax" in out.columns:
        out["pupil_diameter"] = out["major_ax"]
    return out


def resolve_eye_csv_pair(
    analysis_path: Path,
    *,
    source: EyeDataSource = "auto",
) -> tuple[Path | None, Path | None, str]:
    """Return (left_path, right_path, description)."""
    analysis_path = Path(analysis_path)
    le_re_left = analysis_path / "le_df.csv"
    le_re_right = analysis_path / "re_df.csv"

    left_eye, _ = resolve_eye_csv(analysis_path, "left")
    right_eye, _ = resolve_eye_csv(analysis_path, "right")

    if source == "eye_data_csv":
        if left_eye is None or right_eye is None:
            raise FileNotFoundError(
                f"{analysis_path}: expected left/right_eye_data*.csv "
                f"(found left={left_eye}, right={right_eye})."
            )
        return left_eye, right_eye, "left/right_eye_data.csv"

    if source == "le_re_df":
        if not le_re_left.exists() or not le_re_right.exists():
            raise FileNotFoundError(
                f"{analysis_path}: expected le_df.csv and re_df.csv."
            )
        return le_re_left, le_re_right, "le_df.csv / re_df.csv"

    # auto: prefer finalized eye_data exports, fall back to le/re
    if left_eye is not None and right_eye is not None:
        return left_eye, right_eye, f"{left_eye.name} / {right_eye.name}"
    if le_re_left.exists() and le_re_right.exists():
        return le_re_left, le_re_right, "le_df.csv / re_df.csv (fallback)"
    raise FileNotFoundError(
        f"{analysis_path}: no eye tracking CSVs found "
        "(expected left/right_eye_data*.csv or le_df.csv / re_df.csv)."
    )


def load_block_eye_dataframes(
    block,
    *,
    source: EyeDataSource = "auto",
    log=None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load L/R eye data into normalized dataframes; does not mutate ``block``."""
    left_path, right_path, desc = resolve_eye_csv_pair(block.analysis_path, source=source)
    sample_rate = float(getattr(block, "sample_rate", 0) or 0) or None

    if left_path.name.startswith("le_"):
        left_df = load_eye_tracking_df_csv(left_path)
        right_df = load_eye_tracking_df_csv(right_path)
    else:
        left_df = _read_csv(left_path)
        right_df = _read_csv(right_path)

    left_df = normalize_eye_tracking_df(left_df, sample_rate=sample_rate)
    right_df = normalize_eye_tracking_df(right_df, sample_rate=sample_rate)

    for side, df in (("left", left_df), ("right", right_df)):
        missing = [c for c in ("center_x", "center_y", "ms_axis") if c not in df.columns]
        if missing:
            raise KeyError(
                f"Block {block.block_num} {side} eye data ({left_path if side=='left' else right_path}) "
                f"missing columns: {missing}"
            )

    if log is not None:
        log.info(
            f"Block {block.block_num}: loaded eye data from {desc} "
            f"({left_path.name}, {right_path.name})"
        )
    return left_df, right_df, desc


def attach_eye_data_to_block(
    block,
    *,
    source: EyeDataSource = "auto",
    log=None,
) -> str:
    """Populate ``block.le_df`` / ``block.re_df`` from resolved CSV paths."""
    left_df, right_df, desc = load_block_eye_dataframes(block, source=source, log=log)
    block.le_df = left_df
    block.re_df = right_df
    return desc
