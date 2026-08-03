"""Resolve and load per-eye angle CSVs from an analyzed block."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.preprocessing.block_sync_core import (
    drop_pandas_index_artifact_columns,
)

logger = logging.getLogger(__name__)

REQUIRED_ANGLE_COLS = ("k_phi", "k_theta")


@dataclass(frozen=True)
class EyeCsvChoice:
    side: str  # "left" | "right"
    path: Path
    rule: str  # "raw_verified" | "newest"


@dataclass
class LoadedBlockEyes:
    spec: BlockSpec
    left: pd.DataFrame
    right: pd.DataFrame
    left_csv: EyeCsvChoice
    right_csv: EyeCsvChoice


def _candidates(analysis_path: Path, side: str) -> list[Path]:
    pattern = "left_eye_data*.csv" if side == "left" else "right_eye_data*.csv"
    return sorted(analysis_path.glob(pattern))


def _has_angles(path: Path) -> bool:
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        logger.warning("Could not read columns from %s: %s", path, exc)
        return False
    return all(c in cols for c in REQUIRED_ANGLE_COLS)


def resolve_eye_csv(
    analysis_path: Path,
    side: str,
) -> EyeCsvChoice:
    """
    Prefer ``*raw_verified*`` (newest among those), else newest eye CSV.
    Skip candidates lacking k_phi/k_theta.
    """
    analysis_path = Path(analysis_path)
    cands = _candidates(analysis_path, side)
    if not cands:
        raise FileNotFoundError(
            f"{analysis_path}: no {side} eye CSVs matching left/right_eye_data*.csv"
        )

    usable = [p for p in cands if _has_angles(p)]
    if not usable:
        raise FileNotFoundError(
            f"{analysis_path}: no {side} eye CSV with columns {REQUIRED_ANGLE_COLS}; "
            f"candidates={[p.name for p in cands]}"
        )

    raw = [p for p in usable if "raw_verified" in p.name]
    if raw:
        chosen = max(raw, key=lambda p: p.stat().st_mtime)
        rule = "raw_verified"
    else:
        chosen = max(usable, key=lambda p: p.stat().st_mtime)
        rule = "newest"

    return EyeCsvChoice(side=side, path=chosen.resolve(), rule=rule)


def _normalize_eye_df(df: pd.DataFrame) -> pd.DataFrame:
    out = drop_pandas_index_artifact_columns(df.copy())
    if "pupil_diameter" not in out.columns and "major_ax" in out.columns:
        out["pupil_diameter"] = out["major_ax"]
    return out


def load_eye_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_eye_df(df)


def load_block_eyes(spec: BlockSpec, *, log: bool = True) -> LoadedBlockEyes:
    """Load left/right traces for one block; always report chosen CSV paths."""
    left_choice = resolve_eye_csv(spec.analysis_path, "left")
    right_choice = resolve_eye_csv(spec.analysis_path, "right")

    msg = (
        f"[{spec.block_key}] eye CSVs: "
        f"L={left_choice.path.name} ({left_choice.rule}), "
        f"R={right_choice.path.name} ({right_choice.rule})"
    )
    if log:
        print(msg)
    logger.info(msg)
    logger.info("  left path:  %s", left_choice.path)
    logger.info("  right path: %s", right_choice.path)

    return LoadedBlockEyes(
        spec=spec,
        left=load_eye_dataframe(left_choice.path),
        right=load_eye_dataframe(right_choice.path),
        left_csv=left_choice,
        right_csv=right_choice,
    )


def csv_choices_meta(loaded: LoadedBlockEyes) -> dict:
    return {
        "animal": loaded.spec.animal,
        "block_num": loaded.spec.block_num,
        "block_path": str(loaded.spec.block_path),
        "left_csv": str(loaded.left_csv.path),
        "left_rule": loaded.left_csv.rule,
        "right_csv": str(loaded.right_csv.path),
        "right_rule": loaded.right_csv.rule,
    }
