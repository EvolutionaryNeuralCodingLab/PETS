"""Sidecar refined-ellipse CSVs and Promote-to-``eye_data``.

Refined geometry is written to ``{left,right}_eye_data_refined_{tag}.csv``
and never overwrites ``left/right_eye_data.csv`` until
:func:`promote_refined_to_eye_data` copies it across (with a timestamped
backup).
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REFINED_GEOMETRY_COLS = (
    "center_x_refined",
    "center_y_refined",
    "width_refined",
    "height_refined",
    "phi_refined",
)
SPIN_GEOMETRY_COLS = (
    "center_x_spin",
    "center_y_spin",
    "width_spin",
    "height_spin",
    "phi_spin",
)
_PROMOTE_MAP = {
    "center_x": "center_x_refined",
    "center_y": "center_y_refined",
    "width": "width_refined",
    "height": "height_refined",
    "phi": "phi_refined",
}


def _normalize_side(side: str) -> str:
    key = str(side).strip().lower()
    if key in ("left", "l", "le", "left_eye"):
        return "left"
    if key in ("right", "r", "re", "right_eye"):
        return "right"
    raise ValueError(f"Unknown eye side {side!r}")


def _analysis_dir(block_path: Path | str) -> Path:
    path = Path(block_path)
    if path.name == "analysis":
        return path
    return path / "analysis"


def refined_eye_csv_path(
    block_path: Path | str, side: str, tag: str
) -> Path:
    side = _normalize_side(side)
    tag = str(tag).strip() or "default"
    return _analysis_dir(block_path) / f"{side}_eye_data_refined_{tag}.csv"


def list_refined_eye_csvs(block_path: Path | str, side: str) -> list[Path]:
    side = _normalize_side(side)
    analysis = _analysis_dir(block_path)
    if not analysis.is_dir():
        return []
    return sorted(analysis.glob(f"{side}_eye_data_refined_*.csv"))


def latest_refined_eye_csv(block_path: Path | str, side: str) -> Path | None:
    candidates = list_refined_eye_csvs(block_path, side)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def refined_tag_from_name(name: str) -> str | None:
    """Parse ``{side}_eye_data_refined_{tag}.csv`` → tag, else ``None``."""
    stem_name = Path(name).name
    for side in ("left", "right"):
        prefix = f"{side}_eye_data_refined_"
        if stem_name.startswith(prefix) and stem_name.endswith(".csv"):
            return stem_name[len(prefix) : -4]
    return None


def list_refined_tags(block_path: Path | str) -> list[str]:
    tags: set[str] = set()
    for side in ("left", "right"):
        for path in list_refined_eye_csvs(block_path, side):
            tag = refined_tag_from_name(path.name)
            if tag:
                tags.add(tag)
    return sorted(tags)


def write_refined_eye_table(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_refined_eye_table(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def attach_refined_columns(
    eye_df: pd.DataFrame, refined: pd.DataFrame
) -> pd.DataFrame:
    """Left-join ``*_refined`` columns onto ``eye_df`` without changing originals."""
    out = eye_df.copy()
    key = None
    for candidate in ("eye_frame", "OE_timestamp"):
        if candidate in out.columns and candidate in refined.columns:
            key = candidate
            break
    cols = [c for c in REFINED_GEOMETRY_COLS if c in refined.columns]
    if not cols:
        return out
    out = out.drop(columns=[c for c in cols if c in out.columns], errors="ignore")
    if key is not None:
        src = refined[[key, *cols]].drop_duplicates(subset=key)
        return out.merge(src, on=key, how="left")
    for col in cols:
        out[col] = refined[col].to_numpy()[: len(out)]
    return out


def overlay_refined_geometry(
    eye_df: pd.DataFrame, refined: pd.DataFrame
) -> pd.DataFrame:
    """Return a copy of ``eye_df`` whose ellipse columns come from the refined table.

    Used by the Conicoid tab when Ellipse source = refined. Merge key is
    ``eye_frame`` if present, else ``OE_timestamp``, else row alignment.
    """
    out = eye_df.copy()
    key = None
    for candidate in ("eye_frame", "OE_timestamp"):
        if candidate in out.columns and candidate in refined.columns:
            key = candidate
            break
    src = refined
    if key is not None:
        cols = [key, *[c for c in REFINED_GEOMETRY_COLS if c in refined.columns]]
        src = refined[cols].drop_duplicates(subset=key)
        out = out.drop(
            columns=[c for c in REFINED_GEOMETRY_COLS if c in out.columns],
            errors="ignore",
        )
        out = out.merge(src, on=key, how="left")
    else:
        for col in REFINED_GEOMETRY_COLS:
            if col in refined.columns:
                out[col] = refined[col].to_numpy()[: len(out)]
    for dest, src_col in _PROMOTE_MAP.items():
        if src_col not in out.columns:
            continue
        vals = pd.to_numeric(out[src_col], errors="coerce")
        if dest not in out.columns:
            out[dest] = vals
        else:
            mask = vals.notna()
            out.loc[mask, dest] = vals.loc[mask]
    _recompute_axes(out)
    return out


def _recompute_axes(df: pd.DataFrame) -> None:
    if "width" in df.columns and "height" in df.columns:
        w = pd.to_numeric(df["width"], errors="coerce")
        h = pd.to_numeric(df["height"], errors="coerce")
        df["major_ax"] = np.fmax(w, h)
        df["minor_ax"] = np.fmin(w, h)


def promote_refined_to_eye_data(
    block_path: Path | str,
    side: str,
    refined_path: Path | str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Copy refined geometry into ``{side}_eye_data.csv`` after a timestamped backup.

    Returns the backup path.
    """
    side = _normalize_side(side)
    analysis = _analysis_dir(block_path)
    eye_path = analysis / f"{side}_eye_data.csv"
    if not eye_path.is_file():
        raise FileNotFoundError(f"Missing {eye_path}")
    refined_path = Path(refined_path)
    if not refined_path.is_file():
        raise FileNotFoundError(f"Missing refined table {refined_path}")

    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = analysis / f"{side}_eye_data_pre_promote_{ts}.csv"
    shutil.copy2(eye_path, backup)

    eye_df = pd.read_csv(eye_path)
    refined = pd.read_csv(refined_path)
    promoted = overlay_refined_geometry(eye_df, refined)
    drop_refined = [c for c in REFINED_GEOMETRY_COLS if c in promoted.columns]
    if drop_refined:
        promoted = promoted.drop(columns=drop_refined)
    promoted.to_csv(eye_path, index=False)
    return backup


JITTER_NOT_CORRECTED_SUFFIX = "JitterNotCorrected"


def rotation_fixed_eye_csv_path(
    block_path: Path | str,
    side: str,
    *,
    jitter_corrected: bool = True,
) -> Path:
    side = _normalize_side(side)
    analysis = _analysis_dir(block_path)
    if jitter_corrected:
        return analysis / f"{side}_rotation_fixed_eye_data.csv"
    return analysis / f"{side}_rotation_fixed_eye_data_{JITTER_NOT_CORRECTED_SUFFIX}.csv"


def resolve_rotation_fixed_eye_csv(
    block_path: Path | str, side: str
) -> Path | None:
    """Prefer jitter-applied CSV; fall back to the untagged-jitter sidecar."""
    tagged = rotation_fixed_eye_csv_path(block_path, side, jitter_corrected=True)
    if tagged.is_file():
        return tagged
    untagged = rotation_fixed_eye_csv_path(block_path, side, jitter_corrected=False)
    if untagged.is_file():
        return untagged
    return None


def write_rotation_fixed_eye_table(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_rotation_fixed_eye_table(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def attach_spin_columns(eye_df: pd.DataFrame, spin: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``*_spin`` columns onto ``eye_df`` without changing originals."""
    out = eye_df.copy()
    key = None
    for candidate in ("eye_frame", "OE_timestamp"):
        if candidate in out.columns and candidate in spin.columns:
            key = candidate
            break
    cols = [c for c in SPIN_GEOMETRY_COLS if c in spin.columns]
    if not cols:
        return out
    out = out.drop(columns=[c for c in cols if c in out.columns], errors="ignore")
    if key is not None:
        src = spin[[key, *cols]].drop_duplicates(subset=key)
        return out.merge(src, on=key, how="left")
    for col in cols:
        out[col] = spin[col].to_numpy()[: len(out)]
    return out
