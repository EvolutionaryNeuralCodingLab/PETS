"""Shared helpers for building viewer-compatible event tables."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

_PROFILE_LIKE = frozenset(
    {
        "speed_profile_pixel",
        "speed_profile_pixel_calib",
        "speed_profile_angular",
        "diameter_profile",
    }
)


def row_block_key(animal: Any, block: Any) -> str:
    """Rebuild ``BlockSpec.block_key`` from event-table ``animal`` / ``block`` columns."""
    animal_s = str(animal)
    block_s = str(block)
    digits = "".join(c for c in block_s if c.isdigit())
    block_num = digits.zfill(3) if digits else block_s
    return f"{animal_s}_block_{block_num}"


def numeric_event_columns(df: pd.DataFrame | None) -> list[str]:
    """Scalar numeric columns suitable for threshold filtering."""
    if df is None or df.empty:
        return []
    out: list[str] = []
    for col in df.columns:
        name = str(col)
        if name in _PROFILE_LIKE or name.startswith("speed_profile"):
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            out.append(name)
        else:
            sample = series.dropna().head(32)
            if sample.empty:
                continue
            try:
                vals = pd.to_numeric(sample, errors="coerce")
            except Exception:
                continue
            if vals.notna().all():
                out.append(name)
    return sorted(out)


def enrich_events_for_viewer(
    events: pd.DataFrame,
    path_by_key: Mapping[str, str] | None = None,
    *,
    drop_internal_cols: bool = True,
) -> pd.DataFrame:
    """Attach ``block_path`` when missing and drop internal selector columns."""
    if events is None or events.empty:
        return pd.DataFrame()

    out = events.copy()
    mapping = dict(path_by_key or {})
    if mapping and ("block_path" not in out.columns or out["block_path"].isna().any()):
        if "animal" in out.columns and "block" in out.columns:
            paths = [
                mapping.get(row_block_key(animal, block), "")
                for animal, block in zip(out["animal"], out["block"])
            ]
            out["block_path"] = paths

    if drop_internal_cols:
        for col in ("right_peak_v", "left_peak_v", "weight", "block_key"):
            if col in out.columns:
                out = out.drop(columns=[col])
    return out.reset_index(drop=True)


def unique_events_by_identity(events: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate pooled ROI selections on animal/block/eye/onset."""
    if events.empty:
        return events.copy()
    cols = ["animal", "block", "eye", "saccade_on_ms"]
    if not all(c in events.columns for c in cols):
        return events.drop_duplicates().reset_index(drop=True)
    return events.drop_duplicates(subset=cols, keep="last").reset_index(drop=True)
