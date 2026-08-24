"""Shared helpers for building viewer-compatible event tables."""

from __future__ import annotations

import pandas as pd

from eye_tracking_system_tools.analysis.pipeline import EventTables, _row_block_key

_PROFILE_LIKE = frozenset(
    {
        "speed_profile_pixel",
        "speed_profile_pixel_calib",
        "speed_profile_angular",
        "diameter_profile",
    }
)


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
    tables: EventTables,
    events: pd.DataFrame,
    *,
    drop_internal_cols: bool = True,
) -> pd.DataFrame:
    """Attach ``block_path`` and drop internal selector columns."""
    if events is None or events.empty:
        return pd.DataFrame()

    out = events.copy()
    path_by_key = {b.spec.block_key: str(b.spec.block_path) for b in tables.blocks}
    if "block_path" not in out.columns or out["block_path"].isna().any():
        paths = []
        for animal, block in zip(out["animal"], out["block"]):
            key = _row_block_key(animal, block)
            paths.append(path_by_key.get(key, ""))
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
