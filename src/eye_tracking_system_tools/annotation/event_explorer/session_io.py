"""Save/load explorer session JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eye_tracking_system_tools.annotation.event_explorer.models import (
    SESSION_SCHEMA_VERSION,
    ColumnMap,
)


def session_to_dict(
    *,
    sources: dict[str, Any],
    remap_table: dict[str, str],
    visible_columns: list[str],
    filters: dict[str, Any],
    half_window_ms: float,
    stream_toggles: dict[str, bool],
    oe_channels: list[int],
    normalization: str,
    average_stream: str,
    column_overrides: ColumnMap,
    log_file: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "sources": sources,
        "remap_table": remap_table,
        "visible_columns": visible_columns,
        "filters": filters,
        "half_window_ms": half_window_ms,
        "stream_toggles": stream_toggles,
        "oe_channels": oe_channels,
        "normalization": normalization,
        "average_stream": average_stream,
        "column_overrides": {
            "pupil_column": column_overrides.pupil,
            "l_degrees_column": column_overrides.l_degrees,
            "r_degrees_column": column_overrides.r_degrees,
        },
        "log_file": log_file,
    }


def save_session(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_session(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if int(data.get("schema_version", 0)) != SESSION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported session schema in {path}")
    return data


def column_map_from_session(data: dict[str, Any]) -> ColumnMap:
    co = data.get("column_overrides") or {}
    return ColumnMap(
        pupil=co.get("pupil_column"),
        l_degrees=co.get("l_degrees_column"),
        r_degrees=co.get("r_degrees_column"),
    )
