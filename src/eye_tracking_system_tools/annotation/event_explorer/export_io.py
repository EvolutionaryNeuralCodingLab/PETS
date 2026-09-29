"""Export selection to NPZ + sidecar JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from eye_tracking_system_tools.annotation.event_explorer.models import (
    EXPLORER_VERSION,
    EXPORT_SCHEMA_VERSION,
    EventRecord,
    EventSnippet,
)
from eye_tracking_system_tools.annotation.event_explorer.load_log import LoadLog
from eye_tracking_system_tools.annotation.event_explorer.snippet_extractor import (
    STREAM_LABELS,
)


def export_bundle(
    output_dir: Path,
    *,
    npz_name: str,
    json_name: str,
    mode: str,
    stream_name: str,
    half_window_ms: float,
    normalization: str,
    records: list[EventRecord],
    snippets: list[EventSnippet],
    time_grid: np.ndarray | None,
    trials: np.ndarray | None,
    mean: np.ndarray | None,
    sem: np.ndarray | None,
    load_log: LoadLog,
    oe_channels: list[int],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / npz_name
    json_path = output_dir / json_name

    payload: dict[str, Any] = {
        "time_rel_ms": time_grid if time_grid is not None else np.array([]),
        "trial_event_ids": np.array([r.event_id for r in records], dtype=object),
        "stream_name": stream_name,
        "mode": mode,
        "normalization": normalization,
    }
    if trials is not None:
        payload["trials"] = trials
    if mean is not None:
        payload["mean"] = mean
    if sem is not None:
        payload["sem"] = sem
    if mode == "single" and snippets:
        payload["values"] = snippets[0].values
        payload["time_rel_ms"] = snippets[0].time_rel_ms

    np.savez_compressed(npz_path, **payload)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sidecar = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "explorer_version": EXPLORER_VERSION,
        "export_timestamp": now,
        "mode": mode,
        "stream_name": stream_name,
        "stream_label": STREAM_LABELS.get(stream_name, stream_name),
        "half_window_ms": half_window_ms,
        "normalization": normalization,
        "alignment": "timepoint_ms=0",
        "oe_channels": oe_channels,
        "events": [
            {
                "event_id": r.event_id,
                "event_type": r.event_type,
                "timepoint_ms": r.timepoint_ms,
                "block_path": str(r.block_path),
                "annotation_path": str(r.annotation_path),
            }
            for r in records
        ],
        "snippet_meta": [s.meta for s in snippets],
        "load_log_excerpt": load_log.excerpt(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    return npz_path, json_path
