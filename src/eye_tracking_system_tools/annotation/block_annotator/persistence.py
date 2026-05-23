"""Per-block annotation JSON save/load."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eye_tracking_system_tools.annotation.block_annotator.models import (
    AnnotatorConfig,
    AnnotationEvent,
    BlockSession,
)


SCHEMA_VERSION = 1


def annotation_filename(session: BlockSession) -> str:
    date_part = f"{session.experiment_date}_" if session.experiment_date else ""
    return (
        f"{session.animal_call}_{date_part}block_{session.block_num}_annotations.json"
    )


def annotation_path(output_folder: Path, session: BlockSession) -> Path:
    return Path(output_folder) / annotation_filename(session)


def save_annotations(
    output_folder: Path,
    session: BlockSession,
    events: list[AnnotationEvent],
    *,
    created_at: str | None = None,
) -> Path:
    path = annotation_path(output_folder, session)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        created_at = created_at or existing.get("created_at", now)
    else:
        created_at = created_at or now

    payload = {
        "schema_version": SCHEMA_VERSION,
        "animal_call": session.animal_call,
        "experiment_date": session.experiment_date,
        "block_num": session.block_num,
        "block_path": str(session.block_path),
        "created_at": created_at,
        "modified_at": now,
        "sync_source": session.sync_source,
        "sample_rate_hz": session.sample_rate_hz,
        "ms_axis_range": [session.ms_min, session.ms_max],
        "config_snapshot": {"event_types": list(session.config.event_types)},
        "events": [e.to_dict() for e in events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_annotations(output_folder: Path, session: BlockSession) -> list[AnnotationEvent]:
    """Load events from JSON; returns empty list if file missing."""
    path = annotation_path(output_folder, session)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [AnnotationEvent.from_dict(e) for e in data.get("events", [])]
