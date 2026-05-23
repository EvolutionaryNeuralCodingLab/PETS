"""Scan and load Block Annotator *_annotations.json into EventRecord catalog."""

from __future__ import annotations

import json
from pathlib import Path

from eye_tracking_system_tools.annotation.block_annotator.models import AnnotationEvent
from eye_tracking_system_tools.annotation.block_annotator.persistence import SCHEMA_VERSION

from eye_tracking_system_tools.annotation.event_explorer.models import EventRecord


def discover_annotation_files(
    *,
    json_paths: list[Path] | None = None,
    json_list_file: Path | None = None,
    scan_dirs: list[Path] | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Collect unique annotation JSON paths from CLI / dialog inputs."""
    found: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        p = Path(p).resolve()
        if p.suffix.lower() != ".json":
            return
        if not p.name.endswith("_annotations.json"):
            return
        if p not in seen:
            seen.add(p)
            found.append(p)

    for p in json_paths or []:
        if Path(p).is_file():
            add(p)

    if json_list_file and Path(json_list_file).is_file():
        for line in Path(json_list_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                add(Path(line))

    for root in scan_dirs or []:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file() and root.name.endswith("_annotations.json"):
            add(root)
            continue
        pattern = "**/*_annotations.json" if recursive else "*_annotations.json"
        for p in sorted(root.glob(pattern)):
            add(p)

    return found


def load_events_from_file(path: Path) -> list[EventRecord]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    version = int(data.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {version} in {path} (expected {SCHEMA_VERSION})"
        )

    block_path = Path(data["block_path"])
    animal = str(data.get("animal_call", ""))
    exp_date = data.get("experiment_date")
    block_num = str(data.get("block_num", ""))

    records: list[EventRecord] = []
    for raw in data.get("events", []):
        ev = AnnotationEvent.from_dict(raw)
        records.append(
            EventRecord.from_annotation(
                ev,
                animal_call=animal,
                experiment_date=exp_date,
                block_num=block_num,
                block_path=block_path,
                annotation_path=path,
            )
        )
    return records


def build_catalog(paths: list[Path]) -> list[EventRecord]:
    """Load many annotation files; later files do not dedupe events by id."""
    catalog: list[EventRecord] = []
    for p in paths:
        catalog.extend(load_events_from_file(p))
    return catalog


def event_types_in_catalog(catalog: list[EventRecord]) -> list[str]:
    types = sorted({r.event_type for r in catalog}, key=str.lower)
    return types
