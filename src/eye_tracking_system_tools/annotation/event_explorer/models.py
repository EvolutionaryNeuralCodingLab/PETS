"""Data models for the Event Explorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eye_tracking_system_tools.annotation.block_annotator.models import AnnotationEvent

EXPLORER_VERSION = "0.1.0"
SESSION_SCHEMA_VERSION = 1
EXPORT_SCHEMA_VERSION = 1

DEFAULT_HALF_WINDOW_MS = 100.0
DEFAULT_OE_CHANNELS = [1]


class BlockStatus(Flag):
    NONE = 0
    HAS_LE = auto()
    HAS_RE = auto()
    HAS_OE = auto()
    STALE_SYNC = auto()
    LOAD_FAILED = auto()
    SKIPPED = auto()


@dataclass
class EventRecord:
    """One catalog row (one annotated event)."""

    event_id: str
    event_type: str
    animal_call: str
    experiment_date: str | None
    block_num: str
    timepoint_ms: float
    start_ms: float
    end_ms: float
    row_index: int | None
    arena_frame: int | None
    l_eye_frame: int | None
    r_eye_frame: int | None
    note: str
    annotation_path: Path
    block_path: Path
    block_status: BlockStatus = BlockStatus.NONE

    @classmethod
    def from_annotation(
        cls,
        event: AnnotationEvent,
        *,
        animal_call: str,
        experiment_date: str | None,
        block_num: str,
        block_path: Path,
        annotation_path: Path,
    ) -> EventRecord:
        return cls(
            event_id=event.id,
            event_type=event.event_type,
            animal_call=animal_call,
            experiment_date=experiment_date,
            block_num=block_num,
            timepoint_ms=event.timepoint_ms,
            start_ms=event.start_ms,
            end_ms=event.end_ms,
            row_index=event.row_index,
            arena_frame=event.arena_frame,
            l_eye_frame=event.l_eye_frame,
            r_eye_frame=event.r_eye_frame,
            note=event.note,
            annotation_path=Path(annotation_path),
            block_path=Path(block_path),
        )


@dataclass
class ColumnMap:
    pupil: str | None = None
    l_degrees: str | None = None
    r_degrees: str | None = None


@dataclass
class BlockDataCache:
    block_path: Path
    final_sync_df: pd.DataFrame
    ms_axis: np.ndarray
    sample_rate_hz: float
    le_csv_path: Path | None = None
    re_csv_path: Path | None = None
    le_df: pd.DataFrame | None = None
    re_df: pd.DataFrame | None = None
    le_degrees_df: pd.DataFrame | None = None
    re_degrees_df: pd.DataFrame | None = None
    oe_rec: Any = None
    column_map: ColumnMap = field(default_factory=ColumnMap)
    stale_sync_warn: bool = False
    stale_sync_acknowledged: bool = False
    skipped: bool = False


@dataclass
class EventSnippet:
    event_id: str
    stream_id: str
    time_rel_ms: np.ndarray
    values: np.ndarray
    source: str  # frame | ms_axis_fallback | oe
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportSpec:
    output_dir: Path
    npz_name: str
    json_name: str
    half_window_ms: float
    stream_name: str
    mode: str  # single | average
    normalization: str
    oe_channels: list[int]
