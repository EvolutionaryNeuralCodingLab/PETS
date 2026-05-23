"""Data models for the Block Annotator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AnnotatorConfig:
    event_types: list[str] = field(
        default_factory=lambda: ["saccade", "blink", "noise", "pupil event"]
    )
    default_range_half_width_ms: float = 100.0
    playback_fps: float = 60.0
    step_rows: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_types": list(self.event_types),
            "default_range_half_width_ms": self.default_range_half_width_ms,
            "playback_fps": self.playback_fps,
            "step_rows": self.step_rows,
        }

    def add_event_type(self, name: str) -> bool:
        """Append a new event type (case-insensitive duplicate check)."""
        name = name.strip()
        if not name:
            return False
        if any(t.lower() == name.lower() for t in self.event_types):
            return False
        self.event_types.append(name)
        return True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotatorConfig:
        types = data.get("event_types") or ["saccade", "blink", "noise", "pupil event"]
        if types and isinstance(types[0], dict):
            types = [t.get("name", str(t)) for t in types]
        return cls(
            event_types=list(types),
            default_range_half_width_ms=float(
                data.get("default_range_half_width_ms", 100.0)
            ),
            playback_fps=float(data.get("playback_fps", 60.0)),
            step_rows=int(data.get("step_rows", 1)),
        )


@dataclass
class AnnotationEvent:
    event_type: str
    timepoint_ms: float
    start_ms: float
    end_ms: float
    range_half_width_ms: float = 100.0
    row_index: int | None = None
    arena_frame: int | None = None
    l_eye_frame: int | None = None
    r_eye_frame: int | None = None
    note: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "timepoint_ms": self.timepoint_ms,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "range_half_width_ms": self.range_half_width_ms,
            "row_index": self.row_index,
            "arena_frame": self.arena_frame,
            "l_eye_frame": self.l_eye_frame,
            "r_eye_frame": self.r_eye_frame,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnnotationEvent:
        return cls(
            id=str(data.get("id", uuid.uuid4())),
            event_type=str(data["event_type"]),
            timepoint_ms=float(data["timepoint_ms"]),
            start_ms=float(data["start_ms"]),
            end_ms=float(data["end_ms"]),
            range_half_width_ms=float(data.get("range_half_width_ms", 100.0)),
            row_index=data.get("row_index"),
            arena_frame=data.get("arena_frame"),
            l_eye_frame=data.get("l_eye_frame"),
            r_eye_frame=data.get("r_eye_frame"),
            note=str(data.get("note", "")),
        )


@dataclass
class BlockSession:
    """Loaded block with master timeline and media paths."""

    animal_call: str
    experiment_date: str | None
    block_num: str
    block_path: Path
    output_folder: Path
    config: AnnotatorConfig
    final_sync_df: pd.DataFrame
    ms_axis: np.ndarray
    sample_rate_hz: float
    arena_videos: list[Path]
    le_videos: list[Path]
    re_videos: list[Path]
    le_ellipse_df: pd.DataFrame | None = None
    re_ellipse_df: pd.DataFrame | None = None
    oe_rec: Any = None
    sync_source: str = "final_sync_df.csv"

    @property
    def n(self) -> int:
        return len(self.final_sync_df)

    @property
    def ms_min(self) -> float:
        return float(self.ms_axis[0]) if self.n else 0.0

    @property
    def ms_max(self) -> float:
        return float(self.ms_axis[-1]) if self.n else 0.0

    def row_at(self, index: int) -> pd.Series:
        return self.final_sync_df.iloc[int(index)]

    def ms_at(self, index: int) -> float:
        return float(self.ms_axis[int(index)])

    def frame_ids_at(self, index: int) -> tuple[int | None, int | None, int | None]:
        row = self.row_at(index)
        return (
            _safe_frame(row.get("Arena_frame")),
            _safe_frame(row.get("L_eye_frame")),
            _safe_frame(row.get("R_eye_frame")),
        )


# Open Ephys / pandas missing-frame sentinel (INT64_MIN written to CSV).
_INVALID_FRAME_SENTINEL = -9223372036854775808


def _safe_frame(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    iv = int(round(v))
    if iv < 0:
        return None
    if iv == _INVALID_FRAME_SENTINEL or iv < -1_000_000:
        return None
    if iv > 50_000_000:
        return None
    return iv


def first_timeline_index_with_frame(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("Arena_frame", "L_eye_frame", "R_eye_frame"),
) -> int:
    """First useful scrub index: prefer arena + at least one eye, else any valid frame."""
    arena_col, le_col, re_col = "Arena_frame", "L_eye_frame", "R_eye_frame"
    for i in range(len(df)):
        row = df.iloc[i]
        arena = _safe_frame(row.get(arena_col)) if arena_col in df.columns else None
        le = _safe_frame(row.get(le_col)) if le_col in df.columns else None
        re = _safe_frame(row.get(re_col)) if re_col in df.columns else None
        if arena is not None and (le is not None or re is not None):
            return i
    for i in range(len(df)):
        row = df.iloc[i]
        for col in columns:
            if col not in df.columns:
                continue
            if _safe_frame(row.get(col)) is not None:
                return i
    return 0


def compute_ms_axis(df: pd.DataFrame, sample_rate_hz: float) -> np.ndarray:
    """Master timeline: Arena_TTL sample indices → milliseconds."""
    if "ms_axis" in df.columns:
        return df["ms_axis"].to_numpy(dtype=np.float64)
    ttl = df["Arena_TTL"].to_numpy(dtype=np.float64)
    return ttl / (sample_rate_hz / 1000.0)
