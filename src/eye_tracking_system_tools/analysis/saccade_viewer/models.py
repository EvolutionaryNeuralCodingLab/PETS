"""Event models and normalization for the saccade verification viewer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec

VerificationStatus = Literal["good", "bad", "unset"]
PairingTag = Literal["unset", "monocular", "concurrent"]

_IDENTITY_COLS = ("animal", "block", "eye", "onset_ms", "off_ms")
_ONSET_ALIASES = ("onset_ms", "saccade_on_ms", "start_ms", "saccade_start_timestamp")
_OFF_ALIASES = ("off_ms", "saccade_off_ms", "end_ms", "saccade_end_timestamp")


class EventSelector(Protocol):
    """Phase 2 hook: ROI / outlier pickers return an event DataFrame."""

    def select_events(self, tables: Any) -> pd.DataFrame: ...


def first_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _normalize_block_num(raw: Any) -> str:
    text = str(raw).strip()
    digits = "".join(c for c in text if c.isdigit())
    if digits:
        return digits.zfill(3)
    return text


def _normalize_eye(raw: Any) -> str:
    text = str(raw).strip().upper()
    if text.startswith("L"):
        return "L"
    if text.startswith("R"):
        return "R"
    return text


def compute_event_id(
    *,
    animal: str,
    block: str,
    eye: str,
    onset_ms: float,
    off_ms: float,
) -> str:
    """Stable hash from event identity key."""
    key = (
        f"{animal}|{block}|{eye}|{onset_ms:.3f}|{off_ms:.3f}"
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class VerificationEvent:
    """One verifiable event with block context."""

    event_id: str
    animal: str
    block: str
    block_key: str
    block_path: Path
    eye: str
    onset_ms: float
    off_ms: float
    verification_status: VerificationStatus = "unset"
    pairing_tag: PairingTag = "unset"
    notes: str = ""
    row_index: int = -1
    extras: dict[str, Any] = field(default_factory=dict)

    def identity_tuple(self) -> tuple[str, str, str, float, float]:
        return (self.animal, self.block, self.eye, self.onset_ms, self.off_ms)

    def to_tag_row(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "animal": self.animal,
            "block": self.block,
            "eye": self.eye,
            "onset_ms": self.onset_ms,
            "off_ms": self.off_ms,
            "verification_status": self.verification_status,
            "pairing_tag": self.pairing_tag,
            "notes": self.notes,
        }


@dataclass
class EventBatch:
    """Normalized events grouped by block."""

    events: list[VerificationEvent]
    specs_by_key: dict[str, BlockSpec]
    source_df: pd.DataFrame

    @property
    def block_keys(self) -> list[str]:
        seen: dict[str, None] = {}
        for ev in self.events:
            seen.setdefault(ev.block_key, None)
        return list(seen.keys())

    def events_for_block(self, block_key: str) -> list[VerificationEvent]:
        return [e for e in self.events if e.block_key == block_key]

    def spec_for(self, block_key: str) -> BlockSpec | None:
        return self.specs_by_key.get(block_key)


def _resolve_block_path(
    row: pd.Series,
    specs_by_key: dict[str, BlockSpec],
    block_key: str,
) -> Path:
    if "block_path" in row.index and pd.notna(row.get("block_path")):
        return Path(str(row["block_path"])).resolve()
    spec = specs_by_key.get(block_key)
    if spec is not None:
        return spec.block_path.resolve()
    raise KeyError(
        f"No block_path for {block_key!r}; pass block_path column or BlockSpec registry"
    )


def _is_scalar_cell(value) -> bool:
    if isinstance(value, (list, dict, np.ndarray)):
        return False
    try:
        return bool(pd.notna(value))
    except (ValueError, TypeError):
        return False


def normalize_event_batch(
    df: pd.DataFrame,
    specs: list[BlockSpec] | None = None,
    *,
    status_by_id: dict[str, VerificationStatus] | None = None,
) -> EventBatch:
    """
    Coerce an input event table into :class:`VerificationEvent` rows.

    Required columns (aliases accepted for onset/off):
    ``animal``, ``block``, and an onset time column.
    """
    if df is None or df.empty:
        raise ValueError("events DataFrame is empty")

    work = df.reset_index(drop=True).copy()
    if "animal" not in work.columns:
        raise KeyError("events must include an 'animal' column")
    if "block" not in work.columns:
        raise KeyError("events must include a 'block' column")

    onset_col = first_col(work, _ONSET_ALIASES)
    if onset_col is None:
        raise KeyError(
            f"events need an onset column; tried {_ONSET_ALIASES}"
        )
    off_col = first_col(work, _OFF_ALIASES)

    specs_by_key = {s.block_key: s for s in (specs or [])}
    if not specs_by_key and "block_path" not in work.columns:
        raise ValueError(
            "Provide BlockSpec registry and/or a block_path column per row"
        )

    status_by_id = status_by_id or {}
    events: list[VerificationEvent] = []
    skip_cols = {
        "animal",
        "block",
        "block_path",
        onset_col,
        off_col,
        "eye",
        "verification_status",
    }

    for idx, row in work.iterrows():
        animal = str(row["animal"]).strip()
        block = _normalize_block_num(row["block"])
        block_key = f"{animal}_block_{block}"
        eye = _normalize_eye(row["eye"]) if "eye" in row.index and pd.notna(row.get("eye")) else ""
        onset_ms = float(row[onset_col])
        if off_col and pd.notna(row.get(off_col)):
            off_ms = float(row[off_col])
        else:
            off_ms = onset_ms
        if not (np.isfinite(onset_ms) and np.isfinite(off_ms)):
            continue

        block_path = _resolve_block_path(row, specs_by_key, block_key)
        event_id = compute_event_id(
            animal=animal,
            block=block,
            eye=eye,
            onset_ms=onset_ms,
            off_ms=off_ms,
        )
        status: VerificationStatus = "unset"
        if "verification_status" in row.index and pd.notna(row.get("verification_status")):
            raw = str(row["verification_status"]).strip().lower()
            if raw in {"good", "bad", "unset"}:
                status = raw  # type: ignore[assignment]
        elif event_id in status_by_id:
            status = status_by_id[event_id]

        extras = {
            k: row[k]
            for k in work.columns
            if k not in skip_cols and _is_scalar_cell(row[k])
        }
        events.append(
            VerificationEvent(
                event_id=event_id,
                animal=animal,
                block=block,
                block_key=block_key,
                block_path=block_path,
                eye=eye,
                onset_ms=onset_ms,
                off_ms=off_ms,
                verification_status=status,
                row_index=int(idx),
                extras=extras,
            )
        )

    if not events:
        raise ValueError("No valid events after normalization")

    # Fill specs_by_key from events when registry was partial.
    for ev in events:
        if ev.block_key not in specs_by_key:
            specs_by_key[ev.block_key] = BlockSpec(
                animal=ev.animal,
                block_path=ev.block_path,
                block_num=ev.block,
            )

    return EventBatch(events=events, specs_by_key=specs_by_key, source_df=work)


def add_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``event_id`` column from identity columns (in-place copy)."""
    out = df.copy()
    onset_col = first_col(out, _ONSET_ALIASES)
    if onset_col is None:
        raise KeyError(f"onset column required; tried {_ONSET_ALIASES}")
    off_col = first_col(out, _OFF_ALIASES)
    ids = []
    for row in out.itertuples(index=False):
        d = row._asdict()
        animal = str(d["animal"])
        block = _normalize_block_num(d["block"])
        eye = _normalize_eye(d.get("eye", "")) if "eye" in out.columns else ""
        onset = float(d[onset_col])
        off = float(d[off_col]) if off_col and off_col in d and pd.notna(d[off_col]) else onset
        ids.append(
            compute_event_id(
                animal=animal,
                block=block,
                eye=eye,
                onset_ms=onset,
                off_ms=off,
            )
        )
    out["event_id"] = ids
    return out


def count_by_status(events: list[VerificationEvent]) -> dict[str, int]:
    counts = {"good": 0, "bad": 0, "unset": 0}
    for ev in events:
        counts[ev.verification_status] = counts.get(ev.verification_status, 0) + 1
    return counts
