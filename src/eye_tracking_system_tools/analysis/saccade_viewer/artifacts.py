"""Read/write per-block saccade verification tag artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from eye_tracking_system_tools.analysis.saccade_viewer.models import (
    PairingTag,
    VerificationEvent,
    VerificationStatus,
    add_event_ids,
    compute_event_id,
)

VERIFICATION_DIRNAME = "saccade_verification"
TAGS_CSV = "tags.csv"

_TAG_COLUMNS = (
    "event_id",
    "animal",
    "block",
    "eye",
    "onset_ms",
    "off_ms",
    "verification_status",
    "pairing_tag",
    "verified_at",
    "notes",
)

_VALID_PAIRING = frozenset({"unset", "monocular", "concurrent"})


def verification_dir(block_path: Path | str) -> Path:
    return Path(block_path) / "analysis" / VERIFICATION_DIRNAME


def tags_path(block_path: Path | str) -> Path:
    return verification_dir(block_path) / TAGS_CSV


def _empty_tags() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_TAG_COLUMNS))


def _normalize_pairing(raw) -> PairingTag:
    text = str(raw or "unset").strip().lower()
    if text in {"", "nan", "none"}:
        return "unset"
    if text in _VALID_PAIRING:
        return text  # type: ignore[return-value]
    raise ValueError(f"Invalid pairing_tag: {raw!r}")


def _normalize_tags_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_tags()
    out = df.copy()
    for col in _TAG_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col in {"notes"} else None
    if "event_id" not in out.columns or out["event_id"].isna().all():
        out["event_id"] = [
            compute_event_id(
                animal=str(r.animal),
                block=str(r.block),
                eye=str(r.eye),
                onset_ms=float(r.onset_ms),
                off_ms=float(r.off_ms),
            )
            for r in out.itertuples(index=False)
        ]
    out["verification_status"] = (
        out["verification_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unset", "": "unset"})
    )
    for bad in out.loc[~out["verification_status"].isin({"good", "bad", "unset"}), "verification_status"]:
        raise ValueError(f"Invalid verification_status: {bad!r}")
    out["pairing_tag"] = out["pairing_tag"].map(_normalize_pairing)
    out["notes"] = out["notes"].fillna("").astype(str)
    return out[list(_TAG_COLUMNS)]


def read_verification_tags(block_path: Path | str) -> pd.DataFrame:
    path = tags_path(block_path)
    if not path.is_file():
        return _empty_tags()
    df = pd.read_csv(path)
    return _normalize_tags_df(df)


def tag_row_by_id(df: pd.DataFrame) -> dict[str, dict]:
    if df is None or df.empty:
        return {}
    norm = _normalize_tags_df(df)
    out: dict[str, dict] = {}
    for row in norm.itertuples(index=False):
        out[str(row.event_id)] = {
            "verification_status": str(row.verification_status),
            "pairing_tag": _normalize_pairing(row.pairing_tag),
            "notes": str(row.notes or ""),
        }
    return out


def merge_tag_rows(
    existing: pd.DataFrame,
    updates: Iterable[dict],
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Merge updates into existing tags; newer rows win on event_id."""
    ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    base = _normalize_tags_df(existing)
    upd = _normalize_tags_df(pd.DataFrame(list(updates)))
    if upd.empty:
        return base
    upd = upd.copy()
    upd["verified_at"] = ts
    if base.empty:
        return upd
    merged = pd.concat([base, upd], ignore_index=True)
    merged = merged.drop_duplicates(subset=["event_id"], keep="last")
    return merged.sort_values(["block", "onset_ms"]).reset_index(drop=True)


def save_verification_tags(
    block_path: Path | str,
    events: list[VerificationEvent],
    *,
    merge_existing: bool = True,
) -> Path:
    """Write tags for the given block events (autosave path)."""
    path = tags_path(block_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_verification_tags(block_path) if merge_existing else _empty_tags()
    rows = []
    for ev in events:
        row = ev.to_tag_row()
        row["verified_at"] = ""
        rows.append(row)
    merged = merge_tag_rows(existing, rows)
    merged.to_csv(path, index=False)
    return path


def apply_saved_tags(events: list[VerificationEvent], block_path: Path | str) -> list[VerificationEvent]:
    """Return events with saved verification fields loaded from disk."""
    saved = tag_row_by_id(read_verification_tags(block_path))
    if not saved:
        return events
    out: list[VerificationEvent] = []
    for ev in events:
        row = saved.get(ev.event_id, {})
        out.append(
            VerificationEvent(
                event_id=ev.event_id,
                animal=ev.animal,
                block=ev.block,
                block_key=ev.block_key,
                block_path=ev.block_path,
                eye=ev.eye,
                onset_ms=ev.onset_ms,
                off_ms=ev.off_ms,
                verification_status=row.get("verification_status", ev.verification_status),  # type: ignore[arg-type]
                pairing_tag=row.get("pairing_tag", ev.pairing_tag),  # type: ignore[arg-type]
                notes=row.get("notes", ev.notes),
                row_index=ev.row_index,
                extras=dict(ev.extras),
            )
        )
    return out


def merge_verification_tags(
    events_df: pd.DataFrame,
    block_paths: dict[str, Path] | None = None,
    *,
    default_status: VerificationStatus = "unset",
    default_pairing: PairingTag = "unset",
) -> pd.DataFrame:
    """
    Join verification columns from per-block tag CSVs onto an event table.

    Adds ``verification_status``, ``pairing_tag``, and ``notes``.
    """
    if events_df is None or events_df.empty:
        return events_df.copy() if events_df is not None else pd.DataFrame()

    out = events_df.copy()
    block_paths = block_paths or {}

    if "event_id" not in out.columns:
        try:
            out = add_event_ids(out)
        except KeyError:
            out["verification_status"] = default_status
            out["pairing_tag"] = default_pairing
            out["notes"] = ""
            return out

    tag_frames: dict[str, pd.DataFrame] = {}
    for _, row in out.iterrows():
        if "block_path" in out.columns and pd.notna(row.get("block_path")):
            bp = str(row["block_path"])
        else:
            animal = str(row["animal"])
            block = str(row["block"]).zfill(3) if str(row["block"]).isdigit() else str(row["block"])
            key = f"{animal}_block_{block}"
            if key not in block_paths:
                continue
            bp = str(block_paths[key])
        if bp not in tag_frames:
            tag_frames[bp] = read_verification_tags(bp)

    status_by_id: dict[str, str] = {}
    pairing_by_id: dict[str, str] = {}
    notes_by_id: dict[str, str] = {}
    for tags in tag_frames.values():
        for trow in tags.itertuples(index=False):
            eid = str(trow.event_id)
            status_by_id[eid] = str(trow.verification_status)
            pairing_by_id[eid] = str(trow.pairing_tag)
            notes_by_id[eid] = str(trow.notes or "")

    out["verification_status"] = out["event_id"].map(status_by_id).fillna(default_status)
    out["pairing_tag"] = out["event_id"].map(pairing_by_id).fillna(default_pairing)
    out["notes"] = out["event_id"].map(notes_by_id).fillna("")
    return out


def verification_filter_good_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep events that are not explicitly marked bad."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    if "verification_status" not in df.columns:
        return df.copy()
    mask = df["verification_status"].astype(str).str.lower() != "bad"
    return df.loc[mask].copy()
