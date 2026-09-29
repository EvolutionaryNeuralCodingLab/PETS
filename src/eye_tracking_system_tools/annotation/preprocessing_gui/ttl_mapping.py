"""TTL channel mapping helpers for the preprocessing GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eye_tracking_system_tools.annotation.preprocessing_gui.block_session import (
        BlockSyncSession,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync

DEFAULT_CHANNELDICT: dict[int, str] = {
    4: "LED_driver",
    5: "L_eye_TTL",
    1: "Arena_TTL",
    7: "Logical ON/OFF",
    8: "R_eye_TTL",
}


def normalize_channeldict(channeldict: dict[int, str] | dict[str, str] | None) -> dict[int, str]:
    if not channeldict:
        return {}
    return {int(k): str(v) for k, v in channeldict.items()}


def is_default_channeldict(channeldict: dict[int, str] | dict[str, str] | None) -> bool:
    return normalize_channeldict(channeldict) == DEFAULT_CHANNELDICT


def sidecar_path(block_path: Path, oe_dirname: str | None) -> Path | None:
    if not oe_dirname:
        return None
    return Path(block_path) / "oe_files" / oe_dirname / "ttl_manual_mapping.json"


def oe_dirname_for_block(block_path: Path) -> str | None:
    oe_root = Path(block_path) / "oe_files"
    if not oe_root.is_dir():
        return None
    dirs = sorted(p.name for p in oe_root.iterdir() if p.is_dir())
    return dirs[0] if dirs else None


def load_ttl_sidecar(block_path: Path, oe_dirname: str | None) -> dict[str, Any] | None:
    """Load ``manual_line_map`` and ``arena_window`` from a block sidecar, if present."""
    path = sidecar_path(block_path, oe_dirname)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manual_line_map = payload.get("manual_line_map")
        arena_window = payload.get("arena_window")
        if not isinstance(manual_line_map, dict) or not isinstance(arena_window, dict):
            return None
        return {
            "manual_line_map": {str(k): int(v) for k, v in manual_line_map.items()},
            "arena_window": {str(k): int(v) for k, v in arena_window.items()},
            "led_driver_missing": bool(payload.get("led_driver_missing", False)),
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def led_manual_replacement_path(block_path: Path, oe_dirname: str | None) -> Path | None:
    if not oe_dirname:
        return None
    return Path(block_path) / "oe_files" / oe_dirname / "led_manual_replacement.json"


def load_led_manual_replacement(
    block_path: Path, oe_dirname: str | None
) -> dict[str, Any] | None:
    """Load synthetic LED_driver replacement sidecar, if present."""
    path = led_manual_replacement_path(block_path, oe_dirname)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        on_samples = payload.get("on_samples")
        fall_samples = payload.get("fall_samples")
        if not isinstance(on_samples, list) or not isinstance(fall_samples, list):
            return None
        if len(on_samples) == 0:
            return None
        return {
            "first_frame": int(payload.get("first_frame", 0)),
            "interval_s": float(payload.get("interval_s", 60.0)),
            "fps": float(payload.get("fps", 60.0)),
            "on_samples": [int(x) for x in on_samples],
            "fall_samples": [int(x) for x in fall_samples],
            "source": str(payload.get("source", "manual_brightness")),
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def save_led_manual_replacement(
    block_path: Path,
    oe_dirname: str,
    *,
    first_frame: int,
    interval_s: float,
    fps: float,
    on_samples: list[int],
    fall_samples: list[int],
    source: str = "manual_brightness",
) -> Path:
    path = led_manual_replacement_path(block_path, oe_dirname)
    if path is None:
        raise ValueError("oe_dirname is required to save LED manual replacement.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "first_frame": int(first_frame),
        "interval_s": float(interval_s),
        "fps": float(fps),
        "on_samples": [int(x) for x in on_samples],
        "fall_samples": [int(x) for x in fall_samples],
        "source": str(source),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def oe_events_have_led_driver(oe_events: Any) -> bool:
    """True when ``oe_events`` has usable LED_driver (+ fall) columns."""
    if oe_events is None:
        return False
    try:
        cols = oe_events.columns
    except AttributeError:
        return False
    if "LED_driver" not in cols or "LED_driver_fall" not in cols:
        return False
    rise = oe_events["LED_driver"].dropna()
    fall = oe_events["LED_driver_fall"].dropna()
    return len(rise) > 0 and len(fall) > 0


def led_driver_is_ready(blocksync: BlockSync) -> bool:
    """True when ``blocksync.oe_events`` has usable LED_driver (+ fall) columns."""
    return oe_events_have_led_driver(getattr(blocksync, "oe_events", None))


def channeldict_from_line_map(manual_line_map: dict[str, int]) -> dict[int, str]:
    return {int(line): str(role) for role, line in manual_line_map.items()}


def line_map_from_channeldict(channeldict: dict[int, str] | dict[str, str]) -> dict[str, int]:
    return {str(role): int(line) for line, role in normalize_channeldict(channeldict).items()}


def effective_channeldict(blocksync: BlockSync) -> dict[int, str]:
    """Best-known line→role mapping for a block (sidecar overrides ``channeldict``)."""
    oe_dirname = getattr(blocksync, "oe_dirname", None) or oe_dirname_for_block(
        blocksync.block_path
    )
    sidecar = load_ttl_sidecar(blocksync.block_path, oe_dirname)
    if sidecar is not None:
        return channeldict_from_line_map(sidecar["manual_line_map"])
    return normalize_channeldict(getattr(blocksync, "channeldict", None) or DEFAULT_CHANNELDICT)


def effective_line_map(blocksync: BlockSync) -> dict[str, int]:
    return line_map_from_channeldict(effective_channeldict(blocksync))


def line_map_for_block(
    session: BlockSyncSession,
    block: BlockHandle,
) -> dict[str, int] | None:
    """Return a reusable role→line map for *block*, or ``None`` if none is known."""
    oe_dirname = oe_dirname_for_block(block.block_path)
    sidecar = load_ttl_sidecar(block.block_path, oe_dirname)
    if sidecar is not None:
        return dict(sidecar["manual_line_map"])

    if block.channeldict is not None and not is_default_channeldict(block.channeldict):
        return line_map_from_channeldict(block.channeldict)

    if session.has(block):
        channeldict = normalize_channeldict(session.get(block).channeldict)
        if channeldict and not is_default_channeldict(channeldict):
            return line_map_from_channeldict(channeldict)

    return None


def list_mapping_sources(
    session: BlockSyncSession,
    blocks: list[BlockHandle],
    exclude: BlockHandle,
) -> list[tuple[str, dict[str, int]]]:
    """Sibling blocks that expose a known TTL line mapping for the leech dropdown."""
    exclude_key = str(Path(exclude.block_path).resolve())
    sources: list[tuple[str, dict[str, int]]] = []
    for block in blocks:
        if str(Path(block.block_path).resolve()) == exclude_key:
            continue
        line_map = line_map_for_block(session, block)
        if line_map is None:
            continue
        sources.append((block.display_label, line_map))
    return sources
