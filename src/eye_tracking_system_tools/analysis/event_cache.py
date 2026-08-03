"""
Disk cache for :class:`EventTables` (events only — no per-frame traces).

Key is ``sha1(sorted block_keys + saccade/binocular params)``. After a kernel
restart event-only figures are instant; Fig 2c/2d, 2f, and 3* reload traces
for selected blocks via :func:`reload_traces`.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any, Collection

import yaml

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.eye_trace_io import load_block_eyes
from eye_tracking_system_tools.analysis.pipeline import (
    BlockBundle,
    EventTables,
    build_event_tables,
    drop_traces,
)

CACHE_VERSION = 2


def cache_key(specs: list[BlockSpec], params: dict[str, Any]) -> str:
    """Stable hash over block paths + detection-relevant params + finalized fingerprints."""
    from eye_tracking_system_tools.analysis.saccade_export import finalized_fingerprint

    payload = {
        "version": CACHE_VERSION,
        "blocks": sorted(str(s.block_path.resolve()) for s in specs),
        "saccade": params.get("saccade", {}),
        "binocular": params.get("binocular", {}),
        "finalized": finalized_fingerprint(specs),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def cache_path(metadata_dir: Path | str, key: str) -> Path:
    return Path(metadata_dir) / "event_cache" / f"{key}.pkl"


def save_event_cache(
    tables: EventTables,
    metadata_dir: Path | str,
    *,
    specs: list[BlockSpec] | None = None,
    key: str | None = None,
) -> Path:
    """Persist an events-only copy of ``tables`` under ``metadata/event_cache/``."""
    metadata_dir = Path(metadata_dir)
    if key is None:
        if specs is None:
            specs = [b.spec for b in tables.blocks]
        key = cache_key(specs, tables.params)
    path = cache_path(metadata_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = drop_traces(tables)
    payload = {
        "cache_version": CACHE_VERSION,
        "key": key,
        "tables": slim,
        "block_keys": [b.spec.block_key for b in slim.blocks],
        "block_paths": {b.spec.block_key: str(b.spec.block_path) for b in slim.blocks},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    sidecar = path.with_suffix(".meta.yaml")
    with open(sidecar, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "cache_version": CACHE_VERSION,
                "key": key,
                "n_blocks": len(slim.blocks),
                "n_all_saccades": int(len(slim.all_saccades)),
                "n_synced": int(len(slim.synced)),
                "n_non_synced": int(len(slim.non_synced)),
            },
            f,
            sort_keys=False,
        )
    return path


def load_event_cache(path: Path | str) -> EventTables | None:
    """Load a cached :class:`EventTables`, or ``None`` if missing/incompatible."""
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("cache_version") != CACHE_VERSION:
        return None
    tables = payload.get("tables")
    if not isinstance(tables, EventTables):
        return None
    return tables


def build_or_load_event_tables(
    specs: list[BlockSpec],
    params: dict[str, Any],
    metadata_dir: Path | str,
    *,
    keep_traces: bool = False,
    force: bool = False,
    prefer_finalized: bool = True,
) -> tuple[EventTables, Path, bool]:
    """
    Return ``(tables, cache_path, from_cache)``.

    By default builds with ``keep_traces=False`` and caches that slim copy.
    Pass ``keep_traces=True`` to keep frames in the returned object (still
    caches the slim copy).

    ``prefer_finalized`` forwards to :func:`build_event_tables` (load
    ``analysis/saccades/`` when present).
    """
    metadata_dir = Path(metadata_dir)
    key = cache_key(specs, params)
    path = cache_path(metadata_dir, key)
    if not force:
        cached = load_event_cache(path)
        if cached is not None:
            if keep_traces:
                cached = reload_traces(cached, block_keys=[b.block_key for b in specs])
            # Refresh params from the caller's YAML (figure sections may have changed).
            return replace(cached, params=params), path, True

    tables = build_event_tables(
        specs,
        params=params,
        keep_traces=keep_traces,
        prefer_finalized=prefer_finalized,
    )
    save_event_cache(tables, metadata_dir, specs=specs, key=key)
    return tables, path, False


def reload_traces(
    tables: EventTables,
    *,
    block_keys: Collection[str] | None = None,
) -> EventTables:
    """
    Re-load per-frame eye CSVs for the selected blocks (needed by Fig 2c/2d, 2f, 3*).

    Detection is NOT re-run — only ``left`` / ``right`` frames are filled in.
    """
    wanted = {str(k) for k in block_keys} if block_keys is not None else None
    new_bundles: list[BlockBundle] = []
    for b in tables.blocks:
        if wanted is not None and b.spec.block_key not in wanted:
            new_bundles.append(b)
            continue
        if b.left is not None and not b.left.empty and b.right is not None and not b.right.empty:
            new_bundles.append(b)
            continue
        loaded = load_block_eyes(b.spec)
        # Re-apply the same speed-profile columns the detector would have added,
        # without re-detecting events. Fig 2f needs angular_speed_r + ms_axis.
        left = loaded.left.copy()
        right = loaded.right.copy()
        for df in (left, right):
            if "angular_speed_r" not in df.columns and {"k_phi", "k_theta"}.issubset(df.columns):
                dphi = df["k_phi"].diff()
                dth = df["k_theta"].diff()
                df["angular_speed_r"] = (dphi.pow(2) + dth.pow(2)).pow(0.5)
        new_bundles.append(replace(b, left=left, right=right))
    return replace(tables, blocks=new_bundles)


def ensure_traces_for_blocks(
    tables: EventTables,
    block_keys: Collection[str],
) -> EventTables:
    """Convenience wrapper used by the notebook / GUI before building Fig 2f."""
    return reload_traces(tables, block_keys=block_keys)
