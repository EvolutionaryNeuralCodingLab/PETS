"""Helpers for discovering and selecting DeepLabCut CSV exports per eye folder."""

from __future__ import annotations

import os
from pathlib import Path


def list_dlc_csvs(eye_folder: Path) -> list[Path]:
    """Return DLC CSV paths in ``eye_folder``, sorted for stable UI display."""
    eye_folder = Path(eye_folder)
    if not eye_folder.is_dir():
        return []
    paths = [
        eye_folder / name
        for name in os.listdir(eye_folder)
        if "DLC" in name and name.endswith(".csv")
    ]
    return sorted(paths, key=_dlc_sort_key)


def default_dlc_csv(candidates: list[Path]) -> Path:
    """
    Pick the default DLC CSV.

    Prefer the newest file whose name contains ``filtered``; otherwise the newest overall.
    """
    if not candidates:
        raise FileNotFoundError("No DLC csv files found")
    filtered = [p for p in candidates if "filtered" in p.name.lower()]
    pool = filtered if filtered else candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def resolve_dlc_csv(eye_folder: Path, selected: str | Path | None = None) -> Path:
    """Resolve an explicit DLC path or apply :func:`default_dlc_csv`."""
    candidates = list_dlc_csvs(eye_folder)
    if not candidates:
        raise FileNotFoundError(f"No DLC csv under {eye_folder}")
    if selected is None:
        return default_dlc_csv(candidates)

    path = Path(selected)
    if not path.is_absolute():
        path = Path(eye_folder) / path
    resolved = path.resolve()
    candidate_resolved = {c.resolve() for c in candidates}
    if resolved not in candidate_resolved:
        raise ValueError(
            f"DLC csv {path.name!r} is not among candidates in {eye_folder}: "
            f"{[c.name for c in candidates]}"
        )
    return path if path.is_absolute() else next(c for c in candidates if c.resolve() == resolved)


def _dlc_sort_key(path: Path) -> tuple[int, float, str]:
    filtered_rank = 0 if "filtered" in path.name.lower() else 1
    return (filtered_rank, -path.stat().st_mtime, path.name.lower())
