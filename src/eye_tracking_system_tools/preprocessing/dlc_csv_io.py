"""Helpers for discovering and selecting DeepLabCut CSV exports per eye folder."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


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


def _likelihood_columns(data: pd.DataFrame) -> list[str]:
    """Return Pupil*/edge* likelihood column names (every 3rd bodypart field)."""
    cols: list[str] = []
    for token in ("Pupil", "edge"):
        elements = np.array([c for c in data.columns if token in str(c)])
        if len(elements) == 0:
            continue
        cols.extend(elements[np.arange(2, len(elements), 3)].tolist())
    return cols


def load_dlc_likelihood_values(csv_path: Path | str) -> np.ndarray:
    """
    Extract all Pupil/edge likelihood samples from a DLC CSV.

    Matches ``BlockSync.eye_tracking_analysis`` loading: ``header=1``, then
    ``iloc[1:]`` numeric rows, likelihood columns at every 3rd Pupil*/edge* field.
    """
    path = Path(csv_path)
    data = pd.read_csv(path, header=1, low_memory=False)
    data = data.iloc[1:].apply(pd.to_numeric, errors="coerce")
    likelihood_cols = _likelihood_columns(data)
    if not likelihood_cols:
        return np.asarray([], dtype=float)
    values = data[likelihood_cols].to_numpy(dtype=float).ravel()
    return values[~np.isnan(values)]


def load_dlc_likelihood_values_many(csv_paths: list[Path | str]) -> np.ndarray:
    """Concatenate likelihood samples from multiple DLC CSVs."""
    chunks = [load_dlc_likelihood_values(p) for p in csv_paths]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(chunks)


def likelihood_threshold_stats(
    values: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """
    Summarize how many likelihood samples a threshold keeps.

    Uses the same rule as ``eye_tracking_analysis``: keep when ``value > threshold``.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n_total = int(arr.size)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_kept": 0,
            "n_removed": 0,
            "frac_kept": float("nan"),
            "frac_removed": float("nan"),
        }
    n_kept = int(np.sum(arr > float(threshold)))
    n_removed = n_total - n_kept
    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_removed": n_removed,
        "frac_kept": float(n_kept / n_total),
        "frac_removed": float(n_removed / n_total),
    }
