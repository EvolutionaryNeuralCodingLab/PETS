"""Optional head-movement labels for saccade events."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec

logger = logging.getLogger(__name__)


def has_imu(spec: BlockSpec) -> bool:
    imu = spec.block_path / "imu"
    return imu.is_dir() and any(imu.iterdir())


def find_lizmov_mat(spec: BlockSpec) -> Path | None:
    """
    Locate Mark's ``lizMov.mat`` under the block's Open Ephys tree.

    Notebook path pattern::
        block.oe_path / 'analysis' / <subdir containing animal id> / lizMov.mat
    On disk this is typically::
        <block>/oe_files/<recording>/Record Node */analysis/*/lizMov.mat
    """
    oe = spec.block_path / "oe_files"
    if not oe.is_dir():
        return None
    matches = sorted(oe.glob("**/lizMov.mat"))
    if not matches:
        return None
    animal_token = spec.animal.replace("_", "")
    preferred = [
        p
        for p in matches
        if spec.animal in p.as_posix() or animal_token in p.as_posix()
    ]
    return (preferred or matches)[0]


def load_lizmov_times_ms(mat_path: Path) -> np.ndarray:
    """Return movement sample times (ms) from ``lizMov.mat`` (HDF5)."""
    import h5py

    with h5py.File(mat_path, "r") as mat:
        t = np.asarray(mat["t_mov_ms"], dtype=float).reshape(-1)
        mov = np.asarray(mat["movAll"], dtype=float).reshape(-1)
    # Keep times where movement flag is on (movAll > 0), else all sample times.
    if mov.size == t.size and np.nanmax(mov) > 0:
        return t[mov > 0]
    return t


def _label_from_mov_times(events: pd.DataFrame, mov_times: np.ndarray) -> pd.Series:
    """True if any movement sample falls inside [saccade_on_ms, saccade_off_ms]."""
    if events.empty:
        return pd.Series(dtype=bool)
    mov_times = np.asarray(mov_times, dtype=float)
    mov_times = mov_times[np.isfinite(mov_times)]
    flags = []
    for _, row in events.iterrows():
        on = float(row["saccade_on_ms"])
        off = float(row["saccade_off_ms"])
        if not (np.isfinite(on) and np.isfinite(off)):
            flags.append(False)
            continue
        # Binary search window overlap with point samples.
        i0 = np.searchsorted(mov_times, on, side="left")
        i1 = np.searchsorted(mov_times, off, side="right")
        flags.append(i1 > i0)
    return pd.Series(flags, index=events.index, dtype=bool)


def label_saccades_head_movement(
    events: pd.DataFrame,
    spec: BlockSpec,
) -> pd.DataFrame:
    """
    Attach ``head_movement`` column.

    Prefer ``lizMov.mat`` under ``oe_files/`` (paper notebook path). Fall back to
    precomputed movement CSVs; otherwise NaN.
    """
    out = events.copy()
    if out.empty:
        out["head_movement"] = pd.Series(dtype="object")
        return out

    mat_path = find_lizmov_mat(spec)
    if mat_path is not None:
        try:
            mov_times = load_lizmov_times_ms(mat_path)
            # Sort for searchsorted
            mov_times = np.sort(mov_times[np.isfinite(mov_times)])
            out["head_movement"] = _label_from_mov_times(out, mov_times).to_numpy()
            print(
                f"[{spec.block_key}] labeled head_movement from lizMov.mat "
                f"({mat_path.name}; n_mov_samples={mov_times.size})"
            )
            return out
        except Exception as exc:
            logger.warning(
                "%s: failed reading %s (%s); trying CSV fallback",
                spec.block_key,
                mat_path,
                exc,
            )

    candidates = [
        spec.analysis_path / "head_movement_events.csv",
        spec.analysis_path / "lizard_movement_events.csv",
        spec.block_path / "imu" / "head_movement_events.csv",
    ]
    mov_path = next((p for p in candidates if p.exists()), None)
    if mov_path is None:
        out["head_movement"] = np.nan
        msg = (
            f"[{spec.block_key}] no head-movement table / lizMov.mat; "
            f"head_movement=NaN (imu_dir_present={has_imu(spec)})"
        )
        print(msg)
        logger.info(msg)
        return out

    mov = pd.read_csv(mov_path)
    start_col = next(
        (c for c in ("start_ms", "on_ms", "saccade_on_ms") if c in mov.columns), None
    )
    end_col = next(
        (c for c in ("end_ms", "off_ms", "saccade_off_ms") if c in mov.columns), None
    )
    if start_col is None or end_col is None:
        out["head_movement"] = np.nan
        logger.warning("%s: movement file %s missing start/end columns", spec.block_key, mov_path)
        return out

    starts = mov[start_col].to_numpy(dtype=float)
    ends = mov[end_col].to_numpy(dtype=float)
    flags = []
    for _, row in out.iterrows():
        on = float(row["saccade_on_ms"])
        off = float(row["saccade_off_ms"])
        overlap = np.any((starts <= off) & (ends >= on))
        flags.append(bool(overlap))
    out["head_movement"] = flags
    print(f"[{spec.block_key}] labeled head_movement from {mov_path.name}")
    return out
