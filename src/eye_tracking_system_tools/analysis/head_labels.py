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


def load_lizmov_samples(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_mov_ms, movAll)`` from ``lizMov.mat`` with no movement filter."""
    import h5py

    with h5py.File(mat_path, "r") as mat:
        t = np.asarray(mat["t_mov_ms"], dtype=float).reshape(-1)
        mov = np.asarray(mat["movAll"], dtype=float).reshape(-1)
    return t, mov


def load_lizmov_times_ms(mat_path: Path) -> np.ndarray:
    """Return movement sample times (ms) from ``lizMov.mat`` (HDF5)."""
    t, mov = load_lizmov_samples(mat_path)
    # Keep times where movement flag is on (movAll > 0), else all sample times.
    if mov.size == t.size and np.nanmax(mov) > 0:
        return t[mov > 0]
    return t


def bout_onsets_from_times(
    times: np.ndarray,
    *,
    gap_ms: float = 40.0,
) -> np.ndarray:
    """Split movement sample times into bouts; return each bout's first sample.

    ``lizMov.mat`` stores *sparse* times where the accel envelope exceeded
    threshold (typically 4 ms samples). ``movAll`` at those times is almost
    always > 0, so a rising-edge of ``movAll > 0`` collapses to one onset at
    the first sample. Consecutive samples more than ``gap_ms`` apart start a
    new bout.
    """
    t = np.sort(np.asarray(times, dtype=float))
    t = t[np.isfinite(t)]
    if t.size <= 1:
        return t
    starts = np.concatenate([[True], np.diff(t) > float(gap_ms)])
    return t[starts]


def load_lizmov_bout_onsets_ms(
    mat_path: Path,
    *,
    gap_ms: float = 40.0,
) -> np.ndarray:
    """Head-bout onsets from ``lizMov.mat`` (gap-clustered ``t_mov_ms``)."""
    times = load_lizmov_times_ms(mat_path)
    return bout_onsets_from_times(times, gap_ms=gap_ms)


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


def refresh_event_tables_head_labels(tables: "EventTables") -> "EventTables":
    """
    Re-apply ``head_movement`` from ``lizMov.mat`` (or CSV fallbacks) for every block.

    Updates each :class:`~pipeline.BlockBundle` and the pooled ``all_saccades`` frame.
    Use when finalized on-disk events lack the column or you want labels refreshed
    from the current ``lizMov.mat`` on disk.
    """
    from dataclasses import replace

    from eye_tracking_system_tools.analysis.pipeline import EventTables

    if not isinstance(tables, EventTables):
        raise TypeError(f"expected EventTables, got {type(tables)!r}")

    new_blocks = []
    parts: list[pd.DataFrame] = []
    for bundle in tables.blocks:
        ev = bundle.all_saccades.copy()
        if ev.empty:
            new_blocks.append(bundle)
            continue
        ev = label_saccades_head_movement(ev, bundle.spec)
        parts.append(ev)
        if "eye" in ev.columns:
            l_ev = ev[ev["eye"].astype(str) == "L"].reset_index(drop=True)
            r_ev = ev[ev["eye"].astype(str) == "R"].reset_index(drop=True)
        else:
            l_ev = pd.DataFrame()
            r_ev = pd.DataFrame()
        new_blocks.append(
            replace(
                bundle,
                all_saccades=ev,
                l_saccades=l_ev,
                r_saccades=r_ev,
            )
        )
    all_saccades = (
        pd.concat(parts, ignore_index=True) if parts else tables.all_saccades.copy()
    )
    return replace(tables, blocks=new_blocks, all_saccades=all_saccades)
