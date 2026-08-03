"""Per-block behavior-state timelines (``analysis/block_<num>_behavior_state.csv``).

Produced by the preprocessing GUI's Behavior tab (accelerometer → rolling
average → threshold → collapsed segments, see
``annotation.preprocessing_gui.tabs.behavior_tab`` /
``preprocessing.notebook_helpers.create_behavior_df``). Columns are
``start_time``, ``end_time``, ``annotation`` (times in **milliseconds**); some
exports additionally carry redundant ``start_time_ms`` / ``end_time_ms``
columns.

Coverage across the 23 paper blocks: 21/23 have it under ``analysis/``;
``PV_62/block_038`` has no annotation at all; ``PV_126/block_007`` is archived
under ``analysis/older_analyses/`` instead of ``analysis/``.

The paper figures (3e/3f) use the **raw, unsmoothed** state as annotated —
:func:`smooth_behavior_state` is an opt-in denoising helper for exploratory
work only and is never applied automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec

logger = logging.getLogger(__name__)

BEHAVIOR_STATE_FILENAME = "block_{block_num}_behavior_state.csv"
OLDER_ANALYSES_DIRNAME = "older_analyses"
REQUIRED_COLUMNS = ("start_time", "end_time", "annotation")
OPTIONAL_MS_COLUMNS = ("start_time_ms", "end_time_ms")

# Only 'quiet' is spelled out; every other label (including the canonical
# 'active'/'explores' and anything unrecognized) normalizes to 'active'.
_QUIET_LABELS = {"quiet", "stationary", "quiescent", "quite", "rest", "still"}


def behavior_state_path(spec: BlockSpec) -> Path | None:
    """Find ``block_<num>_behavior_state.csv`` in ``analysis/`` or ``analysis/older_analyses/``.

    Returns ``None`` when neither location has the file (e.g. ``PV_62/block_038``).
    """
    filename = BEHAVIOR_STATE_FILENAME.format(block_num=spec.block_num)
    candidates = (
        spec.analysis_path / filename,
        spec.analysis_path / OLDER_ANALYSES_DIRNAME / filename,
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def has_behavior_state(spec: BlockSpec) -> bool:
    """Whether a behavior-state CSV exists for ``spec`` (either location)."""
    return behavior_state_path(spec) is not None


def read_behavior_state(spec: BlockSpec) -> pd.DataFrame | None:
    """Load ``spec``'s behavior-state timeline.

    Returns a DataFrame with (at least) ``start_time``, ``end_time``,
    ``annotation`` columns, sorted by ``start_time``, times in milliseconds.
    Returns ``None`` when the CSV is missing for this block.
    """
    path = behavior_state_path(spec)
    if path is None:
        logger.info(
            "[%s] no behavior_state.csv under analysis/ or analysis/%s/",
            spec.block_key,
            OLDER_ANALYSES_DIRNAME,
        )
        return None

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")

    keep_cols = [c for c in df.columns if c in (*REQUIRED_COLUMNS, *OPTIONAL_MS_COLUMNS)]
    out = df[keep_cols].copy()
    out["start_time"] = out["start_time"].astype(float)
    out["end_time"] = out["end_time"].astype(float)
    out = out.sort_values("start_time").reset_index(drop=True)
    logger.info("[%s] loaded %d behavior_state segments from %s", spec.block_key, len(out), path.name)
    return out


def normalize_label(annotation) -> str:
    """Map a raw annotation string to the canonical ``'quiet'`` / ``'active'``.

    ``quiet``/``stationary`` (and close synonyms: quiescent, quite, rest,
    still) → ``'quiet'``; ``active``/``explores`` → ``'active'``; anything
    else also defaults to ``'active'`` (unrecognized labels are treated as
    non-quiescent, matching the paper notebook's mapping).
    """
    label = str(annotation).strip().lower()
    return "quiet" if label in _QUIET_LABELS else "active"


def label_by_time(state_df: pd.DataFrame | None, ms: float | np.ndarray) -> str | np.ndarray:
    """Return ``'quiet'`` / ``'active'`` / ``'unknown'`` for each timestamp (ms).

    ``ms`` may be a scalar or an array; the return type mirrors the input.
    ``'unknown'`` is returned when ``state_df`` is ``None``/empty or ``ms``
    falls outside every annotated segment.
    """
    is_scalar = np.ndim(ms) == 0
    ms_arr = np.atleast_1d(np.asarray(ms, dtype=float))
    out = np.full(ms_arr.shape, "unknown", dtype=object)

    if state_df is not None and not state_df.empty:
        order = np.argsort(state_df["start_time"].to_numpy(dtype=float))
        starts = state_df["start_time"].to_numpy(dtype=float)[order]
        ends = state_df["end_time"].to_numpy(dtype=float)[order]
        annots = state_df["annotation"].to_numpy()[order]

        # Largest segment start <= ms; then verify ms actually falls inside it.
        idx = np.searchsorted(starts, ms_arr, side="right") - 1
        valid = idx >= 0
        idx_clipped = np.clip(idx, 0, len(starts) - 1)
        within = valid & (ms_arr >= starts[idx_clipped]) & (ms_arr <= ends[idx_clipped])
        labels = np.array([normalize_label(a) for a in annots[idx_clipped]], dtype=object)
        out = np.where(within, labels, "unknown")

    return out[0] if is_scalar else out


def _collapse_runs(segments: list[list]) -> list[list]:
    """Merge adjacent/overlapping same-label segments (in place semantics, returns new list)."""
    if not segments:
        return segments
    out = [segments[0][:]]
    for start, end, label in segments[1:]:
        if label == out[-1][2] and start <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end, label])
    return out


def smooth_behavior_state(
    state_df: pd.DataFrame,
    *,
    min_active_ms: float = 5000.0,
    min_quiet_ms: float = 5000.0,
    gap_bridge_ms: float = 3000.0,
) -> pd.DataFrame:
    """Denoise a raw behavior-state timeline (opt-in; **not** applied by default).

    Two-pass hysteresis filter matching the notebook's ``smooth_behavior_state``
    semantics:

    1. Bridge short opposite-label blips (duration ``<= gap_bridge_ms``)
       sandwiched between two same-label neighbors back into that label.
    2. Merge any segment still shorter than its own state's minimum duration
       (``min_active_ms`` for ``'active'``, ``min_quiet_ms`` for ``'quiet'``)
       into whichever neighbor is closer in time, then re-collapse runs.

    Repeats until stable. The paper figures (3e/3f) deliberately use the
    **raw, unsmoothed** state — call this explicitly only for exploratory work.
    """
    if state_df is None or state_df.empty:
        return state_df

    segments = [
        [float(r["start_time"]), float(r["end_time"]), normalize_label(r["annotation"])]
        for _, r in state_df.sort_values("start_time").iterrows()
    ]
    segments = _collapse_runs(segments)

    for _ in range(100):
        changed = False

        # Pass 1: bridge short opposite-label blips between matching neighbors.
        for i in range(1, len(segments) - 1):
            start, end, label = segments[i]
            prev_label = segments[i - 1][2]
            next_label = segments[i + 1][2]
            if (end - start) <= gap_bridge_ms and prev_label == next_label and label != prev_label:
                segments[i][2] = prev_label
                changed = True
        if changed:
            segments = _collapse_runs(segments)
            continue

        # Pass 2: merge the first too-short segment into its nearer neighbor.
        for i, (start, end, label) in enumerate(segments):
            if len(segments) == 1:
                break
            min_dur = min_active_ms if label == "active" else min_quiet_ms
            if (end - start) >= min_dur:
                continue
            if i == 0:
                target = 1
            elif i == len(segments) - 1:
                target = i - 1
            else:
                dist_prev = start - segments[i - 1][0]
                dist_next = segments[i + 1][1] - end
                target = i - 1 if dist_prev <= dist_next else i + 1
            lo, hi = min(i, target), max(i, target)
            merged_label = segments[target][2]
            segments[lo:hi + 1] = [[segments[lo][0], segments[hi][1], merged_label]]
            changed = True
            break

        segments = _collapse_runs(segments)
        if not changed:
            break

    return pd.DataFrame(segments, columns=["start_time", "end_time", "annotation"])
