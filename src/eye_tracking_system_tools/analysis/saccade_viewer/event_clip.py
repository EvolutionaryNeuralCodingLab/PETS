"""Clip window logic for event-centric playback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eye_tracking_system_tools.analysis.saccade_viewer.models import VerificationEvent


@dataclass(frozen=True)
class ClipBounds:
    """Timeline indices and ms bounds for one event clip."""

    i_start: int
    i_end: int
    t_start_ms: float
    t_end_ms: float
    onset_ms: float
    off_ms: float


def clip_ms_bounds(onset_ms: float, *, pre_ms: float, post_ms: float) -> tuple[float, float]:
    pre = max(0.0, float(pre_ms))
    post = max(0.0, float(post_ms))
    return float(onset_ms) - pre, float(onset_ms) + post


def clip_indices(
    ms_axis: np.ndarray,
    t_start_ms: float,
    t_end_ms: float,
) -> ClipBounds:
    """Map absolute ms clip window to inclusive timeline index range."""
    if ms_axis is None or len(ms_axis) == 0:
        return ClipBounds(0, 0, t_start_ms, t_end_ms, t_start_ms, t_end_ms)

    axis = np.asarray(ms_axis, dtype=float)
    t0 = float(min(t_start_ms, t_end_ms))
    t1 = float(max(t_start_ms, t_end_ms))
    i_start = int(np.searchsorted(axis, t0, side="left"))
    i_end = int(np.searchsorted(axis, t1, side="right")) - 1
    i_start = max(0, min(i_start, len(axis) - 1))
    i_end = max(i_start, min(i_end, len(axis) - 1))
    return ClipBounds(
        i_start=i_start,
        i_end=i_end,
        t_start_ms=t0,
        t_end_ms=t1,
        onset_ms=t_start_ms,
        off_ms=t_end_ms,
    )


def bounds_for_event(
    ms_axis: np.ndarray,
    event: VerificationEvent,
    *,
    pre_ms: float,
    post_ms: float,
) -> ClipBounds:
    t0, t1 = clip_ms_bounds(event.onset_ms, pre_ms=pre_ms, post_ms=post_ms)
    bounds = clip_indices(ms_axis, t0, t1)
    return ClipBounds(
        i_start=bounds.i_start,
        i_end=bounds.i_end,
        t_start_ms=bounds.t_start_ms,
        t_end_ms=bounds.t_end_ms,
        onset_ms=event.onset_ms,
        off_ms=event.off_ms,
    )


def ms_to_index(ms_axis: np.ndarray, ms: float) -> int:
    if ms_axis is None or len(ms_axis) == 0:
        return 0
    idx = int(np.searchsorted(np.asarray(ms_axis, dtype=float), float(ms), side="left"))
    return max(0, min(idx, len(ms_axis) - 1))


def clamp_index(index: int, bounds: ClipBounds) -> int:
    return int(max(bounds.i_start, min(index, bounds.i_end)))


def next_unset_index(
    events: list[VerificationEvent],
    start: int,
    *,
    forward: bool = True,
) -> int | None:
    """Return index of next unset event in list order, or None."""
    if not events:
        return None
    n = len(events)
    order = range(start + 1, n) if forward else range(start - 1, -1, -1)
    for i in order:
        if events[i].verification_status == "unset":
            return i
    # Wrap once.
    wrap = range(0, start) if forward else range(n - 1, start, -1)
    for i in wrap:
        if events[i].verification_status == "unset":
            return i
    return None
