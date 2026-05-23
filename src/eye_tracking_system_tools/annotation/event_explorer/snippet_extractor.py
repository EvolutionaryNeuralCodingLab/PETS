"""Frame-first eye snippets with ms_axis fallback; OE via oe_streams."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eye_tracking_system_tools.annotation.block_annotator.oe_streams import (
    OEStream,
    _fetch_window,
)
from eye_tracking_system_tools.annotation.event_explorer.eye_csv_resolver import (
    frame_column_name,
    pupil_values,
)
from eye_tracking_system_tools.annotation.event_explorer.load_log import LoadLog
from eye_tracking_system_tools.annotation.event_explorer.models import (
    BlockDataCache,
    EventRecord,
    EventSnippet,
)

STREAM_L_PUPIL = "l_pupil"
STREAM_R_PUPIL = "r_pupil"
STREAM_L_DEG = "l_degrees"
STREAM_R_DEG = "r_degrees"
STREAM_EP = "ep"
# Legacy alias
STREAM_PUPIL = STREAM_L_PUPIL

STREAM_LABELS = {
    STREAM_L_PUPIL: "L pupil diameter",
    STREAM_R_PUPIL: "R pupil diameter",
    STREAM_L_DEG: "Left degrees",
    STREAM_R_DEG: "Right degrees",
    STREAM_EP: "EP (HS)",
}

EYE_STREAMS = (STREAM_L_PUPIL, STREAM_R_PUPIL, STREAM_L_DEG, STREAM_R_DEG)


def _ms_column(df: pd.DataFrame) -> str | None:
    if "ms_axis" in df.columns:
        return "ms_axis"
    return None


def _frame_to_row_index(df: pd.DataFrame, frame: int, frame_col: str) -> int | None:
    series = df[frame_col]
    matches = np.where(series.to_numpy() == frame)[0]
    if len(matches):
        return int(matches[0])
    diffs = (series.astype(float) - float(frame)).abs()
    return int(diffs.idxmin()) if len(diffs) else None


def _row_window_from_ms(
    df: pd.DataFrame,
    center_ms: float,
    half_window_ms: float,
    ms_col: str,
) -> tuple[int, int]:
    ms = df[ms_col].to_numpy(dtype=np.float64)
    t0 = center_ms - half_window_ms
    t1 = center_ms + half_window_ms
    i0 = int(np.searchsorted(ms, t0, side="left"))
    i1 = int(np.searchsorted(ms, t1, side="right"))
    i0 = max(0, min(i0, len(ms) - 1))
    i1 = max(i0 + 1, min(i1, len(ms)))
    return i0, i1


def _relative_time_ms(
    df: pd.DataFrame,
    i0: int,
    i1: int,
    anchor_ms: float,
    ms_col: str | None,
    half_window_ms: float,
) -> np.ndarray:
    if ms_col and ms_col in df.columns:
        t = df[ms_col].iloc[i0:i1].to_numpy(dtype=np.float64)
        return t - anchor_ms
    n = i1 - i0
    if n <= 1:
        return np.zeros(1, dtype=np.float64)
    # Eye CSV has no ms_axis: evenly space across requested half-window
    return np.linspace(-half_window_ms, half_window_ms, n, dtype=np.float64)


def extract_eye_snippet(
    record: EventRecord,
    cache: BlockDataCache,
    stream_id: str,
    half_window_ms: float,
    load_log: LoadLog,
) -> EventSnippet | None:
    if stream_id == STREAM_L_PUPIL:
        side = "left"
        df = cache.le_df
        col_spec = cache.column_map.pupil
        eye_frame = record.l_eye_frame
    elif stream_id == STREAM_R_PUPIL:
        side = "right"
        df = cache.re_df
        col_spec = cache.column_map.pupil
        eye_frame = record.r_eye_frame
    elif stream_id == STREAM_L_DEG:
        side = "left"
        df = cache.le_degrees_df if cache.le_degrees_df is not None else cache.le_df
        col_spec = cache.column_map.l_degrees
        eye_frame = record.l_eye_frame
    elif stream_id == STREAM_R_DEG:
        side = "right"
        df = cache.re_degrees_df if cache.re_degrees_df is not None else cache.re_df
        col_spec = cache.column_map.r_degrees
        eye_frame = record.r_eye_frame
    else:
        return None

    if df is None or col_spec is None:
        load_log.warn(
            f"No eye data/column for {stream_id} event={record.event_id} block={cache.block_path.name}"
        )
        return None

    ms_col = _ms_column(df)
    frame_col = frame_column_name(df)
    source = "frame"
    anchor_ms = float(record.timepoint_ms)
    i0: int | None = None
    i1: int | None = None

    if eye_frame is not None and frame_col:
        row_idx = _frame_to_row_index(df, int(eye_frame), frame_col)
        if row_idx is not None and ms_col:
            anchor_ms = float(df[ms_col].iloc[row_idx])
            i0, i1 = _row_window_from_ms(df, anchor_ms, half_window_ms, ms_col)
        elif row_idx is not None:
            half_frames = max(1, int(round(half_window_ms / 33.0)))
            i0 = max(0, row_idx - half_frames)
            i1 = min(len(df), row_idx + half_frames + 1)
        else:
            source = "ms_axis_fallback"
            load_log.warn(
                f"Frame lookup failed for event {record.event_id} frame={eye_frame}; "
                f"using ms_axis slice"
            )
            if ms_col:
                i0, i1 = _row_window_from_ms(
                    df, anchor_ms, half_window_ms, ms_col
                )
    else:
        source = "ms_axis_fallback"
        if eye_frame is None:
            load_log.warn(
                f"Missing eye_frame for event {record.event_id}; ms_axis fallback ({side})"
            )
        if ms_col:
            i0, i1 = _row_window_from_ms(
                df, anchor_ms, half_window_ms, ms_col
            )

    if i0 is None or i1 is None or i1 <= i0:
        load_log.warn(
            f"Could not slice {stream_id} window for event {record.event_id} "
            f"(ms_col={ms_col!r}, frame_col={frame_col!r})"
        )
        return None

    if (i1 - i0) >= max(len(df) - 1, 1):
        load_log.warn(
            f"Eye snippet spans entire CSV ({i1 - i0} rows) for event {record.event_id}; "
            f"retrying narrow window around anchor"
        )
        if ms_col:
            i0, i1 = _row_window_from_ms(df, anchor_ms, half_window_ms, ms_col)
        else:
            return None

    if stream_id in (STREAM_L_PUPIL, STREAM_R_PUPIL):
        values = pupil_values(df, col_spec)
        if values is None:
            load_log.warn(f"Pupil column unavailable: {col_spec}")
            return None
        y = values.iloc[i0:i1].to_numpy(dtype=np.float64)
    else:
        if col_spec not in df.columns:
            load_log.warn(f"Degrees column missing: {col_spec}")
            return None
        y = df[col_spec].iloc[i0:i1].to_numpy(dtype=np.float64)

    t_rel = _relative_time_ms(
        df, i0, i1, anchor_ms, ms_col, half_window_ms
    )
    span = float(t_rel[-1] - t_rel[0]) if len(t_rel) > 1 else 0.0
    if span > 2.5 * half_window_ms:
        load_log.warn(
            f"Eye snippet span {span:.0f} ms > window for event {record.event_id}; "
            f"clipping to ±{half_window_ms:.0f} ms"
        )
        if ms_col:
            i0, i1 = _row_window_from_ms(df, anchor_ms, half_window_ms, ms_col)
            if stream_id in (STREAM_L_PUPIL, STREAM_R_PUPIL):
                y = values.iloc[i0:i1].to_numpy(dtype=np.float64)
            else:
                y = df[col_spec].iloc[i0:i1].to_numpy(dtype=np.float64)
            t_rel = _relative_time_ms(df, i0, i1, anchor_ms, ms_col, half_window_ms)
        else:
            return None

    return EventSnippet(
        event_id=record.event_id,
        stream_id=stream_id,
        time_rel_ms=t_rel,
        values=y,
        source=source,
        meta={
            "column": col_spec,
            "block_path": str(cache.block_path),
            "eye_csv": str(
                (cache.le_csv_path if side == "left" else cache.re_csv_path) or ""
            ),
        },
    )


def extract_ep_snippet(
    record: EventRecord,
    cache: BlockDataCache,
    channel: int,
    half_window_ms: float,
    load_log: LoadLog,
) -> EventSnippet | None:
    if cache.oe_rec is None:
        load_log.warn(f"No OE recording for EP trace event={record.event_id}")
        return None

    stream = OEStream("hs", int(channel), f"HS ch{int(channel)}")
    t0 = record.timepoint_ms - half_window_ms
    window_ms = 2.0 * half_window_ms
    vec = _fetch_window(cache.oe_rec, stream, t0, window_ms)
    if vec is None or len(vec) == 0:
        load_log.warn(f"EP fetch failed ch={channel} event={record.event_id}")
        return None

    sample_ms = float(getattr(cache.oe_rec, "sample_ms", 1.0) or 1.0)
    t_rel = np.arange(len(vec), dtype=np.float64) * sample_ms + t0 - record.timepoint_ms
    return EventSnippet(
        event_id=record.event_id,
        stream_id=STREAM_EP,
        time_rel_ms=t_rel,
        values=vec,
        source="oe",
        meta={"channel": channel, "block_path": str(cache.block_path)},
    )


def normalize_trials(
    trials: list[np.ndarray],
    times: list[np.ndarray],
    mode: str,
    *,
    baseline_ms: tuple[float, float] = (-100.0, 0.0),
) -> list[np.ndarray]:
    """Apply normalization per trial (same length as input trials)."""
    if mode == "none" or not mode:
        return trials
    out: list[np.ndarray] = []
    for y, t in zip(trials, times):
        arr = np.asarray(y, dtype=np.float64)
        if mode == "zscore":
            mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr))
            out.append((arr - mu) / sd if sd > 1e-12 else arr - mu)
        elif mode == "baseline_divide":
            mask = (t >= baseline_ms[0]) & (t <= baseline_ms[1])
            base = float(np.nanmean(arr[mask])) if mask.any() else float(np.nanmean(arr))
            out.append(arr / base if abs(base) > 1e-12 else arr)
        elif mode == "minmax":
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            out.append((arr - lo) / (hi - lo) if hi - lo > 1e-12 else arr * 0)
        else:
            out.append(arr)
    return out


def resample_to_grid(
    snippets: list[EventSnippet],
    n_points: int = 201,
    half_window_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate trials onto a common relative-ms grid (linear).

    When ``half_window_ms`` is set, the grid is fixed to
    ``[-half_window_ms, +half_window_ms]`` so mismatched trial spans cannot
    inflate the plot axis. Values outside each trial's time support are NaN
    (not flat-extrapolated).
    """
    if not snippets:
        raise ValueError("resample_to_grid requires at least one snippet")

    if half_window_ms is not None and half_window_ms > 0:
        grid = np.linspace(-half_window_ms, half_window_ms, n_points)
    else:
        t_min = max(float(s.time_rel_ms[0]) for s in snippets)
        t_max = min(float(s.time_rel_ms[-1]) for s in snippets)
        if t_max <= t_min:
            t_min = min(float(s.time_rel_ms.min()) for s in snippets)
            t_max = max(float(s.time_rel_ms.max()) for s in snippets)
        grid = np.linspace(t_min, t_max, n_points)

    stacked = []
    for s in snippets:
        t = np.asarray(s.time_rel_ms, dtype=np.float64)
        y = np.asarray(s.values, dtype=np.float64)
        order = np.argsort(t)
        t, y = t[order], y[order]
        t_u, uniq_idx = np.unique(t, return_index=True)
        y_u = y[uniq_idx]
        if len(t_u) == 1:
            y_grid = np.full(n_points, y_u[0], dtype=np.float64)
        else:
            y_grid = np.interp(
                grid, t_u, y_u, left=np.nan, right=np.nan
            ).astype(np.float64)
        stacked.append(y_grid)
    return grid, np.vstack(stacked)
