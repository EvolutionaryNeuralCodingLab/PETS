"""Open Ephys stream enumeration and decimated trace loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Time alignment convention (ms_axis ↔ get_data):
# final_sync_df ms_axis = Arena_TTL / (sample_rate_hz / 1000), i.e. milliseconds in the
# same sample-index clock as Open Ephys after OERecording zeroes allTimeStamps at recording
# start (globalStartTime_ms subtracted). Pass this value as start_time_ms to get_data /
# get_analog_data / get_accel_data — matching utility_functions saccade/LFP snippets.


@dataclass(frozen=True)
class OEStream:
    kind: str  # "hs", "adc", "aux"
    channel: int
    label: str

    @property
    def fetcher_kind(self) -> str:
        return self.kind


def _as_sequence(value: Any) -> list:
    """Normalize OERecording attrs (ndarray, list, None) for safe iteration."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def list_oe_streams(oe_rec: Any) -> list[OEStream]:
    streams: list[OEStream] = []
    if oe_rec is None:
        return streams

    for ch in _as_sequence(getattr(oe_rec, "channelNumbers", None)):
        streams.append(OEStream("hs", int(ch), f"HS ch{int(ch)}"))

    for ch in _as_sequence(getattr(oe_rec, "analogChannelNumbers", None)):
        streams.append(OEStream("adc", int(ch), f"ADC ch{int(ch)}"))

    accel_files = _as_sequence(getattr(oe_rec, "accel_files", None))
    for i, fname in enumerate(accel_files):
        stem = fname.replace(".continuous", "").split("_")[-1] if fname else f"ch{i+1}"
        streams.append(OEStream("aux", i + 1, f"AUX {stem}"))

    return streams


def _fetch_window(
    oe_rec: Any,
    stream: OEStream,
    start_ms: float,
    window_ms: float,
) -> np.ndarray | None:
    """Return 1D samples for one window or None if unavailable."""
    start = np.atleast_2d(np.array([start_ms], dtype=np.float64))
    try:
        if stream.kind == "hs":
            data, _ = oe_rec.get_data(
                [stream.channel], start, window_ms, return_timestamps=True
            )
        elif stream.kind == "adc":
            data, _ = oe_rec.get_analog_data(
                [stream.channel], start, window_ms, return_timestamps=True
            )
        else:
            data, _ = oe_rec.get_accel_data(
                [stream.channel], start, window_ms, return_timestamps=True
            )
    except (ValueError, IndexError, TypeError):
        return None

    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    # shape [n_ch, n_windows, n_samples]
    if arr.ndim == 3:
        vec = arr[0, 0, :]
    else:
        vec = arr.ravel()
    return vec.astype(np.float64)


def load_decimated_trace(
    oe_rec: Any,
    stream: OEStream,
    ms_min: float,
    ms_max: float,
    *,
    downsample: int = 50,
    window_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build decimated (time_ms, value) arrays across [ms_min, ms_max].

    Each segment uses get_data at segment start; downsample merges segments.
    """
    if oe_rec is None or ms_max <= ms_min:
        return None

    downsample = max(1, int(downsample))
    sample_ms = float(getattr(oe_rec, "sample_ms", 1.0) or 1.0)
    if window_ms is None:
        window_ms = max(sample_ms * 512, 50.0)

    span = ms_max - ms_min
    n_segments = max(1, int(span / (window_ms * downsample)) + 1)
    step = span / n_segments

    times: list[float] = []
    values: list[float] = []

    for k in range(n_segments):
        t0 = ms_min + k * step
        vec = _fetch_window(oe_rec, stream, t0, window_ms)
        if vec is None or len(vec) == 0:
            continue
        seg_t = t0 + np.arange(len(vec)) * sample_ms
        dec = vec[::downsample]
        dec_t = seg_t[::downsample]
        if len(dec) == 0:
            continue
        times.extend(dec_t.tolist())
        values.extend(dec.tolist())

    if not times:
        return None
    return np.array(times, dtype=np.float64), np.array(values, dtype=np.float64)


def load_overview_trace(
    oe_rec: Any,
    stream: OEStream,
    ms_axis: np.ndarray,
    *,
    downsample: int = 50,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Decimated trace aligned to timeline ms_axis endpoints."""
    if len(ms_axis) == 0:
        return None
    return load_decimated_trace(
        oe_rec,
        stream,
        float(ms_axis[0]),
        float(ms_axis[-1]),
        downsample=downsample,
    )
