# =============================================================================
# Arena block.log + IR sync TTL alignment to Open Ephys timebase
# =============================================================================
#
# Paradigm: Arena PC datetime (in block.log and arena CSVs) can drift relative to
# Open Ephys. We align using:
# 1. block.log: "ARENA-MAIN - trigger is off" appears at start and end of experiment.
# 2. IR sync TTL: Sent to OE exactly 1 second after each "trigger is off".
# 3. So (trigger_off_datetime + 1s) in the log = ir_sync_ttl time in OE.
#
# We get two anchor points (log_datetime, oe_sample), then map any log datetime
# to ms_from_oe_recording_start via linear interpolation (or affine).
# =============================================================================

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# Log line pattern: "2025-12-14 12:30:20.873 - CU-top - DEBUG - ARENA-MAIN - message"
BLOCK_LOG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+-\s+(.+?)\s+-\s+(\w+)\s+-\s+(.*)$"
)
TRIGGER_IS_OFF_MESSAGE = "ARENA-MAIN - trigger is off"
IR_TOGGLE_MESSAGE = "Start IR toggle initial_state: 1, then wait for 1 seconds and set state to 0"


def parse_block_log(log_path: Path) -> pd.DataFrame:
    """
    Parse block.log into a table with timestamp, logger, level, message.

    Parameters
    ----------
    log_path : Path
        Path to block.log (e.g. arena_videos/block.log).

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (datetime64[ns, UTC]), logger, level, message, raw_line.
    """
    log_path = Path(log_path)
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = BLOCK_LOG_PATTERN.match(line)
            if m:
                ts_str, logger, level, message = m.groups()
                try:
                    ts = pd.to_datetime(ts_str, utc=True)
                except Exception:
                    ts = pd.NaT
                rows.append(
                    {
                        "timestamp": ts,
                        "logger": logger.strip(),
                        "level": level.strip(),
                        "message": message.strip() if message else "",
                        "raw_line": line,
                    }
                )
            elif line.strip():
                rows.append(
                    {
                        "timestamp": pd.NaT,
                        "logger": "",
                        "level": "",
                        "message": line.strip(),
                        "raw_line": line,
                    }
                )
    return pd.DataFrame(rows)


def get_trigger_off_plus_one_second_times(log_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the two reference datetimes: trigger_off + 1 second (start and end).

    The IR sync TTL is sent exactly 1 second after each "ARENA-MAIN - trigger is off".
    So we find those two log lines and add 1 second to get the datetimes that
    correspond to the two ir_sync_ttl rising edges in OE.

    Parameters
    ----------
    log_df : pd.DataFrame
        Output of parse_block_log (must have columns message, timestamp).

    Returns
    -------
    (start_dt, end_dt) : tuple of pd.Timestamp
        Start and end reference datetimes (trigger_off + 1s), UTC.
    """
    mask = log_df["message"].str.contains(TRIGGER_IS_OFF_MESSAGE, na=False, regex=False)
    trigger_off = log_df.loc[mask, "timestamp"].dropna()
    if len(trigger_off) < 2:
        raise ValueError(
            f"Expected at least 2 'trigger is off' lines in block.log, found {len(trigger_off)}."
        )
    # First and last occurrence
    t0 = trigger_off.iloc[0]
    t1 = trigger_off.iloc[-1]
    one_sec = pd.Timedelta(seconds=1)
    return (t0 + one_sec, t1 + one_sec)


def detect_ir_sync_ttl_line(
    events_csv_path: Path,
    expected_rising_edges: int = 2,
    ir_sync_line: Optional[int] = None,
) -> Tuple[int, np.ndarray]:
    """
    Find the TTL line in Open Ephys events that has exactly N rising edges (default 2).

    The IR sync TTL fires only at start and end of arena video acquisition,
    so it has exactly two rising edges in the recording.

    Parameters
    ----------
    events_csv_path : Path
        Path to events.csv (Open Ephys export).
    expected_rising_edges : int
        Number of rising edges the IR sync line should have (default 2).
    ir_sync_line : int, optional
        If provided, use this line number as IR sync (must have expected_rising_edges).

    Returns
    -------
    (line_number, sample_numbers) : tuple
        Line index in events.csv and array of OE sample numbers (rising edges).
    """
    path = Path(events_csv_path)
    df = pd.read_csv(path)
    for col in ("line", "state", "sample_number"):
        if col not in df.columns:
            raise ValueError(
                f"events.csv must have columns 'line', 'state', 'sample_number'. Found: {list(df.columns)}"
            )
    rising = df[df["state"] == 1].copy()
    line_num = pd.to_numeric(rising["line"], errors="coerce")
    if ir_sync_line is not None:
        mask = (line_num == ir_sync_line) | (np.abs(line_num - ir_sync_line) < 0.5)
        samples = rising.loc[mask, "sample_number"].values
        if len(samples) < 2:
            raise ValueError(
                f"Line {ir_sync_line} has {len(samples)} rising edges; need at least 2 (first and last)."
            )
        # Use first and last rising edge as start/end of arena acquisition
        samples = np.asarray([samples[0], samples[-1]], dtype=np.int64)
        return (int(ir_sync_line), samples)
    lines = np.unique(line_num.dropna().astype(int))
    candidates = []
    for line in lines:
        mask = (line_num == line) | (np.abs(line_num - line) < 0.5)
        samples = rising.loc[mask, "sample_number"].values
        if len(samples) == expected_rising_edges:
            candidates.append((int(line), np.asarray(samples, dtype=np.int64)))
    if len(candidates) == 0:
        counts = {
            int(ln): int(((line_num == ln) | (np.abs(line_num - ln) < 0.5)).sum())
            for ln in lines
        }
        raise ValueError(
            f"No TTL line with exactly {expected_rising_edges} rising edges found. "
            f"Rising-edge counts per line: {counts}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple TTL lines have exactly {expected_rising_edges} rising edges: {[c[0] for c in candidates]}. "
            "Pass ir_sync_line=<line_number> to specify which is IR sync."
        )
    return candidates[0]


def build_log_datetime_to_oe_ms_mapping(
    ref_start_dt: pd.Timestamp,
    ref_end_dt: pd.Timestamp,
    ir_sync_samples: np.ndarray,
    sample_rate_hz: float,
) -> Tuple[float, float]:
    """
    Build affine mapping from log datetime (Unix ms) to OE ms_from_rec_start.

    Two anchors: (ref_start_dt, oe_ms_1), (ref_end_dt, oe_ms_2).
    Assumes linear relationship: oe_ms = scale * log_unix_ms + offset.
    With two points we get scale and offset.

    Parameters
    ----------
    ref_start_dt, ref_end_dt : pd.Timestamp
        Reference datetimes (trigger_off + 1s) in start and end.
    ir_sync_samples : np.ndarray
        Length-2 array: OE sample numbers for the two ir_sync_ttl rising edges.
    sample_rate_hz : float
        Open Ephys sample rate (Hz).

    Returns
    -------
    (scale, offset) : tuple
        oe_ms = scale * log_unix_ms + offset. So to convert: oe_ms = scale * unix_ms + offset.
    """
    if len(ir_sync_samples) != 2:
        raise ValueError("ir_sync_samples must have length 2.")
    oe_ms_1 = ir_sync_samples[0] / (sample_rate_hz / 1000.0)
    oe_ms_2 = ir_sync_samples[1] / (sample_rate_hz / 1000.0)
    unix_ms_1 = ref_start_dt.timestamp() * 1000.0
    unix_ms_2 = ref_end_dt.timestamp() * 1000.0
    # oe_ms = scale * unix_ms + offset
    # oe_ms_1 = scale * unix_ms_1 + offset, oe_ms_2 = scale * unix_ms_2 + offset
    denom = unix_ms_2 - unix_ms_1
    if abs(denom) < 1e-6:
        raise ValueError("Reference datetimes are too close; cannot build mapping.")
    scale = (oe_ms_2 - oe_ms_1) / denom
    offset = oe_ms_1 - scale * unix_ms_1
    return (float(scale), float(offset))


def log_timestamps_to_oe_ms(
    timestamps: pd.Series,
    scale: float,
    offset: float,
) -> pd.Series:
    """
    Convert a series of log datetimes to ms_from_oe_recording_start.

    Parameters
    ----------
    timestamps : pd.Series
        Datetime series (e.g. from block log or arena CSV time column).
    scale, offset : float
        From build_log_datetime_to_oe_ms_mapping.

    Returns
    -------
    pd.Series
        OE time in milliseconds (ms from recording start).
    """
    dt = pd.to_datetime(timestamps, utc=True, errors="coerce")
    unix_ms = dt.apply(lambda x: x.timestamp() * 1000.0 if pd.notna(x) else np.nan)
    return unix_ms * scale + offset
