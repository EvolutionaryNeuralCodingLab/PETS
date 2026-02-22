# =============================================================================
# Arena video / app data alignment to Open Ephys timebase
# =============================================================================
#
# DATA FORMAT REFERENCE (column meanings)
# ---------------------------------------
#
# final_sync_df (e.g. final_sync_df.csv):
#   - Arena_TTL: Open-ephys sample number for that row (0 = recording start).
#   - Arena_frame: Frame index from the sync scheme. In the current pipeline this
#     is 0-based from the first Arena TTL (oe_events Arena_TTL_frame after shift).
#     Intended mapping: after alignment, row with Arena_frame = n should have the
#     same ms_axis as frame index n in frames_timestamps. To get the actual video
#     file frame index, add get_arena_video_frame_offset().
#   - ms_axis (if present): (Arena_TTL / (sample_rate/1000)) = ms from recording start.
#
# frames_timestamps CSVs (e.g. left_20251214T123030.csv in videos/frames_timestamps):
#   - Column 0 (unlabeled): Frame index; 0 = first frame of the video file.
#   - Column 1 (often titled '0'): Timestamp in Unix seconds (same PC clock as arena CSVs).
#   Video files and these CSVs are matched by name; frame index 0 = first frame of video.
#
# Bug trajectory (bug_trajectory.csv):
#   - time: ISO 8601 datetime with timezone (e.g. 2025-09-04T13:25:08.622000+03:00).
#   - x, y: Bug position. No ms column; we convert time → Unix ms (PC clock) → ms_axis.
#
# Two-clock synchronization:
# 1. Open Ephys clock: ms from recording start (0 ms = OE recording started)
# 2. Arena PC clock: Unix milliseconds (arbitrary reference)
# Synchronization: Match the first arena video frame between the two clocks:
#   - OE: first Arena_TTL sample → ms (via sample / (sample_rate/1000))
#   - PC: first frame timestamp from frames_timestamps (Unix s → ms, median across cameras)
#   - shift = PC_first_frame_ms - OE_first_frame_ms
#   - Apply: OE_ms = PC_ms - shift
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd


# Default CSV names under arena_videos
ARENA_CSV_NAMES = (
    "bug_trajectory.csv",
    "app_events.csv",
    "screen_touches.csv",
    "trials_data.csv",
)

# Only these are written to disk by write_arena_events_ms_axis (after verification in arena_sync_verification).
# app_events is excluded until alignment is verified for it.
ARENA_CSV_NAMES_VERIFIED = (
    "bug_trajectory.csv",
    "trials_data.csv",
    "screen_touches.csv",
)

# Subfolder under block analysis_path for ms_axis-aligned arena CSVs
ARENA_EVENTS_MS_SUBDIR = "arena_events_ms"

# Column name for aligned time (ms from OE recording start)
MS_AXIS = "ms_axis"


def _get_fs(block) -> float:
    """Return Open Ephys sample rate (Hz)."""
    fs = getattr(block, "sample_rate", None)
    if fs is None:
        fs = float(block.get_sample_rate())
        if hasattr(block, "sample_rate"):
            block.sample_rate = fs
    return float(fs)


def get_first_arena_frame_oe_ms(
    block,
    arena_ttl_col: str = "Arena_TTL",
) -> float:
    """
    Get the time (ms) of the first arena TTL in the Open Ephys paradigm.

    Uses the same convention as block sync: ms = OE_sample / (sample_rate/1000),
    i.e. milliseconds from recording start (sample 0).

    Parameters
    ----------
    block
        Block with oe_events and sample_rate (e.g. BlockSync instance).
    arena_ttl_col : str
        Column in oe_events holding arena TTL sample numbers.

    Returns
    -------
    float
        Time in ms of the first arena frame in OE timebase.
    """
    if not hasattr(block, "oe_events") or block.oe_events is None:
        raise RuntimeError("block.oe_events is required. Run block.parse_open_ephys_events() first.")
    if arena_ttl_col not in block.oe_events.columns:
        raise RuntimeError(f"'{arena_ttl_col}' not found in block.oe_events.")
    fs = _get_fs(block)
    arena = block.oe_events[arena_ttl_col].dropna().astype(np.int64)
    if len(arena) == 0:
        raise RuntimeError("No arena TTL events found in block.oe_events.")
    first_sample = int(arena.iloc[0])
    return first_sample / (fs / 1000.0)


def get_first_arena_frame_pc_ms(
    arena_videos_dir: Union[str, Path],
    frames_timestamps_subdir: str = "videos/frames_timestamps",
) -> tuple[float, list[float]]:
    """
    Get the PC-clock timestamp (milliseconds) of the first arena video frame.

    Reads frames_timestamps CSVs (Unix seconds) and converts to milliseconds.
    Returns the median first-frame timestamp across cameras (they should be very close).

    Parameters
    ----------
    arena_videos_dir : path-like
        Path to the block's arena_videos folder.
    frames_timestamps_subdir : str
        Subdirectory under arena_videos_dir containing *_timestamp*.csv files.

    Returns
    -------
    median_first_frame_ms : float
        Median first-frame timestamp in milliseconds (PC clock).
    all_first_frame_ms : list[float]
        All first-frame timestamps from each camera (for accuracy reporting).
    """
    arena_videos_dir = Path(arena_videos_dir)
    ts_dir = arena_videos_dir / frames_timestamps_subdir
    if not ts_dir.is_dir():
        raise FileNotFoundError(f"Frames timestamps directory not found: {ts_dir}")

    csvs = list(ts_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {ts_dir}")

    first_ts_ms_list = []
    for path in csvs:
        df = pd.read_csv(path, header=0)
        if df.empty or len(df.columns) < 2:
            continue
        t_unix_s = df.iloc[0, 1]
        try:
            t_unix_s = float(t_unix_s)
            # Convert Unix seconds to milliseconds (PC clock)
            t_ms = t_unix_s * 1000.0
            first_ts_ms_list.append(t_ms)
        except (TypeError, ValueError):
            continue

    if not first_ts_ms_list:
        raise ValueError(f"Could not read first-frame timestamp from any file in {ts_dir}")

    # Use median (as user requested) - cameras should be very close (same TTL)
    median_first_ms = float(np.median(first_ts_ms_list))
    return median_first_ms, first_ts_ms_list


def arena_datetime_to_pc_ms(series: pd.Series) -> pd.Series:
    """
    Convert arena 'time' column (ISO datetime with timezone) to PC-clock milliseconds.

    Converts datetime → Unix seconds → milliseconds. All arena files use the same
    PC clock, so this gives PC-clock milliseconds (arbitrary reference point).
    For display only, you may convert to a local timezone (e.g. .dt.tz_convert('Asia/Jerusalem'));
    alignment should use this single conversion.

    Parameters
    ----------
    series : pd.Series
        Series of datetime strings (e.g. '2025-12-14T12:30:40.651000+02:00').

    Returns
    -------
    pd.Series
        PC-clock timestamps in milliseconds.
    """
    try:
        dt = pd.to_datetime(series, utc=True, format="ISO8601")
    except (TypeError, ValueError):
        # Older pandas or mixed formats: infer per element
        dt = pd.to_datetime(series, utc=True, format="mixed")
    # Convert to Unix seconds, then to milliseconds
    unix_s = dt.apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)
    return unix_s * 1000.0


def compute_pc_to_oe_shift_ms(
    first_arena_frame_pc_ms: float,
    first_arena_frame_oe_ms: float,
) -> float:
    """
    Compute the shift to convert PC-clock milliseconds to OE-clock milliseconds.

    Synchronization: OE_first_frame_ms = PC_first_frame_ms - shift
    => shift = PC_first_frame_ms - OE_first_frame_ms

    Then for any PC timestamp: OE_ms = PC_ms - shift

    Parameters
    ----------
    first_arena_frame_pc_ms : float
        First arena frame timestamp in PC-clock milliseconds.
    first_arena_frame_oe_ms : float
        First arena frame timestamp in OE-clock milliseconds.

    Returns
    -------
    float
        Shift value (ms). Apply: OE_ms = PC_ms - shift.
    """
    return first_arena_frame_pc_ms - first_arena_frame_oe_ms


def load_and_align_arena_csv(
    path: Path,
    shift_ms: float,
    time_columns: Optional[list] = None,
    ms_axis_suffix: str = "",
) -> pd.DataFrame:
    """
    Load an arena CSV and add ms_axis (or ms_axis_start/ms_axis_end) aligned to OE.

    Converts datetime columns to PC-clock milliseconds, then applies shift:
    OE_ms = PC_ms - shift_ms

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    shift_ms : float
        Shift from compute_pc_to_oe_shift_ms. Applied as: OE_ms = PC_ms - shift_ms.
    time_columns : list of str, optional
        Columns that contain datetime strings to convert. If None, uses ['time'] when present,
        or for trials_data uses ['start_time','end_time'] and adds ms_axis_start, ms_axis_end.
    ms_axis_suffix : str
        Suffix for ms_axis column(s), e.g. '' -> 'ms_axis', '_start' -> 'ms_axis_start'.

    Returns
    -------
    pd.DataFrame
        DataFrame with original data plus ms_axis column(s). Other columns unchanged.
    """
    df = pd.read_csv(path)
    if df.empty:
        df[MS_AXIS + ms_axis_suffix] = pd.Series(dtype=float)
        return df

    if time_columns is None:
        if "start_time" in df.columns and "end_time" in df.columns:
            time_columns = ["start_time", "end_time"]
        else:
            time_columns = ["time"] if "time" in df.columns else []

    if not time_columns:
        df[MS_AXIS + ms_axis_suffix] = np.nan
        return df

    for col in time_columns:
        if col not in df.columns:
            continue
        # Convert datetime → PC-clock milliseconds
        pc_ms = arena_datetime_to_pc_ms(df[col])
        # Apply shift: OE_ms = PC_ms - shift_ms
        oe_ms = pc_ms - shift_ms
        if col == "time":
            df[MS_AXIS + ms_axis_suffix] = oe_ms
        else:
            suffix = "_start" if col == "start_time" else "_end" if col == "end_time" else f"_{col}"
            df[MS_AXIS + suffix] = oe_ms

    return df


def get_arena_video_frame_offset(
    block,
    arena_videos_dir: Union[str, Path],
    arena_ttl_col: str = "Arena_TTL",
    frames_timestamps_subdir: str = "videos/frames_timestamps",
    reference_timestamps_glob: str = "*left*.csv",
) -> int:
    """
    Return the offset to add to final_sync_df['Arena_frame'] to get the video file frame index.

    final_sync_df['Arena_frame'] is 0-based from the first Arena TTL (sync scheme).
    Video files have frame 0 = first frame of the recording. The first TTL may occur
    at a later video frame. So: video_frame_index = Arena_frame + offset.

    Parameters
    ----------
    block
        Block with oe_events and sample_rate.
    arena_videos_dir : path-like
        Path to the block's arena_videos folder.
    arena_ttl_col : str
        Column in oe_events for Arena TTL sample numbers.
    frames_timestamps_subdir : str
        Subdirectory under arena_videos_dir containing timestamp CSVs.
    reference_timestamps_glob : str
        Glob to pick one reference CSV (e.g. left camera) for frame index lookup.

    Returns
    -------
    int
        Offset such that video_frame_index = Arena_frame + offset.
    """
    arena_videos_dir = Path(arena_videos_dir)
    first_oe_ms = get_first_arena_frame_oe_ms(block, arena_ttl_col=arena_ttl_col)
    
    # Get alignment shift to convert OE time to PC time
    median_pc_ms, _ = get_first_arena_frame_pc_ms(
        arena_videos_dir, frames_timestamps_subdir=frames_timestamps_subdir
    )
    shift_ms = compute_pc_to_oe_shift_ms(median_pc_ms, first_oe_ms)
    # Convert first TTL OE time to PC time: PC_ms = OE_ms + shift_ms
    first_ttl_pc_ms = first_oe_ms + shift_ms

    ts_dir = arena_videos_dir / frames_timestamps_subdir
    if not ts_dir.is_dir():
        return 0
    ref_files = list(ts_dir.glob(reference_timestamps_glob))
    if not ref_files:
        ref_files = list(ts_dir.glob("*.csv"))
    if not ref_files:
        return 0

    df = pd.read_csv(ref_files[0], header=0)
    if df.empty or len(df.columns) < 2:
        return 0
    # Column 1: Unix seconds (convert to ms)
    ts_ms = pd.to_numeric(df.iloc[:, 1], errors="coerce") * 1000.0
    valid = np.isfinite(ts_ms)
    if not np.any(valid):
        return 0
    # Find frame index whose timestamp is closest to first_ttl_pc_ms
    idx = np.nanargmin(np.where(valid, np.abs(ts_ms - first_ttl_pc_ms), np.inf))
    first_ttl_video_frame = int(idx)
    # Arena_frame for the first TTL is 0 in the sync scheme
    first_arena_ttl_frame = 0
    return int(first_ttl_video_frame - first_arena_ttl_frame)


def get_arena_alignment_constants(
    block,
    arena_videos_dir: Union[str, Path],
    arena_ttl_col: str = "Arena_TTL",
    frames_timestamps_subdir: str = "videos/frames_timestamps",
) -> Dict[str, Any]:
    """
    Return the alignment constants used to map arena PC-clock → OE-clock.

    Returns diagnostic info including the median and all first-frame timestamps
    from cameras (to verify accuracy).

    Parameters
    ----------
    block
        Block with oe_events and sample_rate (e.g. BlockSync instance).
    arena_videos_dir : path-like
        Path to the block's arena_videos folder.
    arena_ttl_col : str
        Column in oe_events for arena TTL sample numbers.
    frames_timestamps_subdir : str
        Subdirectory under arena_videos_dir containing frame timestamp CSVs.

    Returns
    -------
    dict
        Contains:
        - 'first_arena_frame_oe_ms': OE-clock time of first frame (ms)
        - 'first_arena_frame_pc_ms': PC-clock time of first frame (ms, median)
        - 'all_first_frame_pc_ms': All camera first-frame times (ms, for accuracy check)
        - 'shift_ms': Computed shift (PC_ms - shift_ms = OE_ms)
        - 'camera_accuracy_ms': Max difference between cameras (ms)
    """
    arena_videos_dir = Path(arena_videos_dir)
    first_oe_ms = get_first_arena_frame_oe_ms(block, arena_ttl_col=arena_ttl_col)
    median_pc_ms, all_pc_ms = get_first_arena_frame_pc_ms(
        arena_videos_dir, frames_timestamps_subdir=frames_timestamps_subdir
    )
    shift_ms = compute_pc_to_oe_shift_ms(median_pc_ms, first_oe_ms)
    
    if len(all_pc_ms) > 1:
        camera_accuracy_ms = float(np.max(all_pc_ms) - np.min(all_pc_ms))
    else:
        camera_accuracy_ms = 0.0

    return {
        "first_arena_frame_oe_ms": first_oe_ms,
        "first_arena_frame_pc_ms": median_pc_ms,
        "all_first_frame_pc_ms": all_pc_ms,
        "shift_ms": shift_ms,
        "camera_accuracy_ms": camera_accuracy_ms,
    }


def load_aligned_arena_data(
    block,
    arena_videos_dir: Union[str, Path],
    arena_ttl_col: str = "Arena_TTL",
    frames_timestamps_subdir: str = "videos/frames_timestamps",
    csv_names: tuple = ARENA_CSV_NAMES,
) -> Dict[str, pd.DataFrame]:
    """
    Load all arena CSVs and return them as DataFrames with OE-aligned ms_axis.

    Two-clock synchronization:
    1. Convert frames_timestamps (Unix seconds) → PC-clock milliseconds
    2. Get median first-frame PC_ms across cameras
    3. Get first-frame OE_ms from Arena_TTL
    4. Compute shift = PC_first_ms - OE_first_ms
    5. Apply to all arena CSVs: OE_ms = PC_ms - shift

    All arena files (bug_trajectory, trials_data, app_events, screen_touches) use
    the same PC clock, so they're all synchronized together.

    Parameters
    ----------
    block
        Block with oe_events and sample_rate (e.g. BlockSync instance).
    arena_videos_dir : path-like
        Path to the block's arena_videos folder (e.g. .../block_019/arena_videos).
    arena_ttl_col : str
        Column in oe_events for arena TTL sample numbers.
    frames_timestamps_subdir : str
        Subdirectory under arena_videos_dir containing frame timestamp CSVs.
    csv_names : tuple of str
        Filenames to load (bug_trajectory.csv, app_events.csv, etc.).

    Returns
    -------
    dict
        Keys: base name without .csv (e.g. 'bug_trajectory', 'trials_data').
        Values: DataFrames with original columns plus:
          - For single-time CSVs: 'ms_axis' (ms from OE recording start).
          - For trials_data: 'ms_axis_start' and 'ms_axis_end'.
    """
    arena_videos_dir = Path(arena_videos_dir)
    
    # Get first frame in OE clock
    first_oe_ms = get_first_arena_frame_oe_ms(block, arena_ttl_col=arena_ttl_col)
    
    # Get first frame in PC clock (median across cameras)
    median_pc_ms, all_pc_ms = get_first_arena_frame_pc_ms(
        arena_videos_dir, frames_timestamps_subdir=frames_timestamps_subdir
    )
    
    # Compute shift: OE_ms = PC_ms - shift_ms
    shift_ms = compute_pc_to_oe_shift_ms(median_pc_ms, first_oe_ms)
    
    # Report accuracy
    if len(all_pc_ms) > 1:
        accuracy_ms = np.max(all_pc_ms) - np.min(all_pc_ms)
        print(f"[Arena alignment] First frame PC-clock: {median_pc_ms:.2f} ms (median), "
              f"cameras differ by max {accuracy_ms:.2f} ms")
    else:
        print(f"[Arena alignment] First frame PC-clock: {median_pc_ms:.2f} ms (single camera)")
    print(f"[Arena alignment] First frame OE-clock: {first_oe_ms:.2f} ms")
    print(f"[Arena alignment] Shift: {shift_ms:.2f} ms (OE_ms = PC_ms - shift_ms)")

    out = {}
    for name in csv_names:
        path = arena_videos_dir / name
        if not path.exists():
            continue
        key = path.stem  # e.g. 'bug_trajectory'
        if name == "trials_data.csv":
            df = load_and_align_arena_csv(
                path,
                shift_ms=shift_ms,
                time_columns=["start_time", "end_time"],
                ms_axis_suffix="",
            )
        else:
            df = load_and_align_arena_csv(
                path,
                shift_ms=shift_ms,
                time_columns=None,
                ms_axis_suffix="",
            )
        out[key] = df
    return out


def write_arena_events_ms_axis(
    block,
    arena_videos_dir: Union[str, Path],
    analysis_path: Optional[Union[str, Path]] = None,
    arena_ttl_col: str = "Arena_TTL",
    frames_timestamps_subdir: str = "videos/frames_timestamps",
) -> list:
    """
    Write ms_axis-aligned arena CSVs to the block analysis folder (subfolder arena_events_ms).

    Only CSVs that have been verified for alignment are written (bug_trajectory, trials_data,
    screen_touches). app_events is not written until alignment is verified for it.

    Files are written as: analysis_path / arena_events_ms / <stem>_ms_axis.csv
    (e.g. bug_trajectory_ms_axis.csv, trials_data_ms_axis.csv, screen_touches_ms_axis.csv).

    Parameters
    ----------
    block
        Block with oe_events and sample_rate (e.g. BlockSync instance).
    arena_videos_dir : path-like
        Path to the block's arena_videos folder.
    analysis_path : path-like, optional
        Block analysis folder. If None, uses block.analysis_path.
    arena_ttl_col : str
        Column in oe_events for arena TTL sample numbers.
    frames_timestamps_subdir : str
        Subdirectory under arena_videos_dir containing frame timestamp CSVs.

    Returns
    -------
    list of Path
        Paths to the written CSV files.
    """
    arena_videos_dir = Path(arena_videos_dir)
    if analysis_path is None:
        analysis_path = Path(block.analysis_path)
    else:
        analysis_path = Path(analysis_path)

    out_dir = analysis_path / ARENA_EVENTS_MS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    first_oe_ms = get_first_arena_frame_oe_ms(block, arena_ttl_col=arena_ttl_col)
    median_pc_ms, _ = get_first_arena_frame_pc_ms(
        arena_videos_dir, frames_timestamps_subdir=frames_timestamps_subdir
    )
    shift_ms = compute_pc_to_oe_shift_ms(median_pc_ms, first_oe_ms)

    written = []
    for name in ARENA_CSV_NAMES_VERIFIED:
        path = arena_videos_dir / name
        if not path.exists():
            continue
        stem = path.stem
        out_name = f"{stem}_ms_axis.csv"
        out_path = out_dir / out_name
        if name == "trials_data.csv":
            df = load_and_align_arena_csv(
                path,
                shift_ms=shift_ms,
                time_columns=["start_time", "end_time"],
                ms_axis_suffix="",
            )
        else:
            df = load_and_align_arena_csv(
                path,
                shift_ms=shift_ms,
                time_columns=None,
                ms_axis_suffix="",
            )
        df.to_csv(out_path, index=False)
        written.append(out_path)

    if written:
        print(f"[Arena alignment] Wrote {len(written)} ms_axis-aligned CSV(s) to {out_dir}")
    return written
