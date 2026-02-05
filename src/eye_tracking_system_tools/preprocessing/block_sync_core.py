# ============================================================================
# Block synchronization core logic (sync, arena grid, verification, export, jitter)
# ============================================================================
# Moved from block_synchronization.ipynb for maintainability.
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# simple approach
# --- Simple eye synchronization to the Open Ephys (OE) timebase ---
# Algorithm:
#   (1) Read internal timestamps (seconds) for each eye; delta-analysis to verify stability.
#   (2) Get the FIRST TTL sample for that eye from oe_events.
#   (3) Place frame 0 at that TTL, and place frame i at:  first_TTL_sample + round(fs * (t_sec[i] - t_sec[0])).
#       (i.e., use internal timing deltas verbatim; no regression.)
#   (4) Attach brightness from block.<le/re>_frame_val_list.
#   (5) Return a tidy per-eye DataFrame indexed by OE samples (int64), with OE time in seconds and brightness.



def _get_fs(block) -> float:
    """Return Open Ephys sample rate (Hz)."""
    fs = getattr(block, 'sample_rate', None)
    if fs is None:
        fs = float(block.get_sample_rate())
        block.sample_rate = fs
    return float(fs)

def _locate_eye_timestamps_csv(mp4: Path) -> Path:
    """
    Find <stem>_timestamps.csv even if the mp4 is *_LE.mp4 / *_RE.mp4
    while the CSV is <base>_timestamps.csv.
    Uses case-insensitive matching so Timestamps.csv / .CSV are found on Linux.
    """
    mp4 = Path(mp4)
    stem = mp4.stem
    stripped = re.sub(r'([_\-]?)(LE|RE)$', '', stem, flags=re.IGNORECASE)
    # 1) exact in same folder (try common casing variants)
    for s in (stem, stripped):
        for ext in ("_timestamps.csv", "_Timestamps.csv", "_timestamps.CSV"):
            p = mp4.with_name(s + ext)
            if p.exists():
                return p
    # 2) fuzzy in same folder (case-sensitive glob)
    for pat in (f"{stripped}*timestamp*.csv", f"{stripped}*time*.csv",
                "*timestamp*.csv", "*time*.csv"):
        for p in mp4.parent.glob(pat):
            return p
    # 2b) case-insensitive fallback in same folder (e.g. Linux: Timestamps.csv, .CSV)
    for p in mp4.parent.iterdir():
        if p.is_file() and p.suffix.lower() == ".csv":
            if "timestamp" in p.stem.lower() or "time" in p.stem.lower():
                return p
    # 3) common subfolders
    for sub in ("timestamps", "frames_timestamps"):
        for d in (mp4.parent / sub, mp4.parent.parent / sub):
            if d.exists():
                for pat in (f"{stripped}*timestamp*.csv", f"{stripped}*time*.csv",
                            "*timestamp*.csv", "*time*.csv"):
                    for p in d.glob(pat):
                        return p
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() == ".csv":
                        if "timestamp" in p.stem.lower() or "time" in p.stem.lower():
                            return p
    # 4) LE/RE root recursive
    root = mp4.parents[1]
    for pat in (f"{stripped}*timestamp*.csv", f"{stripped}*time*.csv",
                "*timestamp*.csv", "*time*.csv"):
        for p in root.rglob(pat):
            return p
    for p in root.rglob("*.csv"):
        if p.is_file() and p.suffix.lower() == ".csv":
            if "timestamp" in p.stem.lower() or "time" in p.stem.lower():
                return p
    raise FileNotFoundError(f"Timestamp CSV not found near {mp4}")

def _normalize_to_seconds(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    dt = np.median(np.diff(t))

    # dt expected ~1/60 = 0.0167 s
    # If dt is ~16.7 -> ms
    # If dt is ~16666 -> us
    # If dt is ~1.666e7 -> ns
    if 0.005 < dt < 0.05:
        scale = 1.0          # already seconds
    elif 5 < dt < 50:
        scale = 1e-3         # ms -> s
    elif 5e3 < dt < 5e4:
        scale = 1e-6         # us -> s
    elif 5e6 < dt < 5e7:
        scale = 1e-9         # ns -> s
    else:
        # last resort: if values look like epoch seconds (1e9 range), keep but re-zero
        scale = 1.0

    t_s = t * scale
    # Re-zero to start at 0 to avoid epoch offsets
    t_s = t_s - t_s[0]
    return t_s

def _read_eye_internal_seconds(block, eye: str) -> np.ndarray:
    """
    Load per-frame internal timestamps for 'left' or 'right' eye and return as SECONDS.

    Robust to timestamp CSVs that store time in seconds / ms / us / ns (and to epoch-like offsets),
    by calling your pre-defined _normalize_to_seconds(t_raw).

    Requirements:
      - You already defined _normalize_to_seconds(t_raw: np.ndarray) -> np.ndarray
        which returns a strictly increasing array in seconds, re-zeroed at the first sample.

    Notes:
      - We try hard to pick a reasonable time column:
          1) prefer columns whose name suggests time (and prefer ones suggesting seconds)
          2) otherwise use the first numeric column
          3) last resort: read with header=None and take column 0
      - We validate monotonicity after normalization.
    """
    import re
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # Ensure eye videos are known
    if getattr(block, 'le_videos', None) is None or getattr(block, 're_videos', None) is None:
        block.handle_eye_videos()

    mp4 = Path(block.le_videos[0] if eye == 'left' else block.re_videos[0])
    csvp = _locate_eye_timestamps_csv(mp4)

    # Read CSV (header expected, but tolerate odd formats)
    df = pd.read_csv(csvp, engine="python")

    def _is_numeric_col(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)

    # ---- Choose candidate time columns ----
    cols = list(df.columns)

    # Score columns: higher = better
    def _score_col(cname: str) -> int:
        c = str(cname).strip().lower()
        score = 0

        # must look like time-ish
        if "time" in c or "timestamp" in c or c in ("t", "ts"):
            score += 10
        if "frame" in c and ("time" in c or "timestamp" in c):
            score += 3

        # prefer explicit seconds
        if re.search(r"(\bsec\b|\bsecs\b|\bsecond\b|\bseconds\b|_s\b|\bs\b)", c):
            score += 6
        # de-prefer ms/us/ns if seconds also exist in file
        if re.search(r"(\bms\b|_ms\b|\bmsec\b)", c):
            score += 2
        if re.search(r"(\bus\b|_us\b|\busec\b)", c):
            score += 1
        if re.search(r"(\bns\b|_ns\b|\bnsec\b)", c):
            score += 0

        return score

    # Filter to numeric columns
    numeric_cols = [c for c in cols if _is_numeric_col(df[c])]

    # If we have time-ish numeric columns, pick the best scored among them
    timeish_numeric_cols = [c for c in numeric_cols
                            if ("time" in str(c).lower()) or ("timestamp" in str(c).lower()) or str(c).lower() in ("t", "ts")]
    chosen_col = None

    if timeish_numeric_cols:
        chosen_col = sorted(timeish_numeric_cols, key=_score_col, reverse=True)[0]
    elif numeric_cols:
        # fallback: first numeric column
        chosen_col = numeric_cols[0]
    else:
        # last resort: no header / no usable dtypes
        df2 = pd.read_csv(csvp, header=None, engine="python")
        chosen_col = 0
        df = df2

    # Extract raw time vector
    t_raw = df[chosen_col].to_numpy(dtype='float64')

    # Basic validation pre-normalization
    if t_raw.size < 3 or np.any(~np.isfinite(t_raw)):
        raise ValueError(f"Bad or too-short timestamps in {csvp} (column={chosen_col}).")

    # Normalize units to seconds (your helper)
    t_sec = _normalize_to_seconds(t_raw)

    # Validate output
    if t_sec.size < 3 or np.any(~np.isfinite(t_sec)):
        raise ValueError(f"Normalized timestamps invalid in {csvp} (column={chosen_col}).")
    if not np.all(np.diff(t_sec) > 0):
        # Give a more informative error (how many non-increasing steps)
        d = np.diff(t_sec)
        n_bad = int(np.sum(d <= 0))
        raise ValueError(
            f"Timestamps not strictly increasing after normalization in {csvp} "
            f"(column={chosen_col}); non-increasing steps: {n_bad}."
        )

    return t_sec


def _delta_analysis(t_sec: np.ndarray, label: str, cov_warn: float = 0.05) -> dict:
    """
    Basic delta analysis: fps median, CoV, outlier rate.
    Prints a short report; returns metrics.
    """
    dt = np.diff(t_sec)
    fps = 1.0 / np.median(dt)
    cov = float(np.std(dt) / np.mean(dt)) if np.mean(dt) > 0 else np.inf
    p01, p99 = np.percentile(dt, [1, 99])
    out_frac = float(np.mean((dt < p01) | (dt > p99)))
    print(f"[{label}] frames={len(t_sec):,} | median fps={fps:.3f} | CoV(dt)={cov*100:.2f}% | outliers(±1–99%)={out_frac*100:.2f}%")
    if cov > cov_warn:
        print(f"[WARN] {label}: CoV(dt) > {cov_warn*100:.1f}%. Stream may be unstable.")
    return dict(fps=fps, cov=cov, out_frac=out_frac, dt=dt)

def _first_ttl_sample(block, eye: str) -> int:
    """Get the FIRST TTL (OE samples) for the given eye."""
    col = 'L_eye_TTL' if eye == 'left' else 'R_eye_TTL'
    s = block.oe_events[col].dropna().astype(int).to_numpy()
    if s.size == 0:
        raise RuntimeError(f"No TTLs found for {eye} eye in oe_events['{col}'].")
    return int(s[0])

def build_eye_df_simple(block, eye: str, cov_warn: float = 0.05) -> pd.DataFrame:
    """
    Make a per-eye DataFrame indexed by OE samples using the simple anchor-at-first-TTL approach.
    Columns: ['frame_idx', 'oe_time_s', 'brightness'].
    """
    fs = _get_fs(block)
    t_sec = _read_eye_internal_seconds(block, eye)
    
    _delta_analysis(t_sec, label=eye.upper(), cov_warn=cov_warn)

    # anchor: first TTL sample; place frame 0 at this time; others by internal deltas
    t0_oe = _first_ttl_sample(block, eye)            # samples
    t_rel = t_sec - t_sec[0]                         # seconds relative to first frame
    oe_samples = t0_oe + np.round(fs * t_rel).astype(np.int64)

    # brightness from BlockSync
    b_list = getattr(block, 'le_frame_val_list' if eye == 'left' else 're_frame_val_list')
    b = np.asarray(b_list, dtype='float64')
    n = min(len(b), len(oe_samples))
    if len(b) != len(oe_samples):
        print(f"[INFO] {eye.upper()}: brightness length ({len(b)}) != frames ({len(oe_samples)}); clipping to {n}.")
    oe_samples = oe_samples[:n]
    b = b[:n]

    # Construct DataFrame (deduplicate OE stamps if rounding collided)
    df = pd.DataFrame({'frame_idx': np.arange(n, dtype=int),
                       'oe_sample': oe_samples,
                       'brightness': b})
    # If any duplicate oe_sample due to rounding, keep the first
    df = df.sort_values('oe_sample').drop_duplicates('oe_sample', keep='first')
    df['oe_time_s'] = df['oe_sample'] / fs
    df = df.set_index('oe_sample')
    return df

def simple_sync_build(block, cov_warn: float = 0.05, export: bool = False):
    """
    Run the simple synchronization for both eyes and (optionally) export CSVs.
    Returns (df_left, df_right).
    """
    dfL = build_eye_df_simple(block, 'left', cov_warn=cov_warn)
    dfR = build_eye_df_simple(block, 'right', cov_warn=cov_warn)
    if export:
        outL = Path(block.analysis_path) / "eye_left_simple_sync.csv"
        outR = Path(block.analysis_path) / "eye_right_simple_sync.csv"
        dfL.to_csv(outL); dfR.to_csv(outR)
        print(f"[OK] Saved: {outL}")
        print(f"[OK] Saved: {outR}")
    return dfL, dfR


# ============================================================================
# Core merge / grid / verification / export
# ============================================================================

def _assert_strictly_increasing(name: str, arr: np.ndarray):
    if arr.size < 2 or not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing. Found non-monotonic sequence.")

def _build_arena_grid(block, target_fps: float):
    fs = _get_fs(block)

    arena = block.oe_events[['Arena_TTL','Arena_TTL_frame']].copy()
    arena = arena.dropna(subset=['Arena_TTL'])          # <- key change
    arena['Arena_TTL'] = arena['Arena_TTL'].astype(np.int64)
    arena = arena.sort_values('Arena_TTL')

    if len(arena) < 2:
        raise RuntimeError("Not enough Arena_TTL events to build a grid.")

    step  = int(round(fs / target_fps))
    start = int(arena['Arena_TTL'].iloc[0])
    stop  = int(arena['Arena_TTL'].iloc[-1])
    grid  = np.arange(start, stop + 1, step, dtype=np.int64)
    return fs, grid, step, arena


def _nearest_with_tol(sorted_vec: np.ndarray, queries: np.ndarray, tol: int) -> np.ndarray:
    """
    Return indices into sorted_vec of the nearest element to each query,
    but mark as -1 if the nearest is farther than tol (in samples).
    Assumes sorted_vec is strictly increasing (we assert that).
    """
    _assert_strictly_increasing("sorted_vec", sorted_vec)
    pos = np.searchsorted(sorted_vec, queries, side='left')
    pos0 = np.clip(pos - 1, 0, len(sorted_vec) - 1)
    pos1 = np.clip(pos,     0, len(sorted_vec) - 1)
    d0 = np.abs(sorted_vec[pos0] - queries)
    d1 = np.abs(sorted_vec[pos1] - queries)
    idx = np.where(d0 <= d1, pos0, pos1)
    d = np.minimum(d0, d1)
    idx[d > tol] = -1
    return idx

def _shift_eye_df_by_index(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """
    EXACT slider semantics on an eye df (index=oe_sample; cols: frame_idx, brightness, oe_time_s):
      - time index (oe_sample) and oe_time_s unchanged
      - shift BOTH frame_idx and brightness by `shift` along the current order
      - edges filled with NaN, no wrap
    """
    if shift == 0:
        return df.copy()
    df = df.copy()
    n = len(df)
    fi = df['frame_idx'].to_numpy(dtype=float)
    br = df['brightness'].to_numpy(dtype=float)
    fi_sh = np.full(n, np.nan, dtype=float)
    br_sh = np.full(n, np.nan, dtype=float)
    if shift > 0:
        fi_sh[shift:] = fi[:-shift]
        br_sh[shift:] = br[:-shift]
    else:
        s = -int(shift)
        fi_sh[:-s] = fi[s:]
        br_sh[:-s] = br[s:]
    df['frame_idx']  = fi_sh
    df['brightness'] = br_sh
    return df

def describe_eye_tick(df: pd.DataFrame) -> float:
    """
    Return the median sampling interval in milliseconds for this eye dataframe.
    df must have an 'oe_time_s' column (from simple_sync_build).
    """
    t = df['oe_time_s'].to_numpy(dtype=float)
    if len(t) < 2:
        return float('nan')
    return float(np.median(np.diff(t)) * 1000.0)

def shift_eye_df_by_index(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """
    Apply the SAME 'index-based shift' as the Bokeh slider to an eye dataframe:
      - Keep time ('oe_time_s') and the OE-sample index (df.index) unchanged.
      - Shift BOTH 'frame_idx' and 'brightness' by `shift` along the time-sorted order.
      - Fill the vacated edges with NaN (no wrap-around).
    Assumes df is the output of simple_sync_build (index = oe_sample, cols include 'frame_idx','brightness','oe_time_s').
    """
    if shift == 0:
        return df.copy()

    # Ensure sorted by time (simple_sync_build already gives it, but be safe)
    df = df.sort_index().copy()

    # Build new columns by discrete shift
    n = len(df)
    frame_idx = df['frame_idx'].to_numpy(dtype=float)   # float to allow NaN
    bright    = df['brightness'].to_numpy(dtype=float)

    shifted_idx = np.full(n, np.nan, dtype=float)
    shifted_y   = np.full(n, np.nan, dtype=float)

    if shift > 0:
        shifted_idx[shift:] = frame_idx[:-shift]
        shifted_y[shift:]   = bright[:-shift]
    else:
        s = -int(shift)
        shifted_idx[:-s] = frame_idx[s:]
        shifted_y[:-s]   = bright[s:]

    out = df.copy()
    out['frame_idx'] = shifted_idx
    out['brightness'] = shifted_y
    return out


# =============================================================================
# Automated LED alignment (first-blink to ON edge) and drift correction
# =============================================================================

def _get_led_on_edge_oe_samples(block, fs: float) -> np.ndarray:
    """
    Identify LED ON edges by interval only (robust to flipped rise/fall wiring).

    The LED is ON most of the time and blinks OFF for 34 ms every ~60 s. So:
    - An event followed by a LONG interval (>30 s) is LED_ON (light stays on for a minute).
    - An event followed by a SHORT interval (~34 ms) is LED_OFF (blink: light off then on).

    We do not use rise vs fall; wiring can be flipped so that rising = LED OFF.
    Returns OE sample numbers of each LED ON edge (when the light comes back on).
    """
    if 'LED_driver' not in block.oe_events.columns or 'LED_driver_fall' not in block.oe_events.columns:
        raise ValueError("block.oe_events must have 'LED_driver' and 'LED_driver_fall' columns")
    rise = block.oe_events['LED_driver'].dropna().astype(np.int64).values
    fall = block.oe_events['LED_driver_fall'].dropna().astype(np.int64).values
    samples = np.sort(np.concatenate([rise, fall]))
    if len(samples) < 2:
        return np.array([], dtype=np.int64)
    intervals = np.diff(samples)
    # Event followed by >30 s → LED_ON (light on for the long period)
    long_interval_min = int(30.0 * fs)
    on_edge_samples = []
    for i in range(len(intervals)):
        if intervals[i] >= long_interval_min:
            on_edge_samples.append(int(samples[i]))
    return np.array(on_edge_samples, dtype=np.int64)


def _get_led_off_oe_samples(block, fs: float, on_samples: np.ndarray) -> np.ndarray:
    """
    Return OE sample of LED-OFF (start of blink) for each LED ON.
    The blink is: OFF (light off) then 34 ms later ON (light on). So OFF is 34 ms *before* each ON.
    """
    off_delta = int(0.034 * fs)
    return np.asarray(on_samples, dtype=np.int64) - off_delta


def _get_eye_data_epoch_oe_seconds(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    fs: float,
) -> Tuple[float, float]:
    """
    Return the OE time range (seconds) where *both* eyes have data.
    Eye videos often start after OE recording; we only align LED events in this epoch.

    Returns
    -------
    (t_start, t_end) : float, float
        OE time in seconds. Only LED ON/OFF events with time in [t_start, t_end] are used.
    """
    for df in (df_left, df_right):
        if df is None or len(df) == 0:
            return np.nan, np.nan
    idxL = df_left.sort_index().index.to_numpy(dtype=np.int64)
    idxR = df_right.sort_index().index.to_numpy(dtype=np.int64)
    t_start_L = idxL.min() / fs
    t_end_L = idxL.max() / fs
    t_start_R = idxR.min() / fs
    t_end_R = idxR.max() / fs
    t_start = max(t_start_L, t_start_R)
    t_end = min(t_end_L, t_end_R)
    if t_start >= t_end:
        return np.nan, np.nan
    return float(t_start), float(t_end)


def _filter_led_events_to_eye_epoch(
    on_samples: np.ndarray,
    off_samples: np.ndarray,
    fs: float,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only LED ON/OFF events whose OE time falls within [t_start, t_end].
    Returns (on_filtered, off_filtered) with same length; peak_frames[0:len(on_filtered)] then
    correspond to these events (first blink in video = first LED in epoch).
    """
    if not (np.isfinite(t_start) and np.isfinite(t_end)) or t_start >= t_end:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    on_times_s = on_samples.astype(float) / fs
    mask = (on_times_s >= t_start) & (on_times_s <= t_end)
    return on_samples[mask], off_samples[mask]


def _expected_frame_at_led_on(
    oe_idx: np.ndarray, frame_idx: np.ndarray, peak_frames: np.ndarray, on_s: int
) -> float:
    """
    Which peak lands closest in OE time to this LED ON? Return that peak's frame index as expected.
    Tolerates extra/missing blinks (no strict peak_frames[k] = event k).
    """
    if len(peak_frames) == 0:
        return np.nan
    oe_at_peak = np.zeros(len(peak_frames))
    for j in range(len(peak_frames)):
        pj = int(np.argmin(np.abs(frame_idx - float(peak_frames[j]))))
        oe_at_peak[j] = oe_idx[pj]
    j_star = int(np.argmin(np.abs(oe_at_peak - on_s)))
    return float(peak_frames[j_star])


def compute_led_alignment_shift(
    block,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
) -> Tuple[int, int]:
    """
    Compute the whole-dataset shift (in rows) so that for each eye, the first
    LED blink's peak frame aligns with the *first* Open Ephys LED ON event in
    the epoch where both eyes have data. This keeps the 1:1 mapping (event k
    -> peak_frames[k]) used by drift correction consistent so the first event
    needs no (or minimal) corrections.

    Returns
    -------
    (shift_left, shift_right) : tuple of int
        Apply via shift_eye_df_by_index(dfL, shift_left), shift_eye_df_by_index(dfR, shift_right).
    """
    fs = _get_fs(block)
    on_samples_all = _get_led_on_edge_oe_samples(block, fs)
    if len(on_samples_all) == 0:
        return 0, 0
    on_samples_all = np.asarray(on_samples_all, dtype=np.int64)
    off_samples_all = _get_led_off_oe_samples(block, fs, on_samples_all)
    t_start, t_end = _get_eye_data_epoch_oe_seconds(df_left, df_right, fs)
    on_samples, _ = _filter_led_events_to_eye_epoch(
        on_samples_all, off_samples_all, fs, t_start, t_end
    )
    if len(on_samples) == 0:
        on_samples = on_samples_all

    def shift_for_eye(df: pd.DataFrame, peak_frames: np.ndarray) -> int:
        if len(peak_frames) == 0 or len(on_samples) == 0:
            return 0
        first_blink_frame = int(peak_frames[0])
        df = df.sort_index()
        oe_idx = df.index.to_numpy(dtype=np.int64)
        frame_idx = df['frame_idx'].to_numpy(dtype=float)
        valid = np.isfinite(frame_idx)
        if not np.any(valid):
            return 0
        # q: row where frame_idx is nearest to first_blink_frame (first blink in this eye's trace)
        dist = np.where(valid, np.abs(frame_idx - first_blink_frame), np.inf)
        q = int(np.argmin(dist))
        # Align first blink to the *first* LED ON in the epoch (on_samples[0]), so that
        # drift correction's 1:1 mapping (event k -> peak_frames[k]) is consistent.
        # Using "closest" LED ON would put peak_frames[0] at event j != 0, making event 0
        # appear to need maximal corrections and breaking sync.
        target_on_sample = int(on_samples[0])
        p = int(np.argmin(np.abs(oe_idx - target_on_sample)))
        return int(p - q)

    shift_left = shift_for_eye(df_left, getattr(block, 'led_blink_peak_frames_l', np.array([], dtype=int)))
    shift_right = shift_for_eye(df_right, getattr(block, 'led_blink_peak_frames_r', np.array([], dtype=int)))
    return shift_left, shift_right


def apply_drift_correction(
    block,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    max_event_offset_frac: float = 0.6,
    min_interval_frames: int = 1500,
    drift_threshold_frames: float = 1.0,
    tolerance_seconds: float = 0.017,
    max_corrections_per_event: int = 100,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correct cumulative drift so that each blink's brightness peak (in OE time)
    aligns with the corresponding LED-ON TTL time within a time tolerance.
    Inserts or removes one frame at a time (same as manual pipeline) until
    alignment is within tolerance or a per-event limit is reached.

    Alignment is measured in **time**: for each LED-ON event we find the OE time
    of the row where frame_idx is closest to that blink's peak frame; that time
    must be within tolerance_seconds of the LED-ON TTL time. Greater deviation
    triggers one insert or remove in the blink window, then re-evaluation.

    Parameters
    ----------
    block : BlockSync
        Must have oe_events (LED_driver, LED_driver_fall), led_blink_peak_frames_l/r,
        and sample_rate / get_sample_rate().
    df_left, df_right : pd.DataFrame
        Eye data from simple_sync_build (optionally after compute_led_alignment_shift).
    max_event_offset_frac, min_interval_frames, drift_threshold_frames : float, int
        Unused; kept for API compatibility.
    tolerance_seconds : float
        Acceptable alignment error in seconds (default 0.017 = 17 ms). Alignment
        is correct when |oe_time_of_peak - led_on_time| <= tolerance_seconds.
    max_corrections_per_event : int
        Maximum insert/remove operations per LED event (safety limit).
    verbose : bool
        Print summary and per-event diagnostics when corrections are applied.

    Returns
    -------
    (df_left_corrected, df_right_corrected) : tuple of pd.DataFrame
    """
    from eye_tracking_system_tools.preprocessing.block_sync_visualization import (
        insert_duplicate_frames_slide,
        remove_frame_at_pos,
    )

    fs = _get_fs(block)
    on_samples_all = _get_led_on_edge_oe_samples(block, fs)
    off_samples_all = _get_led_off_oe_samples(block, fs, on_samples_all)
    t_start, t_end = _get_eye_data_epoch_oe_seconds(df_left, df_right, fs)
    on_samples, off_samples = _filter_led_events_to_eye_epoch(
        on_samples_all, off_samples_all, fs, t_start, t_end
    )
    if len(on_samples) < 1:
        if verbose:
            print("[INFO] No LED ON edges in eye-data epoch; skipping drift correction.")
        return df_left.copy(), df_right.copy()

    on_times_s = on_samples.astype(float) / fs
    # Restrict peak search to a window around each blink so we don't pick a row from another blink
    margin_oe_samples = int(1.0 * fs)

    def _oe_time_at_peak_row(
        out: pd.DataFrame,
        peak_frame: float,
        fs: float,
        on_s: Optional[int] = None,
        off_s: Optional[int] = None,
    ) -> Tuple[float, int]:
        """OE time (seconds) of the row where frame_idx is closest to peak_frame.
        If on_s and off_s are given, only consider rows in [off_s - margin, on_s + margin]
        so we find the peak for *this* blink, not another one."""
        oe_idx = out.index.to_numpy(dtype=np.int64)
        frame_idx = out['frame_idx'].to_numpy(dtype=float)
        valid = np.isfinite(frame_idx)
        if not np.any(valid):
            return np.nan, -1
        dist = np.where(valid, np.abs(frame_idx - float(peak_frame)), np.inf)
        if on_s is not None and off_s is not None:
            in_window = (oe_idx >= off_s - margin_oe_samples) & (oe_idx <= on_s + margin_oe_samples)
            dist = np.where(in_window, dist, np.inf)
        if not np.any(np.isfinite(dist)):
            return np.nan, -1
        r = int(np.argmin(dist))
        oe_time = float(oe_idx[r]) / fs
        return oe_time, r

    def correct_drift_eye(df: pd.DataFrame, peak_frames: np.ndarray, label: str) -> pd.DataFrame:
        if len(peak_frames) == 0 or len(on_samples) == 0:
            return df.copy()
        out = df.sort_index().copy()
        n_events = min(len(on_samples), len(peak_frames), len(off_samples))
        peak_frames = peak_frames[:n_events]
        n_ins = 0
        n_rem = 0
        has_brightness = 'brightness' in out.columns
        for k in range(n_events):
            on_s = int(on_samples[k])
            off_s = int(off_samples[k])
            t_led_on = float(on_times_s[k])
            peak_frame_k = float(peak_frames[k])
            if not np.isfinite(peak_frame_k):
                continue
            oe_time_peak, _ = _oe_time_at_peak_row(out, peak_frame_k, fs, on_s=on_s, off_s=off_s)
            dt_start = oe_time_peak - t_led_on if np.isfinite(oe_time_peak) else 0.0
            n_corrected_this_event = 0
            prev_abs_dt = np.inf
            for _ in range(max_corrections_per_event):
                oe_idx = out.index.to_numpy(dtype=np.int64)
                frame_idx = out['frame_idx'].to_numpy(dtype=float)
                oe_time_peak, row_peak = _oe_time_at_peak_row(out, peak_frame_k, fs, on_s=on_s, off_s=off_s)
                if not np.isfinite(oe_time_peak):
                    break
                dt = oe_time_peak - t_led_on
                if abs(dt) <= tolerance_seconds:
                    break
                pos_after_off = int(np.searchsorted(oe_idx, off_s, side='right'))
                pos_at_on = int(np.searchsorted(oe_idx, on_s, side='left'))
                pos_at_on = int(np.clip(pos_at_on, 0, len(oe_idx) - 1))
                if pos_at_on > 0 and abs(oe_idx[pos_at_on - 1] - on_s) < abs(oe_idx[pos_at_on] - on_s):
                    pos_at_on = pos_at_on - 1
                if pos_after_off >= len(oe_idx) or pos_after_off <= 0:
                    break
                if abs(dt) >= prev_abs_dt:
                    break
                prev_abs_dt = abs(dt)
                if dt > 0:
                    # Peak is late (peak OE time > LED ON): remove frame so peak moves earlier in OE time
                    out = remove_frame_at_pos(out, [pos_after_off], mode='pos', verbose=False)
                    n_rem += 1
                    n_corrected_this_event += 1
                else:
                    # Peak is early: insert frame so peak moves later in OE time
                    if has_brightness:
                        brightness = out['brightness'].to_numpy(dtype=float)
                        in_blink = (oe_idx >= off_s) & (oe_idx <= on_s)
                        if np.any(in_blink):
                            b = np.nan_to_num(brightness, nan=np.inf, posinf=np.inf, neginf=np.inf)
                            blink_b = np.where(in_blink, b, np.inf)
                            pos_insert = int(np.argmin(blink_b))
                        else:
                            pos_insert = pos_after_off
                    else:
                        pos_insert = pos_after_off
                    if pos_insert >= pos_at_on:
                        pos_insert = max(0, pos_after_off)
                    out = insert_duplicate_frames_slide(
                        out, [pos_insert], mode='pos', duplicate='current', verbose=False
                    )
                    n_ins += 1
                    n_corrected_this_event += 1
            if verbose and n_corrected_this_event > 0:
                oe_time_peak_end, _ = _oe_time_at_peak_row(out, peak_frame_k, fs, on_s=on_s, off_s=off_s)
                dt_end = oe_time_peak_end - t_led_on if np.isfinite(oe_time_peak_end) else np.nan
                print(
                    f"[DIAG] Drift {label} event k={k}: corrections={n_corrected_this_event}, "
                    f"dt_start={dt_start*1000:.1f}ms, dt_end={dt_end*1000:.1f}ms (tolerance ±{tolerance_seconds*1000:.0f}ms)"
                )
            if n_corrected_this_event >= max_corrections_per_event and verbose:
                oe_time_peak_end, _ = _oe_time_at_peak_row(out, peak_frame_k, fs, on_s=on_s, off_s=off_s)
                dt_end = oe_time_peak_end - t_led_on if np.isfinite(oe_time_peak_end) else np.nan
                if abs(dt_end) > tolerance_seconds:
                    print(
                        f"[WARN] Drift correction {label}: hit limit at event k={k} "
                        f"(remaining dt={dt_end*1000:.1f}ms). Check sync or increase max_corrections_per_event."
                    )
        if verbose and (n_ins or n_rem):
            print(f"[INFO] Drift correction {label} (vs LED): inserted {n_ins}, removed {n_rem} frame(s)")
        return out

    peak_l = getattr(block, 'led_blink_peak_frames_l', np.array([], dtype=int))
    peak_r = getattr(block, 'led_blink_peak_frames_r', np.array([], dtype=int))
    dfL_out = correct_drift_eye(df_left, peak_l, "LEFT")
    dfR_out = correct_drift_eye(df_right, peak_r, "RIGHT")
    return dfL_out, dfR_out


def compute_drift_correction_diagnostics(
    block,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    drift_threshold_frames: float = 1.0,
    tolerance_seconds: float = 0.017,
) -> Dict[str, Any]:
    """
    Compute per-event alignment diagnostics without modifying data (for plots and debugging).
    Uses the same time-based criterion as apply_drift_correction: alignment is the OE time
    of the row where frame_idx is closest to the blink peak, vs LED-ON time.

    Returns
    -------
    dict with keys:
        fs, on_samples, off_samples, on_times_s, off_times_s, eye_epoch_*
        left_events, right_events : list of dicts with k, oe_time_s_led_on, oe_time_s_expected,
        dt_ms (alignment error in ms), within_tolerance, current_frame, expected_frame, etc.
    """
    fs = _get_fs(block)
    on_samples_all = _get_led_on_edge_oe_samples(block, fs)
    off_samples_all = _get_led_off_oe_samples(block, fs, on_samples_all)
    t_start, t_end = _get_eye_data_epoch_oe_seconds(df_left, df_right, fs)
    on_samples, off_samples = _filter_led_events_to_eye_epoch(
        on_samples_all, off_samples_all, fs, t_start, t_end
    )
    on_times_s = on_samples.astype(float) / fs
    off_times_s = off_samples.astype(float) / fs
    margin_oe_samples = int(1.0 * fs)

    def diagnose_eye(df: pd.DataFrame, peak_frames: np.ndarray) -> List[Dict[str, Any]]:
        events = []
        if len(peak_frames) == 0 or len(on_samples) == 0:
            return events
        df = df.sort_index()
        oe_idx = df.index.to_numpy(dtype=np.int64)
        frame_idx = df['frame_idx'].to_numpy(dtype=float)
        oe_time_s = df['oe_time_s'].to_numpy(dtype=float) if 'oe_time_s' in df.columns else (oe_idx.astype(float) / fs)
        n_events = min(len(on_samples), len(peak_frames), len(off_samples))
        peak_frames = peak_frames[:n_events]
        for k in range(n_events):
            on_s = int(on_samples[k])
            off_s = int(off_samples[k])
            pos_at_on = np.searchsorted(oe_idx, on_s, side='left')
            pos_at_on = int(np.clip(pos_at_on, 0, len(oe_idx) - 1))
            if pos_at_on > 0 and abs(oe_idx[pos_at_on - 1] - on_s) < abs(oe_idx[pos_at_on] - on_s):
                pos_at_on = pos_at_on - 1
            current_frame = frame_idx[pos_at_on]
            expected_frame = float(peak_frames[k])
            oe_time_s_current = float(oe_time_s[pos_at_on]) if pos_at_on < len(oe_time_s) else np.nan
            oe_time_s_led_on = on_times_s[k]

            # OE time of the row where the peak frame lands, restricted to this blink's window
            if np.isfinite(expected_frame):
                dist = np.abs(frame_idx - expected_frame)
                in_window = (oe_idx >= off_s - margin_oe_samples) & (oe_idx <= on_s + margin_oe_samples)
                dist = np.where(in_window, dist, np.inf)
                if np.any(np.isfinite(dist)):
                    pos_expected = int(np.argmin(dist))
                    oe_time_s_expected = float(oe_time_s[pos_expected])
                else:
                    oe_time_s_expected = np.nan
            else:
                oe_time_s_expected = np.nan

            dt_s = (oe_time_s_expected - oe_time_s_led_on) if np.isfinite(oe_time_s_expected) else np.nan
            dt_ms = dt_s * 1000.0 if np.isfinite(dt_s) else np.nan
            within_tolerance = abs(dt_s) <= tolerance_seconds if np.isfinite(dt_s) else False
            error_frames = current_frame - expected_frame if (np.isfinite(current_frame) and np.isfinite(expected_frame)) else np.nan
            pos_after_off = np.searchsorted(oe_idx, off_s, side='right')
            if not within_tolerance and 0 < pos_after_off < len(oe_idx):
                correction = 'remove' if dt_s > 0 else 'insert'
            else:
                correction = 'none'

            events.append({
                'k': k,
                'oe_time_s_led_on': oe_time_s_led_on,
                'oe_time_s_current': oe_time_s_current,
                'oe_time_s_expected': oe_time_s_expected,
                'dt_s': dt_s,
                'dt_ms': dt_ms,
                'within_tolerance': within_tolerance,
                'current_frame': current_frame,
                'expected_frame': expected_frame,
                'error_frames': error_frames,
                'correction': correction,
                'pos_at_on': pos_at_on,
                'pos_after_off': pos_after_off,
            })
        return events

    peak_l = getattr(block, 'led_blink_peak_frames_l', np.array([], dtype=int))
    peak_r = getattr(block, 'led_blink_peak_frames_r', np.array([], dtype=int))
    return {
        'fs': fs,
        'on_samples': on_samples,
        'off_samples': off_samples,
        'on_times_s': on_times_s,
        'off_times_s': off_times_s,
        'tolerance_seconds': tolerance_seconds,
        'eye_epoch_t_start': t_start,
        'eye_epoch_t_end': t_end,
        'left_events': diagnose_eye(df_left, peak_l),
        'right_events': diagnose_eye(df_right, peak_r),
    }


def compute_sync_self_verification_correlation(
    block,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    window_seconds: float = 0.5,
    min_correlation_threshold: float = 0.4,
    interp_hz: float = 100.0,
) -> Dict[str, Any]:
    """
    Self-verification: check that corrected L/R brightness vectors correlate around LED blink events.
    When sync is correct, both eyes should dip together with similar magnitude at each blink.

    Parameters
    ----------
    block : BlockSync-like
        Used for LED ON times (via _get_led_on_edge_oe_samples and epoch filtering).
    df_left, df_right : pd.DataFrame
        Corrected eye data with columns oe_time_s (or index/sample rate), brightness.
    window_seconds : float
        Half-window around each LED ON time (s). Correlation is computed in [t_led - window, t_led + window].
    min_correlation_threshold : float
        Sync is considered passed when mean correlation across events >= this (default 0.4).
    interp_hz : float
        Sample rate (Hz) of the common time grid used for interpolation before correlation.

    Returns
    -------
    dict with keys:
        passed : bool
            True if mean_correlation >= min_correlation_threshold.
        mean_correlation : float
            Mean of per-event correlations (NaN if no valid events).
        per_event_correlation : list of float
            Correlation in each blink window.
        min_correlation_threshold : float
            Echo of the threshold used.
        n_events : int
            Number of LED events used.
    """
    fs = _get_fs(block)
    on_samples_all = _get_led_on_edge_oe_samples(block, fs)
    off_samples_all = _get_led_off_oe_samples(block, fs, on_samples_all)
    t_start, t_end = _get_eye_data_epoch_oe_seconds(df_left, df_right, fs)
    on_samples, _ = _filter_led_events_to_eye_epoch(
        on_samples_all, off_samples_all, fs, t_start, t_end
    )
    on_times_s = on_samples.astype(float) / fs

    def _oe_time_s(df: pd.DataFrame) -> np.ndarray:
        if "oe_time_s" in df.columns:
            return df["oe_time_s"].to_numpy(dtype=float)
        return df.index.to_numpy(dtype=np.int64).astype(float) / fs

    tL = _oe_time_s(df_left)
    bL = df_left["brightness"].to_numpy(dtype=float)
    tR = _oe_time_s(df_right)
    bR = df_right["brightness"].to_numpy(dtype=float)

    per_event = []
    for t_led in on_times_s:
        t_lo, t_hi = t_led - window_seconds, t_led + window_seconds
        maskL = (tL >= t_lo) & (tL <= t_hi) & np.isfinite(bL)
        maskR = (tR >= t_lo) & (tR <= t_hi) & np.isfinite(bR)
        if np.sum(maskL) < 5 or np.sum(maskR) < 5:
            per_event.append(np.nan)
            continue
        # Common time grid in this window; interp requires increasing xp
        n_pts = max(10, int(2 * window_seconds * interp_hz))
        t_grid = np.linspace(t_lo, t_hi, n_pts)
        sL = np.argsort(tL[maskL])
        tL_w = np.sort(tL[maskL])
        bL_w = bL[maskL][sL]
        sR = np.argsort(tR[maskR])
        tR_w = np.sort(tR[maskR])
        bR_w = bR[maskR][sR]
        vL = np.interp(t_grid, tL_w, bL_w)
        vR = np.interp(t_grid, tR_w, bR_w)
        if np.std(vL) < 1e-10 or np.std(vR) < 1e-10:
            per_event.append(np.nan)
            continue
        r = np.corrcoef(vL, vR)[0, 1]
        per_event.append(float(r) if np.isfinite(r) else np.nan)

    valid = [c for c in per_event if np.isfinite(c)]
    mean_corr = float(np.mean(valid)) if valid else np.nan
    passed = np.isfinite(mean_corr) and mean_corr >= min_correlation_threshold

    return {
        "passed": bool(passed),
        "mean_correlation": mean_corr,
        "per_event_correlation": per_event,
        "min_correlation_threshold": min_correlation_threshold,
        "n_events": len(on_times_s),
    }


def build_final_sync_df_merge_nearest(
    block,
    df_left:  Optional[pd.DataFrame],
    df_right: Optional[pd.DataFrame],
    target_fps: float = 60.0,
    tol_frac: float = 0.9,          # nearest is accepted if within tol_frac * tick
    pre_shift_left:  int = 0,       # apply EXACT slider-like index shift BEFORE merge
    pre_shift_right: int = 0,
    export_csv: bool = True,
    csv_name: str = "blocksync_df.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build downstream-compatible final_sync_df by merging df_left/df_right onto a 60 Hz Arena grid
    using nearest-with-tolerance, without re-sorting the eye dataframes.

    Expects df_left/df_right from simple_sync_build (index=oe_sample; cols: frame_idx, oe_time_s, brightness).
    """
    # 1) Build Arena grid
    fs, grid, step, arena = _build_arena_grid(block, target_fps=target_fps)
    tol = int(np.ceil(tol_frac * step))  # in samples
    tick_ms = 1000.0 * step / fs

    # 2) Eye dfs: sanity + optional pre-merge slider-like shifts (NO sorting is performed)
    for nm, df in (("LEFT", df_left), ("RIGHT", df_right)):
        if not isinstance(df.index.values, np.ndarray):
            raise ValueError(f"{nm}: df.index must be OE samples.")
        _assert_strictly_increasing(f"{nm} df.index (oe_sample)", df.index.values.astype(np.int64))
        if not {'frame_idx','brightness','oe_time_s'}.issubset(df.columns):
            raise ValueError(f"{nm}: df must include 'frame_idx','brightness','oe_time_s'.")

    if pre_shift_left:
        if verbose: print(f"[INFO] Pre-shifting LEFT by {pre_shift_left} ticks (slider semantics).")
        df_left = _shift_eye_df_by_index(df_left, pre_shift_left)
    if pre_shift_right:
        if verbose: print(f"[INFO] Pre-shifting RIGHT by {pre_shift_right} ticks (slider semantics).")
        df_right = _shift_eye_df_by_index(df_right, pre_shift_right)

    # 3) Map Arena frames to the grid (nearest-with-tolerance)
    a_times  = arena['Arena_TTL'].to_numpy(dtype=np.int64)
    a_frames = arena['Arena_TTL_frame'].to_numpy(dtype=np.int64)
    idxA = _nearest_with_tol(a_times, grid, tol)
    arena_frame = np.full(grid.shape, np.nan, dtype=float)
    okA = idxA >= 0
    arena_frame[okA] = a_frames[idxA[okA]]

    # 4) Nearest-with-tolerance merge for LEFT and RIGHT (NO resort of dfs)
    def _map_eye_to_grid(df_eye: pd.DataFrame):
        t   = df_eye.index.to_numpy(dtype=np.int64)       # oe_sample per frame
        fi  = df_eye['frame_idx'].to_numpy(dtype=float)
        val = df_eye['brightness'].to_numpy(dtype=float)
        idx = _nearest_with_tol(t, grid, tol)             # indices into t
        frames = np.full(grid.shape, np.nan, dtype=float)
        vals   = np.full(grid.shape, np.nan, dtype=float)
        ok = idx >= 0
        frames[ok] = fi[idx[ok]]
        vals[ok]   = val[idx[ok]]
        return frames, vals

    L_eye_frame, L_values = _map_eye_to_grid(df_left)
    R_eye_frame, R_values = _map_eye_to_grid(df_right)

    # 5) Assemble final df (identical column names/format to legacy)
    final_df = pd.DataFrame({
        'Arena_TTL':   grid.astype(float),   # float to mimic legacy style (… .0)
        'Arena_frame': arena_frame,
        'L_eye_frame': L_eye_frame,
        'R_eye_frame': R_eye_frame,
        'L_values':    L_values,
        'R_values':    R_values,
    })

    # 6) Save & attach
    if export_csv:
        outp = Path(block.analysis_path) / csv_name
        final_df.to_csv(outp, index=False)
        if verbose: print(f"[OK] Saved final sync CSV → {outp}")
    block.final_sync_df = final_df

    if verbose:
        print(f"[INFO] Grid rows: {len(grid):,} | tick ≈ {tick_ms:.3f} ms | tol={tol} samp (tol_frac={tol_frac:.2f})")
        nL = int(np.sum(np.isfinite(L_values))); nR = int(np.sum(np.isfinite(R_values)))
        print(f"[INFO] Valid LEFT grid points: {nL:,} | Valid RIGHT grid points: {nR:,}")

    return final_df


@dataclass
class ArenaGridInfo:
    fs_hz: float
    inferred_arena_fps: float
    inferred_arena_step_samp: int
    target_fps: float
    used_pseudo_60hz: bool
    start_samp: int
    end_samp: int
    n_grid: int


def _infer_ttl_fps(samples: np.ndarray, fs_hz: float) -> Tuple[float, int]:
    """
    Robust estimate of TTL cadence: inferred_fps = fs / median(diff(samples)).
    Returns (fps, median_step_samples).
    """
    samples = np.asarray(samples, dtype=np.int64)
    samples = np.sort(samples)
    if samples.size < 3:
        return float("nan"), -1
    dt = np.diff(samples)
    # ignore zeros/negatives just in case (shouldn't happen)
    dt = dt[dt > 0]
    if dt.size < 2:
        return float("nan"), -1
    step = int(np.median(dt))
    fps = float(fs_hz / step) if step > 0 else float("nan")
    return fps, step


def build_arena_grid_df(
    block,
    target_fps: float = 60.0,
    arena_fps_tol_hz: float = 5.0,
    arena_ttl_col: str = "Arena_TTL",
    arena_frame_col: str = "Arena_TTL_frame",
    prefer_valid_window: bool = True,
    verbose: bool = True,
    attach_to_block: bool = True,
) -> Tuple[pd.DataFrame, ArenaGridInfo]:
    """
    Create an arena 'master grid' DataFrame on the Open Ephys sample axis.

    If inferred arena fps is within ±arena_fps_tol_hz of target_fps:
        - use the arena TTL-derived grid (legacy paradigm)
        - the returned df represents the arena's native stream (≈60Hz)
    Else (arena slower/faster than expected):
        - create a pseudo 60Hz grid from arena-valid time window
        - upsample arena frames onto it by nearest assignment (duplicates frames)

    Returns
    -------
    arena_grid_df : DataFrame indexed by OE samples (int64)
        Columns:
          - 'grid_sample' (index duplicate)
          - 'grid_time_s'
          - 'Arena_frame_60'   : 0..N-1 (frame index on the 60Hz grid)
          - 'Arena_frame_src'  : mapped original arena frame number (duplicates allowed)
          - 'Arena_ttl_src'    : OE sample timestamp of the source arena TTL nearest each grid tick
    info : ArenaGridInfo
    """

    if getattr(block, "oe_events", None) is None:
        raise RuntimeError("block.oe_events is missing. Run block.parse_open_ephys_events() first.")
    fs = float(getattr(block, "sample_rate", None) or block.get_sample_rate())

    oe = block.oe_events

    if arena_ttl_col not in oe.columns:
        raise RuntimeError(f"'{arena_ttl_col}' not found in block.oe_events columns.")

    # Use TTL timestamps that exist; optionally restrict to the "valid window" as defined by your parser cleaning
    arena = oe[[arena_ttl_col] + ([arena_frame_col] if arena_frame_col in oe.columns else [])].copy()
    arena = arena.dropna(subset=[arena_ttl_col])
    arena[arena_ttl_col] = arena[arena_ttl_col].astype(np.int64)
    arena = arena.sort_values(arena_ttl_col)

    if len(arena) < 3:
        raise RuntimeError("Not enough arena TTL events to infer fps / build grid.")

    # If parser set arena_frame to NaN outside window, we can use that to find the valid window
    if prefer_valid_window and (arena_frame_col in arena.columns):
        # keep only TTLs that have a non-NaN frame counter (i.e., in-window after your cleaning)
        arena_in = arena.dropna(subset=[arena_frame_col]).copy()
        if len(arena_in) >= 3:
            arena_use = arena_in
        else:
            arena_use = arena
    else:
        arena_use = arena

    a_samp = arena_use[arena_ttl_col].to_numpy(dtype=np.int64)
    fps_a, step_a = _infer_ttl_fps(a_samp, fs)

    if verbose:
        print(f"[arena] inferred fps ≈ {fps_a:.3f} Hz (median step {step_a} samples @ fs={fs:.1f} Hz)")

    # Define the usable sync window for building the grid
    start_samp = int(a_samp[0])
    end_samp   = int(a_samp[-1])

    # Decide: native vs pseudo
    use_native = np.isfinite(fps_a) and (abs(fps_a - target_fps) <= arena_fps_tol_hz)

    if use_native:
        # Native-like grid: based on target_fps step, but spanning native window.
        # (If fps_a is ~60, this matches your legacy expectation.)
        step_grid = int(round(fs / float(target_fps)))
        grid = np.arange(start_samp, end_samp + 1, step_grid, dtype=np.int64)
        used_pseudo = False
    else:
        # Pseudo 60Hz grid spanning the arena window
        step_grid = int(round(fs / float(target_fps)))
        grid = np.arange(start_samp, end_samp + 1, step_grid, dtype=np.int64)
        used_pseudo = True
        if verbose:
            print(f"[arena] arena fps not ~{target_fps}. Building pseudo {target_fps}Hz grid over window.")

    # Map each grid tick to a source arena TTL (nearest)
    # This is the key step that duplicates arena frames when arena is slower (e.g., 15Hz)
    a_samp_full = arena_use[arena_ttl_col].to_numpy(dtype=np.int64)
    a_frame_full = None
    if arena_frame_col in arena_use.columns:
        a_frame_full = arena_use[arena_frame_col].to_numpy(dtype=float)  # may have NaNs
    else:
        # if no frame col, create one as 0..n-1 for the arena TTL list
        a_frame_full = np.arange(len(arena_use), dtype=float)

    # nearest mapping via searchsorted
    pos = np.searchsorted(a_samp_full, grid, side="left")
    pos0 = np.clip(pos - 1, 0, len(a_samp_full) - 1)
    pos1 = np.clip(pos,     0, len(a_samp_full) - 1)
    d0 = np.abs(a_samp_full[pos0] - grid)
    d1 = np.abs(a_samp_full[pos1] - grid)
    idx = np.where(d0 <= d1, pos0, pos1)

    arena_ttl_src = a_samp_full[idx]
    arena_frame_src = a_frame_full[idx]

    arena_grid_df = pd.DataFrame({
        "grid_sample": grid.astype(np.int64),
        "grid_time_s": grid.astype(np.float64) / fs,
        "Arena_frame_60": np.arange(len(grid), dtype=np.int64),  # the 60Hz grid frame index
        "Arena_frame_src": arena_frame_src,                      # original arena frame index (duplicates allowed)
        "Arena_ttl_src": arena_ttl_src.astype(np.int64),
    }).set_index("grid_sample")

    info = ArenaGridInfo(
        fs_hz=fs,
        inferred_arena_fps=float(fps_a),
        inferred_arena_step_samp=int(step_a),
        target_fps=float(target_fps),
        used_pseudo_60hz=bool(used_pseudo),
        start_samp=int(start_samp),
        end_samp=int(end_samp),
        n_grid=int(len(grid)),
    )

    if attach_to_block:
        block.arena_grid_df = arena_grid_df
        block.arena_grid_info = info

    if verbose:
        src_unique = int(pd.Series(arena_frame_src).nunique(dropna=True))
        print(f"[arena] grid rows={len(arena_grid_df):,} | unique source arena frames mapped={src_unique:,}")
        if used_pseudo:
            print("[arena] NOTE: Arena_frame_src will repeat (upsampled arena). Use Arena_frame_60 as the 60Hz master frame index.")

    return arena_grid_df, info


# ============================================================================
# VERIFICATION AND EXPORT FUNCTIONS
# ============================================================================

def verify_final_df_against_sources(block, final_df, df_left, df_right, target_fps=60.0, tol_frac=0.9):
    """Verify that final_df matches recomputed mapping from source eye dataframes."""
    fs, grid, step, arena = _build_arena_grid(block, target_fps)
    tol = int(np.ceil(tol_frac * step))

    ft = np.asarray(final_df['Arena_TTL'], dtype=float)

    if len(ft) != len(grid):
        print("[VERIFY] Length mismatch:")
        print("  len(final_df Arena_TTL) =", len(ft))
        print("  len(recomputed grid)    =", len(grid))
        raise AssertionError("Grid length mismatch.")

    if not np.allclose(ft, grid.astype(float), rtol=0, atol=0.5):
        bad = np.where(np.abs(ft - grid.astype(float)) > 0.5)[0][:10]
        print("[VERIFY] Arena_TTL mismatch at indices:", bad)
        raise AssertionError("final_df['Arena_TTL'] does not match recomputed grid.")

    def _nearest_with_tol(sorted_vec, queries, tol):
        _assert_strictly_increasing("sorted_vec", sorted_vec)
        pos = np.searchsorted(sorted_vec, queries, 'left')
        pos0 = np.clip(pos-1, 0, len(sorted_vec)-1)
        pos1 = np.clip(pos,   0, len(sorted_vec)-1)
        d0 = np.abs(sorted_vec[pos0] - queries)
        d1 = np.abs(sorted_vec[pos1] - queries)
        idx = np.where(d0 <= d1, pos0, pos1)
        d = np.minimum(d0, d1)
        idx[d > tol] = -1
        return idx

    def map_eye(df_eye):
        t = df_eye.sort_index().index.to_numpy(dtype=np.int64)
        _assert_strictly_increasing("df_eye.index (oe_sample)", t)
        fi = df_eye.sort_index()['frame_idx'].to_numpy(dtype=float)
        y  = df_eye.sort_index()['brightness'].to_numpy(dtype=float)
        idx = _nearest_with_tol(t, grid, tol)
        frames = np.full(grid.shape, np.nan, dtype=float)
        vals   = np.full(grid.shape, np.nan, dtype=float)
        ok = idx >= 0
        frames[ok] = fi[idx[ok]]
        vals[ok]   = y[idx[ok]]
        return frames, vals

    Lf, Lv = map_eye(df_left)
    Rf, Rv = map_eye(df_right)

    Lf0 = final_df['L_eye_frame'].to_numpy(dtype=float)
    Lv0 = final_df['L_values'].to_numpy(dtype=float)
    Rf0 = final_df['R_eye_frame'].to_numpy(dtype=float)
    Rv0 = final_df['R_values'].to_numpy(dtype=float)

    def nan_equal(a,b):
        return ((a==b) | (np.isnan(a) & np.isnan(b)))

    mLf = nan_equal(Lf, Lf0).mean()
    mLv = nan_equal(Lv, Lv0).mean()
    mRf = nan_equal(Rf, Rf0).mean()
    mRv = nan_equal(Rv, Rv0).mean()

    n = len(grid)
    print(f"[VERIFY] Grid length={n}, tick≈{1000.0*step/fs:.3f} ms, tol={tol} samples")
    print(f"[VERIFY] Left  frame match: {mLf*100:.3f}%   Left  values match: {mLv*100:.3f}%")
    print(f"[VERIFY] Right frame match: {mRf*100:.3f}%   Right values match: {mRv*100:.3f}%")

    return dict(
        tick_ms=1000.0*step/fs, tol_samples=tol,
        left_frame_match=mLf, left_values_match=mLv,
        right_frame_match=mRf, right_values_match=mRv
    )


def export_final_sync_df(block,
                         final_df: pd.DataFrame,
                         overwrite: bool = True,
                         filenames = ("blocksync_df.csv", "final_sync_df.csv"),
                         ms_axis=True) -> None:
    """
    Save `final_df` into block.analysis_path and set block.final_sync_df and block.blocksync_df.
    """
    required = ['Arena_TTL','Arena_frame','L_eye_frame','R_eye_frame','L_values','R_values']
    missing = [c for c in required if c not in final_df.columns]
    if missing:
        raise ValueError(f"final_df missing required columns: {missing}")

    out = final_df.copy()
    out['Arena_TTL'] = out['Arena_TTL'].astype(float)
    if ms_axis:
        out['ms_axis'] = out['Arena_TTL'] / (block.sample_rate/1000)
    
    ap = Path(block.analysis_path)
    ap.mkdir(parents=True, exist_ok=True)

    for name in filenames:
        p = ap / name
        existed = p.exists()
        if existed and not overwrite:
            print(f"[SKIP] {p.name} exists and overwrite=False")
            continue
        out.to_csv(p, index=False)
        print(f"[OK] {'Overwrote' if existed else 'Wrote'} {p}")

    block.final_sync_df = out
    block.blocksync_df = out
    print("[OK] block.final_sync_df (and block.blocksync_df) set.")


def load_final_sync_df(block, filename=None, verbose=True):
    """
    Load a downstream-compatible final sync dataframe from disk and set block.final_sync_df and block.blocksync_df.
    """
    ap = Path(block.analysis_path)
    candidates = [filename] if filename else ["final_sync_df.csv", "blocksync_df.csv"]
    path = None
    for name in candidates:
        p = ap / name
        if p.exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"No sync file found. Tried: {', '.join(str(ap / n) for n in candidates)}")

    df = pd.read_csv(path)
    required = ['Arena_TTL','Arena_frame','L_eye_frame','R_eye_frame','L_values','R_values']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df.copy()
    df['Arena_TTL'] = df['Arena_TTL'].astype(float)

    block.final_sync_df = df
    block.blocksync_df = df
    if verbose:
        print(f"[OK] Loaded {path.name} → block.final_sync_df (rows={len(df):,})")

    return df


# ============================================================================
# JITTER ANALYSIS FUNCTIONS
# ============================================================================

def create_distance_plot(distances, top_dist_to_show=500):
    """Create cumulative distribution and histogram plots for jitter distances."""
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, figsize=(6, 6), dpi=150)

    axs[0].set_title('Cumulative Euclidean Distances for Camera Jitter', fontsize=15)
    axs[0].set_ylabel('Cumulative \n % of Frames')
    axs[0].set_xlim(0, top_dist_to_show)
    axs[0].grid(False)

    sns.kdeplot(distances, cumulative=True, label='Left Eye', ax=axs[0], linewidth=4, c='black')

    axs[1].hist(distances, bins=np.linspace(0, top_dist_to_show, 20), log=False, color='black')
    axs[1].set_title('Image displacement histogram', fontsize=15)
    axs[1].set_xlabel(r'Euclidean Displacement [$\mu$m]', fontsize=15)
    axs[1].set_ylabel('Frame count', fontsize=15)
    axs[1].tick_params(axis='x', which='major', labelsize=15)
    axs[1].set_facecolor('white')
    axs[1].title.set_color('black')
    axs[1].xaxis.label.set_color('black')
    axs[1].yaxis.label.set_color('black')
    axs[1].tick_params(colors='black')
    axs[1].grid(False)

    plt.tight_layout()
    return fig, axs


def add_intermediate_elements(input_vector, gap_to_bridge):
    """Add intervening elements to bridge gaps in a vector."""
    differences = np.diff(input_vector)
    output_vector = [input_vector[0]]
    for i, diff in enumerate(differences):
        if diff < gap_to_bridge:
            output_vector.extend(range(input_vector[i] + 1, input_vector[i + 1]))
        output_vector.append(input_vector[i + 1])
    return np.sort(np.unique(output_vector))


def find_jittery_frames(block, eye, max_distance, diff_threshold, gap_to_bridge=6):
    """Find jittery frames based on distance thresholds and return indices to remove."""
    if eye not in ['left', 'right']:
        print(f'eye can only be left/right, your input: {eye}')
        return None
    
    if eye == 'left':
        jitter_dict = block.le_jitter_dict
        eye_frame_col = 'L_eye_frame'
    elif eye == 'right':
        jitter_dict = block.re_jitter_dict
        eye_frame_col = 'R_eye_frame'

    df_dict = {'left':block.le_df, 'right':block.re_df}
    df = pd.DataFrame.from_dict(jitter_dict)
    
    indices_of_highest_drift = df.query("top_correlation_dist > @max_distance").index.values
    diff_vec = np.diff(df['top_correlation_dist'].values)
    diff_peaks_indices = np.where(diff_vec > diff_threshold)[0]
    video_indices = np.concatenate((diff_peaks_indices, indices_of_highest_drift))
    print(f'the diff based jitter frame exclusion gives: {np.shape(diff_peaks_indices)}')
    print(f'the threshold based jitter frame exclusion gives: {np.shape(indices_of_highest_drift)}')

    video_indices = add_intermediate_elements(video_indices, gap_to_bridge=gap_to_bridge)
    df_indices_to_remove = df_dict[eye].loc[df_dict[eye][eye_frame_col].isin(video_indices)].index.values

    return df_indices_to_remove, video_indices


# ============================================================================
# FINAL EXPORT FUNCTIONS
# ============================================================================

def export_eye_data_2d(block):
    """
    Save the eye dataframes to two CSV files.
    :param block: The current blocksync class with verified re/le dfs
    :return: None
    """
    block.right_eye_data.to_csv(block.analysis_path / 'right_eye_data.csv')
    block.left_eye_data.to_csv(block.analysis_path / 'left_eye_data.csv')
    print(f'eye dataframes saved to: {block.analysis_path}')
