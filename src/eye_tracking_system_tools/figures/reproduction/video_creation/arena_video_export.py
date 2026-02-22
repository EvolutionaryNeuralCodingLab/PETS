# Arena video export: minimal (arena-only) and sync clip (arena + eye cameras).
# Minimal export uses PC clock only (arena_sync_verification.ipynb logic).
# Sync clip uses block + final_sync_df + shift to ms_axis; writes log with frame ranges and shift.
# See ARENA_VIDEO_WORKPLAN.md.

from pathlib import Path
from typing import Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd

# Default subdir under block analysis_path for sync video exports
SYNC_VIDEO_TRIALS_SUBDIR = "sync_video_trials"


def _first_bug_time_pc_ms(arena_videos_dir: Path) -> float:
    """First bug appearance time in PC-clock milliseconds (from bug_trajectory.csv)."""
    bug_path = arena_videos_dir / "bug_trajectory.csv"
    if not bug_path.exists():
        raise FileNotFoundError(f"bug_trajectory.csv not found: {bug_path}")
    from eye_tracking_system_tools.preprocessing.arena_alignment import arena_datetime_to_pc_ms

    bug_df = pd.read_csv(bug_path)
    valid = bug_df["x"].notna() & bug_df["y"].notna()
    if not valid.any():
        raise ValueError("No valid bug positions in bug_trajectory.csv")
    first_idx = bug_df[valid].index[0]
    pc_ms = arena_datetime_to_pc_ms(pd.Series([bug_df.loc[first_idx, "time"]]))
    return float(pc_ms.iloc[0])


def _load_left_timestamps_pc_ms(arena_videos_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load left camera frames_timestamps; return (df with pc_ms column, pc_ms array)."""
    ts_dir = arena_videos_dir / "videos" / "frames_timestamps"
    if not ts_dir.is_dir():
        raise FileNotFoundError(f"Frames timestamps dir not found: {ts_dir}")
    left_files = list(ts_dir.glob("*left*.csv"))
    if not left_files:
        left_files = [f for f in ts_dir.glob("*.csv") if "left" in f.stem.lower()]
    if not left_files:
        raise FileNotFoundError(f"No left camera timestamp file in {ts_dir}")
    df = pd.read_csv(left_files[0], header=0)
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"Invalid timestamp file: {left_files[0]}")
    unix_s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df = df.copy()
    df["pc_ms"] = unix_s * 1000.0
    pc_ms = df["pc_ms"].values
    return df, pc_ms


def _left_video_path(arena_videos_dir: Path) -> Path:
    """Path to left arena video file (videos/*left*.mp4)."""
    videos_dir = arena_videos_dir / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    left_files = list(videos_dir.glob("*left*.mp4"))
    if not left_files:
        left_files = [f for f in videos_dir.glob("*.mp4") if "left" in f.stem.lower()]
    if not left_files:
        raise FileNotFoundError(f"No left arena video in {videos_dir}")
    return left_files[0]


class _FrameReader:
    """Read video frames by index; seeks forward by reading (no backward seek)."""

    def __init__(self, path: Path, label: str = "video"):
        self.path = Path(path)
        self.label = label
        self._cap = None
        self._cur_idx = -1
        self._cur_frame = None

    def _ensure_open(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open {self.label}: {self.path}")
            self._cur_idx = -1
            self._cur_frame = None

    def read_at(self, target_idx: int):
        if target_idx < 0:
            return None
        target_idx = int(target_idx)
        self._ensure_open()
        if target_idx == self._cur_idx and self._cur_frame is not None:
            return self._cur_frame.copy()
        if target_idx < self._cur_idx:
            self._cap.release()
            self._cap = None
            self._ensure_open()
        while self._cur_idx < target_idx:
            ok = self._cap.grab()
            if not ok:
                return None
            self._cur_idx += 1
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._cur_idx += 1
        self._cur_frame = frame
        return frame.copy()

    def close(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._cur_idx = -1
        self._cur_frame = None


def export_arena_left_10s_around_first_bug(
    arena_videos_dir: Path,
    out_path: Optional[Path] = None,
    duration_sec: float = 10.0,
    fps_out: Optional[float] = None,
    codec: str = "mp4v",
) -> Path:
    """
    Export a short clip of the left arena camera centered on the first bug appearance.

    Uses the same logic as arena_sync_verification.ipynb: PC clock only (bug_trajectory
    time and left camera frames_timestamps). No Open Ephys, no final_sync_df.

    Parameters
    ----------
    arena_videos_dir : Path
        Block's arena_videos directory (contains bug_trajectory.csv and videos/).
    out_path : Path, optional
        Output MP4 path. Default: arena_videos_dir / "videos" / "left_first_bug_10s.mp4".
    duration_sec : float
        Total clip duration in seconds (centered on first bug). Default 10 (5 s before, 5 s after).
    fps_out : float, optional
        Output video FPS. Default: use source video FPS.
    codec : str
        OpenCV fourcc codec. Default "mp4v".

    Returns
    -------
    Path
        Path to the written video file.
    """
    arena_videos_dir = Path(arena_videos_dir)

    # 1) First bug time (PC ms)
    first_bug_pc_ms = _first_bug_time_pc_ms(arena_videos_dir)

    # 2) Left timestamps (PC ms per frame)
    left_ts_df, frame_pc_ms = _load_left_timestamps_pc_ms(arena_videos_dir)
    n_frames_ts = len(left_ts_df)
    valid = np.isfinite(frame_pc_ms)
    if not np.any(valid):
        raise ValueError("No valid timestamps in left camera frames_timestamps.")

    # 3) Closest frame to first bug
    diffs = np.where(valid, np.abs(frame_pc_ms - first_bug_pc_ms), np.inf)
    closest_idx = int(np.argmin(diffs))

    # 4) Frame range for duration_sec (centered on closest frame)
    # Use median frame interval to convert duration to frame count
    dt_ms = np.median(np.diff(frame_pc_ms[valid]))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        dt_ms = 1000.0 / 60.0  # fallback ~60 Hz
    half_frames = int(round((duration_sec * 1000.0 * 0.5) / dt_ms))
    frame_start = max(0, closest_idx - half_frames)
    frame_end = min(n_frames_ts, closest_idx + half_frames + 1)
    if frame_end <= frame_start:
        frame_end = frame_start + 1

    # 5) Left video file
    video_path = _left_video_path(arena_videos_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 60.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Clamp to video length
    if frame_end > total_video_frames:
        frame_end = total_video_frames
    if frame_start >= frame_end:
        frame_start = max(0, frame_end - 1)

    # 6) Output path
    if out_path is None:
        out_path = arena_videos_dir / "videos" / "left_first_bug_10s.mp4"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 7) Read frames sequentially and write (monotone forward to avoid seek issues)
    use_fps = float(fps_out) if fps_out is not None and fps_out > 0 else src_fps
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*codec),
        use_fps,
        (w, h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

    # Read from start up to frame_start (skip), then read frame_start..frame_end-1
    for _ in range(frame_start):
        if not cap.grab():
            break
    n_written = 0
    for _ in range(frame_end - frame_start):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        n_written += 1

    cap.release()
    writer.release()

    if n_written == 0:
        raise RuntimeError(f"No frames written; frame range [{frame_start}, {frame_end}) may be invalid.")

    return out_path


def _ensure_final_sync_df_and_ms_axis(block) -> pd.DataFrame:
    """Ensure block has final_sync_df; load from disk if needed. Add ms_axis if missing."""
    from eye_tracking_system_tools.preprocessing.block_sync_core import load_final_sync_df

    if not getattr(block, "final_sync_df", None) or block.final_sync_df is None:
        load_final_sync_df(block, verbose=False)
    df = block.final_sync_df
    if "ms_axis" not in df.columns:
        fs = getattr(block, "sample_rate", None) or float(block.get_sample_rate())
        df = df.copy()
        df["ms_axis"] = (df["Arena_TTL"].astype(float) / fs) * 1000.0
        block.final_sync_df = df
    return block.final_sync_df


def export_sync_clip_arena_and_eyes(
    block: Any,
    duration_sec: float = 10.0,
    out_dir: Optional[Path] = None,
    out_basename: str = "first_bug_10s",
    fps_out: Optional[float] = None,
    codec: str = "mp4v",
    eye_panel_height: Optional[int] = None,
) -> Tuple[Path, Path]:
    """
    Export a synchronized clip (left arena + left eye + right eye) centered on first bug,
    with a log file listing frame ranges and the arena timestamp shift.

    Uses ms_axis (OE timebase): first bug time from bug_trajectory is converted to
    ms_axis via the PC→OE shift; final_sync_df rows in the window are used to get
    Arena_frame + offset (left), L_eye_frame, R_eye_frame per output frame.

    Convention: ms_from_rec_start = (Arena_TTL / open_ephys_sample_rate) * 1000
    Shift: OE_ms = PC_ms - shift_ms (applied to arena PC timestamps to get ms_axis).

    Parameters
    ----------
    block
        Block with block_path, analysis_path, final_sync_df (or loadable), oe_events,
        sample_rate, le_videos, re_videos. arena_videos_dir = block.block_path / "arena_videos".
    duration_sec : float
        Total clip duration in seconds (centered on first bug in ms_axis). Default 10.
    out_dir : Path, optional
        Output directory. Default: block.block_path / "analysis" / sync_video_trials.
    out_basename : str
        Basename for video and log (e.g. "first_bug_10s" → first_bug_10s.mp4, first_bug_10s.log).
    fps_out : float, optional
        Output FPS. Default: use arena source FPS.
    codec : str
        OpenCV fourcc. Default "mp4v".
    eye_panel_height : int, optional
        Height of each eye panel (same as arena row height if None).

    Returns
    -------
    (video_path, log_path) : Tuple[Path, Path]
    """
    from eye_tracking_system_tools.preprocessing.arena_alignment import (
        compute_pc_to_oe_shift_ms,
        get_arena_video_frame_offset,
        get_first_arena_frame_oe_ms,
        get_first_arena_frame_pc_ms,
    )

    block_path = Path(block.block_path)
    arena_videos_dir = block_path / "arena_videos"
    analysis_path = Path(block.analysis_path)

    if out_dir is None:
        out_dir = analysis_path / SYNC_VIDEO_TRIALS_SUBDIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{out_basename}.mp4"
    out_log = out_dir / f"{out_basename}.log"

    # 1) First bug in PC ms, then shift to get ms_axis
    first_bug_pc_ms = _first_bug_time_pc_ms(arena_videos_dir)
    median_pc_ms, _ = get_first_arena_frame_pc_ms(arena_videos_dir)
    first_oe_ms = get_first_arena_frame_oe_ms(block)
    shift_ms = compute_pc_to_oe_shift_ms(median_pc_ms, first_oe_ms)
    first_bug_ms_axis = first_bug_pc_ms - shift_ms

    # 2) final_sync_df with ms_axis
    fsync = _ensure_final_sync_df_and_ms_axis(block)
    ms_all = np.asarray(fsync["ms_axis"], dtype=float)
    half_ms = (duration_sec * 1000.0) * 0.5
    t0, t1 = first_bug_ms_axis - half_ms, first_bug_ms_axis + half_ms
    mask = np.isfinite(ms_all) & (ms_all >= t0) & (ms_all <= t1)
    if not np.any(mask):
        raise ValueError(f"No final_sync_df rows in ms_axis window [{t0}, {t1}] ms")
    idx_rows = np.where(mask)[0]
    win = fsync.iloc[idx_rows].copy()
    ms_win = ms_all[idx_rows]

    # 3) Frame indices: arena (left) = Arena_frame + offset; eye = L_eye_frame, R_eye_frame
    arena_offset = get_arena_video_frame_offset(block, arena_videos_dir)
    arena_frame = pd.to_numeric(win["Arena_frame"], errors="coerce")
    valid = (arena_frame.notna()) & (arena_frame >= 0)
    win_valid = win.loc[valid].copy()
    ms_win = ms_win[valid.values]
    if len(win_valid) == 0:
        raise ValueError("No valid Arena_frame in sync window")
    A_idx = (win_valid["Arena_frame"].astype(int) + arena_offset).to_numpy()
    L_idx = win_valid["L_eye_frame"].astype(int).to_numpy()
    R_idx = win_valid["R_eye_frame"].astype(int).to_numpy()

    # Frame ranges for log
    arena_range = (int(np.min(A_idx)), int(np.max(A_idx)))
    left_eye_range = (int(np.min(L_idx)), int(np.max(L_idx)))
    right_eye_range = (int(np.min(R_idx)), int(np.max(R_idx)))

    # 4) Paths
    arena_path = _left_video_path(arena_videos_dir)
    if not getattr(block, "le_videos", None) or not getattr(block, "re_videos", None):
        raise RuntimeError("block.le_videos and block.re_videos required. Run block.handle_eye_videos() if needed.")
    left_eye_path = Path(block.le_videos[0])
    right_eye_path = Path(block.re_videos[0])

    # 5) Video dimensions: arena defines row height; eye panels resized to same height
    cap_a = cv2.VideoCapture(str(arena_path))
    if not cap_a.isOpened():
        raise RuntimeError(f"Cannot open arena video: {arena_path}")
    Wa, Ha = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap_a.get(cv2.CAP_PROP_FPS)
    cap_a.release()
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 60.0
    use_fps = float(fps_out) if fps_out is not None and fps_out > 0 else src_fps

    H_row = eye_panel_height if eye_panel_height is not None and eye_panel_height > 0 else Ha
    cap_l = cv2.VideoCapture(str(left_eye_path))
    cap_r = cv2.VideoCapture(str(right_eye_path))
    Wl, Hl = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Wr, Hr = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_l.release()
    cap_r.release()
    # Scale arena to H_row if we forced eye_panel_height
    if H_row != Ha:
        Wa_out = max(1, int(round(Wa * (H_row / float(Ha)))))
        Ha_out = H_row
    else:
        Wa_out, Ha_out = Wa, Ha
    Wl_out = max(1, int(round(Wl * (H_row / float(Hl))))) if Hl else Wl
    Wr_out = max(1, int(round(Wr * (H_row / float(Hr))))) if Hr else Wr
    W_total = Wa_out + Wl_out + Wr_out
    H_total = H_row

    def resize_to_height(img, w_out, h_out):
        h, w = img.shape[:2]
        if w == w_out and h == h_out:
            return img
        return cv2.resize(img, (w_out, h_out), interpolation=cv2.INTER_AREA)

    # 6) Readers and writer
    r_arena = _FrameReader(arena_path, "arena_left")
    r_left = _FrameReader(left_eye_path, "left_eye")
    r_right = _FrameReader(right_eye_path, "right_eye")
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*codec),
        use_fps,
        (W_total, H_total),
    )
    if not writer.isOpened():
        r_arena.close()
        r_left.close()
        r_right.close()
        raise RuntimeError(f"Cannot open VideoWriter: {out_video}")

    try:
        n = len(A_idx)
        for i in range(n):
            fa = r_arena.read_at(A_idx[i])
            fl = r_left.read_at(L_idx[i])
            fr = r_right.read_at(R_idx[i])
            if fa is None:
                fa = np.zeros((Ha, Wa, 3), dtype=np.uint8)
            if fl is None:
                fl = np.zeros((Hl, Wl, 3), dtype=np.uint8)
            if fr is None:
                fr = np.zeros((Hr, Wr, 3), dtype=np.uint8)
            fa = resize_to_height(fa, Wa_out, Ha_out)
            fl = resize_to_height(fl, Wl_out, H_row)
            fr = resize_to_height(fr, Wr_out, H_row)
            row = np.hstack([fa, fl, fr])
            writer.write(row)
    finally:
        r_arena.close()
        r_left.close()
        r_right.close()
        writer.release()

    # 7) Log file: frame ranges, shift, sample rate, formula
    sample_rate = getattr(block, "sample_rate", None) or float(block.get_sample_rate())
    log_lines = [
        "# Sync clip export log",
        f"# Video: {out_video.name}",
        f"# Duration: {duration_sec} s around first bug (ms_axis)",
        "",
        "[Frame ranges] (min_frame_index, max_frame_index) per source",
        f"arena_left: {arena_range[0]}, {arena_range[1]}",
        f"left_eye:  {left_eye_range[0]}, {left_eye_range[1]}",
        f"right_eye: {right_eye_range[0]}, {right_eye_range[1]}",
        "",
        "[Arena timestamp shift for ms_axis]",
        "Conversion: ms_from_rec_start = (Arena_TTL / open_ephys_sample_rate) * 1000",
        "Arena PC timestamps are aligned to OE by: OE_ms = PC_ms - shift_ms",
        f"shift_ms: {shift_ms}",
        f"open_ephys_sample_rate: {sample_rate}",
        "",
        "[First bug]",
        f"first_bug_pc_ms: {first_bug_pc_ms}",
        f"first_bug_ms_axis: {first_bug_ms_axis}",
        f"sync_window_ms: [{t0}, {t1}]",
        f"output_frames: {n}",
    ]
    with open(out_log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return out_video, out_log
