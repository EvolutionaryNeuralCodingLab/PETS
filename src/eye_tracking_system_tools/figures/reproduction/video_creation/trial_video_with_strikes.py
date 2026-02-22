# Trial video export with strike (screen-touch) markers for sync verification.
# Used by arena_trial_summary_and_video.ipynb. Logic copied from synchronized_video_creation / trial_based_video_creation.

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm


class MonotoneFrameReader:
    """Frame-exact reader for mostly-nondecreasing frame indices."""
    def __init__(self, path, label="video"):
        self.path = str(path)
        self.label = label
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {label}: {path}")
        self.cur_idx = -1
        self.cur_frame = None

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass

    def _reopen_and_seek(self, target_idx: int):
        self.close()
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot reopen {self.label}: {self.path}")
        self.cur_idx = -1
        self.cur_frame = None
        if target_idx > 0:
            for _ in range(target_idx):
                ok = self.cap.grab()
                if not ok:
                    return None

    def read_at(self, target_idx: Optional[int]):
        if target_idx is None or target_idx < 0:
            return None
        target_idx = int(target_idx)
        if target_idx == self.cur_idx and self.cur_frame is not None:
            return self.cur_frame
        if target_idx < self.cur_idx:
            self._reopen_and_seek(target_idx)
        while self.cur_idx < target_idx:
            ok, frame = self.cap.read()
            if not ok:
                return None
            self.cur_idx += 1
            self.cur_frame = frame
        return self.cur_frame


def export_trial_video_rolling_window(
    block: object,
    trial_row: pd.Series,
    bug_traj_df: pd.DataFrame,
    out_path: Union[Path, str],
    *,
    fps: float = 60.0,
    lfp_channel: int = 1,
    trace_window_half_s: float = 5.0,
    arena_video: Optional[Union[int, str]] = None,
    top_banner_h: int = 60,
    trace_h: int = 300,
    trace_scale: float = 2.0,
    flip_eyes_vertical: bool = True,
    codec: str = "mp4v",
    show_debug_prints: bool = True,
    strike_times_ms_rel: Optional[Sequence[float]] = None,
    get_arena_video_frame_offset=None,
) -> Path:
    """
    Export a single trial video with rolling-window trace panel.
    If strike_times_ms_rel is provided, a vertical line is drawn on the trace panel
    at each strike time (screen touch during trial) for synchronization verification.
    """
    trace_window_half_ms = trace_window_half_s * 1000.0
    strike_times_ms_rel = list(strike_times_ms_rel) if strike_times_ms_rel else []

    start_ms = float(trial_row["ms_axis_start"])
    end_ms = float(trial_row["ms_axis_end"])
    if end_ms <= start_ms:
        raise ValueError(f"Invalid trial time range: {start_ms} to {end_ms} ms")

    fsync = block.final_sync_df
    ms_all = fsync["ms_axis"].to_numpy(dtype=float)
    mask = np.isfinite(ms_all) & (ms_all >= start_ms) & (ms_all <= end_ms)
    if not np.any(mask):
        raise ValueError(f"No final_sync_df rows in trial window [{start_ms}, {end_ms}] ms")

    idx_rows = np.where(mask)[0]
    fs_win = fsync.iloc[idx_rows].copy()
    t_ms = ms_all[idx_rows].astype(float)
    t_ms_rel = t_ms - start_ms

    if len(t_ms) > 5:
        dt = np.median(np.diff(t_ms))
        fps_master = 1000.0 / dt if dt > 0 else float("nan")
        if np.isfinite(fps_master) and fps_master > 0:
            stride = int(round(fps_master / float(fps))) if float(fps) <= fps_master else 1
            stride = max(1, stride)
            if stride > 1:
                fs_win = fs_win.iloc[::stride].copy()
                t_ms = t_ms[::stride]
                t_ms_rel = t_ms_rel[::stride]

    af_num = pd.to_numeric(fs_win["Arena_frame"], errors="coerce")
    valid_mask = (af_num.notna()) & (af_num >= 0)
    fs_win_valid = fs_win[valid_mask].copy()
    t_ms = t_ms[valid_mask.values]
    t_ms_rel = t_ms_rel[valid_mask.values]

    A_frames = fs_win_valid["Arena_frame"].astype(int).to_numpy()
    L_frames = fs_win_valid["L_eye_frame"].to_numpy(dtype=int)
    R_frames = fs_win_valid["R_eye_frame"].to_numpy(dtype=int)

    arena_videos_dir = Path(block.block_path) / "arena_videos"
    if get_arena_video_frame_offset is not None:
        arena_frame_offset = get_arena_video_frame_offset(block, arena_videos_dir)
    else:
        from eye_tracking_system_tools.preprocessing.arena_alignment import get_arena_video_frame_offset
        arena_frame_offset = get_arena_video_frame_offset(block, arena_videos_dir)
    A_frames = A_frames + arena_frame_offset

    left_df = block.left_eye_data_centered
    right_df = block.right_eye_data_centered
    Ltab = left_df.drop_duplicates(subset=["eye_frame"], keep="first").set_index("eye_frame", drop=False)
    Rtab = right_df.drop_duplicates(subset=["eye_frame"], keep="first").set_index("eye_frame", drop=False)
    L_eye_idx = pd.Index(L_frames, name="eye_frame")
    R_eye_idx = pd.Index(R_frames, name="eye_frame")

    L_phi = Ltab.reindex(L_eye_idx)["k_phi_recentered"].to_numpy(dtype=float)
    R_phi = Rtab.reindex(R_eye_idx)["k_phi_recentered"].to_numpy(dtype=float)
    L_theta = Ltab.reindex(L_eye_idx)["k_theta_recentered"].to_numpy(dtype=float)
    R_theta = Rtab.reindex(R_eye_idx)["k_theta_recentered"].to_numpy(dtype=float)
    L_pupil = Ltab.reindex(L_eye_idx)["pupil_diameter"].to_numpy(dtype=float)
    R_pupil = Rtab.reindex(R_eye_idx)["pupil_diameter"].to_numpy(dtype=float)

    bug_mask = (bug_traj_df["ms_axis"] >= start_ms) & (bug_traj_df["ms_axis"] <= end_ms)
    bug_trial = bug_traj_df.loc[bug_mask].copy()
    bug_trial = bug_trial.sort_values("ms_axis").reset_index(drop=True)
    bug_t_ms_rel = bug_trial["ms_axis"].values - start_ms
    if len(bug_trial) > 0:
        bug_x_interp = np.interp(t_ms_rel, bug_t_ms_rel, bug_trial["x"].values)
        bug_y_interp = np.interp(t_ms_rel, bug_t_ms_rel, bug_trial["y"].values)
    else:
        bug_x_interp = np.full_like(t_ms_rel, np.nan)
        bug_y_interp = np.full_like(t_ms_rel, np.nan)

    window_ms = end_ms - start_ms
    global_start_ms = float(getattr(block.oe_rec, "globalStartTime_ms", 0))
    oe_start_ms = start_ms + global_start_ms
    start_arr = np.atleast_2d(np.array([oe_start_ms], dtype=float))
    try:
        lfp_data, lfp_timestamps = block.oe_rec.get_data(
            channels=[lfp_channel],
            start_time_ms=start_arr,
            window_ms=window_ms,
            convert_microvolts=True,
            return_timestamps=True,
            repress_output=True,
        )
        if lfp_data is None or lfp_data.size == 0:
            raise ValueError("No LFP data returned")
        lfp_trace = lfp_data[0, 0, :]
        lfp_t_ms = lfp_timestamps[0, :]
        lfp_t_ms_rel = lfp_t_ms - lfp_t_ms[0]
    except Exception as e:
        if show_debug_prints:
            print(f"Warning: Could not load LFP data: {e}")
        lfp_trace = None
        lfp_t_ms_rel = None

    rv_raw = Path(block.re_videos[0])
    lv_raw = Path(block.le_videos[0])
    av_list = block.arena_videos
    if arena_video is not None:
        if isinstance(arena_video, int):
            arena_path = Path(av_list[arena_video])
        else:
            arena_path = next(Path(p) for p in av_list if str(arena_video).lower() in Path(p).name.lower())
    else:
        arena_path = Path(av_list[0])

    capR = cv2.VideoCapture(str(rv_raw))
    capL = cv2.VideoCapture(str(lv_raw))
    capA = cv2.VideoCapture(str(arena_path))
    rR = MonotoneFrameReader(rv_raw, "right_eye")
    rL = MonotoneFrameReader(lv_raw, "left_eye")
    rA = MonotoneFrameReader(arena_path, "arena")

    Wr, Hr = int(capR.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capR.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Wl, Hl = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Wa, Ha = int(capA.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capA.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Heye = int(min(Hr, Hl))
    trace_h_eff = max(1, int(round(float(trace_h) * float(trace_scale))))
    Wr_out = max(1, int(round(Wr * (Heye / float(Hr)))))
    Wl_out = max(1, int(round(Wl * (Heye / float(Hl)))))
    Wa_out = max(1, int(round(Wa * (Heye / float(Ha)))))
    Hrow = Heye
    Wtotal = Wr_out + Wa_out + Wl_out
    Htotal = top_banner_h + Hrow + trace_h_eff

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), float(fps), (Wtotal, Htotal))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    def _safe_put_text(img, text, org, color, scale=0.6, thickness=2):
        x, y = org
        cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _make_banner(W: int, title: str) -> np.ndarray:
        banner = np.zeros((top_banner_h, W, 3), dtype=np.uint8)
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x = max(12, (W - tw) // 2)
        _safe_put_text(banner, title, (x, 34), (255, 255, 255), scale=0.9, thickness=2)
        return banner

    def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == target_h:
            return img
        new_w = max(1, int(round(w * (target_h / float(h)))))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    def _map_to_axis(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return (vals - lo) / (hi - lo + 1e-12)

    def _limits_minmax_std(x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return (-1.0, 1.0)
        mn, mx = float(np.min(x)), float(np.max(x))
        sd = float(np.std(x))
        if not np.isfinite(sd) or sd < 1e-12:
            sd = max(1.0, 0.05 * (mx - mn) if (mx - mn) > 0 else 1.0)
        lo, hi = mn - sd, mx + sd
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            lo, hi = mn - 1.0, mx + 1.0
        if abs(hi - lo) < 1e-12:
            lo -= 1.0
            hi += 1.0
        return lo, hi

    # Strike line color: magenta (BGR)
    STRIKE_LINE_COLOR = (255, 0, 255)

    def _draw_trace_panel_rolling(
        W: int,
        t_ms: float,
        t_grid: np.ndarray,
        L_phi: np.ndarray,
        R_phi: np.ndarray,
        L_theta: np.ndarray,
        R_theta: np.ndarray,
        L_pupil: np.ndarray,
        R_pupil: np.ndarray,
        bug_x: np.ndarray,
        bug_y: np.ndarray,
        lfp_trace: Optional[np.ndarray],
        lfp_t_ms: Optional[np.ndarray],
        trace_window_half_ms: float,
        trace_h_eff: int,
        strike_times_ms_rel: Sequence[float],
    ) -> np.ndarray:
        panel = np.zeros((trace_h_eff, W, 3), dtype=np.uint8)
        w0 = t_ms - trace_window_half_ms
        w1 = t_ms + trace_window_half_ms
        idx = np.where((t_grid >= w0) & (t_grid <= w1))[0]
        if idx.size < 2:
            cursor_x = W // 2
            cv2.line(panel, (cursor_x, 0), (cursor_x, trace_h_eff - 1), (120, 120, 120), 1)
            _safe_put_text(panel, f"t = {t_ms:.0f} ms", (12, 26), (255, 255, 255), scale=0.7, thickness=2)
            return panel

        tg = t_grid[idx]
        x = (tg - w0) / (w1 - w0 + 1e-12)
        xpix = (x * (W - 1)).astype(int)

        n_axes = 9 if lfp_trace is not None else 8
        pad_y = 12
        axis_h = max(50, (trace_h_eff - 2 * pad_y) // max(1, n_axes))
        color_L, color_R = (255, 0, 0), (0, 0, 255)
        color_bug, color_lfp = (0, 255, 0), (255, 255, 0)

        traces = [
            ("phi_L", L_phi[idx], color_L, "k_phi L"),
            ("phi_R", R_phi[idx], color_R, "k_phi R"),
            ("theta_L", L_theta[idx], color_L, "k_theta L"),
            ("theta_R", R_theta[idx], color_R, "k_theta R"),
            ("pupil_L", L_pupil[idx], color_L, "pupil L"),
            ("pupil_R", R_pupil[idx], color_R, "pupil R"),
            ("bug_x", bug_x[idx], color_bug, "bug_x"),
            ("bug_y", bug_y[idx], color_bug, "bug_y"),
        ]
        if lfp_trace is not None and lfp_t_ms is not None:
            lfp_interp = np.interp(tg, lfp_t_ms, lfp_trace)
            traces.append(("lfp", lfp_interp, color_lfp, "LFP"))

        for j, (name, vals, color, label) in enumerate(traces):
            y0 = pad_y + j * axis_h
            y1 = min(trace_h_eff - pad_y, y0 + axis_h) - 10
            yy0, yy1 = int(y0 + 20), int(y1 - 10)
            Hax = max(2, yy1 - yy0)
            cv2.rectangle(panel, (0, y0), (W - 1, y1), (20, 20, 20), 1)
            vals_f = vals.astype(float)
            lo, hi = _limits_minmax_std(vals_f)
            vals_n = _map_to_axis(vals_f, lo, hi)
            y_f = yy0 + (1.0 - np.clip(vals_n, 0.0, 1.0)) * (Hax - 1)
            y_i = y_f.astype(np.int32)
            m = np.isfinite(y_f)
            for k in range(1, len(xpix)):
                if m[k - 1] and m[k]:
                    cv2.line(
                        panel,
                        (int(xpix[k - 1]), int(y_i[k - 1])),
                        (int(xpix[k]), int(y_i[k])),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            _safe_put_text(panel, f"{label} [{lo:.2f},{hi:.2f}]", (12, y0 + 18), (200, 200, 200), scale=0.5, thickness=1)

        # Strike markers: vertical lines at each strike time in the visible window
        for s_ms in strike_times_ms_rel:
            if w0 <= s_ms <= w1:
                sx = int(round((s_ms - w0) / (w1 - w0 + 1e-12) * (W - 1)))
                sx = max(0, min(W - 1, sx))
                cv2.line(panel, (sx, 0), (sx, trace_h_eff - 1), STRIKE_LINE_COLOR, 2, cv2.LINE_AA)

        cursor_x = W // 2
        cv2.line(panel, (cursor_x, 0), (cursor_x, trace_h_eff - 1), (120, 120, 120), 1)
        _safe_put_text(
            panel,
            f"t = {t_ms:.0f} ms  [±{trace_window_half_ms/1000:.1f}s]",
            (12, 26),
            (255, 255, 255),
            scale=0.7,
            thickness=2,
        )
        return panel

    trial_id = trial_row.get("trial_db_id", "?")
    banner = _make_banner(Wtotal, f"Trial {trial_id} (strike=magenta)")
    prev_R = np.zeros((Hr, Wr, 3), dtype=np.uint8)
    prev_L = np.zeros((Hl, Wl, 3), dtype=np.uint8)
    prev_A = np.zeros((Ha, Wa, 3), dtype=np.uint8)

    try:
        for i in tqdm(range(len(t_ms)), desc="Exporting trial video (rolling)", unit="frame"):
            tcur_rel = float(t_ms_rel[i])
            idxR = int(R_frames[i]) if R_frames[i] >= 0 else None
            idxL = int(L_frames[i]) if L_frames[i] >= 0 else None
            idxA = int(A_frames[i]) if A_frames[i] >= 0 else None
            fR = rR.read_at(idxR) if idxR is not None else prev_R.copy()
            fL = rL.read_at(idxL) if idxL is not None else prev_L.copy()
            fA = rA.read_at(idxA) if idxA is not None else prev_A.copy()
            if fR is not None:
                prev_R = fR.copy()
            if fL is not None:
                prev_L = fL.copy()
            if fA is not None:
                prev_A = fA.copy()
            if flip_eyes_vertical:
                fR = cv2.flip(fR, 0)
                fL = cv2.flip(fL, 0)
            _safe_put_text(fR, "RIGHT", (12, 24), (255, 255, 255), scale=0.75, thickness=2)
            _safe_put_text(fA, "ARENA", (12, 24), (255, 255, 255), scale=0.75, thickness=2)
            _safe_put_text(fL, "LEFT", (12, 24), (255, 255, 255), scale=0.75, thickness=2)
            fR = _resize_to_height(fR, Heye)
            fA = _resize_to_height(fA, Heye)
            fL = _resize_to_height(fL, Heye)
            row_img = np.concatenate([fR, fA, fL], axis=1)
            if row_img.shape[1] != Wtotal:
                if row_img.shape[1] < Wtotal:
                    row_img = cv2.copyMakeBorder(row_img, 0, 0, 0, Wtotal - row_img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
                else:
                    row_img = row_img[:, :Wtotal, :]

            trace = _draw_trace_panel_rolling(
                W=Wtotal,
                t_ms=tcur_rel,
                t_grid=t_ms_rel,
                L_phi=L_phi,
                R_phi=R_phi,
                L_theta=L_theta,
                R_theta=R_theta,
                L_pupil=L_pupil,
                R_pupil=R_pupil,
                bug_x=bug_x_interp,
                bug_y=bug_y_interp,
                lfp_trace=lfp_trace,
                lfp_t_ms=lfp_t_ms_rel,
                trace_window_half_ms=trace_window_half_ms,
                trace_h_eff=trace_h_eff,
                strike_times_ms_rel=strike_times_ms_rel,
            )

            frame = np.zeros((Htotal, Wtotal, 3), dtype=np.uint8)
            frame[0:top_banner_h, :, :] = banner
            frame[top_banner_h : top_banner_h + Hrow, :, :] = row_img
            frame[top_banner_h + Hrow : top_banner_h + Hrow + trace_h_eff, :, :] = trace
            writer.write(frame)
        return out_path
    finally:
        rR.close()
        rL.close()
        rA.close()
        try:
            writer.release()
        except Exception:
            pass
        for cap in (capR, capL, capA):
            try:
                cap.release()
            except Exception:
                pass


def create_synchronized_trial_videos(
    block: object,
    trial_ids: Sequence[int],
    arena_data: Optional[dict] = None,
    arena_videos_dir: Optional[Path] = None,
    output_dir: Optional[Union[Path, str]] = None,
    *,
    fps: float = 60.0,
    lfp_channel: int = 1,
    trace_window_half_s: float = 5.0,
    arena_video: Optional[Union[int, str]] = None,
    top_banner_h: int = 60,
    trace_h: int = 300,
    trace_scale: float = 2.0,
    flip_eyes_vertical: bool = True,
    codec: str = "mp4v",
    show_debug_prints: bool = True,
) -> list:
    """
    Create synchronized trial videos for the given trial IDs.
    Uses aligned arena data (trials_data, screen_touches, bug_trajectory) and draws
    a vertical line on the trace panel at each screen-touch (strike) time for verification.

    Parameters
    ----------
    block : BlockSync
        Loaded block with final_sync_df, eye data, arena/eye videos.
    trial_ids : sequence of int
        trial_db_id values to export.
    arena_data : dict, optional
        Must contain 'trials_data', 'screen_touches', 'bug_trajectory' with ms_axis columns.
        If None, will be loaded via load_aligned_arena_data(block, arena_videos_dir).
    arena_videos_dir : Path, optional
        Required if arena_data is None (block.block_path / "arena_videos" used if not provided).
    output_dir : Path or str, optional
        Directory for output videos. Default: block.analysis_path / "trial_videos_sync_verified".
    fps, lfp_channel, trace_window_half_s, arena_video, top_banner_h, trace_h,
    trace_scale, flip_eyes_vertical, codec, show_debug_prints
        Passed through to export_trial_video_rolling_window.

    Returns
    -------
    list of Path
        Paths to written video files.
    """
    if arena_data is None:
        if arena_videos_dir is None:
            arena_videos_dir = Path(block.block_path) / "arena_videos"
        from eye_tracking_system_tools.preprocessing import load_aligned_arena_data
        arena_data = load_aligned_arena_data(block, arena_videos_dir)
    trials_df = arena_data.get("trials_data")
    screen_touches_df = arena_data.get("screen_touches")
    bug_traj_df = arena_data.get("bug_trajectory")
    if trials_df is None or bug_traj_df is None:
        raise ValueError("arena_data must contain 'trials_data' and 'bug_trajectory'")

    trial_summary = build_trial_summary(trials_df, screen_touches_df)
    summary_by_id = trial_summary.set_index("trial_db_id")

    if output_dir is None:
        output_dir = Path(block.analysis_path) / "trial_videos_sync_verified"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_by_id = trials_df.set_index("trial_db_id")
    written = []
    for tid in trial_ids:
        if tid not in trials_by_id.index:
            if show_debug_prints:
                print(f"Skipping trial {tid}: not in trials_data")
            continue
        trial_row = trials_by_id.loc[tid]
        if pd.isna(trial_row.get("ms_axis_start")) or pd.isna(trial_row.get("ms_axis_end")):
            if show_debug_prints:
                print(f"Skipping trial {tid}: missing ms_axis_start/ms_axis_end")
            continue
        strike_times_ms_rel = summary_by_id.loc[tid, "strike_times_ms_rel"] if tid in summary_by_id.index else []
        if isinstance(strike_times_ms_rel, (list, tuple)):
            pass
        else:
            strike_times_ms_rel = list(strike_times_ms_rel) if hasattr(strike_times_ms_rel, "__iter__") and not isinstance(strike_times_ms_rel, str) else []
        out_path = output_dir / f"trial_{tid:05d}_sync_verified.mp4"
        if show_debug_prints:
            print(f"Exporting trial {tid} (strikes: {len(strike_times_ms_rel)})...")
        try:
            export_trial_video_rolling_window(
                block=block,
                trial_row=trial_row,
                bug_traj_df=bug_traj_df,
                out_path=out_path,
                fps=fps,
                lfp_channel=lfp_channel,
                trace_window_half_s=trace_window_half_s,
                arena_video=arena_video,
                top_banner_h=top_banner_h,
                trace_h=trace_h,
                trace_scale=trace_scale,
                flip_eyes_vertical=flip_eyes_vertical,
                codec=codec,
                show_debug_prints=show_debug_prints,
                strike_times_ms_rel=strike_times_ms_rel,
            )
            written.append(out_path)
            if show_debug_prints:
                print(f"  Saved: {out_path}")
        except Exception as e:
            if show_debug_prints:
                print(f"  Failed: {e}")
            raise
    return written


def build_trial_summary(trials_df: pd.DataFrame, screen_touches_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build per-trial summary: touch_during_trial, success, n_touches, strike_times_ms_rel."""
    rows = []
    touch_ms = touch_trial_id = None
    if screen_touches_df is not None and "ms_axis" in screen_touches_df.columns and len(screen_touches_df) > 0:
        touch_ms = screen_touches_df["ms_axis"].values
        touch_trial_id = screen_touches_df["trial_id"].values if "trial_id" in screen_touches_df.columns else None
        if touch_trial_id is None and "in_block_trial_id" in screen_touches_df.columns:
            touch_trial_id = screen_touches_df["in_block_trial_id"].values
    has_reward_col = screen_touches_df is not None and (
        "is_reward_bug" in screen_touches_df.columns or "is_reward_any_touch" in screen_touches_df.columns
    )

    for _, row in trials_df.iterrows():
        tid = row.get("trial_db_id", row.name)
        start_ms = row.get("ms_axis_start")
        end_ms = row.get("ms_axis_end")
        if pd.isna(start_ms) or pd.isna(end_ms):
            rows.append({"trial_db_id": tid, "touch_during_trial": False, "success": False, "n_touches": 0, "strike_times_ms_rel": []})
            continue
        start_ms, end_ms = float(start_ms), float(end_ms)
        in_window = (touch_ms >= start_ms) & (touch_ms <= end_ms) if touch_ms is not None else np.array([])
        if touch_trial_id is not None and in_window.any():
            in_window = in_window & (touch_trial_id == tid)
        n_touches = int(np.sum(in_window))
        touch_during = n_touches > 0
        strike_times_ms_rel = []
        if touch_during and touch_ms is not None:
            strike_times_ms_rel = (touch_ms[in_window] - start_ms).tolist()
        success = False
        if touch_during and has_reward_col and screen_touches_df is not None:
            touch_idx = np.where(in_window)[0]
            if "is_reward_bug" in screen_touches_df.columns:
                success = bool(np.any(screen_touches_df.iloc[touch_idx]["is_reward_bug"].fillna(False).astype(bool)))
            elif "is_reward_any_touch" in screen_touches_df.columns:
                success = bool(np.any(screen_touches_df.iloc[touch_idx]["is_reward_any_touch"].fillna(False).astype(bool)))
        rows.append({"trial_db_id": tid, "touch_during_trial": touch_during, "success": success, "n_touches": n_touches, "strike_times_ms_rel": strike_times_ms_rel})
    return pd.DataFrame(rows)
