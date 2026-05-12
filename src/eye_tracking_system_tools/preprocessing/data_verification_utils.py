"""
Utilities for the data verification pipeline (scripted or notebook).
Used by run_batch_dlc_and_verification.py for interactive verification of eye data.
"""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import cv2


def load_eye_data(block):
    """
    Load the eye dataframes from CSV files created by the synchronization pipeline.
    :param block: The current BlockSync instance
    :return: None
    """
    block.left_eye_data = pd.read_csv(
        block.analysis_path / "left_eye_data.csv", index_col=0, engine="python"
    )
    block.right_eye_data = pd.read_csv(
        block.analysis_path / "right_eye_data.csv", index_col=0, engine="python"
    )
    print(f"Loaded eye data for block {block.block_num}")


def horizontal_flip_eye_data(df: pd.DataFrame, frame_width: int) -> pd.DataFrame:
    df2 = df.copy()
    df2["center_x"] = frame_width - df2["center_x"]
    df2["phi"] = (180 - df2["phi"]) % 360
    return df2


def rotate_phi_only(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["phi"] = (df2["phi"] + 90) % 360
    return df2


def flip_x_only(df: pd.DataFrame, frame_width: int) -> pd.DataFrame:
    df2 = df.copy()
    df2["center_x"] = frame_width - df2["center_x"]
    return df2


def interactive_ellipse_corrector(
    df: pd.DataFrame,
    video_path: str | Path,
    eye: str,
    ref_point_xy: Optional[tuple[int, int]] = None,
    *,
    block: Any = None,
    on_save: Optional[Callable[[pd.DataFrame, Optional[tuple[int, int]]], None]] = None,
) -> tuple[pd.DataFrame, Optional[tuple[int, int]]]:
    """
    Interactive video + ellipse editor (same UI as the synced pipeline).

    If ``block`` is given, Save updates ``block.left_eye_data`` / ``block.right_eye_data``
    and ``block.kerr_ref_*`` (same behavior as ``interactive_eye_data_corrector_synced``).

    If ``block`` is None and ``on_save`` is set, Save calls
    ``on_save(df_corrected.copy(), ref_xy_or_none)``.
    """
    eye_lc = eye.lower()
    if eye_lc not in ("left", "right"):
        raise ValueError("eye must be 'left' or 'right'")
    if block is None and on_save is None:
        print(
            "[interactive_ellipse_corrector] Warning: no block and no on_save — "
            "Save only updates in-memory data unless you assign a callback."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip_frames = int(fps * 60)

    df_current = df.copy()
    frame_col = "eye_frame" if "eye_frame" in df_current.columns else "frame"

    buttons = {
        "Play": ((10, 10), (180, 60)),
        "Pause": ((10, 80), (180, 130)),
        "X-flip": ((10, 150), (180, 200)),
        "Phi+90": ((10, 220), (180, 270)),
        "FlipX-only": ((10, 290), (180, 340)),
        "Flip Dot": ((10, 360), (180, 410)),
        "Bwd": ((10, 430), (180, 480)),
        "Fwd": ((10, 490), (180, 540)),
        "Save": ((10, 550), (180, 600)),
        "Quit": ((10, 610), (180, 660)),
    }
    ctrl_h, ctrl_w = 680, 200

    def draw_controls():
        img = np.zeros((ctrl_h, ctrl_w, 3), dtype=np.uint8)
        for name, ((x1, y1), (x2, y2)) in buttons.items():
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
            cv2.putText(
                img, name, (x1 + 5, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA
            )
        return img

    controls_img = draw_controls()
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    running = True
    playing = False
    current_ref = ref_point_xy
    last_frame = None

    def on_mouse_controls(event, x, y, flags, param):
        nonlocal df_current, running, playing, current_ref, last_frame
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, ((x1, y1), (x2, y2)) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "Play":
                    playing = True
                elif name == "Pause":
                    playing = False
                elif name == "X-flip":
                    df_current = horizontal_flip_eye_data(df_current, W)
                elif name == "Phi+90":
                    df_current = rotate_phi_only(df_current)
                elif name == "FlipX-only":
                    df_current = flip_x_only(df_current, W)
                elif name == "Flip Dot" and current_ref is not None:
                    x0, y0 = current_ref
                    current_ref = (W - x0, y0)
                elif name == "Bwd":
                    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    new_idx = max(idx - skip_frames, 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                    last_frame = None
                elif name == "Fwd":
                    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    new_idx = min(idx + skip_frames, N - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                    last_frame = None
                elif name == "Save":
                    if block is not None:
                        if eye_lc == "left":
                            block.left_eye_data = df_current.copy()
                        else:
                            block.right_eye_data = df_current.copy()
                        if current_ref is not None:
                            rx = int(round(current_ref[0]))
                            ry = int(round(current_ref[1]))
                            if eye_lc == "left":
                                block.kerr_ref_l_x = rx
                                block.kerr_ref_l_y = ry
                                print(f"Saved left-eye reference to block: ({rx}, {ry})")
                            else:
                                block.kerr_ref_r_x = rx
                                block.kerr_ref_r_y = ry
                                print(f"Saved right-eye reference to block: ({rx}, {ry})")
                        print(f"{eye.capitalize()} eye data saved.")
                    elif on_save is not None:
                        on_save(df_current.copy(), current_ref)
                        print(f"{eye.capitalize()} eye data saved (on_save callback).")
                    else:
                        print(f"{eye.capitalize()} eye data: in-memory only (no block/on_save).")
                elif name == "Quit":
                    running = False
                break

    def on_mouse_frame(event, x, y, flags, param):
        nonlocal current_ref
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        y_raw = H - 1 - y
        x_raw = x
        current_ref = (int(x_raw), int(y_raw))
        print(f"Picked reference (raw coords): ({current_ref[0]}, {current_ref[1]})")

    cv2.setMouseCallback("Controls", on_mouse_controls)
    cv2.setMouseCallback("Frame", on_mouse_frame)

    while running:
        if playing or last_frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()

        current_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        current_idx = max(current_idx, 0)

        annotated = frame.copy()
        if current_ref is not None:
            cv2.circle(
                annotated,
                (int(current_ref[0]), int(current_ref[1])),
                5, (255, 0, 0), -1,
            )

        mask = df_current[frame_col] == current_idx
        if mask.any():
            row = df_current[mask].iloc[0]
            cx, cy = row["center_x"], row["center_y"]
            if not (pd.isna(cx) or pd.isna(cy)):
                x = int(round(cx))
                y = int(round(cy))
                w = int(row.get("width", 0))
                h = int(row.get("height", 0))
                phi = float(row.get("phi", 0.0))
                w = max(w, 1)
                h = max(h, 1)
                cv2.ellipse(annotated, (x, y), (w, h), phi, 0, 360, (0, 255, 0), 2)

        disp = cv2.flip(annotated, 0)
        cv2.imshow("Frame", disp)
        cv2.imshow("Controls", controls_img)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return df_current, current_ref


def interactive_eye_data_corrector_synced(block, eye, ref_point_xy=None):
    """
    Interactive synchronized video + ellipse editor with Play/Pause, correction, Save,
    Flip-Dot, and Skip-forward/backward (1 minute) buttons.
    Click on the Frame window to set reference point; Save updates block.kerr_ref_*.
    """
    eye_lc = eye.lower()
    if eye_lc == "left":
        df_orig = block.left_eye_data.copy()
        video = block.le_videos[0]
    elif eye_lc == "right":
        df_orig = block.right_eye_data.copy()
        video = block.re_videos[0]
    else:
        raise ValueError("eye must be 'left' or 'right'")

    interactive_ellipse_corrector(df_orig, video, eye_lc, ref_point_xy=ref_point_xy, block=block)


def export_corrected_eye_data(block, include_rotation_pickle=False):
    """
    Overwrite the eye-data CSVs in block.analysis_path.
    If include_rotation_pickle is True and block has rotation attributes, also write
    rotate_eye_data_params.pkl (e.g. when using rotation in verification).
    """
    analysis_path = Path(block.analysis_path)
    analysis_path.mkdir(parents=True, exist_ok=True)
    block.left_eye_data.to_csv(analysis_path / "left_eye_data.csv", index=True)
    block.right_eye_data.to_csv(analysis_path / "right_eye_data.csv", index=True)
    if include_rotation_pickle and hasattr(block, "left_rotation_matrix"):
        rot_dict = {
            "left_rotation_matrix": block.left_rotation_matrix,
            "left_rotation_angle": getattr(block, "left_rotation_angle", 0),
            "right_rotation_matrix": block.right_rotation_matrix,
            "right_rotation_angle": getattr(block, "right_rotation_angle", 0),
        }
        with open(analysis_path / "rotate_eye_data_params.pkl", "wb") as f:
            pickle.dump(rot_dict, f)
    print(f"Exported corrected eye data to {analysis_path}")
