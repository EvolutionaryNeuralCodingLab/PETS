"""Locate DLC tables, eye videos, and jitter reports without BlockSync / Qt."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from eye_tracking_system_tools.preprocessing.conicoid.refined_io import _analysis_dir
from eye_tracking_system_tools.preprocessing.conicoid.spin_max import (
    raw_ellipse_table_from_dlc_df,
)

MAX_SEQUENTIAL_ADVANCE = 100_000


def analysis_dir(block_path: Path | str) -> Path:
    return _analysis_dir(block_path)


def dlc_csv_path(block_path: Path | str, side: str) -> Path:
    name = "le_df.csv" if side == "left" else "re_df.csv"
    return analysis_dir(block_path) / name


def jitter_report_path(block_path: Path | str) -> Path:
    return analysis_dir(block_path) / "jitter_report_dict.pkl"


def discover_eye_video(block_path: Path | str, side: str) -> Path | None:
    """First non-DLC ``*.mp4`` under ``eye_videos/LE`` or ``eye_videos/RE``."""
    folder = "LE" if side == "left" else "RE"
    root = Path(block_path) / "eye_videos" / folder
    if not root.is_dir():
        return None
    videos = sorted(
        p for p in root.rglob("*.mp4") if "DLC" not in str(p)
    )
    return videos[0] if videos else None


def load_raw_dlc_table(block_path: Path | str, side: str) -> pd.DataFrame:
    path = dlc_csv_path(block_path, side)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Run Read DLC so spin-max can use raw ellipses."
        )
    df = pd.read_csv(path)
    return raw_ellipse_table_from_dlc_df(df, side)


def load_jitter_eye(block_path: Path | str, side: str) -> dict[str, Any] | None:
    path = jitter_report_path(block_path)
    if not path.is_file():
        return None
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None
    key = "left_eye" if side == "left" else "right_eye"
    data = payload.get(key)
    return data if isinstance(data, dict) else None


class SequentialVideoReader:
    """OpenCV capture that prefers sequential ``read()`` over random seeks."""

    def __init__(
        self,
        path: Path | str | None,
        *,
        max_sequential_advance: int = MAX_SEQUENTIAL_ADVANCE,
    ):
        self.path = Path(path) if path else None
        self._cap: cv2.VideoCapture | None = None
        self._nframes = 0
        self._next_frame: int | None = None
        self._max_sequential_advance = max(0, int(max_sequential_advance))
        if self.path is not None and self.path.exists():
            self._cap = cv2.VideoCapture(str(self.path))
            self._nframes = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def nframes(self) -> int:
        return self._nframes

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._next_frame = None

    def _decode(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_bgr(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _seek_and_read(self, frame_idx: int) -> np.ndarray | None:
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        bgr = self._read_bgr()
        if bgr is None:
            self._next_frame = None
            return None
        self._next_frame = frame_idx + 1
        return self._decode(bgr)

    def read_frame(self, frame_idx: int | None) -> np.ndarray | None:
        if self._cap is None or frame_idx is None:
            return None
        frame_idx = int(frame_idx)
        if frame_idx < 0 or (self._nframes > 0 and frame_idx >= self._nframes):
            return None
        if self._next_frame is not None and frame_idx >= self._next_frame:
            advance = frame_idx - self._next_frame
            if advance <= self._max_sequential_advance:
                for _ in range(advance):
                    bgr = self._read_bgr()
                    if bgr is None:
                        self._next_frame = None
                        return self._seek_and_read(frame_idx)
                    self._next_frame += 1
                bgr = self._read_bgr()
                if bgr is None:
                    self._next_frame = None
                    return self._seek_and_read(frame_idx)
                self._next_frame = frame_idx + 1
                return self._decode(bgr)
        return self._seek_and_read(frame_idx)

    def frame_width(self) -> float | None:
        if self._cap is None:
            return None
        width = float(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return width if width > 0 else None
