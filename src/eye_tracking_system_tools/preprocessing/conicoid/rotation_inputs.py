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
        self._last_idx: int | None = None
        self._last_gray: bool | None = None
        self._last_image: np.ndarray | None = None
        if self.path is not None and self.path.exists():
            self._cap = self._open(self.path)
            if self._cap is not None and self._cap.isOpened():
                self._nframes = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self._cap = None

    @staticmethod
    def _open(path: Path) -> cv2.VideoCapture | None:
        attempts: list[int | None] = []
        if hasattr(cv2, "CAP_FFMPEG"):
            attempts.append(int(cv2.CAP_FFMPEG))
        attempts.append(None)
        for backend in attempts:
            cap = (
                cv2.VideoCapture(str(path))
                if backend is None
                else cv2.VideoCapture(str(path), backend)
            )
            if cap.isOpened():
                return cap
            cap.release()
        return None

    @property
    def nframes(self) -> int:
        return self._nframes

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._next_frame = None
        self._last_idx = None
        self._last_gray = None
        self._last_image = None

    def _decode(self, bgr: np.ndarray, *, as_gray: bool) -> np.ndarray:
        if as_gray:
            if bgr.ndim == 2:
                return bgr
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_bgr(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _seek_and_read(self, frame_idx: int, *, as_gray: bool) -> np.ndarray | None:
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        bgr = self._read_bgr()
        if bgr is None:
            self._next_frame = None
            return None
        self._next_frame = frame_idx + 1
        return self._decode(bgr, as_gray=as_gray)

    def read_frame(
        self, frame_idx: int | None, *, as_gray: bool = False
    ) -> np.ndarray | None:
        if self._cap is None or frame_idx is None:
            return None
        frame_idx = int(frame_idx)
        if frame_idx < 0 or (self._nframes > 0 and frame_idx >= self._nframes):
            return None
        if (
            self._last_idx == frame_idx
            and self._last_gray is as_gray
            and self._last_image is not None
        ):
            return self._last_image
        image: np.ndarray | None
        if self._next_frame is not None and frame_idx >= self._next_frame:
            advance = frame_idx - self._next_frame
            if advance <= self._max_sequential_advance:
                for _ in range(advance):
                    bgr = self._read_bgr()
                    if bgr is None:
                        self._next_frame = None
                        image = self._seek_and_read(frame_idx, as_gray=as_gray)
                        break
                    self._next_frame += 1
                else:
                    bgr = self._read_bgr()
                    if bgr is None:
                        self._next_frame = None
                        image = self._seek_and_read(frame_idx, as_gray=as_gray)
                    else:
                        self._next_frame = frame_idx + 1
                        image = self._decode(bgr, as_gray=as_gray)
            else:
                image = self._seek_and_read(frame_idx, as_gray=as_gray)
        else:
            image = self._seek_and_read(frame_idx, as_gray=as_gray)
        if image is not None:
            self._last_idx = frame_idx
            self._last_gray = as_gray
            self._last_image = image
        return image

    def frame_width(self) -> float | None:
        if self._cap is None:
            return None
        width = float(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return width if width > 0 else None
