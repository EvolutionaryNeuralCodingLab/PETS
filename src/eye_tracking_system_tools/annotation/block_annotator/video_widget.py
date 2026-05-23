"""Video panel: cv2 capture, LRU cache, flip, ellipse overlay."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets


MISSING_TEXT = "missing frame"
CACHE_LIMIT = 32
# Forward steps within this range use read(); larger jumps use CAP_PROP_POS_FRAMES.
MAX_SEQUENTIAL_ADVANCE = 8


class VideoReader:
    """OpenCV video reader with LRU cache and sequential read() when possible."""

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        self._cap: cv2.VideoCapture | None = None
        self._nframes = 0
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        # Index of the frame the next read() would return; None if unknown.
        self._next_frame: int | None = None
        if self.path and self.path.exists():
            self._cap = cv2.VideoCapture(str(self.path))
            self._nframes = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._cache.clear()
        self._next_frame = None

    @property
    def nframes(self) -> int:
        return self._nframes

    def _store_frame(self, frame_idx: int, rgb: np.ndarray) -> np.ndarray:
        self._cache[frame_idx] = rgb
        if len(self._cache) > CACHE_LIMIT:
            self._cache.popitem(last=False)
        return rgb

    def _decode_bgr(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _read_bgr_frame(self) -> np.ndarray | None:
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
        bgr = self._read_bgr_frame()
        if bgr is None:
            self._next_frame = None
            return None
        self._next_frame = frame_idx + 1
        return self._store_frame(frame_idx, self._decode_bgr(bgr))

    def _read_sequential_to(self, frame_idx: int) -> np.ndarray | None:
        """Advance with read() until frame_idx; assumes frame_idx >= _next_frame."""
        assert self._cap is not None and self._next_frame is not None
        steps = frame_idx - self._next_frame
        bgr = None
        for _ in range(steps):
            bgr = self._read_bgr_frame()
            if bgr is None:
                self._next_frame = None
                return self._seek_and_read(frame_idx)
            self._next_frame += 1
        bgr = self._read_bgr_frame()
        if bgr is None:
            self._next_frame = None
            return self._seek_and_read(frame_idx)
        self._next_frame = frame_idx + 1
        return self._store_frame(frame_idx, self._decode_bgr(bgr))

    def read_frame(self, frame_idx: int | None) -> np.ndarray | None:
        if self._cap is None or frame_idx is None:
            return None
        frame_idx = int(frame_idx)
        if frame_idx < 0 or (self._nframes > 0 and frame_idx >= self._nframes):
            return None
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            # Capture position no longer reliable after a cache-only hit.
            if self._next_frame is not None and self._next_frame != frame_idx + 1:
                self._next_frame = None
            return self._cache[frame_idx]

        if self._next_frame is not None and frame_idx >= self._next_frame:
            advance = frame_idx - self._next_frame
            if advance <= MAX_SEQUENTIAL_ADVANCE:
                return self._read_sequential_to(frame_idx)

        return self._seek_and_read(frame_idx)


def apply_display_transforms(
    arr: np.ndarray,
    *,
    ellipse_df: pd.DataFrame | None = None,
    frame_col: str = "L_eye_frame",
    frame_idx: int | None = None,
    show_annotations: bool = False,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    """
    Build display image: ellipse overlay on raw frame coords, then display-only flips.
    """
    out = arr
    if show_annotations and frame_idx is not None:
        out = draw_ellipse_overlay(out, ellipse_df, frame_col, int(frame_idx))
    if flip_horizontal:
        out = cv2.flip(out, 1)
    if flip_vertical:
        out = cv2.flip(out, 0)
    return out


def draw_ellipse_overlay(
    frame: np.ndarray,
    df: pd.DataFrame | None,
    frame_col: str,
    frame_idx: int,
) -> np.ndarray:
    if df is None or frame_col not in df.columns:
        return frame
    mask = df[frame_col] == frame_idx
    if not mask.any():
        return frame
    row = df.loc[mask].iloc[0]
    cx, cy = row.get("center_x"), row.get("center_y")
    if pd.isna(cx) or pd.isna(cy):
        return frame
    out = frame.copy()
    x = int(round(float(cx)))
    y = int(round(float(cy)))
    w = max(int(row.get("width", 1) or 1), 1)
    h = max(int(row.get("height", 1) or 1), 1)
    phi = float(row.get("phi", 0.0))
    cv2.ellipse(out, (x, y), (w, h), phi, 0, 360, (0, 255, 0), 2)
    return out


def _qimage_format(name: str) -> int:
    """PyQt6 uses QImage.Format.Format_*; older bindings used QImage.Format_*."""
    fmt_enum = QtGui.QImage.Format
    value = getattr(fmt_enum, f"Format_{name}", None)
    if value is not None:
        return value
    legacy = getattr(QtGui.QImage, f"Format_{name}", None)
    if legacy is not None:
        return legacy
    raise AttributeError(f"No QImage format constant for {name}")


def numpy_rgb_to_qpixmap(arr: np.ndarray | None) -> QtGui.QPixmap:
    if arr is None:
        return QtGui.QPixmap()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    h, w, c = arr.shape
    arr = np.ascontiguousarray(arr)
    if c == 4:
        fmt = _qimage_format("RGBA8888")
    else:
        fmt = _qimage_format("RGB888")
    bytes_per_line = int(arr.strides[0])
    qimg = QtGui.QImage(arr.data, w, h, bytes_per_line, fmt)
    return QtGui.QPixmap.fromImage(qimg.copy())


def missing_frame_pixmap(width: int = 320, height: int = 240) -> QtGui.QPixmap:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    pix = numpy_rgb_to_qpixmap(img)
    return pix


class VideoPanel(QtWidgets.QWidget):
    """Single video display with optional title and controls hook."""

    def __init__(
        self,
        title: str = "",
        *,
        min_width: int = 200,
        min_height: int = 150,
        parent=None,
    ):
        super().__init__(parent)
        self._reader = VideoReader(None)
        self._flip_h = False
        self._flip_v = False
        self._show_annotations = False
        self._ellipse_df: pd.DataFrame | None = None
        self._frame_col = "L_eye_frame"
        self._max_display_width = 0
        self._max_display_height = 0
        self._no_upscale = True
        self._last_frame_idx: int | None = None

        layout = QtWidgets.QVBoxLayout(self)
        self._title = QtWidgets.QLabel(title)
        self._title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label = QtWidgets.QLabel()
        self._label.setMinimumSize(min_width, min_height)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background-color: black; color: white;")
        self._label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._info = QtWidgets.QLabel("")
        layout.addWidget(self._title)
        layout.addWidget(self._label, stretch=1)
        layout.addWidget(self._info)

    def set_video(self, path: Path | str | None) -> None:
        self._reader.close()
        self._reader = VideoReader(path)
        self._label.clear()

    def set_ellipse_data(
        self, df: pd.DataFrame | None, frame_col: str = "L_eye_frame"
    ) -> None:
        self._ellipse_df = df
        self._frame_col = frame_col

    def set_flip_horizontal(self, enabled: bool) -> None:
        self._flip_h = bool(enabled)

    def set_flip_vertical(self, enabled: bool) -> None:
        self._flip_v = bool(enabled)

    def set_show_annotations(self, enabled: bool) -> None:
        self._show_annotations = bool(enabled)

    def set_display_limits(
        self,
        max_width: int = 0,
        max_height: int = 0,
        *,
        no_upscale: bool = True,
    ) -> None:
        """Cap display size in pixels (0 = no cap beyond label/splitter size)."""
        self._max_display_width = max(0, int(max_width))
        self._max_display_height = max(0, int(max_height))
        self._no_upscale = bool(no_upscale)
        if self._last_frame_idx is not None:
            self.show_frame(self._last_frame_idx, self._info.text())

    def _target_size(self, pix: QtGui.QPixmap) -> QtCore.QSize:
        label_sz = self._label.size()
        w, h = label_sz.width(), label_sz.height()
        if self._max_display_width > 0:
            w = min(w, self._max_display_width)
        if self._max_display_height > 0:
            h = min(h, self._max_display_height)
        if self._no_upscale:
            w = min(w, pix.width())
            h = min(h, pix.height())
        return QtCore.QSize(max(1, w), max(1, h))

    def _scale_pixmap(self, pix: QtGui.QPixmap) -> QtGui.QPixmap:
        target = self._target_size(pix)
        return pix.scaled(
            target,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

    def show_frame(self, frame_idx: int | None, info: str = "") -> None:
        self._info.setText(info)
        self._last_frame_idx = frame_idx
        arr = self._reader.read_frame(frame_idx)
        if arr is None:
            self._label.setPixmap(QtGui.QPixmap())
            self._label.setText(MISSING_TEXT)
            return

        arr = apply_display_transforms(
            arr,
            ellipse_df=self._ellipse_df,
            frame_col=self._frame_col,
            frame_idx=frame_idx,
            show_annotations=self._show_annotations,
            flip_horizontal=self._flip_h,
            flip_vertical=self._flip_v,
        )

        pix = numpy_rgb_to_qpixmap(arr)
        self._label.setText("")
        self._label.setPixmap(self._scale_pixmap(pix))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_frame_idx is not None and self._label.text() != MISSING_TEXT:
            self.show_frame(self._last_frame_idx, self._info.text())
