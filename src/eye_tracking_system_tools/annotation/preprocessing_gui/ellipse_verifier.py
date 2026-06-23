"""Native Qt ellipse verifier (replaces cv2.imshow interactive corrector)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
    VideoReader,
    draw_ellipse_overlay,
    numpy_rgb_to_qpixmap,
)
from eye_tracking_system_tools.preprocessing.data_verification_utils import (
    flip_x_only,
    horizontal_flip_eye_data,
    rotate_phi_only,
)


class FramePickerView(QtWidgets.QGraphicsView):
    """Display a video frame and emit click positions in scene (display) coordinates."""

    clicked_scene = QtCore.pyqtSignal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            self.clicked_scene.emit(float(pos.x()), float(pos.y()))
        super().mousePressEvent(event)


class EllipseVerifierWidget(QtWidgets.QWidget):
    """Interactive eye-video verifier with ellipse overlay and Kerr ref picking."""

    def __init__(
        self,
        df: pd.DataFrame,
        video_path: str | Path,
        eye: str,
        ref_point_xy: tuple[int, int] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        eye_lc = eye.lower()
        if eye_lc not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")

        self._eye = eye_lc
        self._df_current = df.copy()
        self._reader = VideoReader(video_path)
        self._current_ref = ref_point_xy
        self._frame_idx = 0
        self._playing = False
        self._frame_w = 0
        self._frame_h = 0
        self._skip_frames = 30 * 60
        self._probe_video_geometry(video_path)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_one_frame)

        self._build_ui()
        self._frame_view.clicked_scene.connect(self._on_frame_clicked)
        self._show_frame(self._frame_idx)

    def _probe_video_geometry(self, video_path: str | Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            self._frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            self._skip_frames = max(1, int(fps * 60))
        cap.release()
        if self._frame_h <= 0:
            probe = self._reader.read_frame(0)
            if probe is not None:
                self._frame_h, self._frame_w = probe.shape[:2]

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        title = "Left eye" if self._eye == "left" else "Right eye"
        layout.addWidget(QtWidgets.QLabel(f"<b>{title}</b> — click frame to set Kerr ref"))

        self._frame_view = FramePickerView()
        self._frame_view.setMinimumHeight(280)
        layout.addWidget(self._frame_view, stretch=1)

        transport = QtWidgets.QHBoxLayout()
        self._btn_play = QtWidgets.QPushButton("Play")
        self._btn_pause = QtWidgets.QPushButton("Pause")
        self._btn_bwd = QtWidgets.QPushButton("Bwd 1 min")
        self._btn_fwd = QtWidgets.QPushButton("Fwd 1 min")
        transport.addWidget(self._btn_play)
        transport.addWidget(self._btn_pause)
        transport.addWidget(self._btn_bwd)
        transport.addWidget(self._btn_fwd)
        layout.addLayout(transport)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(self._reader.nframes - 1, 0))
        layout.addWidget(self._slider)

        self._counter = QtWidgets.QLabel("0 / 0")
        layout.addWidget(self._counter)

        transforms = QtWidgets.QHBoxLayout()
        self._btn_xflip = QtWidgets.QPushButton("X-flip")
        self._btn_phi = QtWidgets.QPushButton("Phi+90")
        self._btn_flipx = QtWidgets.QPushButton("FlipX-only")
        self._btn_flip_dot = QtWidgets.QPushButton("Flip Dot")
        for btn in (
            self._btn_xflip,
            self._btn_phi,
            self._btn_flipx,
            self._btn_flip_dot,
        ):
            transforms.addWidget(btn)
        layout.addLayout(transforms)

        self._btn_play.clicked.connect(self._on_play)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_bwd.clicked.connect(self._on_bwd)
        self._btn_fwd.clicked.connect(self._on_fwd)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._btn_xflip.clicked.connect(self._on_xflip)
        self._btn_phi.clicked.connect(self._on_phi)
        self._btn_flipx.clicked.connect(self._on_flipx)
        self._btn_flip_dot.clicked.connect(self._on_flip_dot)

    @property
    def frame_height(self) -> int:
        return int(self._frame_h)

    def _frame_col(self) -> str:
        return "eye_frame" if "eye_frame" in self._df_current.columns else "frame"

    def df(self) -> pd.DataFrame:
        return self._df_current

    def ref_xy(self) -> tuple[int, int] | None:
        if self._current_ref is None:
            return None
        return int(self._current_ref[0]), int(self._current_ref[1])

    def pick_ref_from_display_xy(self, x_view: float, y_view: float) -> tuple[int, int]:
        """Map a click on the vertically flipped display to raw video coordinates."""
        y_raw = self._frame_h - 1 - int(round(y_view))
        x_raw = int(round(x_view))
        self._current_ref = (x_raw, y_raw)
        return self._current_ref

    def _on_frame_clicked(self, x_view: float, y_view: float) -> None:
        self.pick_ref_from_display_xy(x_view, y_view)
        self._show_frame(self._frame_idx)

    def _render_frame_rgb(self, frame_idx: int) -> np.ndarray | None:
        rgb = self._reader.read_frame(frame_idx)
        if rgb is None:
            return None
        annotated = draw_ellipse_overlay(
            rgb,
            self._df_current,
            self._frame_col(),
            int(frame_idx),
        )
        if self._current_ref is not None:
            cv2.circle(
                annotated,
                (int(self._current_ref[0]), int(self._current_ref[1])),
                5,
                (255, 0, 0),
                -1,
            )
        return cv2.flip(annotated, 0)

    def _show_frame(self, frame_idx: int) -> None:
        nframes = max(self._reader.nframes, 1)
        frame_idx = int(max(0, min(frame_idx, nframes - 1)))
        self._frame_idx = frame_idx
        self._slider.blockSignals(True)
        self._slider.setValue(frame_idx)
        self._slider.blockSignals(False)
        self._counter.setText(f"{frame_idx} / {max(nframes - 1, 0)}")

        rgb = self._render_frame_rgb(frame_idx)
        if rgb is None:
            self._frame_view.set_pixmap(QtGui.QPixmap())
            return
        self._frame_view.set_pixmap(numpy_rgb_to_qpixmap(rgb))

    def _advance_one_frame(self) -> None:
        nframes = self._reader.nframes
        if nframes <= 0:
            return
        next_idx = self._frame_idx + 1
        if next_idx >= nframes:
            self._playing = False
            self._timer.stop()
            return
        self._show_frame(next_idx)

    def _on_play(self) -> None:
        self._playing = True
        self._timer.start()

    def _on_pause(self) -> None:
        self._playing = False
        self._timer.stop()

    def _on_bwd(self) -> None:
        self._show_frame(self._frame_idx - self._skip_frames)

    def _on_fwd(self) -> None:
        self._show_frame(self._frame_idx + self._skip_frames)

    def _on_slider_changed(self, value: int) -> None:
        if value != self._frame_idx:
            self._show_frame(value)

    def _on_xflip(self) -> None:
        self._df_current = horizontal_flip_eye_data(self._df_current, self._frame_w)
        self._show_frame(self._frame_idx)

    def _on_phi(self) -> None:
        self._df_current = rotate_phi_only(self._df_current)
        self._show_frame(self._frame_idx)

    def _on_flipx(self) -> None:
        self._df_current = flip_x_only(self._df_current, self._frame_w)
        self._show_frame(self._frame_idx)

    def _on_flip_dot(self) -> None:
        if self._current_ref is None:
            return
        x0, y0 = self._current_ref
        self._current_ref = (self._frame_w - int(x0), int(y0))
        self._show_frame(self._frame_idx)
