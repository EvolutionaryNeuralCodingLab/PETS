"""Qt rubber-band ROI picker for eye-video brightness extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
    numpy_rgb_to_qpixmap,
)


def load_first_video_frame_bgr(video_path: Path | str) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame from {video_path}")
    return frame


class RoiGraphicsView(QtWidgets.QGraphicsView):
    """Display a still frame and collect one rectangular ROI via rubber-band drag."""

    roi_changed = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, pixmap: QtGui.QPixmap, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

        self._origin: QtCore.QPoint | None = None
        self._rubber = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle, self.viewport()
        )
        self._roi: tuple[int, int, int, int] | None = None

    @property
    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def set_roi_rect(self, x: int, y: int, w: int, h: int) -> None:
        """Programmatic ROI (used by pytest and re-open flows)."""
        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))
        self._roi = (x, y, w, h)
        self.roi_changed.emit(x, y, w, h)
        self._show_rubber_for_roi()

    def _show_rubber_for_roi(self) -> None:
        if self._roi is None:
            self._rubber.hide()
            return
        x, y, w, h = self._roi
        top_left = self.mapFromScene(QtCore.QPointF(x, y))
        bottom_right = self.mapFromScene(QtCore.QPointF(x + w, y + h))
        rect = QtCore.QRect(top_left, bottom_right).normalized()
        self._rubber.setGeometry(rect)
        self._rubber.show()

    def _scene_rect_from_viewport(self, viewport_rect: QtCore.QRect) -> QtCore.QRectF:
        p1 = self.mapToScene(viewport_rect.topLeft())
        p2 = self.mapToScene(viewport_rect.bottomRight())
        return QtCore.QRectF(p1, p2).normalized()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._origin = event.pos()
            self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber.show()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._origin is not None:
            rect = QtCore.QRect(self._origin, event.pos()).normalized()
            self._rubber.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._origin is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            rect = self._rubber.geometry()
            scene_rect = self._scene_rect_from_viewport(rect)
            x = int(round(scene_rect.left()))
            y = int(round(scene_rect.top()))
            w = max(1, int(round(scene_rect.width())))
            h = max(1, int(round(scene_rect.height())))
            self._roi = (x, y, w, h)
            self.roi_changed.emit(x, y, w, h)
            self._origin = None
        super().mouseReleaseEvent(event)


class QtRoiPickerDialog(QtWidgets.QDialog):
    """Modal dialog: drag a rectangle on the first video frame; returns ``(x, y, w, h)``."""

    def __init__(
        self,
        frame_bgr: np.ndarray,
        *,
        title: str = "Select ROI",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 520)
        self._roi: tuple[int, int, int, int] | None = None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pix = numpy_rgb_to_qpixmap(rgb)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Drag a rectangle over the LED region. Minimum size is 1×1 pixel."
            )
        )
        self._view = RoiGraphicsView(pix)
        self._view.setMinimumHeight(360)
        layout.addWidget(self._view, stretch=1)

        self._roi_label = QtWidgets.QLabel("ROI: (not selected)")
        layout.addWidget(self._roi_label)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self._btn_ok = QtWidgets.QPushButton("Use ROI")
        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_ok.setEnabled(False)
        row.addWidget(self._btn_ok)
        row.addWidget(self._btn_cancel)
        layout.addLayout(row)

        self._view.roi_changed.connect(self._on_roi_changed)
        self._btn_ok.clicked.connect(self._on_accept)
        self._btn_cancel.clicked.connect(self.reject)

    def _on_roi_changed(self, x: int, y: int, w: int, h: int) -> None:
        self._roi = (x, y, w, h)
        self._roi_label.setText(f"ROI: x={x}, y={y}, w={w}, h={h}")
        self._btn_ok.setEnabled(True)

    def set_selection_rect(self, x: int, y: int, w: int, h: int) -> None:
        """Programmatically set ROI geometry (pytest helper)."""
        self._view.set_roi_rect(x, y, w, h)

    def selected_roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def _on_accept(self) -> None:
        if self._roi is None:
            QtWidgets.QMessageBox.warning(self, "Select ROI", "Drag a rectangle first.")
            return
        self.accept()


def pick_roi_for_video(
    video_path: Path | str,
    *,
    title: str = "Select ROI",
    parent: QtWidgets.QWidget | None = None,
) -> tuple[int, int, int, int] | None:
    """Open the picker on the first frame; return ROI or ``None`` if cancelled."""
    frame = load_first_video_frame_bgr(video_path)
    dlg = QtRoiPickerDialog(frame, title=title, parent=parent)
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return dlg.selected_roi()


def extract_brightness_with_roi_fallback(
    blocksync,
    *,
    threshold_value: float = 30.0,
    export: bool = True,
    use_auto_roi: bool = True,
    force: bool = False,
    parent: QtWidgets.QWidget | None = None,
) -> None:
    """Try auto ROI; open Qt picker per eye when auto fails."""
    import glob
    import pickle

    if blocksync.le_videos is None:
        blocksync.le_videos = [
            vid
            for vid in glob.glob(str(blocksync.block_path) + r"\eye_videos\LE\**\*.mp4")
            if "DLC" not in vid
        ]
    if blocksync.re_videos is None:
        blocksync.re_videos = [
            vid
            for vid in glob.glob(str(blocksync.block_path) + r"\eye_videos\RE\**\*.mp4")
            if "DLC" not in vid
        ]

    p = blocksync.analysis_path / "eye_brightness_values_dict.pkl"
    if p.is_file() and not force:
        with open(p, "rb") as file:
            eye_brightness_dict = pickle.load(file)
            blocksync.le_frame_val_list = eye_brightness_dict.get("left_eye", None)
            blocksync.re_frame_val_list = eye_brightness_dict.get("right_eye", None)
        return

    rois: dict[str, tuple[int, int, int, int]] = {}
    eye_specs = [
        ("Left Eye", "le_videos", blocksync.le_videos),
        ("Right Eye", "re_videos", blocksync.re_videos),
    ]
    for label, _attr, videos in eye_specs:
        if not videos:
            raise RuntimeError(f"No video found for {label}.")
        vid = videos[0]
        roi = None
        if use_auto_roi:
            roi = blocksync.get_roi_auto_brightest_2x2(vid, threshold_value)
        if roi is None:
            picked = pick_roi_for_video(
                vid,
                title=f"Select ROI for {label}",
                parent=parent,
            )
            if picked is None:
                raise RuntimeError(f"ROI selection cancelled for {label}.")
            roi = picked
        rois[label] = roi

    blocksync.le_frame_val_list = blocksync.produce_frame_val_list_with_roi(
        blocksync.le_videos[0], rois["Left Eye"], threshold_value
    )
    blocksync.re_frame_val_list = blocksync.produce_frame_val_list_with_roi(
        blocksync.re_videos[0], rois["Right Eye"], threshold_value
    )
    if export:
        frame_val_dict = {
            "left_eye": blocksync.le_frame_val_list,
            "right_eye": blocksync.re_frame_val_list,
        }
        with open(p, "wb") as file:
            pickle.dump(frame_val_dict, file)


def load_eye_brightness_lists(blocksync) -> tuple[list, list]:
    """Load left/right brightness vectors from pkl or in-memory BlockSync state."""
    import pickle

    p = Path(blocksync.analysis_path) / "eye_brightness_values_dict.pkl"
    if p.is_file():
        with open(p, "rb") as file:
            eye_brightness_dict = pickle.load(file)
        left = eye_brightness_dict.get("left_eye")
        right = eye_brightness_dict.get("right_eye")
        if left is not None and right is not None:
            return list(left), list(right)

    left = getattr(blocksync, "le_frame_val_list", None)
    right = getattr(blocksync, "re_frame_val_list", None)
    if left is not None and right is not None:
        return list(left), list(right)

    raise RuntimeError(
        "No eye brightness data found. Run 'Extract brightness' first, or use "
        "'Re-run eye brightness with manual ROIs'."
    )


def summarize_brightness_traces(left_values, right_values) -> str:
    """Human-readable quality summary to help decide whether ROI re-run is needed."""
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    lines = [
        f"Left eye:  {len(left)} frames | max={np.nanmax(left):.2f} | "
        f"mean={np.nanmean(left):.2f}",
        f"Right eye: {len(right)} frames | max={np.nanmax(right):.2f} | "
        f"mean={np.nanmean(right):.2f}",
    ]

    def _flat_fraction(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 1.0
        return float(np.mean(arr < 1.0))

    flat_l = _flat_fraction(left)
    flat_r = _flat_fraction(right)
    if flat_l > 0.5 or flat_r > 0.5:
        lines.append(
            "Warning: one or both traces are mostly near zero — the ROI may miss the LED. "
            "Consider 'Re-run eye brightness with manual ROIs'."
        )
    elif np.nanmax(left) < 5.0 or np.nanmax(right) < 5.0:
        lines.append(
            "Warning: peak brightness is very low — verify ROI placement or threshold."
        )
    else:
        lines.append(
            "Traces look plausible (clear LED pulses). Re-run only if sync looks wrong later."
        )
    return "\n".join(lines)


class EyeBrightnessPreviewDialog(QtWidgets.QDialog):
    """Show stored per-frame brightness traces and a short quality summary."""

    def __init__(
        self,
        left_values,
        right_values,
        *,
        source_path: Path | str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Eye brightness preview")
        self.resize(860, 520)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                f"Source: {source_path}\n"
                "Inspect LED pulses per frame. Flat or near-zero traces usually mean a bad ROI."
            )
        )

        from eye_tracking_system_tools.annotation.preprocessing_gui.pyqtgraph_helpers import (
            EyeBrightnessPreviewPlot,
        )

        self._plot = EyeBrightnessPreviewPlot()
        self._plot.setMinimumHeight(320)
        self._plot.set_traces(left_values, right_values)
        layout.addWidget(self._plot, stretch=1)

        self._summary = QtWidgets.QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(110)
        self._summary.setPlainText(summarize_brightness_traces(left_values, right_values))
        layout.addWidget(self._summary)

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)


def show_eye_brightness_preview(blocksync, parent: QtWidgets.QWidget | None = None) -> None:
    """Load brightness vectors and open the preview dialog."""
    pkl = Path(blocksync.analysis_path) / "eye_brightness_values_dict.pkl"
    left, right = load_eye_brightness_lists(blocksync)
    dlg = EyeBrightnessPreviewDialog(
        left,
        right,
        source_path=pkl if pkl.is_file() else "in-memory BlockSync state",
        parent=parent,
    )
    dlg.exec()
