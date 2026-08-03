"""Native Qt ellipse verifier (Kerr ref + pupil perimeter + image QC)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from eye_tracking_system_tools.preprocessing.noise_epochs import (
    CATEGORY_PUPIL_PERIMETER,
    append_epochs,
    frames_outside_perimeter,
)
from eye_tracking_system_tools.preprocessing.pupil_perimeter import (
    normalize_perimeter,
    perimeter_summary,
    raw_circle_from_display_drag,
    raw_rect_from_display_drag,
)


def _apply_color_filters(
    rgb: np.ndarray,
    *,
    contrast: float,
    saturation: float,
    gamma: float,
) -> np.ndarray:
    """Contrast / saturation / gamma on an RGB uint8 image."""
    img = rgb.astype(np.float32) / 255.0
    # Contrast around mid-grey
    img = (img - 0.5) * float(contrast) + 0.5
    img = np.clip(img, 0.0, 1.0)
    # Saturation: lerp toward grayscale
    gray = (
        0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    )[..., None]
    sat = float(saturation)
    img = gray + (img - gray) * sat
    img = np.clip(img, 0.0, 1.0)
    # Gamma
    g = float(gamma) if float(gamma) > 1e-6 else 1.0
    img = np.power(img, 1.0 / g)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


class FramePickerView(QtWidgets.QGraphicsView):
    """Video frame view: click (Kerr) or drag rubber-band (perimeter / zoom)."""

    clicked_scene = QtCore.pyqtSignal(float, float)
    drag_finished_scene = QtCore.pyqtSignal(float, float, float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._interaction = "click"  # "click" | "drag"
        self._origin_view: QtCore.QPoint | None = None
        self._zoom_rect: QtCore.QRectF | None = None
        self._rubber = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Shape.Rectangle, self.viewport()
        )

    def set_interaction(self, mode: str) -> None:
        self._interaction = "drag" if mode == "drag" else "click"
        self._rubber.hide()
        self._origin_view = None

    def set_zoom_rect(self, rect: QtCore.QRectF | None) -> None:
        self._zoom_rect = QtCore.QRectF(rect) if rect is not None else None
        self._apply_view()

    def clear_zoom(self) -> None:
        self.set_zoom_rect(None)

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self._apply_view()

    def _apply_view(self) -> None:
        if self._pixmap_item is None:
            return
        if self._zoom_rect is not None and self._zoom_rect.width() > 2:
            self.fitInView(self._zoom_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.fitInView(
                self._pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_view()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._interaction == "click":
                pos = self.mapToScene(event.pos())
                self.clicked_scene.emit(float(pos.x()), float(pos.y()))
            else:
                self._origin_view = event.pos()
                self._rubber.setGeometry(QtCore.QRect(self._origin_view, QtCore.QSize()))
                self._rubber.show()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._interaction == "drag" and self._origin_view is not None:
            rect = QtCore.QRect(self._origin_view, event.pos()).normalized()
            self._rubber.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._interaction == "drag"
            and self._origin_view is not None
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            p0 = self.mapToScene(self._origin_view)
            p1 = self.mapToScene(event.pos())
            self._rubber.hide()
            self._origin_view = None
            self.drag_finished_scene.emit(
                float(p0.x()), float(p0.y()), float(p1.x()), float(p1.y())
            )
        super().mouseReleaseEvent(event)


class EllipseVerifierWidget(QtWidgets.QWidget):
    """Interactive eye-video verifier: Kerr ref, perimeter commit, zoom, color QC."""

    def __init__(
        self,
        df: pd.DataFrame,
        video_path: str | Path,
        eye: str,
        ref_point_xy: tuple[int, int] | None = None,
        parent: QtWidgets.QWidget | None = None,
        perimeter: dict[str, Any] | None = None,
        block_path: Path | str | None = None,
    ):
        super().__init__(parent)
        eye_lc = eye.lower()
        if eye_lc not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")

        self._eye = eye_lc
        self._block_path = Path(block_path) if block_path is not None else None
        self._df_current = df.copy()
        self._reader = VideoReader(video_path)
        self._current_ref = ref_point_xy
        self._perimeter: dict[str, Any] | None = normalize_perimeter(perimeter)
        self._shape_kind = "rect"
        self._mode = "kerr"  # kerr | perimeter | zoom
        self._frame_idx = 0
        self._playing = False
        self._frame_w = 0
        self._frame_h = 0
        self._skip_frames = 30 * 60
        self._contrast = 1.0
        self._saturation = 1.0
        self._gamma = 1.0
        self._probe_video_geometry(video_path)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_one_frame)

        self._build_ui()
        self._frame_view.clicked_scene.connect(self._on_frame_clicked)
        self._frame_view.drag_finished_scene.connect(self._on_drag_finished)
        self._show_frame(self._frame_idx)
        self._update_perimeter_status()

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
        self._title = QtWidgets.QLabel(f"<b>{title}</b>")
        layout.addWidget(self._title)

        mode_row = QtWidgets.QHBoxLayout()
        self._rb_kerr = QtWidgets.QRadioButton("Kerr ref")
        self._rb_perim = QtWidgets.QRadioButton("Perimeter")
        self._rb_zoom = QtWidgets.QRadioButton("Zoom region")
        self._rb_kerr.setChecked(True)
        self._rb_kerr.setToolTip("Click the frame to set the Kerr reference point")
        self._rb_perim.setToolTip(
            "Drag on the frame to draw a physiological pupil bound (rect or circle)"
        )
        self._rb_zoom.setToolTip("Drag a rectangle to zoom into that image region")
        self._mode_group = QtWidgets.QButtonGroup(self)
        for rb in (self._rb_kerr, self._rb_perim, self._rb_zoom):
            self._mode_group.addButton(rb)
            mode_row.addWidget(rb)
        mode_row.addSpacing(12)
        self._rb_rect = QtWidgets.QRadioButton("Rectangle")
        self._rb_circle = QtWidgets.QRadioButton("Circle")
        self._rb_rect.setChecked(True)
        self._shape_group = QtWidgets.QButtonGroup(self)
        self._shape_group.addButton(self._rb_rect)
        self._shape_group.addButton(self._rb_circle)
        mode_row.addWidget(self._rb_rect)
        mode_row.addWidget(self._rb_circle)
        self._btn_reset_zoom = QtWidgets.QPushButton("Reset zoom")
        mode_row.addWidget(self._btn_reset_zoom)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        perim_row = QtWidgets.QHBoxLayout()
        self._btn_clear_perim = QtWidgets.QPushButton("Clear perimeter")
        self._btn_commit_perim = QtWidgets.QPushButton("Commit bad datapoints")
        self._btn_commit_perim.setToolTip(
            "Write noise_epochs_{eye}.csv rows (category=pupil_perimeter) for "
            "frames whose center lies outside the perimeter — does not NaN the data"
        )
        self._perim_status = QtWidgets.QLabel("Perimeter: (none)")
        self._perim_status.setWordWrap(True)
        perim_row.addWidget(self._btn_clear_perim)
        perim_row.addWidget(self._btn_commit_perim)
        perim_row.addWidget(self._perim_status, stretch=1)
        layout.addLayout(perim_row)

        filt = QtWidgets.QGroupBox("Image filters")
        filt_lay = QtWidgets.QGridLayout(filt)
        self._slider_contrast = self._make_filter_slider(50, 200, 100)
        self._slider_sat = self._make_filter_slider(0, 200, 100)
        self._slider_gamma = self._make_filter_slider(30, 300, 100)
        self._lbl_contrast = QtWidgets.QLabel("1.00")
        self._lbl_sat = QtWidgets.QLabel("1.00")
        self._lbl_gamma = QtWidgets.QLabel("1.00")
        filt_lay.addWidget(QtWidgets.QLabel("Contrast"), 0, 0)
        filt_lay.addWidget(self._slider_contrast, 0, 1)
        filt_lay.addWidget(self._lbl_contrast, 0, 2)
        filt_lay.addWidget(QtWidgets.QLabel("Saturation"), 1, 0)
        filt_lay.addWidget(self._slider_sat, 1, 1)
        filt_lay.addWidget(self._lbl_sat, 1, 2)
        filt_lay.addWidget(QtWidgets.QLabel("Gamma"), 2, 0)
        filt_lay.addWidget(self._slider_gamma, 2, 1)
        filt_lay.addWidget(self._lbl_gamma, 2, 2)
        self._btn_reset_filters = QtWidgets.QPushButton("Reset filters")
        filt_lay.addWidget(self._btn_reset_filters, 0, 3, 3, 1)
        layout.addWidget(filt)

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

        preview_row = QtWidgets.QHBoxLayout()
        self._btn_angles_span = QtWidgets.QPushButton("Compute angles span")
        self._btn_angles_span.setToolTip(
            "Run Kerr angles with the current red-dot reference (not yet saved) "
            "and show φ/θ histograms for this eye"
        )
        preview_row.addWidget(self._btn_angles_span)
        preview_row.addStretch(1)
        layout.addLayout(preview_row)

        self._btn_play.clicked.connect(self._on_play)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_bwd.clicked.connect(self._on_bwd)
        self._btn_fwd.clicked.connect(self._on_fwd)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._btn_xflip.clicked.connect(self._on_xflip)
        self._btn_phi.clicked.connect(self._on_phi)
        self._btn_flipx.clicked.connect(self._on_flipx)
        self._btn_flip_dot.clicked.connect(self._on_flip_dot)
        self._btn_angles_span.clicked.connect(self._on_compute_angles_span)
        self._rb_kerr.toggled.connect(self._on_mode_changed)
        self._rb_perim.toggled.connect(self._on_mode_changed)
        self._rb_zoom.toggled.connect(self._on_mode_changed)
        self._rb_rect.toggled.connect(self._on_shape_changed)
        self._rb_circle.toggled.connect(self._on_shape_changed)
        self._btn_clear_perim.clicked.connect(self._on_clear_perimeter)
        self._btn_commit_perim.clicked.connect(self._on_commit_perimeter)
        self._btn_reset_zoom.clicked.connect(self._on_reset_zoom)
        self._btn_reset_filters.clicked.connect(self._on_reset_filters)
        self._slider_contrast.valueChanged.connect(self._on_filters_changed)
        self._slider_sat.valueChanged.connect(self._on_filters_changed)
        self._slider_gamma.valueChanged.connect(self._on_filters_changed)
        self._sync_mode_ui()

    @staticmethod
    def _make_filter_slider(lo: int, hi: int, val: int) -> QtWidgets.QSlider:
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

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

    def perimeter(self) -> dict[str, Any] | None:
        return normalize_perimeter(self._perimeter)

    def set_perimeter(self, d: dict[str, Any] | None) -> None:
        self._perimeter = normalize_perimeter(d)
        self._update_perimeter_status()
        self._show_frame(self._frame_idx)

    def set_block_path(self, block_path: Path | str | None) -> None:
        self._block_path = Path(block_path) if block_path is not None else None

    def pick_ref_from_display_xy(self, x_view: float, y_view: float) -> tuple[int, int]:
        y_raw = self._frame_h - 1 - int(round(y_view))
        x_raw = int(round(x_view))
        self._current_ref = (x_raw, y_raw)
        return self._current_ref

    def _sync_mode_ui(self) -> None:
        perim = self._rb_perim.isChecked()
        zoom = self._rb_zoom.isChecked()
        self._rb_rect.setEnabled(perim)
        self._rb_circle.setEnabled(perim)
        self._btn_clear_perim.setEnabled(True)
        self._btn_commit_perim.setEnabled(self._perimeter is not None)
        self._frame_view.set_interaction("drag" if (perim or zoom) else "click")
        eye_title = "Left eye" if self._eye == "left" else "Right eye"
        if zoom:
            self._title.setText(f"<b>{eye_title}</b> — Zoom: drag a region")
        elif perim:
            tip = (
                "drag center→rim for circle"
                if self._rb_circle.isChecked()
                else "drag a rectangle"
            )
            self._title.setText(f"<b>{eye_title}</b> — Perimeter mode ({tip})")
        else:
            self._title.setText(f"<b>{eye_title}</b> — click frame to set Kerr ref")

    def _on_mode_changed(self, _checked: bool = False) -> None:
        if self._rb_zoom.isChecked():
            self._mode = "zoom"
        elif self._rb_perim.isChecked():
            self._mode = "perimeter"
        else:
            self._mode = "kerr"
        self._sync_mode_ui()

    def _on_shape_changed(self, _checked: bool = False) -> None:
        self._shape_kind = "circle" if self._rb_circle.isChecked() else "rect"
        self._sync_mode_ui()

    def _update_perimeter_status(self) -> None:
        self._perim_status.setText(f"Perimeter: {perimeter_summary(self._perimeter)}")
        self._btn_commit_perim.setEnabled(self._perimeter is not None)

    def _on_clear_perimeter(self) -> None:
        self._perimeter = None
        self._update_perimeter_status()
        self._show_frame(self._frame_idx)

    def _on_commit_perimeter(self) -> None:
        if self._perimeter is None:
            QtWidgets.QMessageBox.information(
                self, "Perimeter", "Draw a perimeter first."
            )
            return
        if self._block_path is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Perimeter",
                "Block path unknown — cannot write noise epochs.",
            )
            return
        frames = frames_outside_perimeter(
            self._df_current,
            self._perimeter,
            frame_col=self._frame_col(),
        )
        path, n_frames, n_epochs = append_epochs(
            self._block_path,
            self._eye,
            category=CATEGORY_PUPIL_PERIMETER,
            frames=frames,
            replace_category=True,
        )
        self._perim_status.setText(
            f"Perimeter: {perimeter_summary(self._perimeter)} — "
            f"committed {n_frames} frames ({n_epochs} epochs) → {path.name}"
        )
        QtWidgets.QMessageBox.information(
            self,
            "Bad datapoints committed",
            f"Wrote {n_epochs} epoch(s) covering {n_frames} frame(s) as "
            f"'{CATEGORY_PUPIL_PERIMETER}' to {path.name}.\n"
            "Eye CSVs were not NaN'd. Use Sync → Apply selected noise to apply later.",
        )

    def _on_reset_zoom(self) -> None:
        self._frame_view.clear_zoom()

    def _on_reset_filters(self) -> None:
        for s, v in (
            (self._slider_contrast, 100),
            (self._slider_sat, 100),
            (self._slider_gamma, 100),
        ):
            s.blockSignals(True)
            s.setValue(v)
            s.blockSignals(False)
        self._on_filters_changed()

    def _on_filters_changed(self, _v: int = 0) -> None:
        self._contrast = self._slider_contrast.value() / 100.0
        self._saturation = self._slider_sat.value() / 100.0
        self._gamma = self._slider_gamma.value() / 100.0
        self._lbl_contrast.setText(f"{self._contrast:.2f}")
        self._lbl_sat.setText(f"{self._saturation:.2f}")
        self._lbl_gamma.setText(f"{self._gamma:.2f}")
        self._show_frame(self._frame_idx)

    def _on_frame_clicked(self, x_view: float, y_view: float) -> None:
        if self._mode != "kerr":
            return
        self.pick_ref_from_display_xy(x_view, y_view)
        self._show_frame(self._frame_idx)

    def _on_drag_finished(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        if self._mode == "zoom":
            rect = QtCore.QRectF(
                QtCore.QPointF(min(x0, x1), min(y0, y1)),
                QtCore.QPointF(max(x0, x1), max(y0, y1)),
            )
            if rect.width() < 4 or rect.height() < 4:
                return
            self._frame_view.set_zoom_rect(rect)
            return
        if self._mode != "perimeter":
            return
        if self._shape_kind == "circle":
            self._perimeter = raw_circle_from_display_drag(
                x0, y0, x1, y1, self._frame_h
            )
        else:
            self._perimeter = raw_rect_from_display_drag(
                x0, y0, x1, y1, self._frame_h
            )
        self._perimeter = normalize_perimeter(self._perimeter)
        self._update_perimeter_status()
        self._show_frame(self._frame_idx)

    def _draw_perimeter_overlay(self, annotated: np.ndarray) -> None:
        peri = normalize_perimeter(self._perimeter)
        if peri is None:
            return
        color = (0, 255, 255)
        if peri["shape"] == "rect":
            x0 = int(round(peri["x"]))
            y0 = int(round(peri["y"]))
            x1 = int(round(peri["x"] + peri["w"]))
            y1 = int(round(peri["y"] + peri["h"]))
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
        else:
            cx = int(round(peri["cx"]))
            cy = int(round(peri["cy"]))
            r = int(round(peri["r"]))
            cv2.circle(annotated, (cx, cy), max(r, 1), color, 2)

    def _render_frame_rgb(self, frame_idx: int) -> np.ndarray | None:
        rgb = self._reader.read_frame(frame_idx)
        if rgb is None:
            return None
        filtered = _apply_color_filters(
            rgb,
            contrast=self._contrast,
            saturation=self._saturation,
            gamma=self._gamma,
        )
        annotated = draw_ellipse_overlay(
            filtered,
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
        self._draw_perimeter_overlay(annotated)
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

    def _on_compute_angles_span(self) -> None:
        ref = self.ref_xy()
        if ref is None:
            QtWidgets.QMessageBox.information(
                self,
                "Compute angles span",
                "Set a Kerr reference point first (Kerr ref mode → click the frame).",
            )
            return

        from eye_tracking_system_tools.annotation.preprocessing_gui.kerr_angles_span_dialog import (
            KerrAnglesSpanDialog,
        )
        from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
            preview_kerr_angles,
        )

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            preview = preview_kerr_angles(self._df_current, ref[0], ref[1])
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(
                self,
                "Compute angles span",
                f"Kerr preview failed:\n{exc}",
            )
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        dialog = KerrAnglesSpanDialog(
            preview,
            eye=self._eye,
            eye_df=self._df_current,
            parent=self,
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_ref = dialog.ref_xy()
            self._current_ref = new_ref
            self._show_frame(self._frame_idx)
