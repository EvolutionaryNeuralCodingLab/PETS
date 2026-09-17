"""Ellipse rotation angle correction dialog for the Refine tab."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

import cv2

from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
    numpy_rgb_to_qpixmap,
)
from eye_tracking_system_tools.preprocessing.conicoid.spin_max import (
    SpinMaxSettings,
    binarize_gray,
    crop_roi,
    score_curve,
    spin_maximize_frame,
)
from eye_tracking_system_tools.preprocessing.conicoid.rotation_params import (
    RotationCorrectionParams,
)
from eye_tracking_system_tools.preprocessing.ellipse_fit import canonicalize_ellipse_phi


def _draw_ellipse_on_crop(
    crop: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    height: float,
    phi: float,
    color: tuple[int, int, int],
) -> np.ndarray:
    out = np.ascontiguousarray(crop.copy())
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    if not all(np.isfinite(v) for v in (cx, cy, width, height, phi)):
        return out
    cv2.ellipse(
        out,
        (int(round(cx)), int(round(cy))),
        (max(int(round(width)), 1), max(int(round(height)), 1)),
        float(np.degrees(phi)),
        0,
        360,
        color,
        1,
    )
    return out


def _score_plot_rgb(
    phis: np.ndarray,
    scores: np.ndarray,
    chosen: float | None,
    size: tuple[int, int] = (280, 100),
) -> np.ndarray:
    w, h = size
    img = np.full((h, w, 3), 32, dtype=np.uint8)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return img
    s = scores[finite]
    p = phis[finite]
    lo, hi = float(np.min(s)), float(np.max(s))
    span = max(hi - lo, 1e-6)
    xs = np.clip(((p / np.pi) * (w - 3)).astype(int), 0, w - 1)
    ys = np.clip(
        (h - 3 - ((s - lo) / span) * (h - 6)).astype(int), 0, h - 1
    )
    for i in range(1, xs.size):
        cv2.line(img, (int(xs[i - 1]), int(ys[i - 1])), (int(xs[i]), int(ys[i])), (200, 200, 200), 1)
    if chosen is not None and np.isfinite(chosen):
        x = int(round((float(np.mod(chosen, np.pi)) / np.pi) * (w - 3)))
        cv2.line(img, (x, 0), (x, h - 1), (80, 180, 255), 1)
    cv2.putText(
        img,
        "score vs phi [0, pi)",
        (6, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return img


class _EyeTuneColumn(QtWidgets.QWidget):
    """Per-eye threshold, X-flip, and frame slider over raw DLC ellipses."""

    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        title: str,
        df: pd.DataFrame,
        frame_fn: Callable[[int], np.ndarray | None],
        *,
        nframes: int,
        frame0: int,
        frame_width: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._df = df
        self._frame_fn = frame_fn
        self._nframes = max(int(nframes), 1)
        self._frame_width = float(frame_width) if np.isfinite(frame_width) and frame_width > 0 else None
        self._frame_idx = int(max(0, min(frame0, self._nframes - 1)))
        self._band_width = 5.0
        self._median_k = 5
        self._build_ui(title)
        self.refresh()

    def settings(self) -> SpinMaxSettings:
        thresh = None if self._chk_otsu.isChecked() else int(self._slider_thr.value())
        return SpinMaxSettings(
            threshold=thresh,
            roi_mult=float(self._spin_roi.value()),
            x_flip=self._chk_xflip.isChecked(),
            frame_width=self._frame_width,
            band_width=float(self._band_width),
            median_k=int(self._median_k),
        )

    def apply_settings(self, settings: SpinMaxSettings) -> None:
        self._band_width = float(settings.band_width)
        self._median_k = int(settings.median_k)
        if settings.frame_width is not None and np.isfinite(settings.frame_width):
            self._frame_width = float(settings.frame_width)
        self._chk_xflip.blockSignals(True)
        self._spin_roi.blockSignals(True)
        self._chk_otsu.blockSignals(True)
        self._slider_thr.blockSignals(True)
        try:
            self._chk_xflip.setChecked(bool(settings.x_flip))
            self._spin_roi.setValue(float(settings.roi_mult))
            if settings.threshold is None:
                self._chk_otsu.setChecked(True)
                self._slider_thr.setEnabled(False)
            else:
                self._chk_otsu.setChecked(False)
                self._slider_thr.setEnabled(True)
                self._slider_thr.setValue(int(np.clip(int(settings.threshold), 0, 255)))
        finally:
            self._chk_xflip.blockSignals(False)
            self._spin_roi.blockSignals(False)
            self._chk_otsu.blockSignals(False)
            self._slider_thr.blockSignals(False)

    def _row_for_frame(self, frame_idx: int) -> pd.Series | None:
        if self._df is None or self._df.empty or "eye_frame" not in self._df.columns:
            return None
        frames = pd.to_numeric(self._df["eye_frame"], errors="coerce")
        mask = frames == int(frame_idx)
        if not mask.any():
            return None
        return self._df.loc[mask].iloc[0]

    def _geometry(self, row: pd.Series) -> tuple[float, float, float, float, float]:
        try:
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            w = float(row["width"])
            h = float(row["height"])
            phi = float(row["phi"])
        except (TypeError, ValueError, KeyError):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        w, h, phi = canonicalize_ellipse_phi(w, h, phi)
        if self._chk_xflip.isChecked() and self._frame_width is not None:
            cx = float(self._frame_width) - cx
            phi = np.pi - phi
            w, h, phi = canonicalize_ellipse_phi(w, h, phi)
        return cx, cy, w, h, phi

    def _build_ui(self, title: str) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))

        self._chk_xflip = QtWidgets.QCheckBox("X-flip overlay (W − x, π − φ)")
        layout.addWidget(self._chk_xflip)

        roi_row = QtWidgets.QHBoxLayout()
        roi_row.addWidget(QtWidgets.QLabel("Crop factor"))
        self._spin_roi = QtWidgets.QDoubleSpinBox()
        self._spin_roi.setRange(0.8, 3.0)
        self._spin_roi.setSingleStep(0.1)
        self._spin_roi.setValue(1.2)
        self._spin_roi.setToolTip(
            "ROI side = 2 × crop factor × max(width, height). 1.0 just contains the ellipse."
        )
        roi_row.addWidget(self._spin_roi)
        roi_row.addStretch(1)
        layout.addLayout(roi_row)

        self._chk_otsu = QtWidgets.QCheckBox("Auto (Otsu)")
        self._chk_otsu.setChecked(True)
        layout.addWidget(self._chk_otsu)

        thr_row = QtWidgets.QHBoxLayout()
        thr_row.addWidget(QtWidgets.QLabel("Thr"))
        self._slider_thr = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider_thr.setRange(0, 255)
        self._slider_thr.setValue(80)
        self._slider_thr.setEnabled(False)
        self._lbl_thr = QtWidgets.QLabel("auto")
        self._lbl_thr.setMinimumWidth(48)
        thr_row.addWidget(self._slider_thr, stretch=1)
        thr_row.addWidget(self._lbl_thr)
        layout.addLayout(thr_row)

        pics = QtWidgets.QHBoxLayout()
        self._lbl_crop = QtWidgets.QLabel()
        self._lbl_bin = QtWidgets.QLabel()
        for lab in (self._lbl_crop, self._lbl_bin):
            lab.setMinimumSize(160, 140)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            pics.addWidget(lab)
        layout.addLayout(pics)
        self._lbl_plot = QtWidgets.QLabel()
        self._lbl_plot.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_plot)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(self._nframes - 1)
        self._slider.setValue(self._frame_idx)
        layout.addWidget(self._slider)
        self._counter = QtWidgets.QLabel("")
        layout.addWidget(self._counter)
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._chk_xflip.toggled.connect(self.refresh)
        self._chk_otsu.toggled.connect(self._on_otsu)
        self._slider_thr.valueChanged.connect(self.refresh)
        self._spin_roi.valueChanged.connect(self.refresh)
        self._slider.valueChanged.connect(self._on_slider)

    def _on_otsu(self, checked: bool) -> None:
        self._slider_thr.setEnabled(not checked)
        self.refresh()

    def _on_slider(self, value: int) -> None:
        self._frame_idx = int(value)
        self.refresh()

    def refresh(self) -> None:
        try:
            self._refresh_impl()
        except Exception as exc:
            self._status.setText(f"Cannot preview this frame: {exc}")
            self._clear_pix()
            self.changed.emit()

    def _refresh_impl(self) -> None:
        self._counter.setText(f"{self._frame_idx} / {self._nframes - 1}")
        image = self._frame_fn(self._frame_idx)
        if image is not None and self._frame_width is None:
            self._frame_width = float(image.shape[1])
        row = self._row_for_frame(self._frame_idx)
        if image is None:
            self._status.setText("No video frame.")
            self._clear_pix()
            self.changed.emit()
            return
        if row is None:
            self._status.setText("No raw DLC ellipse on this frame.")
            pix = _scaled_pix(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            self._lbl_crop.setPixmap(pix)
            self._lbl_bin.clear()
            self._lbl_plot.clear()
            self.changed.emit()
            return
        cx, cy, w, h, phi = self._geometry(row)
        if not all(np.isfinite(v) for v in (cx, cy, w, h, phi)):
            self._status.setText("No ellipse on this frame (NaN geometry).")
            pix = _scaled_pix(
                image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            )
            self._lbl_crop.setPixmap(pix)
            self._lbl_bin.clear()
            self._lbl_plot.clear()
            self.changed.emit()
            return
        settings = self.settings()
        crop, x0, y0 = crop_roi(image, cx, cy, w, h, settings.roi_mult)
        binary, used = binarize_gray(crop, settings.threshold)
        if settings.threshold is None:
            self._lbl_thr.setText(f"Otsu {used}")
        else:
            self._lbl_thr.setText(str(used))
        lx, ly = cx - x0, cy - y0
        crop_rgb = _draw_ellipse_on_crop(crop, lx, ly, w, h, phi, (0, 255, 0))
        _, _, ang, score = spin_maximize_frame(
            image, cx, cy, w, h, phi, settings=settings
        )
        if np.isfinite(ang):
            crop_rgb = _draw_ellipse_on_crop(
                crop_rgb, lx, ly, w, h, ang, (80, 180, 255)
            )
        bin_rgb = _draw_ellipse_on_crop(
            binary, lx, ly, w, h, ang if np.isfinite(ang) else phi, (80, 180, 255)
        )
        # Match Verify: draw in raw camera coords, then flip vertically for display.
        crop_rgb = cv2.flip(crop_rgb, 0)
        bin_rgb = cv2.flip(bin_rgb, 0)
        phis, scores = score_curve(
            binary, lx, ly, w, h, band_width=settings.band_width, step_deg=5.0
        )
        plot = _score_plot_rgb(phis, scores, ang if np.isfinite(ang) else None)
        self._lbl_crop.setPixmap(_scaled_pix(crop_rgb, 220))
        self._lbl_bin.setPixmap(_scaled_pix(bin_rgb, 220))
        self._lbl_plot.setPixmap(_scaled_pix(plot, 280))
        flip_note = "  [X-flip on]" if settings.x_flip else ""
        if not np.isfinite(ang):
            self._status.setText("No usable ellipse geometry." + flip_note)
        elif not np.isfinite(score):
            self._status.setText(
                f"phi = {np.degrees(ang):.1f}° (flat score){flip_note}"
            )
        else:
            self._status.setText(
                f"phi = {np.degrees(ang):.1f}°  score = {score:.3f}{flip_note}"
            )
        self.changed.emit()

    def _clear_pix(self) -> None:
        self._lbl_crop.clear()
        self._lbl_bin.clear()
        self._lbl_plot.clear()


class SpinMaximizerDialog(QtWidgets.QDialog):
    """Tune per-eye threshold / X-flip and save ``rotation_correction_params.yaml``."""

    def __init__(
        self,
        *,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_frame_fn: Callable[[int], np.ndarray | None],
        right_frame_fn: Callable[[int], np.ndarray | None],
        left_nframes: int,
        right_nframes: int,
        left_frame0: int,
        right_frame0: int,
        left_width: float,
        right_width: float,
        parent: QtWidgets.QWidget | None = None,
        initial_left: SpinMaxSettings | None = None,
        initial_right: SpinMaxSettings | None = None,
        initial_apply_jitter: bool = True,
        block_label: str | None = None,
    ):
        super().__init__(parent)
        title = "Ellipse rotation angle correction"
        if block_label:
            title = f"{title} — {block_label}"
        self.setWindowTitle(title)
        self.setMinimumWidth(780)
        self._build_ui(
            left_df,
            right_df,
            left_frame_fn,
            right_frame_fn,
            left_nframes,
            right_nframes,
            left_frame0,
            right_frame0,
            left_width,
            right_width,
        )
        if initial_left is not None:
            self._left.apply_settings(initial_left)
        if initial_right is not None:
            self._right.apply_settings(initial_right)
        self._chk_jitter.setChecked(bool(initial_apply_jitter))
        if initial_left is not None or initial_right is not None:
            self._left.refresh()
            self._right.refresh()

    def eye_settings(self) -> dict[str, SpinMaxSettings]:
        return {
            "left": self._left.settings(),
            "right": self._right.settings(),
        }

    def apply_jitter_after(self) -> bool:
        return self._chk_jitter.isChecked()

    def correction_params(self) -> RotationCorrectionParams:
        settings = self.eye_settings()
        return RotationCorrectionParams(
            left=settings["left"],
            right=settings["right"],
            apply_jitter=self.apply_jitter_after(),
        )

    def _build_ui(
        self,
        left_df,
        right_df,
        left_frame_fn,
        right_frame_fn,
        left_nframes,
        right_nframes,
        left_frame0,
        right_frame0,
        left_width,
        right_width,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Uses raw DLC xy (on the pupil in the video), not jitter-corrected. "
            "X-flip mirrors ellipse data in camera pixels like the Verify tab "
            "(W − x, π − φ) without flipping the video. Crops use the same vertical "
            "display flip as Verify. Threshold and crop factor are per eye. "
            "Save writes analysis/rotation_correction_params.yaml only — it does not "
            "run the recording. Run rotation correction from the Refine tab or the "
            "batch notebook / CLI. Check apply-jitter so the batch job adds jitter "
            "when jitter_report_dict.pkl is present."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        cols = QtWidgets.QHBoxLayout()
        self._left = _EyeTuneColumn(
            "Left eye",
            left_df,
            left_frame_fn,
            nframes=left_nframes,
            frame0=left_frame0,
            frame_width=left_width,
            parent=self,
        )
        self._right = _EyeTuneColumn(
            "Right eye",
            right_df,
            right_frame_fn,
            nframes=right_nframes,
            frame0=right_frame0,
            frame_width=right_width,
            parent=self,
        )
        cols.addWidget(self._left)
        cols.addWidget(self._right)
        layout.addLayout(cols)

        self._chk_jitter = QtWidgets.QCheckBox(
            "After correction, apply jitter (rotation-corrected xy → Verify eye_data xy)"
        )
        layout.addWidget(self._chk_jitter)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            "Save rotation-correction parameters"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _scaled_pix(rgb: np.ndarray, max_side: int = 220) -> QtGui.QPixmap:
    pix = numpy_rgb_to_qpixmap(rgb)
    return pix.scaled(
        max_side,
        max_side,
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )
