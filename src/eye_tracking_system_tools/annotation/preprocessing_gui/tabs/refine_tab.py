"""Stage 2b -- Swirski image refine with dual original/refined ellipse overlay."""

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
from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    LoadReport,
    refine_artifact_profile,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.ellipse_verifier import (
    FramePickerView,
    _apply_color_filters,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.workers import CallableWorker
from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import load_eye_data
from eye_tracking_system_tools.preprocessing.block_sync_core import load_eye_tracking_df_csv
from eye_tracking_system_tools.preprocessing.conicoid.camera import resolve_intrinsics
from eye_tracking_system_tools.preprocessing.conicoid.refine import (
    RefineSettings,
    extract_pupil_inliers,
    refine_eye_table,
)
from eye_tracking_system_tools.preprocessing.conicoid.refined_io import (
    attach_refined_columns,
    attach_spin_columns,
    latest_refined_eye_csv,
    promote_refined_to_eye_data,
    read_refined_eye_table,
    read_rotation_fixed_eye_table,
    refined_eye_csv_path,
    resolve_rotation_fixed_eye_csv,
    rotation_fixed_eye_csv_path,
    write_refined_eye_table,
    write_rotation_fixed_eye_table,
)
from eye_tracking_system_tools.preprocessing.conicoid.refraction import (
    refraction_npz_path,
    try_load_refraction_maps,
)
from eye_tracking_system_tools.preprocessing.conicoid.batch_rotation import run_block
from eye_tracking_system_tools.preprocessing.conicoid.rotation_params import (
    try_read_rotation_params,
    write_rotation_params,
)
from eye_tracking_system_tools.preprocessing.conicoid.spin_max import (
    apply_jitter_to_spin_table,
    canonicalize_ellipse_table,
    raw_ellipse_table_from_dlc_df,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.spin_max_dialog import (
    SpinMaximizerDialog,
)
from eye_tracking_system_tools.preprocessing.dlc_csv_io import resolve_dlc_csv


class RefinePreviewWidget(QtWidgets.QWidget):
    """Verify-like transport with green original / magenta refined ellipses."""

    def __init__(
        self,
        df: pd.DataFrame,
        video_path: str | Path,
        eye: str,
        parent: QtWidgets.QWidget | None = None,
        *,
        show_gaze: bool = False,
    ):
        super().__init__(parent)
        eye_lc = eye.lower()
        if eye_lc not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        self._eye = eye_lc
        self._df = df.copy()
        self._reader = VideoReader(video_path)
        self._frame_idx = 0
        self._playing = False
        self._skip_frames = 30 * 60
        self._contrast = 1.0
        self._saturation = 1.0
        self._gamma = 1.0
        self._show_original = True
        self._show_refined = True
        self._show_spin = True
        self._show_gaze = bool(show_gaze)
        self._flip_vertical = True

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_one_frame)
        self._build_ui()
        self._show_frame(0)

    def df(self) -> pd.DataFrame:
        return self._df

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._show_frame(self._frame_idx)

    def frame_idx(self) -> int:
        return int(self._frame_idx)

    def current_row(self) -> pd.Series | None:
        col = self._frame_col()
        if col not in self._df.columns:
            return None
        mask = self._df[col] == self._frame_idx
        if not mask.any():
            return None
        return self._df.loc[mask].iloc[0]

    def set_overlay_flags(
        self, *, original: bool, refined: bool, spin: bool | None = None
    ) -> None:
        self._show_original = bool(original)
        self._show_refined = bool(refined)
        if spin is not None:
            self._show_spin = bool(spin)
        self._show_frame(self._frame_idx)

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._reader.close()
        super().closeEvent(event)

    def _frame_col(self) -> str:
        return "eye_frame" if "eye_frame" in self._df.columns else "frame"

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        title = "Left eye" if self._eye == "left" else "Right eye"
        layout.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))

        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.setContentsMargins(0, 0, 0, 0)
        overlay_row.setSpacing(6)
        self._chk_orig = QtWidgets.QCheckBox("Original")
        self._chk_orig.setChecked(True)
        self._chk_ref = QtWidgets.QCheckBox("Refined")
        self._chk_ref.setChecked(True)
        self._chk_spin = QtWidgets.QCheckBox("Rotation-corrected")
        self._chk_spin.setChecked(True)
        overlay_row.addWidget(self._chk_orig)
        overlay_row.addWidget(self._chk_ref)
        overlay_row.addWidget(self._chk_spin)
        overlay_row.addStretch(1)
        layout.addLayout(overlay_row)
        self._chk_orig.toggled.connect(self._on_overlay_toggled)
        self._chk_ref.toggled.connect(self._on_overlay_toggled)
        self._chk_spin.toggled.connect(self._on_overlay_toggled)

        filt_row = QtWidgets.QHBoxLayout()
        filt_row.setContentsMargins(0, 0, 0, 0)
        filt_row.setSpacing(4)
        self._slider_contrast = self._make_filter_slider(50, 200, 100)
        self._slider_sat = self._make_filter_slider(0, 200, 100)
        self._slider_gamma = self._make_filter_slider(30, 300, 100)
        self._lbl_contrast = QtWidgets.QLabel("1.00")
        self._lbl_sat = QtWidgets.QLabel("1.00")
        self._lbl_gamma = QtWidgets.QLabel("1.00")
        for label, slider, value in (
            ("C", self._slider_contrast, self._lbl_contrast),
            ("S", self._slider_sat, self._lbl_sat),
            ("G", self._slider_gamma, self._lbl_gamma),
        ):
            filt_row.addWidget(QtWidgets.QLabel(label))
            filt_row.addWidget(slider, stretch=1)
            filt_row.addWidget(value)
        layout.addLayout(filt_row)
        self._slider_contrast.valueChanged.connect(self._on_filters_changed)
        self._slider_sat.valueChanged.connect(self._on_filters_changed)
        self._slider_gamma.valueChanged.connect(self._on_filters_changed)

        self._frame_view = FramePickerView()
        self._frame_view.setMinimumHeight(240)
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

        self._btn_play.clicked.connect(self._on_play)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_bwd.clicked.connect(lambda: self._show_frame(self._frame_idx - self._skip_frames))
        self._btn_fwd.clicked.connect(lambda: self._show_frame(self._frame_idx + self._skip_frames))
        self._slider.valueChanged.connect(self._on_slider_changed)

    @staticmethod
    def _make_filter_slider(lo: int, hi: int, val: int) -> QtWidgets.QSlider:
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def _on_overlay_toggled(self, _checked: bool = False) -> None:
        self.set_overlay_flags(
            original=self._chk_orig.isChecked(),
            refined=self._chk_ref.isChecked(),
            spin=self._chk_spin.isChecked(),
        )

    def _on_filters_changed(self, _value: int = 0) -> None:
        self._contrast = self._slider_contrast.value() / 100.0
        self._saturation = self._slider_sat.value() / 100.0
        self._gamma = self._slider_gamma.value() / 100.0
        self._lbl_contrast.setText(f"{self._contrast:.2f}")
        self._lbl_sat.setText(f"{self._saturation:.2f}")
        self._lbl_gamma.setText(f"{self._gamma:.2f}")
        self._show_frame(self._frame_idx)

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
        col = self._frame_col()
        if self._show_original:
            filtered = draw_ellipse_overlay(
                filtered,
                self._df,
                col,
                int(frame_idx),
                color=(0, 255, 0),
                phi_unit="radians",
            )
        if self._show_refined:
            filtered = draw_ellipse_overlay(
                filtered,
                self._df,
                col,
                int(frame_idx),
                color=(255, 0, 255),
                column_suffix="refined",
                phi_unit="radians",
            )
        if self._show_spin:
            filtered = draw_ellipse_overlay(
                filtered,
                self._df,
                col,
                int(frame_idx),
                color=(0, 0, 255),
                column_suffix="spin",
                phi_unit="radians",
            )
        if self._show_gaze:
            from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
                draw_gaze_line,
            )

            filtered = draw_gaze_line(filtered, self._df, col, int(frame_idx))
        if self._flip_vertical:
            return cv2.flip(filtered, 0)
        return filtered

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

    def _on_slider_changed(self, value: int) -> None:
        if value != self._frame_idx:
            self._show_frame(value)

    def nframes(self) -> int:
        return max(int(self._reader.nframes), 0)

    def raw_frame(self, frame_idx: int | None = None) -> np.ndarray | None:
        idx = self._frame_idx if frame_idx is None else int(frame_idx)
        return self._reader.read_frame(idx)


class RefineTab(BaseTab):
    tab_id = "refine"
    tab_label = "Refine"

    def __init__(self, state, config, parent=None):
        self._worker: CallableWorker | None = None
        self._left: RefinePreviewWidget | None = None
        self._right: RefinePreviewWidget | None = None
        self._sphere_cache: dict = {}
        super().__init__(state, config, parent)

    def artifact_profile(self):
        return refine_artifact_profile(self._name_tag_value_safe())

    def _name_tag_value_safe(self) -> str:
        if hasattr(self, "_name_tag"):
            return self._name_tag.text().strip() or "refined"
        return "refined"

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._build_artifact_panel())

        self._info = QtWidgets.QLabel(
            "Fit the ellipse to the pupil boundary first (blue = rotation-corrected "
            "raw DLC xy). Apply jitter afterwards so blue xy matches green Verify "
            "eye_data; phi stays the corrected angle. Then Swirski 2D. "
            "Original eye CSVs are untouched until Promote."
        )
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setContentsMargins(0, 0, 0, 0)
        btn_grid.setHorizontalSpacing(4)
        btn_grid.setVerticalSpacing(3)
        self._btn_preview = self._small_button("Preview this frame")
        self._btn_visible = self._small_button("Run visible range")
        self._btn_full = self._small_button("Run full recording")
        self._btn_save = self._small_button("Save refined CSV")
        self._btn_promote = self._small_button("Promote to eye_data")
        self._btn_spin = self._small_button("Correct ellipse rotation…")
        self._btn_spin_run = self._small_button("Run rotation correction")
        self._btn_spin_jitter = self._small_button("Apply jitter to rotation-corrected")
        for i, btn in enumerate(
            (
                self._btn_preview,
                self._btn_visible,
                self._btn_full,
                self._btn_save,
                self._btn_promote,
                self._btn_spin,
                self._btn_spin_run,
                self._btn_spin_jitter,
            )
        ):
            btn_grid.addWidget(btn, i // 3, i % 3)
        layout.addLayout(btn_grid)

        settings = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(settings)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)

        self._name_tag = QtWidgets.QLineEdit("refined")
        self._name_tag.setPlaceholderText("e.g. refined")
        self._name_tag.setMaximumWidth(140)
        self._name_tag.textChanged.connect(lambda _: self._refresh_artifact_ui())

        self._metric = QtWidgets.QComboBox()
        self._metric.addItems(["Region contrast", "Edge distance", "Both"])
        self._band = QtWidgets.QDoubleSpinBox()
        self._band.setRange(1.0, 20.0)
        self._band.setValue(5.0)
        self._eps = QtWidgets.QDoubleSpinBox()
        self._eps.setRange(0.05, 5.0)
        self._eps.setSingleStep(0.05)
        self._eps.setValue(0.5)
        self._downsample = QtWidgets.QComboBox()
        self._downsample.addItems(["1.0", "0.5"])
        self._dlc_lik = QtWidgets.QDoubleSpinBox()
        self._dlc_lik.setRange(0.0, 1.0)
        self._dlc_lik.setSingleStep(0.01)
        self._dlc_lik.setValue(0.95)
        self._sphere_method = QtWidgets.QComboBox()
        self._sphere_method.addItems(["Swirski 2D", "Dierkes 3D"])
        self._lock_sphere = QtWidgets.QCheckBox("Lock sphere")
        self._lock_sphere.setChecked(True)
        self._eye_z = QtWidgets.QDoubleSpinBox()
        self._eye_z.setRange(1.0, 40.0)
        self._eye_z.setValue(13.0)
        self._ransac = QtWidgets.QCheckBox("RANSAC")
        self._ransac.setChecked(True)
        self._refraction = QtWidgets.QCheckBox("Refraction maps")
        self._refraction.setEnabled(False)

        def _add(row: int, col: int, label: str, widget: QtWidgets.QWidget) -> None:
            cell = QtWidgets.QHBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(4)
            if label:
                lab = QtWidgets.QLabel(label)
                lab.setStyleSheet("font-size: 11px;")
                cell.addWidget(lab)
            cell.addWidget(widget)
            wrap = QtWidgets.QWidget()
            wrap.setLayout(cell)
            grid.addWidget(wrap, row, col)

        _add(0, 0, "name_tag", self._name_tag)
        _add(0, 1, "Metric", self._metric)
        _add(0, 2, "Band px", self._band)
        _add(0, 3, "ε px", self._eps)
        _add(1, 0, "Downsample", self._downsample)
        _add(1, 1, "DLC lik.", self._dlc_lik)
        _add(1, 2, "Sphere", self._sphere_method)
        _add(1, 3, "eye_z mm", self._eye_z)
        checks = QtWidgets.QHBoxLayout()
        checks.setContentsMargins(0, 0, 0, 0)
        checks.addWidget(self._lock_sphere)
        checks.addWidget(self._ransac)
        checks.addWidget(self._refraction)
        checks.addStretch(1)
        check_wrap = QtWidgets.QWidget()
        check_wrap.setLayout(checks)
        grid.addWidget(check_wrap, 2, 0, 1, 4)
        layout.addWidget(settings)

        self._preview_host = QtWidgets.QWidget()
        self._preview_layout = QtWidgets.QHBoxLayout(self._preview_host)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._preview_host, stretch=1)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

        self._btn_preview.clicked.connect(self._preview_this_frame)
        self._btn_visible.clicked.connect(lambda: self._run_range("visible"))
        self._btn_full.clicked.connect(lambda: self._run_range("full"))
        self._btn_save.clicked.connect(self._save_refined)
        self._btn_promote.clicked.connect(self._promote)
        self._btn_spin.clicked.connect(self._open_spin_maximizer)
        self._btn_spin_run.clicked.connect(self._run_rotation_from_yaml)
        self._btn_spin_jitter.clicked.connect(self._apply_spin_jitter)

    @staticmethod
    def _small_button(text: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        btn.setStyleSheet("QPushButton { padding: 2px 8px; min-height: 18px; max-height: 24px; }")
        return btn

    def status_signature(self, block: BlockHandle) -> list[Path]:
        tag = self._name_tag_value_safe()
        ap = block.analysis_path
        return [
            ap / f"left_eye_data_refined_{tag}.csv",
            ap / f"right_eye_data_refined_{tag}.csv",
            ap / "rotation_correction_params.yaml",
            resolve_rotation_fixed_eye_csv(ap, "left")
            or rotation_fixed_eye_csv_path(ap, "left"),
            resolve_rotation_fixed_eye_csv(ap, "right")
            or rotation_fixed_eye_csv_path(ap, "right"),
        ]

    def set_block(self, block: BlockHandle | None) -> None:
        self._clear_previews()
        self._sphere_cache = {}
        if block is None:
            self._info.setText("No block loaded.")
            self._status.setText("")
            self._refraction.setEnabled(False)
        else:
            self._info.setText(f"Active block: {block.display_label}")
            npz = refraction_npz_path(block.block_path)
            self._refraction.setEnabled(npz.is_file())
            if not npz.is_file():
                self._refraction.setChecked(False)
            self._status.setText("Load prev analysis (eye CSVs) then open videos.")
        super().set_block(block)

    def _after_load_artifacts(self, report: LoadReport) -> None:
        if self._block is None:
            return
        try:
            self._load_previews(self._block)
            self._status.setText("Loaded eye videos. Preview a frame or run refine.")
        except Exception as exc:
            self._status.setText(f"Cannot load refine preview: {exc}")

    def _clear_previews(self) -> None:
        while self._preview_layout.count():
            item = self._preview_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._left = None
        self._right = None

    def _load_previews(self, block: BlockHandle) -> None:
        blocksync = self._require_blocksync()
        self._session.ensure_eye_videos(blocksync)
        load_eye_data(blocksync)
        if not blocksync.le_videos or not blocksync.re_videos:
            raise RuntimeError("Eye videos not found — run Prepare data on Sync tab.")
        left_df = canonicalize_ellipse_table(blocksync.left_eye_data.copy())
        right_df = canonicalize_ellipse_table(blocksync.right_eye_data.copy())
        for side, df in (("left", left_df), ("right", right_df)):
            latest = latest_refined_eye_csv(block.analysis_path, side)
            if latest is not None:
                refined = read_refined_eye_table(latest)
                df = attach_refined_columns(df, refined)
                if side == "left":
                    left_df = df
                else:
                    right_df = df
            spin_path = resolve_rotation_fixed_eye_csv(block.analysis_path, side)
            if spin_path is not None:
                spin = read_rotation_fixed_eye_table(spin_path)
                df = attach_spin_columns(df, spin)
                if side == "left":
                    left_df = df
                else:
                    right_df = df
        self._clear_previews()
        self._left = RefinePreviewWidget(
            left_df, blocksync.le_videos[0], "left", parent=self
        )
        self._right = RefinePreviewWidget(
            right_df, blocksync.re_videos[0], "right", parent=self
        )
        self._preview_layout.addWidget(self._left)
        self._preview_layout.addWidget(self._right)

    def _settings(self) -> RefineSettings:
        metric_map = {
            "Region contrast": "contrast",
            "Edge distance": "edge",
            "Both": "both",
        }
        sphere_map = {"Dierkes 3D": "dierkes_3d", "Swirski 2D": "swirski_2d"}
        maps = None
        apply = False
        if self._refraction.isChecked() and self._block is not None:
            maps = try_load_refraction_maps(self._block.block_path)
            apply = maps is not None
        return RefineSettings(
            metric=metric_map[self._metric.currentText()],
            band_width=float(self._band.value()),
            epsilon=float(self._eps.value()),
            downsample=float(self._downsample.currentText()),
            dlc_likelihood=float(self._dlc_lik.value()),
            sphere_method=sphere_map[self._sphere_method.currentText()],
            lock_sphere=self._lock_sphere.isChecked(),
            eye_z=float(self._eye_z.value()),
            use_ransac=self._ransac.isChecked(),
            apply_refraction=apply,
            refraction_maps=maps,
        )

    def _inliers_for_side(self, side: str) -> dict[int, np.ndarray]:
        blocksync = self._require_blocksync()
        folder = Path(blocksync.l_e_path if side == "left" else blocksync.r_e_path)
        try:
            csv = resolve_dlc_csv(folder)
        except (FileNotFoundError, ValueError):
            return {}
        return extract_pupil_inliers(csv, likelihood=float(self._dlc_lik.value()))

    def _preview_this_frame(self) -> None:
        if self._left is None or self._right is None or self._block is None:
            return
        settings = self._settings()
        blocksync = self._require_blocksync()
        updated = []
        for side, widget, attr in (
            ("left", self._left, "left_eye_data"),
            ("right", self._right, "right_eye_data"),
        ):
            row = widget.current_row()
            image = widget.raw_frame()
            if row is None or image is None:
                continue
            camera = resolve_intrinsics(self._block.block_path, side)
            df = widget.df()
            col = "eye_frame" if "eye_frame" in df.columns else "frame"
            mask = df[col] == widget.frame_idx()
            if not mask.any():
                continue
            pos = int(np.flatnonzero(mask.to_numpy())[0])
            inliers = self._inliers_for_side(side)
            out, fit = refine_eye_table(
                df,
                camera,
                frame_image=lambda idx, w=widget: w.raw_frame(idx),
                inliers_by_frame=inliers,
                settings=settings,
                frame_indices=[pos],
            )
            widget.set_dataframe(out)
            setattr(blocksync, attr, out)
            self._sphere_cache[side] = fit.sphere
            updated.append(side)
        self._status.setText(
            "Preview refined current frame for: " + (", ".join(updated) or "none")
        )

    def _visible_indices(self, widget: RefinePreviewWidget) -> list[int]:
        df = widget.df()
        col = "eye_frame" if "eye_frame" in df.columns else "frame"
        if col not in df.columns:
            return list(range(len(df)))
        lo = max(0, widget.frame_idx() - 30)
        hi = widget.frame_idx() + 30
        mask = (df[col] >= lo) & (df[col] <= hi)
        return [int(i) for i in np.flatnonzero(mask.to_numpy())]

    def _run_range(self, mode: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._left is None or self._right is None or self._block is None:
            return
        settings = self._settings()
        block_path = self._block.block_path
        left_df = self._left.df().copy()
        right_df = self._right.df().copy()
        left_idx = None if mode == "full" else self._visible_indices(self._left)
        right_idx = None if mode == "full" else self._visible_indices(self._right)
        left_video = Path(self._require_blocksync().le_videos[0])
        right_video = Path(self._require_blocksync().re_videos[0])
        left_inliers = self._inliers_for_side("left")
        right_inliers = self._inliers_for_side("right")

        def work():
            results = {}
            for side, df, video, inliers, indices in (
                ("left", left_df, left_video, left_inliers, left_idx),
                ("right", right_df, right_video, right_inliers, right_idx),
            ):
                camera = resolve_intrinsics(block_path, side)
                reader = VideoReader(video)
                try:
                    col = "eye_frame" if "eye_frame" in df.columns else "frame"

                    def grab(idx, _reader=reader, _col=col, _df=df):
                        return _reader.read_frame(int(idx))

                    out, fit = refine_eye_table(
                        df,
                        camera,
                        frame_image=grab,
                        inliers_by_frame=inliers,
                        settings=settings,
                        frame_indices=indices,
                    )
                finally:
                    reader.close()
                results[side] = (out, fit)
            return results

        self._status.setText("Running refine…")
        worker = CallableWorker(work, self)

        def on_ok(results):
            blocksync = self._require_blocksync()
            if "left" in results and self._left is not None:
                out, fit = results["left"]
                self._left.set_dataframe(out)
                blocksync.left_eye_data = out
                self._sphere_cache["left"] = fit.sphere
            if "right" in results and self._right is not None:
                out, fit = results["right"]
                self._right.set_dataframe(out)
                blocksync.right_eye_data = out
                self._sphere_cache["right"] = fit.sphere
            self._status.setText(f"Refine {mode} finished.")
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str):
            self._status.setText(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Refine tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _save_refined(self) -> None:
        if self._block is None or self._left is None or self._right is None:
            return
        tag = self._name_tag_value_safe()
        left_path = refined_eye_csv_path(self._block.block_path, "left", tag)
        right_path = refined_eye_csv_path(self._block.block_path, "right", tag)
        write_refined_eye_table(self._left.df(), left_path)
        write_refined_eye_table(self._right.df(), right_path)
        self._status.setText(f"Wrote {left_path.name} and {right_path.name}")
        self._refresh_artifact_ui()

    def _promote(self) -> None:
        if self._block is None:
            return
        tag = self._name_tag_value_safe()
        try:
            backups = []
            for side in ("left", "right"):
                path = refined_eye_csv_path(self._block.block_path, side, tag)
                if not path.is_file():
                    latest = latest_refined_eye_csv(self._block.block_path, side)
                    if latest is None:
                        raise FileNotFoundError(f"No refined CSV for {side}")
                    path = latest
                backups.append(
                    promote_refined_to_eye_data(self._block.block_path, side, path)
                )
            self._status.setText(
                "Promoted refined geometry into left/right_eye_data.csv. "
                f"Backups: {', '.join(p.name for p in backups)}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Promote", str(exc))
            self._status.setText(f"Promote failed: {exc}")

    def _raw_dlc_table(self, side: str) -> pd.DataFrame:
        """Ellipse table as it exits ``read_dlc_data`` (raw xy, not jitter-corrected)."""
        if self._block is None:
            raise RuntimeError("No block loaded")
        blocksync = self._require_blocksync()
        attr = "le_df" if side == "left" else "re_df"
        df = getattr(blocksync, attr, None)
        if df is None or getattr(df, "empty", True) or "center_x" not in getattr(df, "columns", []):
            name = "le_df.csv" if side == "left" else "re_df.csv"
            path = self._block.analysis_path / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"{name} is missing. Run Read DLC so spin-max can use raw ellipses."
                )
            df = load_eye_tracking_df_csv(path)
            setattr(blocksync, attr, df)
        return raw_ellipse_table_from_dlc_df(df, side)

    def _open_spin_maximizer(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Ellipse rotation angle correction",
                "A correction run is already in progress. Wait for it to finish.",
            )
            return
        if self._left is None or self._right is None or self._block is None:
            return
        load_errors = []
        try:
            left_df = self._raw_dlc_table("left")
        except Exception as exc:
            left_df = pd.DataFrame()
            load_errors.append(f"left: {exc}")
        try:
            right_df = self._raw_dlc_table("right")
        except Exception as exc:
            right_df = pd.DataFrame()
            load_errors.append(f"right: {exc}")
        left_img = self._left.raw_frame()
        right_img = self._right.raw_frame()
        if left_img is None and right_img is None:
            QtWidgets.QMessageBox.information(
                self,
                "Ellipse rotation angle correction",
                "Load videos and seek to a frame with a raw DLC ellipse.",
            )
            return
        if left_df.empty and right_df.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "Ellipse rotation angle correction",
                "Could not load raw DLC ellipses.\n" + "\n".join(load_errors),
            )
            return
        left_w = float(left_img.shape[1]) if left_img is not None else 0.0
        right_w = float(right_img.shape[1]) if right_img is not None else 0.0
        existing = try_read_rotation_params(self._block.analysis_path)
        dialog = SpinMaximizerDialog(
            left_df=left_df,
            right_df=right_df,
            left_frame_fn=self._left.raw_frame,
            right_frame_fn=self._right.raw_frame,
            left_nframes=self._left.nframes(),
            right_nframes=self._right.nframes(),
            left_frame0=self._left.frame_idx(),
            right_frame0=self._right.frame_idx(),
            left_width=left_w,
            right_width=right_w,
            parent=self,
            initial_left=None if existing is None else existing.left,
            initial_right=None if existing is None else existing.right,
            initial_apply_jitter=True if existing is None else existing.apply_jitter,
            block_label=self._block.display_label,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        path = write_rotation_params(
            self._block.analysis_path, dialog.correction_params()
        )
        self._status.setText(f"Saved rotation-correction parameters: {path}")
        QtWidgets.QMessageBox.information(
            self,
            "Ellipse rotation angle correction",
            f"Wrote {path}\n\n"
            "This does not run the recording. Press Run rotation correction "
            "to apply these parameters.",
        )
        self._refresh_artifact_ui()

    def _jitter_eye(self, side: str):
        if self._block is None:
            return None
        blocksync = self._require_blocksync()
        attr = "le_jitter_dict" if side == "left" else "re_jitter_dict"
        data = getattr(blocksync, attr, None)
        if data:
            return data
        pkl = self._block.analysis_path / "jitter_report_dict.pkl"
        if not pkl.is_file():
            return None
        import pickle

        with open(pkl, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            return None
        key = "left_eye" if side == "left" else "right_eye"
        return payload.get(key)

    def _run_rotation_from_yaml(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._left is None or self._right is None or self._block is None:
            return
        if try_read_rotation_params(self._block.analysis_path) is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Run rotation correction",
                "No rotation_correction_params.yaml for this block. "
                "Open Correct ellipse rotation… and save parameters first.",
            )
            return
        block_path = Path(self._block.block_path)
        animal = getattr(self._block, "animal_call", None)

        def work():
            return run_block(
                block_path,
                overwrite=True,
                animal=animal,
                progress=worker.report,
                show_tqdm=True,
            )

        self._status.setText(
            "Running ellipse rotation angle correction from YAML…"
        )
        self._set_rotation_busy(True)
        worker = CallableWorker(work, self)

        def on_ok(result):
            body = f"{result.status}: {result.message}"
            if result.status in ("ok", "ok_jitter_missing"):
                for side, widget in (("left", self._left), ("right", self._right)):
                    spin_path = resolve_rotation_fixed_eye_csv(
                        self._block.analysis_path, side
                    )
                    if widget is None or spin_path is None:
                        continue
                    spin = read_rotation_fixed_eye_table(spin_path)
                    widget.set_dataframe(attach_spin_columns(widget.df(), spin))
                QtWidgets.QMessageBox.information(
                    self, "Ellipse rotation angle correction", body
                )
            elif result.status == "skipped_exists":
                QtWidgets.QMessageBox.information(
                    self, "Ellipse rotation angle correction", body
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Ellipse rotation angle correction", body
                )
            self._status.setText(body.replace("\n", " | "))
            self._refresh_artifact_ui()
            self._set_rotation_busy(False)
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str):
            self._status.setText(f"Ellipse rotation angle correction error: {msg}")
            QtWidgets.QMessageBox.warning(
                self, "Ellipse rotation angle correction", msg
            )
            self._set_rotation_busy(False)
            self._worker = None
            worker.deleteLater()

        worker.progress.connect(self._status.setText)
        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _set_rotation_busy(self, busy: bool) -> None:
        for btn in (self._btn_spin, self._btn_spin_run, self._btn_spin_jitter):
            btn.setEnabled(not busy)

    def _apply_spin_jitter(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Apply jitter",
                "Ellipse rotation angle correction is still running. Wait for it to finish.",
            )
            return
        if self._left is None or self._right is None or self._block is None:
            return
        left_jitter = self._jitter_eye("left")
        right_jitter = self._jitter_eye("right")
        if left_jitter is None and right_jitter is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Apply jitter",
                "jitter_report_dict.pkl is missing. Run jitter on the Sync tab first.",
            )
            return
        lines = []
        wrote_any = False
        for side, widget, jitter in (
            ("left", self._left, left_jitter),
            ("right", self._right, right_jitter),
        ):
            src = resolve_rotation_fixed_eye_csv(self._block.analysis_path, side)
            dest = rotation_fixed_eye_csv_path(
                self._block.analysis_path, side, jitter_corrected=True
            )
            if src is None:
                lines.append(
                    f"{side.capitalize()}: skipped — no rotation-fixed CSV. "
                    "Run ellipse rotation angle correction first."
                )
                continue
            if jitter is None:
                lines.append(
                    f"{side.capitalize()}: skipped — no jitter report for this eye."
                )
                continue
            try:
                spun = read_rotation_fixed_eye_table(src)
                spun = apply_jitter_to_spin_table(spun, jitter)
                write_rotation_fixed_eye_table(spun, dest)
                widget.set_dataframe(attach_spin_columns(widget.df(), spun))
                wrote_any = True
                lines.append(
                    f"{side.capitalize()}: jitter applied\n  wrote {dest.resolve()}"
                )
            except Exception as exc:
                lines.append(f"{side.capitalize()}: FAILED — {exc}")
        body = "\n".join(lines)
        self._status.setText(body.replace("\n", " | "))
        if wrote_any:
            QtWidgets.QMessageBox.information(self, "Apply jitter", body)
        else:
            QtWidgets.QMessageBox.warning(self, "Apply jitter", body)
        self._refresh_artifact_ui()
