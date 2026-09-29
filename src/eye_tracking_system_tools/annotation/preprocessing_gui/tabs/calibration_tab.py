"""Pixel-size calibration tab (LR_pix_size.csv) — separate from Saccades."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.analysis.pixel_calibration import (
    DEFAULT_KNOWN_DIST_MM,
    find_eye_videos,
    grab_frame,
    has_pixel_calibration,
    manual_calibration,
    pix_size_from_roi,
    read_pixel_size,
    write_pixel_size,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    CALIBRATION_ARTIFACT_PROFILE,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker import (
    QtRoiPickerDialog,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab


_LANDMARK_INSTRUCTION = (
    "Drag a rectangle whose <b>diagonal</b> spans the known landmark distance "
    "on this eye frame, then click Use ROI."
)


class CalibrationTab(BaseTab):
    tab_id = "calibration"
    tab_label = "Calibration"

    def artifact_profile(self):
        return CALIBRATION_ARTIFACT_PROFILE

    def status_signature(self, block: BlockHandle) -> list[Path]:
        return [block.analysis_path / "LR_pix_size.csv"]

    def build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self._build_artifact_panel())

        self._info = QtWidgets.QLabel(
            "Pixel size calibration writes <code>analysis/LR_pix_size.csv</code> "
            "(mm/px). Used by pupil / jitter figures — not by angular saccade detection."
        )
        self._info.setWordWrap(True)
        self._info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        root.addWidget(self._info)

        self._status = QtWidgets.QLabel("No calibration loaded.")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        form = QtWidgets.QFormLayout()
        self._known_mm = QtWidgets.QDoubleSpinBox()
        self._known_mm.setRange(0.1, 100.0)
        self._known_mm.setDecimals(2)
        self._known_mm.setValue(DEFAULT_KNOWN_DIST_MM)
        self._known_mm.setSuffix(" mm")
        self._known_mm.setToolTip("Real-world length spanned by the ROI diagonal")
        form.addRow("Known landmark distance:", self._known_mm)

        self._manual_l = QtWidgets.QDoubleSpinBox()
        self._manual_r = QtWidgets.QDoubleSpinBox()
        for spin in (self._manual_l, self._manual_r):
            spin.setRange(0.01, 1e6)
            spin.setDecimals(3)
            spin.setValue(100.0)
        self._manual_l.setToolTip("Measured landmark length in left-eye pixels")
        self._manual_r.setToolTip("Measured landmark length in right-eye pixels")
        form.addRow("Manual L landmark length (px):", self._manual_l)
        form.addRow("Manual R landmark length (px):", self._manual_r)
        root.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        self._btn_roi = QtWidgets.QPushButton("Launch landmark ROI tool")
        self._btn_roi.setToolTip(
            "Qt dialog per eye: drag a box whose diagonal spans the known distance "
            "(avoids OpenCV HighGUI conflicts with PyQt)."
        )
        self._btn_manual = QtWidgets.QPushButton("Save manual calibration")
        self._btn_refresh = QtWidgets.QPushButton("Refresh")
        row.addWidget(self._btn_roi)
        row.addWidget(self._btn_manual)
        row.addWidget(self._btn_refresh)
        root.addLayout(row)

        self._msg = QtWidgets.QLabel("")
        self._msg.setWordWrap(True)
        root.addWidget(self._msg)
        root.addStretch(1)

        self._btn_roi.clicked.connect(self._run_qt_roi_calibration)
        self._btn_manual.clicked.connect(self._run_manual_calibration)
        self._btn_refresh.clicked.connect(self._refresh_status)

    def set_block(self, block: BlockHandle | None) -> None:
        if block is None:
            self._status.setText("No block loaded.")
            self._msg.setText("")
        else:
            self._refresh_status()
            self._msg.setText("")
        super().set_block(block)

    def _after_load_artifacts(self, report) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._block is None:
            self._status.setText("No block loaded.")
            return
        if not has_pixel_calibration(self._block.block_path):
            self._status.setText(
                "No LR_pix_size.csv yet — launch the landmark tool or enter manual px lengths."
            )
            return
        try:
            ps = read_pixel_size(self._block.block_path)
            assert ps is not None
            self._status.setText(
                f"L={ps.l_mm_per_px:.6g} mm/px ({ps.l_um_per_px:.2f} µm/px)  |  "
                f"R={ps.r_mm_per_px:.6g} mm/px ({ps.r_um_per_px:.2f} µm/px)\n"
                f"({ps.source})"
            )
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Calibration file present but unreadable: {exc}")

    def _run_qt_roi_calibration(self) -> None:
        """Main-thread Qt ROI picker (OpenCV selectROI crashes under PyQt)."""
        if self._block is None:
            return
        if has_pixel_calibration(self._block.block_path):
            ans = QtWidgets.QMessageBox.question(
                self,
                "Overwrite calibration?",
                "LR_pix_size.csv already exists. Overwrite?",
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        videos = find_eye_videos(self._block.block_path)
        missing = [side for side, p in videos.items() if p is None]
        if missing:
            QtWidgets.QMessageBox.critical(
                self,
                "Videos missing",
                f"No raw mp4 under eye_videos for: {', '.join(missing)}.\n"
                "Use manual calibration instead.",
            )
            return

        known = float(self._known_mm.value())
        sizes: dict[str, float] = {}
        try:
            for side in ("right", "left"):
                frame = grab_frame(videos[side], frame_index=1)
                dlg = QtRoiPickerDialog(
                    frame,
                    title=f"Landmark ROI — {side} eye ({known:g} mm diagonal)",
                    instruction=_LANDMARK_INSTRUCTION,
                    parent=self,
                )
                if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    self._msg.setText("Calibration cancelled.")
                    return
                roi = dlg.selected_roi()
                if roi is None:
                    self._msg.setText("Calibration cancelled (no ROI).")
                    return
                sizes[side] = pix_size_from_roi(known, roi)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Calibration failed", str(exc))
            return

        write_pixel_size(
            self._block.block_path,
            sizes["left"],
            sizes["right"],
            known_dist_mm=known,
            method="roi_diagonal_qt",
            extra_meta={
                "left_video": str(videos["left"]),
                "right_video": str(videos["right"]),
                "frame_index": 1,
            },
        )
        self._refresh_status()
        self._msg.setText("Pixel calibration saved (Qt ROI).")
        self._notify_status()

    def _run_manual_calibration(self) -> None:
        if self._block is None:
            return
        try:
            ps = manual_calibration(
                self._block.block_path,
                left_px=float(self._manual_l.value()),
                right_px=float(self._manual_r.value()),
                known_dist_mm=float(self._known_mm.value()),
            )
            self._refresh_status()
            self._msg.setText(
                f"Saved manual calibration: L={ps.l_mm_per_px:.6g} R={ps.r_mm_per_px:.6g} mm/px"
            )
            self._notify_status()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Calibration failed", str(exc))

    def _notify_status(self) -> None:
        parent = self.window()
        bus = getattr(parent, "_status_bus", None)
        if bus is not None:
            bus.refresh_all()
