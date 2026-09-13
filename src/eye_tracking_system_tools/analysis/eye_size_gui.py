"""
Qt UI: mark eye ROIs on representative lizard / mouse / turtle blocks.

Thin wrapper around
:class:`~eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker.QtRoiPickerDialog`
(same picker as the preprocessing Calibration tab).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.analysis.eye_size_on_sensor import (
    CAMERA_SPECS_TEXT,
    EyeRoiMeasurement,
    SpeciesBlock,
    SpeciesEyeSize,
    average_species_eye_sizes,
    default_species_blocks,
    format_measurements_report,
    measure_eye_roi,
    pick_large_pupil_frame_index,
)
from eye_tracking_system_tools.analysis.eye_trace_io import resolve_eye_csv
from eye_tracking_system_tools.analysis.pixel_calibration import find_eye_videos, grab_frame
from eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker import (
    QtRoiPickerDialog,
)

_EYE_INSTRUCTION = (
    "Drag a rectangle that tightly encloses the <b>eye</b> (globe / visible "
    "ocular surface) on this frame, then click Use ROI."
)


def _ensure_qapplication() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _RoiPickerWithReroll(QtRoiPickerDialog):
    """ROI picker with an extra Re-roll button to get a different frame."""

    def __init__(self, frame_bgr, *, title, instruction, parent=None):
        super().__init__(frame_bgr, title=title, instruction=instruction, parent=parent)
        self._result: str = "cancel"
        # Insert a Re-roll button next to Use ROI / Cancel.
        self._btn_reroll = QtWidgets.QPushButton("Re-roll frame")
        self._btn_reroll.setToolTip("Pick a different random frame with a large pupil")
        # Find the button row layout (last QHBoxLayout in the dialog).
        for i in range(self.layout().count() - 1, -1, -1):
            item = self.layout().itemAt(i)
            if item and item.layout() is not None:
                item.layout().insertWidget(0, self._btn_reroll)
                break
        self._btn_reroll.clicked.connect(self._on_reroll)

    def _on_reroll(self) -> None:
        self._result = "reroll"
        self.reject()

    def _on_accept(self) -> None:
        if self._roi is None:
            QtWidgets.QMessageBox.warning(self, "Select ROI", "Drag a rectangle first.")
            return
        self._result = "accept"
        self.accept()

    def run(self) -> str:
        """Execute the dialog; returns ``'accept'``, ``'cancel'``, or ``'reroll'``."""
        self.exec()
        return self._result


class _SpeciesRow(QtWidgets.QWidget):
    """One species: path status + Mark L / Mark R."""

    roi_changed = QtCore.pyqtSignal()

    def __init__(self, block: SpeciesBlock, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.block = block
        self.rois: dict[str, EyeRoiMeasurement] = {}

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel(
            f"<b>{block.species}</b>  {block.animal} / {block.block_path.name}"
        )
        title.setMinimumWidth(280)
        root.addWidget(title)

        self._path_lbl = QtWidgets.QLabel()
        self._path_lbl.setWordWrap(True)
        root.addWidget(self._path_lbl, stretch=1)

        self._btn_l = QtWidgets.QPushButton("Mark left")
        self._btn_r = QtWidgets.QPushButton("Mark right")
        self._btn_clear = QtWidgets.QPushButton("Clear")
        root.addWidget(self._btn_l)
        root.addWidget(self._btn_r)
        root.addWidget(self._btn_clear)

        self._btn_l.clicked.connect(lambda: self._pick("left"))
        self._btn_r.clicked.connect(lambda: self._pick("right"))
        self._btn_clear.clicked.connect(self.clear_rois)

        self._refresh_path_status()

    def _refresh_path_status(self) -> None:
        bp = self.block.block_path
        if not bp.is_dir():
            self._path_lbl.setText(
                f"<span style='color:#a11'>Missing block:</span> {bp}"
            )
            self._btn_l.setEnabled(False)
            self._btn_r.setEnabled(False)
            return
        videos = find_eye_videos(bp)
        missing = [s for s, p in videos.items() if p is None]
        if missing:
            self._path_lbl.setText(
                f"<span style='color:#a11'>No eye video for: {', '.join(missing)}</span>"
            )
        else:
            self._path_lbl.setText(f"<span style='color:#177245'>OK</span>  {bp}")
        self._btn_l.setEnabled(videos.get("left") is not None)
        self._btn_r.setEnabled(videos.get("right") is not None)
        self._update_button_labels()

    def _update_button_labels(self) -> None:
        for side, btn in (("left", self._btn_l), ("right", self._btn_r)):
            m = self.rois.get(side)
            if m is None:
                btn.setText(f"Mark {side}")
            else:
                btn.setText(f"{side[0].upper()}: {m.width_px}×{m.height_px}")

    def clear_rois(self) -> None:
        self.rois.clear()
        self._update_button_labels()
        self.roi_changed.emit()

    def _pick(self, side: str) -> None:
        videos = find_eye_videos(self.block.block_path)
        video = videos.get(side)
        if video is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Video missing",
                f"No {side} eye mp4 under {self.block.block_path / 'eye_videos'}",
            )
            return

        # Resolve the eye CSV for pupil-diameter filtering.
        analysis = self.block.block_path / "analysis"
        try:
            csv_choice = resolve_eye_csv(analysis, side)
        except FileNotFoundError as exc:
            QtWidgets.QMessageBox.critical(self, "Eye CSV missing", str(exc))
            return

        while True:
            try:
                frame_idx = pick_large_pupil_frame_index(csv_choice.path)
                frame = grab_frame(video, frame_index=frame_idx)
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(
                    self, "Cannot load frame", str(exc),
                )
                return

            dlg = _RoiPickerWithReroll(
                frame,
                title=(
                    f"Eye ROI — {self.block.species} {side} "
                    f"({self.block.animal}) [frame {frame_idx}]"
                ),
                instruction=_EYE_INSTRUCTION,
                parent=self,
            )
            result = dlg.run()
            if result == "reroll":
                continue
            if result == "cancel":
                return
            # result == "accept"
            roi = dlg.selected_roi()
            if roi is None:
                return
            h, w = frame.shape[:2]
            self.rois[side] = measure_eye_roi(
                roi,
                species=self.block.species,
                animal=self.block.animal,
                side=side,
                block_path=self.block.block_path,
                video_path=video,
                frame_shape_hw=(h, w),
            )
            self._update_button_labels()
            self.roi_changed.emit()
            return


_DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[3] / "outputs"


class EyeSizeOnSensorWindow(QtWidgets.QMainWindow):
    """Main window: mark ROIs → Calculate → report eye size / frame %."""

    def __init__(
        self,
        blocks: list[SpeciesBlock] | None = None,
        out_root: Path | str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Eye size on sensor (review reply)")
        self.resize(980, 640)
        self._blocks = list(blocks) if blocks is not None else default_species_blocks()
        self._out_root = Path(out_root) if out_root is not None else _DEFAULT_OUT_ROOT
        self._rows: list[_SpeciesRow] = []
        self._last_averages: list[SpeciesEyeSize] = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        intro = QtWidgets.QLabel(
            "Mark <b>both</b> eyes on one representative block per species "
            "(lizard PV_126 / mouse M_002 / turtle T_18), then press "
            "<b>Calculate</b>. Report is the L/R average: "
            "<b>area</b> in px² and <b>diagonal</b> as % of the 640×480 frame diagonal."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(intro)

        specs = QtWidgets.QPlainTextEdit(CAMERA_SPECS_TEXT)
        specs.setReadOnly(True)
        specs.setMaximumHeight(160)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        specs.setFont(font)
        layout.addWidget(specs)

        for block in self._blocks:
            row = _SpeciesRow(block)
            row.roi_changed.connect(self._on_roi_changed)
            self._rows.append(row)
            layout.addWidget(row)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_calc = QtWidgets.QPushButton("Calculate")
        self._btn_calc.setDefault(True)
        self._btn_copy = QtWidgets.QPushButton("Copy report")
        self._btn_save = QtWidgets.QPushButton("Save CSV…")
        btn_row.addWidget(self._btn_calc)
        btn_row.addWidget(self._btn_copy)
        btn_row.addWidget(self._btn_save)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._report = QtWidgets.QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setFont(font)
        self._report.setPlaceholderText(
            "Mark left and right for at least one animal, then press Calculate."
        )
        layout.addWidget(self._report, stretch=1)

        self._btn_calc.clicked.connect(self.calculate)
        self._btn_copy.clicked.connect(self._copy_report)
        self._btn_save.clicked.connect(self._save_csv)
        self._btn_calc.setEnabled(False)
        self._btn_copy.setEnabled(False)
        self._btn_save.setEnabled(False)

    def _all_measurements(self) -> list[EyeRoiMeasurement]:
        out: list[EyeRoiMeasurement] = []
        for row in self._rows:
            out.extend(row.rois.values())
        return out

    def _complete_pair_count(self) -> int:
        return sum(1 for row in self._rows if set(row.rois) >= {"left", "right"})

    def _on_roi_changed(self) -> None:
        self._btn_calc.setEnabled(self._complete_pair_count() > 0)

    def calculate(self) -> None:
        measurements = self._all_measurements()
        self._last_averages = average_species_eye_sizes(
            measurements, require_both_eyes=True
        )
        if not self._last_averages:
            QtWidgets.QMessageBox.warning(
                self,
                "Need both eyes",
                "Mark left and right ROIs for at least one animal before calculating.",
            )
            self._report.setPlainText(
                format_measurements_report(measurements, require_both_eyes=True)
            )
            self._btn_copy.setEnabled(False)
            self._btn_save.setEnabled(False)
            return
        text = format_measurements_report(measurements, require_both_eyes=True)
        self._report.setPlainText(text)
        self._btn_copy.setEnabled(True)
        self._btn_save.setEnabled(True)

        # Prompt for tag and save to outputs/eye_size_on_sensor_<tag>/
        self._auto_save_to_outputs(text)

    def _auto_save_to_outputs(self, report_text: str) -> None:
        tag, ok = QtWidgets.QInputDialog.getText(
            self,
            "Output tag",
            "Tag for the output folder (eye_size_on_sensor_<tag>):",
            QtWidgets.QLineEdit.EchoMode.Normal,
            "latest",
        )
        if not ok or not tag.strip():
            return
        tag = tag.strip().replace(" ", "_")
        out_dir = self._out_root / f"eye_size_on_sensor_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)

        report_path = out_dir / "eye_size_on_sensor_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        csv_path = out_dir / "eye_size_on_sensor.csv"
        if self._last_averages:
            fieldnames = list(self._last_averages[0].to_dict().keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self._last_averages:
                    writer.writerow(row.to_dict())

        self.statusBar().showMessage(f"Saved to {out_dir}", 8000)

    def _copy_report(self) -> None:
        text = self._report.toPlainText().strip()
        if not text:
            return
        QtWidgets.QApplication.clipboard().setText(text)

    def _save_csv(self) -> None:
        if not self._last_averages:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save eye-size measurements",
            "eye_size_on_sensor.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        fieldnames = list(self._last_averages[0].to_dict().keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._last_averages:
                writer.writerow(row.to_dict())
        QtWidgets.QMessageBox.information(self, "Saved", f"Wrote {path}")


def launch_eye_size_gui(
    blocks: list[SpeciesBlock] | None = None,
) -> EyeSizeOnSensorWindow:
    """Open the eye-size window (blocks until closed outside IPython)."""
    app = _ensure_qapplication()
    win = EyeSizeOnSensorWindow(blocks=blocks)
    win.show()
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        in_ipython = get_ipython() is not None
    except ImportError:
        in_ipython = False
    if not in_ipython:
        app.exec()
    return win


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for future CLI flags
    launch_eye_size_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
