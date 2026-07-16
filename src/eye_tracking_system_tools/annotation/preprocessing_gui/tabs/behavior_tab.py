"""Stage 4 -- Accelerometer-based behavior annotation."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    BEHAVIOR_ARTIFACT_PROFILE,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.pyqtgraph_helpers import (
    BehaviorThresholdPlot,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.workers import CallableWorker
from eye_tracking_system_tools.preprocessing.accel_calibration import (
    load_accel_calibration,
    list_headstages,
)
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.lizard_movement import (
    compute_and_save_lizard_movement,
    oe_rec_has_accel_channels,
)
from eye_tracking_system_tools.preprocessing.notebook_helpers import (
    create_behavior_df,
    rolling_window_analysis,
)


def resolve_liz_mov_mat_path_for_block(block: BlockHandle) -> Path | None:
    """Return ``lizMov.mat`` path without constructing BlockSync."""
    oe_files = block.block_path / "oe_files"
    if not oe_files.is_dir():
        return None
    for mat in oe_files.rglob("lizMov.mat"):
        if "analysis" in mat.parts:
            return mat
    return None


def resolve_liz_mov_mat_path(blocksync: BlockSync) -> Path | None:
    """Return ``lizMov.mat`` path using the same layout as ``block_get_lizard_movement``."""
    p = blocksync.oe_path / "analysis"
    if not p.is_dir():
        return None
    try:
        analysis_list = os.listdir(p)
    except OSError:
        return None
    matches = [name for name in analysis_list if blocksync.animal_call in name]
    if not matches:
        return None
    mat_path = p / matches[0] / "lizMov.mat"
    return mat_path if mat_path.is_file() else None


def has_liz_mov(block: BlockHandle) -> bool:
    return resolve_liz_mov_mat_path_for_block(block) is not None


class BehaviorTab(BaseTab):
    tab_id = "behavior"
    tab_label = "Behavior"

    def __init__(self, state, config, parent=None):
        self._rolling_df: pd.DataFrame | None = None
        self._worker: CallableWorker | None = None
        super().__init__(state, config, parent)

    def artifact_profile(self):
        return BEHAVIOR_ARTIFACT_PROFILE

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_artifact_panel())

        self._lizmov_banner = QtWidgets.QLabel(
            "<b>lizMov.mat not found.</b> Select accelerometer calibration below, "
            "then use <b>Compute lizMov (Python)</b> (or run MATLAB <code>getLizMovement</code>)."
        )
        self._lizmov_banner.setWordWrap(True)
        self._lizmov_banner.setStyleSheet(
            "background-color: #fff3cd; color: #664d03; padding: 8px; border-radius: 4px;"
        )
        self._lizmov_banner.hide()
        layout.addWidget(self._lizmov_banner)

        calib_box = QtWidgets.QGroupBox("Accelerometer calibration")
        calib_lay = QtWidgets.QFormLayout(calib_box)
        calib_row = QtWidgets.QHBoxLayout()
        self._calib_path_edit = QtWidgets.QLineEdit(
            self._config.accel_calibration_mat_path or ""
        )
        self._calib_path_edit.setPlaceholderText("calibration_results.mat")
        self._btn_calib_browse = QtWidgets.QPushButton("Browse…")
        calib_row.addWidget(self._calib_path_edit, stretch=1)
        calib_row.addWidget(self._btn_calib_browse)
        calib_lay.addRow("Calibration file:", calib_row)
        self._headstage_combo = QtWidgets.QComboBox()
        self._headstage_combo.setEditable(False)
        calib_lay.addRow("Headstage:", self._headstage_combo)
        layout.addWidget(calib_box)

        self._info = QtWidgets.QLabel("Load a block with accelerometer movement data.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        params = QtWidgets.QFormLayout()
        self._window_size_ms = QtWidgets.QSpinBox()
        self._window_size_ms.setRange(100, 600_000)
        self._window_size_ms.setValue(int(self._config.behavior_window_size_ms))
        self._step_size_ms = QtWidgets.QSpinBox()
        self._step_size_ms.setRange(10, 60_000)
        self._step_size_ms.setValue(int(self._config.behavior_step_size_ms))
        params.addRow("Window size (ms):", self._window_size_ms)
        params.addRow("Step size (ms):", self._step_size_ms)
        layout.addLayout(params)

        row = QtWidgets.QHBoxLayout()
        self._btn_load = QtWidgets.QPushButton("Load lizMov.mat")
        self._btn_compute = QtWidgets.QPushButton("Compute lizMov (Python)")
        self._btn_rolling = QtWidgets.QPushButton("Compute rolling average")
        self._btn_export = QtWidgets.QPushButton("Export behavior state")
        self._btn_export.setEnabled(False)
        row.addWidget(self._btn_load)
        row.addWidget(self._btn_compute)
        row.addWidget(self._btn_rolling)
        row.addWidget(self._btn_export)
        layout.addLayout(row)

        thresh_row = QtWidgets.QHBoxLayout()
        thresh_row.addWidget(QtWidgets.QLabel("Threshold:"))
        self._threshold = QtWidgets.QDoubleSpinBox()
        self._threshold.setRange(0.0, 1e6)
        self._threshold.setDecimals(4)
        self._threshold.setSingleStep(0.01)
        self._threshold.setValue(float(self._config.behavior_threshold))
        thresh_row.addWidget(self._threshold)
        thresh_row.addStretch(1)
        layout.addLayout(thresh_row)

        self._movement_plot = BehaviorThresholdPlot()
        self._movement_plot.setMinimumHeight(280)
        layout.addWidget(self._movement_plot, stretch=1)

        self._preview = QtWidgets.QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setMaximumHeight(120)
        self._preview.setPlaceholderText("Exported behavior segments preview…")
        layout.addWidget(self._preview)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

        self._btn_load.clicked.connect(self._run_load_lizmov)
        self._btn_compute.clicked.connect(self._run_compute_lizmov)
        self._btn_rolling.clicked.connect(self._run_rolling_average)
        self._btn_export.clicked.connect(self._run_export)
        self._btn_calib_browse.clicked.connect(self._browse_calibration_file)
        self._calib_path_edit.editingFinished.connect(self._on_calibration_path_changed)
        self._headstage_combo.currentTextChanged.connect(self._on_headstage_changed)
        self._threshold.valueChanged.connect(self._on_threshold_spin)
        self._movement_plot.threshold_changed.connect(self._on_threshold_plot)
        self._refresh_headstage_combo()

    def _calibration_path(self) -> Path | None:
        text = self._calib_path_edit.text().strip()
        if not text:
            return None
        p = Path(text)
        return p if p.is_file() else None

    def _persist_calibration_config(self) -> None:
        path = self._calib_path_edit.text().strip()
        self._config.accel_calibration_mat_path = path or None
        hs = self._headstage_combo.currentText().strip()
        self._config.accel_calibration_headstage = hs or None

    def _browse_calibration_file(self) -> None:
        start = self._calib_path_edit.text().strip() or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Accelerometer calibration file",
            start,
            "MAT files (*.mat);;All files (*)",
        )
        if path:
            self._calib_path_edit.setText(path)
            self._on_calibration_path_changed()

    def _on_calibration_path_changed(self) -> None:
        self._persist_calibration_config()
        self._refresh_headstage_combo()

    def _on_headstage_changed(self, _text: str) -> None:
        self._persist_calibration_config()

    def _refresh_headstage_combo(self) -> None:
        saved = self._config.accel_calibration_headstage
        self._headstage_combo.blockSignals(True)
        self._headstage_combo.clear()
        path = self._calibration_path()
        if path is not None:
            try:
                for hs in list_headstages(path):
                    self._headstage_combo.addItem(hs)
            except (OSError, KeyError) as e:
                self._status.setText(f"Cannot read calibration file: {e}")
        if saved and self._headstage_combo.findText(saved) >= 0:
            self._headstage_combo.setCurrentText(saved)
        elif self._headstage_combo.count() > 0:
            self._headstage_combo.setCurrentIndex(0)
        self._headstage_combo.blockSignals(False)
        self._persist_calibration_config()

    def _load_calibration_or_warn(self):
        path = self._calibration_path()
        if path is None:
            raise FileNotFoundError(
                "Select a valid accelerometer calibration file (calibration_results.mat)."
            )
        hs = self._headstage_combo.currentText().strip()
        if not hs:
            raise ValueError("Select a headstage from the calibration file.")
        return load_accel_calibration(path, hs)

    def _set_compute_busy(self, busy: bool) -> None:
        for btn in (
            self._btn_compute,
            self._btn_load,
            self._btn_rolling,
            self._btn_export,
            self._btn_calib_browse,
        ):
            btn.setEnabled(not busy)
        self._calib_path_edit.setEnabled(not busy)
        self._headstage_combo.setEnabled(not busy)
        if not busy and self._block is not None:
            present = has_liz_mov(self._block)
            self._set_workflow_enabled(present)

    def _run_compute_lizmov(self) -> None:
        if self._block is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            calibration = self._load_calibration_or_warn()
            blocksync = self._require_blocksync()
            if not oe_rec_has_accel_channels(blocksync.oe_rec):
                raise ValueError(
                    "No AUX accelerometer channels found in this recording "
                    "(need 3 AUX .continuous files or 3 analog channels)."
                )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Behavior tab", str(e))
            return

        def work():
            bs = self._require_blocksync()
            compute_and_save_lizard_movement(
                bs,
                overwrite=True,
                calibration=calibration,
            )

        self._set_compute_busy(True)
        self._status.setText("Computing lizMov from OE accelerometer data…")
        worker = CallableWorker(work, self)

        def on_ok(_=None):
            self._set_compute_busy(False)
            self._lizmov_banner.hide()
            self._set_workflow_enabled(True)
            self.setToolTip("")
            try:
                self._run_load_lizmov(silent=True)
                self._status.setText("Computed and loaded lizMov.mat (Python pipeline).")
            except Exception as exc:
                self._status.setText(f"Computed lizMov but load failed: {exc}")
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str):
            self._set_compute_busy(False)
            self._status.setText(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Behavior tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def status_signature(self, block: BlockHandle) -> list[Path]:
        return [block.analysis_path / f"block_{block.block_num}_behavior_state.csv"]

    def _workflow_widgets(self) -> list[QtWidgets.QWidget]:
        return [
            self._window_size_ms,
            self._step_size_ms,
            self._btn_load,
            self._btn_rolling,
            self._btn_export,
            self._threshold,
            self._movement_plot,
            self._preview,
        ]

    def _set_workflow_enabled(self, enabled: bool) -> None:
        for widget in self._workflow_widgets():
            widget.setEnabled(enabled)
        if not enabled:
            self._btn_export.setEnabled(False)

    def set_block(self, block: BlockHandle | None) -> None:
        self._rolling_df = None
        self._preview.clear()
        self._btn_export.setEnabled(False)

        if block is None:
            self._info.setText("No block loaded.")
            self._lizmov_banner.hide()
            self._set_workflow_enabled(False)
            self.setEnabled(True)
            self.setToolTip("")
            self._status.setText("")
        else:
            self._info.setText(f"Active block: {block.display_label}")
            present = has_liz_mov(block)
            self._lizmov_banner.setVisible(not present)
            self._set_workflow_enabled(present)
            self.setEnabled(True)
            self.setToolTip(
                ""
                if present
                else "lizMov.mat is missing — set calibration and use Compute lizMov (Python)."
            )
            if present:
                self._status.setText(
                    "Load lizMov.mat or use 'Load prev analysis' for saved behavior state."
                )
            else:
                self._status.setText("")
        super().set_block(block)

    def _persist_tab_config(self) -> None:
        """Called by main window on close to save behavior-tab settings."""
        self._config.behavior_window_size_ms = int(self._window_size_ms.value())
        self._config.behavior_step_size_ms = int(self._step_size_ms.value())
        self._config.behavior_threshold = float(self._threshold.value())
        self._persist_calibration_config()

    def _after_load_artifacts(self, report) -> None:
        if self._block is None:
            return
        path = self._block.analysis_path / f"block_{self._block.block_num}_behavior_state.csv"
        if path.is_file():
            self._preview.setPlainText(
                pd.read_csv(path).head(20).to_string(index=False)
            )
            self._btn_export.setEnabled(True)
            self._status.setText(f"Loaded preview from {path.name}.")

    def _run_load_lizmov(self, *, silent: bool = False) -> None:
        try:
            blocksync = self._require_blocksync()
            mat_path = resolve_liz_mov_mat_path(blocksync)
            if mat_path is None:
                raise FileNotFoundError(
                    "lizMov.mat not found under oe_files/.../analysis/. "
                    "Run the MATLAB getLizMovement function first."
                )
            blocksync.block_get_lizard_movement()
            if getattr(blocksync, "liz_mov_df", None) is None:
                raise RuntimeError("block_get_lizard_movement did not set liz_mov_df.")
            self._rolling_df = None
            self._btn_export.setEnabled(False)
            sample_count = len(blocksync.liz_mov_df)
            self._run_rolling_average(silent=True)
            if not silent:
                self._status.setText(
                    f"Loaded lizMov ({sample_count} samples) from {mat_path.name}; "
                    f"rolling average plotted ({len(self._rolling_df or [])} windows)."
                )
        except Exception as e:
            if not silent:
                QtWidgets.QMessageBox.warning(self, "Behavior tab", str(e))
            raise

    def _fit_threshold_to_data(self) -> None:
        """Pick a visible threshold when the config default is far above the data scale."""
        if self._rolling_df is None or self._rolling_df.empty:
            return
        ymax = float(self._rolling_df["average_movAll"].max())
        if ymax <= 0:
            return
        current = float(self._threshold.value())
        if current <= ymax * 1.5:
            return
        suggested = float(self._rolling_df["average_movAll"].median())
        self._threshold.blockSignals(True)
        self._threshold.setValue(suggested)
        self._threshold.blockSignals(False)

    def _run_rolling_average(self, *, silent: bool = False) -> None:
        try:
            blocksync = self._require_blocksync()
            if getattr(blocksync, "liz_mov_df", None) is None:
                self._run_load_lizmov(silent=True)
                blocksync = self._require_blocksync()
            window = int(self._window_size_ms.value())
            step = int(self._step_size_ms.value())
            self._rolling_df = rolling_window_analysis(
                blocksync.liz_mov_df,
                window_size=window,
                step_size=step,
            )
            self._movement_plot.set_rolling_data(self._rolling_df, step_ms=step)
            self._fit_threshold_to_data()
            self._movement_plot.set_threshold(
                float(self._threshold.value()), emit=False
            )
            self._btn_export.setEnabled(True)
            if not silent:
                self._status.setText(
                    f"Rolling average ready ({len(self._rolling_df)} windows)."
                )
        except Exception as e:
            self._status.setText(f"Error: {e}")
            if not silent:
                QtWidgets.QMessageBox.warning(self, "Behavior tab", str(e))
            raise

    def _on_threshold_spin(self, value: float) -> None:
        self._movement_plot.set_threshold(float(value), emit=False)

    def _on_threshold_plot(self, value: float) -> None:
        self._threshold.blockSignals(True)
        self._threshold.setValue(float(value))
        self._threshold.blockSignals(False)

    def _behavior_df_from_threshold(self) -> pd.DataFrame:
        if self._rolling_df is None:
            raise RuntimeError("Compute rolling average first.")
        threshold = float(self._threshold.value())
        df = self._rolling_df.copy()
        df["behavior"] = df["average_movAll"].apply(
            lambda x: "active" if x > threshold else "quiet"
        )
        return create_behavior_df(df)

    def _run_export(self) -> None:
        try:
            if self._block is None:
                raise RuntimeError("No block loaded.")
            behavior_df = self._behavior_df_from_threshold()
            out_path = (
                self._block.analysis_path
                / f"block_{self._block.block_num}_behavior_state.csv"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            behavior_df.to_csv(out_path, index=False)
            self._preview.setPlainText(behavior_df.head(20).to_string(index=False))
            self._status.setText(f"Exported {out_path.name} ({len(behavior_df)} segments).")
        except Exception as e:
            self._status.setText(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Behavior tab", str(e))
