"""Stage 4 -- Accelerometer-based behavior annotation."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from PyQt6 import QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.pyqtgraph_helpers import (
    BehaviorThresholdPlot,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.notebook_helpers import (
    create_behavior_df,
    rolling_window_analysis,
)


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
    try:
        blocksync = BlockSync(
            block.animal_call,
            block.experiment_date,
            block.block_num,
            block.path_to_animal_folder,
            channeldict=block.channeldict,
        )
    except Exception:
        return False
    return resolve_liz_mov_mat_path(blocksync) is not None


class BehaviorTab(BaseTab):
    tab_id = "behavior"
    tab_label = "Behavior"

    def __init__(self, state, config, parent=None):
        self._block: BlockHandle | None = None
        self._blocksync: BlockSync | None = None
        self._rolling_df: pd.DataFrame | None = None
        super().__init__(state, config, parent)

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self._lizmov_banner = QtWidgets.QLabel(
            "<b>lizMov.mat not found.</b> Run the MATLAB <code>getLizMovement</code> "
            "function for this block first (expected under "
            "<code>oe_files/&lt;exp&gt;/&lt;Record Node&gt;/analysis/</code>)."
        )
        self._lizmov_banner.setWordWrap(True)
        self._lizmov_banner.setStyleSheet(
            "background-color: #fff3cd; color: #664d03; padding: 8px; border-radius: 4px;"
        )
        self._lizmov_banner.hide()
        layout.addWidget(self._lizmov_banner)

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
        self._btn_rolling = QtWidgets.QPushButton("Compute rolling average")
        self._btn_export = QtWidgets.QPushButton("Export behavior state")
        self._btn_export.setEnabled(False)
        row.addWidget(self._btn_load)
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
        self._btn_rolling.clicked.connect(self._run_rolling_average)
        self._btn_export.clicked.connect(self._run_export)
        self._threshold.valueChanged.connect(self._on_threshold_spin)
        self._movement_plot.threshold_changed.connect(self._on_threshold_plot)

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
        self._block = block
        self._blocksync = None
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
            return

        self._info.setText(f"Active block: {block.display_label}")
        present = has_liz_mov(block)
        self._lizmov_banner.setVisible(not present)
        self._set_workflow_enabled(present)
        self.setEnabled(True)
        self.setToolTip(
            ""
            if present
            else "lizMov.mat is missing for this block — run MATLAB getLizMovement first."
        )
        if not present:
            self._status.setText("")
            return

        try:
            self._run_load_lizmov(silent=True)
            if self._rolling_df is not None:
                self._status.setText(
                    f"Movement trace ready ({len(self._rolling_df)} windows)."
                )
        except Exception as e:
            self._status.setText(f"Error loading lizMov: {e}")
            self._set_workflow_enabled(False)

    def _blocksync_for_handle(self, handle: BlockHandle) -> BlockSync:
        return BlockSync(
            handle.animal_call,
            handle.experiment_date,
            handle.block_num,
            handle.path_to_animal_folder,
            channeldict=handle.channeldict,
        )

    def _require_blocksync(self) -> BlockSync:
        if self._block is None:
            raise RuntimeError("No block loaded.")
        if self._blocksync is None:
            self._blocksync = self._blocksync_for_handle(self._block)
        return self._blocksync

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
            self._blocksync = blocksync
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
