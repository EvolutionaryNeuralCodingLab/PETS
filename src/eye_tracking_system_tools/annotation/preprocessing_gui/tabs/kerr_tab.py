"""Stage 3 -- Kerr-degree conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6 import QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.batch_runner import (
    SequentialBatchWorker,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.workers import CallableWorker
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
    append_angle_data,
    export_eye_data_w_angles,
    load_eye_data,
    load_self_kerr_refs,
)


class KerrTab(BaseTab):
    tab_id = "kerr"
    tab_label = "Kerr"

    def __init__(self, state, config, parent=None):
        self._block: BlockHandle | None = None
        self._blocksync: BlockSync | None = None
        self._worker: CallableWorker | None = None
        self._batch_worker: SequentialBatchWorker | None = None
        super().__init__(state, config, parent)

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self._refs_banner = QtWidgets.QLabel()
        self._refs_banner.setWordWrap(True)
        self._refs_banner.setStyleSheet(
            "background-color: #f8d7da; color: #842029; padding: 8px; border-radius: 4px;"
        )
        self._refs_banner.hide()
        layout.addWidget(self._refs_banner)

        self._info = QtWidgets.QLabel("Load a block to calculate Kerr angles.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        form = QtWidgets.QFormLayout()
        self._name_tag = QtWidgets.QLineEdit(str(self._config.kerr_name_tag))
        self._name_tag.setPlaceholderText("e.g. raw_verified")
        form.addRow("name_tag:", self._name_tag)
        layout.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        self._btn_calculate = QtWidgets.QPushButton("Calculate Kerr angles")
        self._btn_export = QtWidgets.QPushButton("Export merged eye data")
        self._btn_export.setEnabled(False)
        row.addWidget(self._btn_calculate)
        row.addWidget(self._btn_export)
        layout.addLayout(row)

        batch_box = QtWidgets.QGroupBox("Batch — all loaded blocks")
        batch_layout = QtWidgets.QVBoxLayout(batch_box)
        batch_row = QtWidgets.QHBoxLayout()
        self._btn_batch = QtWidgets.QPushButton("Run calculate + export for all blocks")
        self._btn_batch_cancel = QtWidgets.QPushButton("Cancel batch")
        self._btn_batch_cancel.setEnabled(False)
        batch_row.addWidget(self._btn_batch)
        batch_row.addWidget(self._btn_batch_cancel)
        batch_layout.addLayout(batch_row)
        self._batch_log = QtWidgets.QPlainTextEdit()
        self._batch_log.setReadOnly(True)
        self._batch_log.setMaximumHeight(120)
        batch_layout.addWidget(self._batch_log)
        layout.addWidget(batch_box)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._btn_calculate.clicked.connect(self._run_calculate)
        self._btn_export.clicked.connect(self._run_export_merged)
        self._btn_batch.clicked.connect(self._run_batch)
        self._btn_batch_cancel.clicked.connect(self._cancel_batch)

    def status_signature(self, block: BlockHandle) -> list[Path]:
        tag = self._name_tag.text().strip() or self._config.kerr_name_tag
        ap = block.analysis_path
        return [
            ap / f"left_kerr_angle_{tag}.csv",
            ap / f"right_kerr_angle_{tag}.csv",
        ]

    def set_block(self, block: BlockHandle | None) -> None:
        self._block = block
        self._blocksync = None
        self._btn_export.setEnabled(False)
        if block is None:
            self._info.setText("No block loaded.")
            self._refs_banner.hide()
            self._status.setText("")
            return

        self._info.setText(f"Active block: {block.display_label}")
        try:
            blocksync = self._blocksync_for_handle(block)
            load_eye_data(blocksync)
            self._blocksync = blocksync
            if load_self_kerr_refs(blocksync):
                self._refs_banner.hide()
                self._status.setText("Eye data and Kerr refs loaded.")
            else:
                self._refs_banner.setText(
                    "Pick Kerr refs in the Verify tab first (self_kerr_refs.csv is missing)."
                )
                self._refs_banner.show()
                self._status.setText("")
        except FileNotFoundError as e:
            self._refs_banner.setText(str(e))
            self._refs_banner.show()
            self._status.setText("")

    def _blocksync_for_handle(self, handle: BlockHandle) -> BlockSync:
        b = BlockSync(
            handle.animal_call,
            handle.experiment_date,
            handle.block_num,
            handle.path_to_animal_folder,
            channeldict=handle.channeldict,
        )
        return b

    def _require_blocksync(self) -> BlockSync:
        if self._block is None:
            raise RuntimeError("No block loaded.")
        if self._blocksync is None:
            self._blocksync = self._blocksync_for_handle(self._block)
            load_eye_data(self._blocksync)
        return self._blocksync

    def _name_tag_value(self) -> str:
        tag = self._name_tag.text().strip()
        if not tag:
            raise RuntimeError("Provide a name_tag.")
        return tag

    def _ensure_kerr_refs(self, blocksync: BlockSync) -> None:
        if not load_self_kerr_refs(blocksync):
            raise RuntimeError(
                "self_kerr_refs.csv is missing. Complete the Verify tab and save Kerr refs first."
            )
        if getattr(blocksync, "kerr_ref_l_x", None) is None or getattr(
            blocksync, "kerr_ref_r_x", None
        ) is None:
            raise RuntimeError("Kerr reference coordinates are not set on the block.")

    def _set_busy(self, busy: bool) -> None:
        for btn in (
            self._btn_calculate,
            self._btn_export,
            self._btn_batch,
        ):
            btn.setEnabled(not busy)
        self._btn_batch_cancel.setEnabled(busy)

    def _run_calculate(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        def work():
            b = self._require_blocksync()
            self._ensure_kerr_refs(b)
            tag = self._name_tag_value()
            b.calculate_kerr_angles(name_tag=tag)

        self._set_busy(True)
        worker = CallableWorker(work, self)

        def on_ok(_=None):
            self._set_busy(False)
            self._btn_export.setEnabled(True)
            self._status.setText(f"Calculated Kerr angles (tag={self._name_tag_value()}).")
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str):
            self._set_busy(False)
            self._status.setText(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Kerr tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _run_export_merged(self) -> None:
        try:
            b = self._require_blocksync()
            tag = self._name_tag_value()
            ap = Path(b.analysis_path)
            left_angle = ap / f"left_kerr_angle_{tag}.csv"
            right_angle = ap / f"right_kerr_angle_{tag}.csv"
            if not left_angle.is_file() or not right_angle.is_file():
                raise RuntimeError(
                    f"Kerr angle CSVs for tag {tag!r} not found. Run Calculate first."
                )
            left_angles = pd.read_csv(left_angle)
            right_angles = pd.read_csv(right_angle)
            b.left_eye_data = append_angle_data(b.left_eye_data, left_angles)
            b.right_eye_data = append_angle_data(b.right_eye_data, right_angles)
            export_eye_data_w_angles(b, name_tag=tag)
            self._status.setText(f"Exported left/right_eye_data_{tag}.csv")
        except Exception as e:
            self._status.setText(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Kerr tab", str(e))

    def _batch_blocks(self) -> list[BlockHandle]:
        blocks = list(self._state.blocks)
        if len(blocks) < 2:
            raise RuntimeError("Load at least two blocks to use batch mode.")
        return blocks

    def _run_batch(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            return
        try:
            blocks = self._batch_blocks()
        except RuntimeError as e:
            QtWidgets.QMessageBox.warning(self, "Kerr tab", str(e))
            return

        tag = self._name_tag_value()
        self._batch_log.clear()
        self._batch_log.appendPlainText(
            f"Starting Kerr batch: calculate + export (tag={tag}, {len(blocks)} blocks)"
        )
        self._set_busy(True)
        self._batch_worker = SequentialBatchWorker(blocks, self._batch_one_block, self)
        self._batch_worker.progress.connect(self._batch_log.appendPlainText)
        self._batch_worker.block_done.connect(self._on_batch_block_done)
        self._batch_worker.finished_all.connect(self._on_batch_finished)
        self._batch_worker.cancelled.connect(self._on_batch_cancelled)
        self._batch_worker.start()

    def _batch_one_block(self, handle: BlockHandle) -> str:
        b = self._blocksync_for_handle(handle)
        load_eye_data(b)
        self._ensure_kerr_refs(b)
        tag = self._name_tag_value()
        b.calculate_kerr_angles(name_tag=tag)
        left_angle = Path(b.analysis_path) / f"left_kerr_angle_{tag}.csv"
        right_angle = Path(b.analysis_path) / f"right_kerr_angle_{tag}.csv"
        left_angles = pd.read_csv(left_angle)
        right_angles = pd.read_csv(right_angle)
        b.left_eye_data = append_angle_data(b.left_eye_data, left_angles)
        b.right_eye_data = append_angle_data(b.right_eye_data, right_angles)
        export_eye_data_w_angles(b, name_tag=tag)
        return f"tag={tag} exported"

    def _cancel_batch(self) -> None:
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_log.appendPlainText("Cancel requested…")
            self._batch_worker.request_cancel()

    def _on_batch_block_done(self, label: str, ok: bool, detail: str) -> None:
        prefix = "OK" if ok else "FAIL"
        self._batch_log.appendPlainText(f"  {prefix} {label}: {detail}")

    def _on_batch_finished(self) -> None:
        self._batch_log.appendPlainText("Batch finished.")
        self._finish_batch_ui()

    def _on_batch_cancelled(self) -> None:
        self._batch_log.appendPlainText("Batch cancelled.")
        self._finish_batch_ui()

    def _finish_batch_ui(self) -> None:
        self._set_busy(False)
        if self._batch_worker is not None:
            self._batch_worker.deleteLater()
            self._batch_worker = None
        self._status.setText("Kerr batch complete.")
