"""Stage 3 -- Kerr-degree conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    kerr_artifact_profile,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.batch_runner import (
    SequentialBatchWorker,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.workers import CallableWorker
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
    append_angle_data,
    check_appended_angles,
    export_eye_data_w_angles,
    load_eye_data,
    load_self_kerr_refs,
)
from eye_tracking_system_tools.preprocessing.noise_epochs import (
    list_categories,
    mask_eye_df_by_epochs,
    read_noise_epochs,
    resolve_frame_col,
)


class KerrTab(BaseTab):
    tab_id = "kerr"
    tab_label = "Kerr"

    def __init__(self, state, config, parent=None):
        self._worker: CallableWorker | None = None
        self._batch_worker: SequentialBatchWorker | None = None
        super().__init__(state, config, parent)

    def artifact_profile(self):
        return kerr_artifact_profile(self._name_tag_value_safe())

    def _name_tag_value_safe(self) -> str:
        if hasattr(self, "_name_tag"):
            return self._name_tag.text().strip() or self._config.kerr_name_tag
        return self._config.kerr_name_tag

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_artifact_panel())

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
        self._name_tag.textChanged.connect(lambda _: self._refresh_artifact_ui())
        form.addRow("name_tag:", self._name_tag)
        layout.addLayout(form)

        noise_box = QtWidgets.QGroupBox("Noise epochs (optional)")
        noise_lay = QtWidgets.QVBoxLayout(noise_box)
        self._chk_exclude_noise = QtWidgets.QCheckBox("Exclude noise epochs")
        self._chk_exclude_noise.setToolTip(
            "When checked, Kerr runs on an in-memory copy with geometry NaN'd for "
            "selected categories. Disk eye CSVs are not modified."
        )
        self._chk_exclude_noise.setChecked(False)
        noise_lay.addWidget(self._chk_exclude_noise)
        self._noise_cats = QtWidgets.QListWidget()
        self._noise_cats.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self._noise_cats.setMaximumHeight(90)
        self._noise_cats.setEnabled(False)
        noise_lay.addWidget(self._noise_cats)
        noise_hint = QtWidgets.QLabel(
            "Default: none checked → raw data. Categories come from "
            "analysis/noise_epochs_{left,right}.csv."
        )
        noise_hint.setWordWrap(True)
        noise_lay.addWidget(noise_hint)
        layout.addWidget(noise_box)
        self._chk_exclude_noise.toggled.connect(self._on_exclude_noise_toggled)

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
        self._btn_export.setEnabled(False)
        if block is None:
            self._info.setText("No block loaded.")
            self._refs_banner.hide()
            self._status.setText("")
            self._refresh_noise_category_list(None)
        else:
            self._info.setText(f"Active block: {block.display_label}")
            left = block.analysis_path / "left_eye_data.csv"
            if not left.is_file():
                self._refs_banner.setText(
                    "left_eye_data.csv is missing. Complete Sync tab step 5 first."
                )
                self._refs_banner.show()
            elif not (block.analysis_path / "self_kerr_refs.csv").is_file():
                self._refs_banner.setText(
                    "Pick Kerr refs in the Verify tab first (self_kerr_refs.csv is missing)."
                )
                self._refs_banner.show()
            else:
                self._refs_banner.hide()
            self._status.setText(
                "Use 'Load prev analysis' to hydrate eye data and Kerr outputs from disk."
            )
            self._refresh_noise_category_list(block)
        super().set_block(block)

    def _on_exclude_noise_toggled(self, checked: bool) -> None:
        self._noise_cats.setEnabled(bool(checked))

    def _refresh_noise_category_list(self, block: BlockHandle | None) -> None:
        self._noise_cats.clear()
        if block is None:
            return
        for cat in list_categories(block.block_path):
            item = QtWidgets.QListWidgetItem(cat)
            item.setFlags(
                item.flags()
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self._noise_cats.addItem(item)

    def _selected_noise_categories(self) -> list[str]:
        if not self._chk_exclude_noise.isChecked():
            return []
        out: list[str] = []
        for i in range(self._noise_cats.count()):
            item = self._noise_cats.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _apply_noise_mask_in_memory(self, blocksync: BlockSync, categories: list[str]) -> None:
        if not categories:
            return
        block_path = Path(blocksync.block_path)
        for eye, attr in (("left", "left_eye_data"), ("right", "right_eye_data")):
            df = getattr(blocksync, attr, None)
            if df is None or (hasattr(df, "empty") and df.empty):
                continue
            epochs = read_noise_epochs(block_path, eye)
            frame_col = resolve_frame_col(df, eye)
            masked, _ = mask_eye_df_by_epochs(
                df, epochs, frame_col=frame_col, categories=categories
            )
            setattr(blocksync, attr, masked)

    def _after_load_artifacts(self, report) -> None:
        if self._block is None:
            return
        try:
            b = self._require_blocksync()
            if load_self_kerr_refs(b):
                self._refs_banner.hide()
                self._btn_export.setEnabled(True)
                self._status.setText("Eye data and Kerr refs loaded from disk.")
        except FileNotFoundError as e:
            self._refs_banner.setText(str(e))
            self._refs_banner.show()

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
            load_eye_data(b)
            cats = self._selected_noise_categories()
            self._apply_noise_mask_in_memory(b, cats)
            tag = self._name_tag_value()
            b.calculate_kerr_angles(name_tag=tag)
            return cats

        self._set_busy(True)
        worker = CallableWorker(work, self)

        def on_ok(cats=None):
            self._set_busy(False)
            self._btn_export.setEnabled(True)
            extra = ""
            if cats:
                extra = f" (excluded noise: {', '.join(cats)})"
            self._status.setText(
                f"Calculated Kerr angles (tag={self._name_tag_value()}){extra}."
            )
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

    def _merge_angles_onto_eye_data(self, blocksync, tag: str) -> list[str]:
        """
        Append Kerr angle CSVs onto current in-memory eye frames (no disk reload).

        Returns human-readable passive-check lines (column names + finite counts).
        Raises if canonical ``k_phi``/``k_theta`` are missing after append.
        """
        ap = Path(blocksync.analysis_path)
        left_angle = ap / f"left_kerr_angle_{tag}.csv"
        right_angle = ap / f"right_kerr_angle_{tag}.csv"
        if not left_angle.is_file() or not right_angle.is_file():
            raise RuntimeError(
                f"Kerr angle CSVs for tag {tag!r} not found. Run Calculate first."
            )
        left_angles = pd.read_csv(left_angle)
        right_angles = pd.read_csv(right_angle)
        blocksync.left_eye_data = append_angle_data(
            blocksync.left_eye_data, left_angles
        )
        blocksync.right_eye_data = append_angle_data(
            blocksync.right_eye_data, right_angles
        )
        checks = [
            check_appended_angles(
                blocksync.left_eye_data, left_angles, side="left"
            ),
            check_appended_angles(
                blocksync.right_eye_data, right_angles, side="right"
            ),
        ]
        if any(not c.columns_ok for c in checks):
            detail = "; ".join(c.summary() for c in checks)
            raise RuntimeError(
                "Appended eye data is missing canonical k_phi/k_theta columns. "
                f"{detail}"
            )
        return [c.summary() for c in checks]

    def _run_export_merged(self) -> None:
        try:
            b = self._require_blocksync()
            tag = self._name_tag_value()
            check_lines = self._merge_angles_onto_eye_data(b, tag)
            export_eye_data_w_angles(b, name_tag=tag)
            warn = [line for line in check_lines if ": warn" in line]
            status = f"Exported left/right_eye_data_{tag}.csv"
            if warn:
                status += " | " + " · ".join(warn)
                self._status.setText(status)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Kerr tab — angle append check",
                    "Export wrote files, but the passive angle check reported:\n\n"
                    + "\n".join(check_lines),
                )
            else:
                self._status.setText(status + " | " + " · ".join(check_lines))
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
        b = self._session.get(handle)
        load_eye_data(b)
        self._ensure_kerr_refs(b)
        cats = self._selected_noise_categories()
        self._apply_noise_mask_in_memory(b, cats)
        tag = self._name_tag_value()
        b.calculate_kerr_angles(name_tag=tag)
        check_lines = self._merge_angles_onto_eye_data(b, tag)
        export_eye_data_w_angles(b, name_tag=tag)
        extra = f", excluded={cats}" if cats else ""
        checks = " · ".join(check_lines)
        return f"tag={tag} exported{extra} | {checks}"

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
