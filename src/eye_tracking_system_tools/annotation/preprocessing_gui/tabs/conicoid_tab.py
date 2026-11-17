"""Stage 2c -- Conicoid (Safaee-Rad / Swirski / Dierkes) vector estimation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    conicoid_artifact_profile,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.batch_runner import (
    SequentialBatchWorker,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.refine_tab import (
    RefinePreviewWidget,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.workers import CallableWorker
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.calculate_conicoid_angles import (
    EllipseGeometryError,
    append_conicoid_angle_data,
    calculate_conicoid_angles_for_block,
    check_block_geometry,
    export_eye_data_w_conicoid,
)
from eye_tracking_system_tools.preprocessing.compare_conicoid_kerr import (
    compare_conicoid_kerr_for_block,
    comparison_paths,
    conicoid_outputs_available,
    format_summary,
    kerr_outputs_available,
)
from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import load_eye_data
from eye_tracking_system_tools.preprocessing.conicoid.refined_io import (
    list_refined_tags,
)
from eye_tracking_system_tools.preprocessing.conicoid.refraction import (
    refraction_npz_path,
    try_load_refraction_maps,
)
from eye_tracking_system_tools.preprocessing.noise_epochs import (
    list_categories,
    mask_eye_df_by_epochs,
    read_noise_epochs,
    resolve_frame_col,
)


_V2_TABLE_LABEL = "Eye table, geometry v2 (main or analysis/ellipse_v2)"
_SWIRSKI_LABEL = "Swirski 2D"
_DIERKES_LABEL = "Dierkes 3D (provisional, not validated)"


class ConicoidTab(BaseTab):
    tab_id = "conicoid"
    tab_label = "Conicoid"

    def __init__(self, state, config, parent=None):
        self._worker: CallableWorker | None = None
        self._batch_worker: SequentialBatchWorker | None = None
        self._preview: RefinePreviewWidget | None = None
        super().__init__(state, config, parent)

    def _on_name_tag_changed(self) -> None:
        self._refresh_artifact_ui()
        if hasattr(self, "_btn_compare"):
            self._refresh_compare_ui()

    def artifact_profile(self):
        return conicoid_artifact_profile(self._name_tag_value_safe())

    def _name_tag_value_safe(self) -> str:
        if hasattr(self, "_name_tag"):
            return self._name_tag.text().strip() or "raw_verified"
        return "raw_verified"

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_artifact_panel())

        self._info = QtWidgets.QLabel(
            "Fit a 3D pupil-on-sphere model and write the gaze normal c_nx / c_ny / c_nz "
            "(plus c_phi / c_theta). Kerr k_phi / k_theta are not replaced; Kerr stays "
            "the check (run Kerr with the same name_tag, then Compare)."
        )
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        form = QtWidgets.QFormLayout()
        self._name_tag = QtWidgets.QLineEdit("raw_verified")
        self._name_tag.setPlaceholderText("e.g. raw_verified")
        self._name_tag.textChanged.connect(lambda _: self._on_name_tag_changed())
        form.addRow("name_tag:", self._name_tag)

        self._ellipse_source = QtWidgets.QComboBox()
        self._ellipse_source.addItem(_V2_TABLE_LABEL, "original")
        form.addRow("Ellipse source:", self._ellipse_source)

        self._geometry_status = QtWidgets.QLabel("")
        self._geometry_status.setWordWrap(True)
        form.addRow("Geometry:", self._geometry_status)

        self._sphere_method = QtWidgets.QComboBox()
        self._sphere_method.addItems([_SWIRSKI_LABEL, _DIERKES_LABEL])
        form.addRow("Sphere method:", self._sphere_method)
        self._dierkes_note = QtWidgets.QLabel(
            "Dierkes 3D is provisional: it has not been validated against the "
            "synthetic ground truth. Use Swirski 2D for results."
        )
        self._dierkes_note.setWordWrap(True)
        self._dierkes_note.setStyleSheet("color: #b36b00;")
        self._dierkes_note.setVisible(False)
        form.addRow("", self._dierkes_note)
        self._sphere_method.currentTextChanged.connect(
            lambda text: self._dierkes_note.setVisible(text == _DIERKES_LABEL)
        )

        self._eye_z = QtWidgets.QDoubleSpinBox()
        self._eye_z.setRange(1.0, 40.0)
        self._eye_z.setValue(13.0)
        form.addRow("eye_z (mm):", self._eye_z)

        self._ransac = QtWidgets.QCheckBox("RANSAC")
        self._ransac.setChecked(True)
        form.addRow("", self._ransac)

        self._refraction = QtWidgets.QCheckBox("Apply refraction maps")
        self._refraction.setEnabled(False)
        form.addRow("", self._refraction)
        layout.addLayout(form)

        noise_box = QtWidgets.QGroupBox("Noise epochs (optional)")
        noise_lay = QtWidgets.QVBoxLayout(noise_box)
        self._chk_exclude_noise = QtWidgets.QCheckBox("Exclude noise epochs")
        self._chk_exclude_noise.setChecked(False)
        noise_lay.addWidget(self._chk_exclude_noise)
        self._noise_cats = QtWidgets.QListWidget()
        self._noise_cats.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self._noise_cats.setMaximumHeight(90)
        self._noise_cats.setEnabled(False)
        noise_lay.addWidget(self._noise_cats)
        layout.addWidget(noise_box)
        self._chk_exclude_noise.toggled.connect(
            lambda checked: self._noise_cats.setEnabled(bool(checked))
        )

        row = QtWidgets.QHBoxLayout()
        self._btn_calculate = QtWidgets.QPushButton("Calculate conicoid angles")
        self._btn_export = QtWidgets.QPushButton("Export merged eye data")
        self._btn_export.setEnabled(False)
        row.addWidget(self._btn_calculate)
        row.addWidget(self._btn_export)
        layout.addLayout(row)

        compare_box = QtWidgets.QGroupBox("Compare with Kerr (3D angle between gaze normals)")
        compare_layout = QtWidgets.QVBoxLayout(compare_box)
        compare_row = QtWidgets.QHBoxLayout()
        self._btn_compare = QtWidgets.QPushButton("Compare with Kerr")
        self._btn_compare.setEnabled(False)
        self._btn_open_plot = QtWidgets.QPushButton("Open plot")
        self._btn_open_plot.setEnabled(False)
        compare_row.addWidget(self._btn_compare)
        compare_row.addWidget(self._btn_open_plot)
        compare_row.addStretch(1)
        compare_layout.addLayout(compare_row)
        self._compare_summary = QtWidgets.QLabel("")
        self._compare_summary.setWordWrap(True)
        self._compare_summary.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        compare_layout.addWidget(self._compare_summary)
        layout.addWidget(compare_box)

        batch_box = QtWidgets.QGroupBox("Batch — all loaded blocks")
        batch_layout = QtWidgets.QVBoxLayout(batch_box)
        self._chk_batch_compare = QtWidgets.QCheckBox(
            "Also compare with Kerr where Kerr outputs for this name_tag exist"
        )
        self._chk_batch_compare.setChecked(True)
        batch_layout.addWidget(self._chk_batch_compare)
        batch_row = QtWidgets.QHBoxLayout()
        self._btn_batch = QtWidgets.QPushButton("Run calculate + export for all blocks")
        self._btn_batch_cancel = QtWidgets.QPushButton("Cancel batch")
        self._btn_batch_cancel.setEnabled(False)
        batch_row.addWidget(self._btn_batch)
        batch_row.addWidget(self._btn_batch_cancel)
        batch_layout.addLayout(batch_row)
        self._batch_log = QtWidgets.QPlainTextEdit()
        self._batch_log.setReadOnly(True)
        self._batch_log.setMaximumHeight(80)
        batch_layout.addWidget(self._batch_log)
        layout.addWidget(batch_box)

        self._preview_host = QtWidgets.QWidget()
        self._preview_layout = QtWidgets.QHBoxLayout(self._preview_host)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._preview_host, stretch=1)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

        self._btn_calculate.clicked.connect(self._run_calculate)
        self._btn_export.clicked.connect(self._run_export_merged)
        self._btn_batch.clicked.connect(self._run_batch)
        self._btn_batch_cancel.clicked.connect(self._cancel_batch)
        self._btn_compare.clicked.connect(self._run_compare)
        self._btn_open_plot.clicked.connect(self._open_compare_plot)

    def status_signature(self, block: BlockHandle) -> list[Path]:
        tag = self._name_tag_value_safe()
        ap = block.analysis_path
        return [
            ap / f"left_conicoid_angle_{tag}.csv",
            ap / f"right_conicoid_angle_{tag}.csv",
        ]

    def set_block(self, block: BlockHandle | None) -> None:
        self._btn_export.setEnabled(False)
        self._clear_preview()
        if block is None:
            self._info.setText("No block loaded.")
            self._status.setText("")
            self._refresh_noise_category_list(None)
            self._refresh_ellipse_sources(None)
            self._refraction.setEnabled(False)
            self._geometry_status.setText("")
        else:
            self._info.setText(f"Active block: {block.display_label}")
            self._refresh_noise_category_list(block)
            self._refresh_ellipse_sources(block)
            npz = refraction_npz_path(block.block_path)
            self._refraction.setEnabled(npz.is_file())
            if not npz.is_file():
                self._refraction.setChecked(False)
            self._status.setText(
                "Use 'Load prev analysis' to hydrate eye data, then calculate."
            )
        super().set_block(block)
        self._refresh_geometry_status()
        self._refresh_compare_ui()

    def _refresh_geometry_status(self) -> bool:
        """Show which v2 eye tables will be fitted; disable Calculate if none."""
        if self._block is None:
            self._btn_calculate.setEnabled(False)
            return False
        try:
            sources = check_block_geometry(
                self._block.analysis_path, block_label=self._block.display_label
            )
        except EllipseGeometryError as exc:
            self._geometry_status.setText(f"Not geometry v2. {exc}")
            self._geometry_status.setStyleSheet("color: #b00020;")
            self._btn_calculate.setEnabled(False)
            return False
        self._geometry_status.setText(
            f"v2 — left: {sources['left']}; right: {sources['right']}"
        )
        self._geometry_status.setStyleSheet("")
        self._btn_calculate.setEnabled(True)
        return True

    def _refresh_compare_ui(self) -> None:
        if self._block is None:
            self._btn_compare.setEnabled(False)
            self._btn_open_plot.setEnabled(False)
            self._compare_summary.setText("")
            return
        tag = self._name_tag_value_safe()
        ap = self._block.analysis_path
        have_c = conicoid_outputs_available(ap, tag)
        have_k = kerr_outputs_available(ap, tag)
        self._btn_compare.setEnabled(have_c and have_k)
        paths = comparison_paths(ap, tag)
        self._btn_open_plot.setEnabled(paths.plot.is_file())
        if paths.summary.is_file():
            try:
                self._compare_summary.setText(format_summary(pd.read_csv(paths.summary)))
                return
            except Exception as exc:
                self._compare_summary.setText(f"Could not read {paths.summary.name}: {exc}")
                return
        missing = [name for name, ok in (("Conicoid", have_c), ("Kerr", have_k)) if not ok]
        self._compare_summary.setText(
            f"Needs {' and '.join(missing)} outputs for name_tag={tag!r}." if missing else
            "Ready to compare."
        )

    def _refresh_ellipse_sources(self, block: BlockHandle | None) -> None:
        self._ellipse_source.blockSignals(True)
        self._ellipse_source.clear()
        self._ellipse_source.addItem(_V2_TABLE_LABEL, "original")
        if block is not None:
            tags = list_refined_tags(block.analysis_path)
            if tags:
                self._ellipse_source.addItem("Latest refined CSV", "refined")
            for tag in tags:
                self._ellipse_source.addItem(
                    f"Refined ({tag})", f"refined:{tag}"
                )
        self._ellipse_source.blockSignals(False)

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
            self._require_blocksync()
            self._refresh_ellipse_sources(self._block)
            self._status.setText("Eye data loaded from disk.")
        except FileNotFoundError as e:
            self._status.setText(str(e))

    def _name_tag_value(self) -> str:
        tag = self._name_tag.text().strip()
        if not tag:
            raise RuntimeError("Provide a name_tag.")
        return tag

    def _ellipse_source_value(self) -> str:
        data = self._ellipse_source.currentData()
        return str(data) if data else "original"

    def _sphere_method_value(self) -> str:
        return (
            "dierkes_3d"
            if self._sphere_method.currentText() == _DIERKES_LABEL
            else "swirski_2d"
        )

    def _set_busy(self, busy: bool) -> None:
        for btn in (self._btn_calculate, self._btn_export, self._btn_batch, self._btn_compare):
            btn.setEnabled(not busy)
        self._btn_batch_cancel.setEnabled(busy)
        if not busy:
            self._refresh_geometry_status()
            self._refresh_compare_ui()

    def _run_calculate(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        def work():
            b = self._require_blocksync()
            load_eye_data(b)
            cats = self._selected_noise_categories()
            self._apply_noise_mask_in_memory(b, cats)
            maps = None
            if self._refraction.isChecked() and self._block is not None:
                maps = try_load_refraction_maps(self._block.block_path)
            calculate_conicoid_angles_for_block(
                b,
                name_tag=self._name_tag_value(),
                load_eye_data_flag=False,
                export_flag=False,
                eye_z=float(self._eye_z.value()),
                use_ransac=self._ransac.isChecked(),
                sphere_method=self._sphere_method_value(),
                ellipse_source=self._ellipse_source_value(),
                refraction_maps=maps,
            )
            return cats

        self._set_busy(True)
        worker = CallableWorker(work, self)

        def on_ok(cats=None):
            self._set_busy(False)
            self._btn_export.setEnabled(True)
            extra = f" (excluded noise: {', '.join(cats)})" if cats else ""
            self._status.setText(
                f"Calculated conicoid angles (tag={self._name_tag_value()}){extra}."
            )
            self._refresh_preview()
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str):
            self._set_busy(False)
            self._status.setText(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Conicoid tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _run_compare(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._block is None:
            return
        tag = self._name_tag_value()
        analysis_path = self._block.analysis_path

        def work():
            return compare_conicoid_kerr_for_block(analysis_path, name_tag=tag)

        self._set_busy(True)
        worker = CallableWorker(work, self)

        def on_ok(summary=None):
            self._set_busy(False)
            self._status.setText(
                f"Wrote {comparison_paths(analysis_path, tag).summary.name} and plot."
            )
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str):
            self._set_busy(False)
            self._status.setText(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Conicoid tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _open_compare_plot(self) -> None:
        if self._block is None:
            return
        path = comparison_paths(self._block.analysis_path, self._name_tag_value_safe()).plot
        if not path.is_file():
            self._status.setText(f"No plot at {path}")
            return
        from PyQt6 import QtGui

        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _run_export_merged(self) -> None:
        try:
            b = self._require_blocksync()
            tag = self._name_tag_value()
            ap = Path(b.analysis_path)
            left_angle = ap / f"left_conicoid_angle_{tag}.csv"
            right_angle = ap / f"right_conicoid_angle_{tag}.csv"
            if not left_angle.is_file() or not right_angle.is_file():
                raise RuntimeError(
                    f"Conicoid angle CSVs for tag {tag!r} not found. Run Calculate first."
                )
            b.left_eye_data = append_conicoid_angle_data(
                b.left_eye_data, pd.read_csv(left_angle)
            )
            b.right_eye_data = append_conicoid_angle_data(
                b.right_eye_data, pd.read_csv(right_angle)
            )
            export_eye_data_w_conicoid(b, name_tag=tag)
            self._status.setText(
                f"Exported left/right_eye_data_degrees_{tag}_conicoid.csv"
            )
        except Exception as e:
            self._status.setText(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Conicoid tab", str(e))

    def _run_batch(self) -> None:
        if self._batch_worker is not None and self._batch_worker.isRunning():
            return
        blocks = list(self._state.blocks)
        if not blocks:
            self._status.setText("No blocks loaded.")
            return
        tag = self._name_tag_value()
        eye_z = float(self._eye_z.value())
        use_ransac = self._ransac.isChecked()
        sphere_method = self._sphere_method_value()
        ellipse_source = self._ellipse_source_value()
        apply_refraction = self._refraction.isChecked()
        also_compare = self._chk_batch_compare.isChecked()
        self._batch_log.clear()

        def one(handle: BlockHandle):
            session = self._state.ensure_session()
            bs = session.ensure_blocksync(handle)
            load_eye_data(bs)
            maps = try_load_refraction_maps(handle.block_path) if apply_refraction else None
            calculate_conicoid_angles_for_block(
                bs,
                name_tag=tag,
                load_eye_data_flag=False,
                export_flag=True,
                eye_z=eye_z,
                use_ransac=use_ransac,
                sphere_method=sphere_method,
                ellipse_source=ellipse_source,
                refraction_maps=maps,
            )
            if also_compare and kerr_outputs_available(bs, tag):
                summary = compare_conicoid_kerr_for_block(bs, name_tag=tag)
                medians = ", ".join(
                    f"{r.eye} {r.median_angle_deg:.2f} deg" for r in summary.itertuples()
                )
                return f"{handle.display_label} (vs Kerr median: {medians})"
            return handle.display_label

        self._set_busy(True)
        worker = SequentialBatchWorker(blocks, one, self)
        self._batch_worker = worker

        def on_item(label: str, ok: bool, detail: str):
            prefix = "OK" if ok else "FAIL"
            self._batch_log.appendPlainText(f"  {prefix} {label}: {detail}")

        def on_done():
            self._set_busy(False)
            self._btn_export.setEnabled(True)
            self._status.setText("Conicoid batch complete.")
            self._batch_worker = None
            worker.deleteLater()

        worker.progress.connect(self._batch_log.appendPlainText)
        worker.block_done.connect(on_item)
        worker.finished_all.connect(on_done)
        worker.cancelled.connect(on_done)
        worker.start()

    def _cancel_batch(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.request_cancel()

    def _clear_preview(self) -> None:
        while self._preview_layout.count():
            item = self._preview_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._preview = None

    def _refresh_preview(self) -> None:
        if self._block is None:
            return
        try:
            blocksync = self._require_blocksync()
            self._session.ensure_eye_videos(blocksync)
            if not blocksync.le_videos:
                return
            self._clear_preview()
            self._preview = RefinePreviewWidget(
                blocksync.left_eye_data,
                blocksync.le_videos[0],
                "left",
                parent=self,
                show_gaze=True,
            )
            self._preview_layout.addWidget(self._preview)
        except Exception:
            return
