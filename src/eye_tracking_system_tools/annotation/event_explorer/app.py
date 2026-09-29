"""Event Explorer main window and application entry."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.event_explorer.block_data_loader import (
    BlockDataManager,
)
from eye_tracking_system_tools.annotation.event_explorer.catalog import (
    build_catalog,
    discover_annotation_files,
    event_types_in_catalog,
)
from eye_tracking_system_tools.annotation.event_explorer.event_table import (
    COLUMN_SPECS,
    EventTableWidget,
)
from eye_tracking_system_tools.annotation.event_explorer.export_io import export_bundle
from eye_tracking_system_tools.annotation.event_explorer.load_log import (
    LoadLog,
    LoadLogPanel,
    default_log_path,
)
from eye_tracking_system_tools.annotation.event_explorer.models import (
    ColumnMap,
    EventRecord,
    EXPLORER_VERSION,
)
from eye_tracking_system_tools.annotation.event_explorer.plot_panel import PlotPanel
from eye_tracking_system_tools.annotation.event_explorer.session_io import (
    column_map_from_session,
    load_session,
    save_session,
    session_to_dict,
)
from eye_tracking_system_tools.annotation.event_explorer.snippet_extractor import (
    STREAM_EP,
    STREAM_L_DEG,
    STREAM_L_PUPIL,
    STREAM_R_DEG,
    STREAM_R_PUPIL,
    extract_ep_snippet,
    extract_eye_snippet,
)


class _PreviewWorker(QtCore.QThread):
    """Load snippets off the UI thread so the preview status can update."""

    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, work_fn, parent=None) -> None:
        super().__init__(parent)
        self._work_fn = work_fn

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._work_fn())
        except Exception as exc:
            self.failed.emit(str(exc))
from eye_tracking_system_tools.annotation.event_explorer.source_dialog import (
    AddSourcesDialog,
)


class FilterBar(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        self.event_types = QtWidgets.QListWidget()
        self.event_types.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        self.event_types.setMaximumHeight(80)
        self.animal = QtWidgets.QComboBox()
        self.animal.setEditable(True)
        self.block = QtWidgets.QComboBox()
        self.block.setEditable(True)
        self.date_from = QtWidgets.QLineEdit()
        self.date_to = QtWidgets.QLineEdit()
        self.date_from.setPlaceholderText("yyyy_mm_dd")
        self.date_to.setPlaceholderText("yyyy_mm_dd")
        self.ms_min = QtWidgets.QDoubleSpinBox()
        self.ms_max = QtWidgets.QDoubleSpinBox()
        self.ms_min.setRange(0, 1e9)
        self.ms_max.setRange(0, 1e9)
        self.ms_max.setValue(1e9)
        self.note = QtWidgets.QLineEdit()
        self.chk_le = QtWidgets.QCheckBox("Has L eye")
        self.chk_re = QtWidgets.QCheckBox("Has R eye")
        self.chk_oe = QtWidgets.QCheckBox("Has OE")
        form.addRow("Event types", self.event_types)
        form.addRow("Animal", self.animal)
        form.addRow("Block", self.block)
        form.addRow("Date from", self.date_from)
        form.addRow("Date to", self.date_to)
        form.addRow("Timepoint ≥ (ms)", self.ms_min)
        form.addRow("Timepoint ≤ (ms)", self.ms_max)
        form.addRow("Note contains", self.note)
        form.addRow(self.chk_le)
        form.addRow(self.chk_re)
        form.addRow(self.chk_oe)

    def apply_to_proxy(self, proxy) -> None:
        types = {i.text() for i in self.event_types.selectedItems()}
        animal = self.animal.currentText().strip()
        block = self.block.currentText().strip()
        proxy.set_filters(
            event_types=types if types else None,
            animals={animal} if animal else None,
            blocks={block} if block else None,
            date_from=self.date_from.text().strip() or None,
            date_to=self.date_to.text().strip() or None,
            ms_min=float(self.ms_min.value()) if self.ms_min.value() > 0 else None,
            ms_max=float(self.ms_max.value())
            if self.ms_max.value() < 1e9
            else None,
            note_text=self.note.text(),
            require_le=self.chk_le.isChecked(),
            require_re=self.chk_re.isChecked(),
            require_oe=self.chk_oe.isChecked(),
        )


class EventExplorerWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        initial_paths: list[Path] | None = None,
        log_file: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"PETS Event Explorer v{EXPLORER_VERSION}")
        self._catalog: list[EventRecord] = []
        self._sources: dict = {"json_paths": [], "scan_dirs": []}
        self._session_path: Path | None = None
        self._dirty = False
        self._half_window_ms = 100.0

        self._load_log = LoadLog(log_file or default_log_path())
        self._block_mgr = BlockDataManager(self._load_log, parent_widget=self)
        self._preview_worker: _PreviewWorker | None = None

        self._build_ui()
        self._build_menus()
        self._wire()

        if initial_paths:
            self._ingest_paths(initial_paths)
        else:
            self._show_empty_state()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        self._empty_label = QtWidgets.QLabel(
            "No annotations loaded.\nUse File → Add sources… or the button below."
        )
        self._empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._add_btn = QtWidgets.QPushButton("Add sources…")
        left.addWidget(self._empty_label)
        left.addWidget(self._add_btn)

        self._filter_bar = FilterBar()
        self._filter_bar.hide()
        left.addWidget(self._filter_bar)

        self._table = EventTableWidget()
        self._table.hide()
        left.addWidget(self._table, stretch=1)

        self._plot = PlotPanel()
        main_layout.addLayout(left, stretch=2)
        main_layout.addWidget(self._plot, stretch=3)

        log_dock = QtWidgets.QDockWidget("Load log", self)
        self._log_panel = LoadLogPanel(self._load_log)
        log_dock.setWidget(self._log_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        file_m = mb.addMenu("&File")
        file_m.addAction("Add sources…", self._add_sources)
        file_m.addAction("Preview selection", self._preview_selection)
        file_m.addSeparator()
        file_m.addAction("Export selection…", self._export_selection)
        file_m.addSeparator()
        file_m.addAction("Save session…", self._save_session)
        file_m.addAction("Load session…", self._load_session_dialog)
        file_m.addSeparator()
        file_m.addAction("E&xit", self.close)

        view_m = mb.addMenu("&View")
        view_m.addAction("Refresh preview", self._refresh_preview)
        view_m.addAction("Column visibility…", self._column_visibility_menu)

        help_m = mb.addMenu("&Help")
        help_m.addAction("About", self._about)

    def _wire(self) -> None:
        self._add_btn.clicked.connect(self._add_sources)
        self._table.selection_changed.connect(self._on_selection_changed)
        self._plot.preview_requested.connect(self._refresh_preview)
        self._plot._half_window.valueChanged.connect(
            lambda: (self._mark_dirty(), self._preview_selection())
        )
        for w in (
            self._plot._chk_lpupil,
            self._plot._chk_rpupil,
            self._plot._chk_ldeg,
            self._plot._chk_rdeg,
            self._plot._chk_ep,
        ):
            w.toggled.connect(lambda: (self._mark_dirty(), self._preview_selection()))
        self._plot._norm_combo.currentIndexChanged.connect(
            lambda: (self._mark_dirty(), self._preview_selection())
        )
        for rb in (
            self._plot._avg_lpupil,
            self._plot._avg_rpupil,
            self._plot._avg_ldeg,
            self._plot._avg_rdeg,
            self._plot._avg_ep,
        ):
            rb.toggled.connect(lambda: (self._mark_dirty(), self._preview_selection()))

        fb = self._filter_bar
        for w in (
            fb.animal,
            fb.block,
            fb.date_from,
            fb.date_to,
            fb.note,
            fb.chk_le,
            fb.chk_re,
            fb.chk_oe,
            fb.ms_min,
            fb.ms_max,
        ):
            if isinstance(w, QtWidgets.QLineEdit):
                w.textChanged.connect(self._apply_filters)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentTextChanged.connect(self._apply_filters)
            elif isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(self._apply_filters)
            elif isinstance(w, QtWidgets.QDoubleSpinBox):
                w.valueChanged.connect(self._apply_filters)
        fb.event_types.itemSelectionChanged.connect(self._apply_filters)

    def _show_empty_state(self) -> None:
        self._empty_label.show()
        self._add_btn.show()
        self._table.hide()
        self._filter_bar.hide()

    def _show_catalog_ui(self) -> None:
        self._empty_label.hide()
        self._table.show()
        self._filter_bar.show()

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _ingest_paths(self, paths: list[Path]) -> None:
        new_records = build_catalog(paths)
        if not new_records:
            QtWidgets.QMessageBox.warning(
                self, "Catalog", "No events found in selected annotation files."
            )
            return
        self._catalog.extend(new_records)
        for p in paths:
            sp = str(p.resolve())
            if sp not in self._sources["json_paths"]:
                self._sources["json_paths"].append(sp)
        self._refresh_catalog_ui()
        self._load_log.info(f"Loaded {len(new_records)} events from {len(paths)} file(s)")
        self._mark_dirty()

    def _refresh_catalog_ui(self) -> None:
        if not self._catalog:
            self._show_empty_state()
            return
        self._show_catalog_ui()
        self._block_mgr.update_record_statuses(self._catalog)
        self._table.set_records(self._catalog)
        self._populate_filter_widgets()
        self._apply_filters()

    def _populate_filter_widgets(self) -> None:
        fb = self._filter_bar
        fb.event_types.clear()
        for t in event_types_in_catalog(self._catalog):
            fb.event_types.addItem(t)
        animals = sorted({r.animal_call for r in self._catalog})
        blocks = sorted({r.block_num for r in self._catalog})
        fb.animal.clear()
        fb.animal.addItem("")
        fb.animal.addItems(animals)
        fb.block.clear()
        fb.block.addItem("")
        fb.block.addItems(blocks)

    def _apply_filters(self) -> None:
        self._filter_bar.apply_to_proxy(self._table.filter_proxy())
        self._mark_dirty()

    def _add_sources(self) -> None:
        dlg = AddSourcesDialog(self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            self._ingest_paths(dlg.discovered_paths)

    def _column_visibility_menu(self) -> None:
        header = self._table.table_view().horizontalHeader()
        self._table.show_column_menu(header.mapToGlobal(header.rect().bottomLeft()))

    def _on_selection_changed(self, records: list[EventRecord]) -> None:
        self._plot.set_selection_mode(len(records))
        if records:
            self._preview_selection()

    def _collect_snippets_single(self, record: EventRecord) -> list:
        cache = self._block_mgr.ensure_block(record)
        if cache is None or cache.skipped:
            return []
        half = self._plot.half_window_ms()
        snippets = []
        for sid in self._plot.enabled_single_streams():
            if sid == STREAM_EP:
                for ch in self._plot.oe_channel_list():
                    sn = extract_ep_snippet(record, cache, ch, half, self._load_log)
                    if sn:
                        snippets.append(sn)
            else:
                sn = extract_eye_snippet(record, cache, sid, half, self._load_log)
                if sn:
                    snippets.append(sn)
        return snippets

    def _collect_snippets_average(self, records: list[EventRecord]) -> list:
        sid = self._plot.average_stream()
        half = self._plot.half_window_ms()
        snippets = []
        for rec in records:
            cache = self._block_mgr.ensure_block(rec)
            if cache is None or cache.skipped:
                self._load_log.warn(
                    f"Skipping trial {rec.event_id} (block unavailable)"
                )
                continue
            self._load_log.info(
                f"Average trial {rec.event_id} block={cache.block_path.name}"
            )
            if sid == STREAM_EP:
                ch = self._plot.oe_channel_list()[0]
                sn = extract_ep_snippet(rec, cache, ch, half, self._load_log)
            else:
                sn = extract_eye_snippet(rec, cache, sid, half, self._load_log)
            if sn:
                snippets.append(sn)
        return snippets

    def _refresh_preview(self) -> None:
        self._block_mgr.clear_cache()
        self._block_mgr.update_record_statuses(self._catalog)
        self._table.set_records(self._catalog)
        self._load_log.info("Block cache cleared; refreshing preview")
        self._preview_selection()

    def _preview_selection(self) -> None:
        records = self._table.selected_records()
        if not records:
            self._plot.clear_plot()
            self._plot._set_status_ready("No events selected")
            return

        if self._preview_worker is not None and self._preview_worker.isRunning():
            return

        self._plot._set_status_loading(
            f"Loading {len(records)} event(s)…"
        )
        # Block load + dialogs must run on the UI thread before background extract.
        for rec in records:
            self._block_mgr.ensure_block(rec)
        self._block_mgr.update_record_statuses(self._catalog)
        self._table.set_records(self._catalog)

        def work():
            norm = self._plot.normalization_mode()
            if len(records) == 1:
                snippets = self._collect_snippets_single(records[0])
                return ("single", snippets, norm)
            snippets = self._collect_snippets_average(records)
            return ("average", snippets, norm)

        self._preview_worker = _PreviewWorker(work, self)
        self._preview_worker.finished_ok.connect(self._on_preview_finished)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.start()

    def _on_preview_finished(self, payload) -> None:
        self._block_mgr.update_record_statuses(self._catalog)
        self._table.set_records(self._catalog)
        mode, snippets, norm = payload
        if mode == "single":
            self._plot.plot_single(snippets, norm)
        else:
            self._plot.plot_average(snippets, norm)

    def _on_preview_failed(self, message: str) -> None:
        self._load_log.warn(f"Preview failed: {message}")
        self._plot._set_status_error(f"Preview failed: {message}")

    def _export_selection(self) -> None:
        records = self._table.selected_records()
        if not records:
            QtWidgets.QMessageBox.information(self, "Export", "Select one or more events.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export folder")
        if not folder:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npz_name, json_name = f"explorer_export_{stamp}.npz", f"explorer_export_{stamp}.json"

        if len(records) == 1:
            snippets = self._collect_snippets_single(records[0])
            mode = "single"
            stream = ",".join(self._plot.enabled_single_streams()) or "none"
            norm = self._plot.normalization_mode()
            grid = mean = sem = trials = None
            if snippets:
                grid = snippets[0].time_rel_ms
        else:
            snippets = self._collect_snippets_average(records)
            mode = "average"
            stream = self._plot.average_stream()
            norm = self._plot.normalization_mode()
            grid = mean = sem = trials = None
            if snippets:
                grid, trials, mean, sem = self._plot.last_average_arrays(snippets, norm)

        export_bundle(
            Path(folder),
            npz_name=npz_name,
            json_name=json_name,
            mode=mode,
            stream_name=stream,
            half_window_ms=self._plot.half_window_ms(),
            normalization=norm,
            records=records,
            snippets=snippets,
            time_grid=grid,
            trials=trials,
            mean=mean,
            sem=sem,
            load_log=self._load_log,
            oe_channels=self._plot.oe_channel_list(),
        )
        QtWidgets.QMessageBox.information(
            self, "Export", f"Wrote {npz_name} and {json_name} to {folder}"
        )

    def _session_dict(self) -> dict:
        return session_to_dict(
            sources=self._sources,
            remap_table=self._block_mgr.remap_table,
            visible_columns=self._table._model.visible_column_keys(),
            filters={},
            half_window_ms=self._plot.half_window_ms(),
            stream_toggles={
                "l_pupil": self._plot._chk_lpupil.isChecked(),
                "r_pupil": self._plot._chk_rpupil.isChecked(),
                "l_deg": self._plot._chk_ldeg.isChecked(),
                "r_deg": self._plot._chk_rdeg.isChecked(),
                "ep": self._plot._chk_ep.isChecked(),
            },
            oe_channels=self._plot.oe_channel_list(),
            normalization=self._plot.normalization_mode(),
            average_stream=self._plot.average_stream(),
            column_overrides=self._block_mgr._column_overrides,
            log_file=str(self._load_log.log_file) if self._load_log.log_file else None,
        )

    def _save_session(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session", filter="Explorer session (*.explorer_session.json)"
        )
        if not path:
            return
        if not path.endswith(".explorer_session.json"):
            path += ".explorer_session.json"
        self._session_path = Path(path)
        log_dir = self._session_path.parent
        if self._load_log.log_file is None:
            self._load_log.set_log_file(default_log_path(log_dir))
        save_session(self._session_path, self._session_dict())
        self._dirty = False
        QtWidgets.QMessageBox.information(self, "Session", f"Saved {path}")

    def _load_session_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load session", filter="Explorer session (*.explorer_session.json)"
        )
        if path:
            self._restore_session(Path(path))

    def _restore_session(self, path: Path) -> None:
        data = load_session(path)
        self._session_path = path
        if data.get("log_file"):
            self._load_log.set_log_file(Path(data["log_file"]))
        self._catalog.clear()
        sources = data.get("sources") or {}
        paths = discover_annotation_files(
            json_paths=[Path(p) for p in sources.get("json_paths", [])],
            scan_dirs=[Path(p) for p in sources.get("scan_dirs", [])],
        )
        self._sources = sources
        self._catalog = build_catalog(paths) if paths else []
        self._block_mgr.set_remap_table(data.get("remap_table") or {})
        self._block_mgr.set_column_overrides(column_map_from_session(data))
        vis = data.get("visible_columns")
        if vis:
            self._table._model.set_visible_columns(vis)
        toggles = data.get("stream_toggles") or {}
        pupil_legacy = toggles.get("pupil", None)
        self._plot._chk_lpupil.setChecked(
            toggles.get("l_pupil", pupil_legacy if pupil_legacy is not None else True)
        )
        self._plot._chk_rpupil.setChecked(
            toggles.get("r_pupil", pupil_legacy if pupil_legacy is not None else True)
        )
        self._plot._chk_ldeg.setChecked(toggles.get("l_deg", True))
        self._plot._chk_rdeg.setChecked(toggles.get("r_deg", True))
        self._plot._chk_ep.setChecked(toggles.get("ep", False))
        chans = data.get("oe_channels") or [1]
        self._plot._oe_channels.setText(",".join(str(c) for c in chans))
        self._plot._half_window.setValue(float(data.get("half_window_ms", 100)))
        self._refresh_catalog_ui()
        self._dirty = False
        self._load_log.info(f"Restored session from {path}")

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "Event Explorer",
            f"PETS Event Explorer v{EXPLORER_VERSION}\n"
            "Browse Block Annotator events and time-aligned traces.",
        )

    def closeEvent(self, event) -> None:
        if not self._dirty:
            event.accept()
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Exit")
        box.setText("Save session before exit?")
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        ans = box.exec()
        if ans == QtWidgets.QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if ans == QtWidgets.QMessageBox.StandardButton.Yes:
            self._save_session()
        event.accept()

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PETS Event Explorer — browse annotated events and traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eye_tracking_system_tools.annotation.event_explorer
  python -m eye_tracking_system_tools.annotation.event_explorer --json path/to/block_annotations.json
  python -m eye_tracking_system_tools.annotation.event_explorer --scan-dir ./_annotator_out
        """,
    )
    parser.add_argument(
        "--json",
        type=Path,
        action="append",
        dest="json_paths",
        help="Annotation JSON file (repeatable)",
    )
    parser.add_argument(
        "--json-list",
        type=Path,
        help="Text file with one annotation JSON path per line",
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        action="append",
        dest="scan_dirs",
        help="Directory to scan for *_annotations.json (repeatable, recursive)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)

    paths = discover_annotation_files(
        json_paths=args.json_paths,
        json_list_file=args.json_list,
        scan_dirs=args.scan_dirs,
    )

    win = EventExplorerWindow(initial_paths=paths or None)
    win.resize(1400, 900)
    win.show()

    if not paths:
        QtCore.QTimer.singleShot(0, win._add_sources)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
