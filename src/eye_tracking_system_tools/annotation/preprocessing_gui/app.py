"""Preprocessing GUI main window and application entry.

Phase 0 ships:

* MainWindow with a tabbed dashboard (Sync / Verify / Kerr / Behavior / Sync-free / Data Exploration).
* StartupDialog asking for experiment path, animal, block(s), output folder.
* argparse + env-var support so the dialog can be skipped.
* Per-tab status-icon plumbing via :class:`StatusBus`.

Later phases (1-8) fill in the tab contents.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.block_picker import (
    BlockPicker,
    discover_blocks,
    infer_fields_from_block_folder,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.add_blocks_dialog import (
    AddBlocksDialog,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    PreprocConfig,
    ensure_config_template,
    load_config,
    save_config,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
    StageStatus,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.status_bus import StatusBus
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs import (
    BehaviorTab,
    ExploreTab,
    KerrTab,
    SyncFreeTab,
    SyncTab,
    VerifyTab,
)


_TAB_CLASSES = (SyncTab, VerifyTab, KerrTab, BehaviorTab, SyncFreeTab, ExploreTab)


_STATUS_DOTS = {
    StageStatus.NOT_STARTED: ("\u25cb", "#888888", "not started"),
    StageStatus.PARTIAL: ("\u25d0", "#d9a407", "partial"),
    StageStatus.COMPLETE: ("\u25cf", "#2e8b57", "complete"),
    StageStatus.STALE: ("\u25cf", "#c0392b", "stale (upstream newer)"),
}


def _make_status_icon(status: StageStatus) -> QtGui.QIcon:
    """Build a tiny coloured circle icon for a tab status."""
    glyph, color, _ = _STATUS_DOTS[status]
    pix = QtGui.QPixmap(16, 16)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QtGui.QBrush(QtGui.QColor(color)))
    painter.setPen(QtGui.QPen(QtGui.QColor(color)))
    painter.drawEllipse(2, 2, 12, 12)
    painter.end()
    return QtGui.QIcon(pix)


class StartupDialog(QtWidgets.QDialog):
    """Pick experiment path, animal, block(s), output folder."""

    def __init__(self, defaults: PreprocConfig | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preprocessing GUI - Setup")
        self.experiment_path: Path | None = None
        self.animal: str | None = None
        self.blocks: list[str] = []
        self.output_folder: Path | None = None

        defaults = defaults or PreprocConfig()
        layout = QtWidgets.QFormLayout(self)

        self._exp_edit = QtWidgets.QLineEdit(defaults.last_experiment_path or "")
        exp_btn = QtWidgets.QPushButton("Browse...")
        block_btn = QtWidgets.QPushButton("Pick block folder...")
        exp_row = QtWidgets.QHBoxLayout()
        exp_row.addWidget(self._exp_edit)
        exp_row.addWidget(exp_btn)
        exp_row.addWidget(block_btn)
        layout.addRow("Experiment path:", exp_row)

        self._animal_edit = QtWidgets.QLineEdit(defaults.last_animal or "")
        layout.addRow("Animal:", self._animal_edit)

        self._blocks_edit = QtWidgets.QLineEdit(
            ",".join(defaults.last_blocks) if defaults.last_blocks else ""
        )
        self._blocks_edit.setPlaceholderText("Comma-separated, e.g. 015, 016, 017")
        layout.addRow("Block(s):", self._blocks_edit)

        self._out_edit = QtWidgets.QLineEdit()
        out_btn = QtWidgets.QPushButton("Browse...")
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self._out_edit)
        out_row.addWidget(out_btn)
        layout.addRow("Output folder:", out_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addRow(buttons)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        exp_btn.clicked.connect(self._pick_experiment)
        block_btn.clicked.connect(self._pick_block_folder)
        out_btn.clicked.connect(self._pick_output)

    def _pick_experiment(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Experiment path")
        if path:
            self._exp_edit.setText(path)

    def _pick_block_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select block folder")
        if not path:
            return
        inferred = infer_fields_from_block_folder(Path(path))
        if inferred is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Setup",
                "Could not infer experiment/animal/block from this folder.\n"
                "Expected a path like .../<animal>/<date>/block_NNN (or .../<animal>/block_NNN).",
            )
            return
        experiment_path, animal, block = inferred
        self._exp_edit.setText(str(experiment_path))
        self._animal_edit.setText(animal)
        self._blocks_edit.setText(block)

    def _pick_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder")
        if path:
            self._out_edit.setText(path)

    def _accept(self) -> None:
        exp = self._exp_edit.text().strip()
        if not exp or not Path(exp).is_dir():
            QtWidgets.QMessageBox.warning(
                self, "Setup", "A valid experiment path is required."
            )
            return
        animal = self._animal_edit.text().strip()
        if not animal:
            QtWidgets.QMessageBox.warning(self, "Setup", "Animal is required.")
            return
        raw_blocks = self._blocks_edit.text().strip()
        blocks = [b.strip() for b in raw_blocks.split(",") if b.strip()]
        if not blocks:
            QtWidgets.QMessageBox.warning(
                self, "Setup", "At least one block number is required."
            )
            return
        out = self._out_edit.text().strip()
        if not out:
            QtWidgets.QMessageBox.warning(
                self, "Setup", "An output folder is required."
            )
            return

        self.experiment_path = Path(exp)
        self.animal = animal
        self.blocks = blocks
        self.output_folder = Path(out)
        self.accept()


class PreprocessingGuiWindow(QtWidgets.QMainWindow):
    """Main window: tabbed dashboard with one tab per preprocessing stage."""

    def __init__(
        self,
        state: GuiState,
        config: PreprocConfig,
        config_path: Path,
    ):
        super().__init__()
        self.setWindowTitle("PETS Preprocessing GUI")
        self._state = state
        self._config = config
        self._config_path = config_path
        self._status_bus = StatusBus(self)
        self._tabs: dict[str, QtWidgets.QWidget] = {}

        self._build_ui()
        self._wire_signals()
        self._populate_block_picker()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self._block_picker = BlockPicker()
        root.addWidget(self._block_picker)

        self._tab_widget = QtWidgets.QTabWidget()
        root.addWidget(self._tab_widget, stretch=1)

        for cls in _TAB_CLASSES:
            tab = cls(self._state, self._config)
            self._tabs[tab.tab_id] = tab
            idx = self._tab_widget.addTab(tab, tab.tab_label)
            self._tab_widget.setTabIcon(idx, _make_status_icon(StageStatus.NOT_STARTED))
            self._status_bus.register_tab(
                tab.tab_id,
                tab.status_signature,
                upstream_tabs=_upstream_tabs(tab.tab_id),
            )

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Add block from folder…", self._open_block_folder)
        file_menu.addAction("Reload current block", self._reload_block)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("User guide...", self._show_help_guide)
        help_menu.addAction("About...", self._show_about)

        self.statusBar().showMessage("Ready")

    def _wire_signals(self) -> None:
        self._block_picker.block_changed.connect(self._on_block_changed)
        self._block_picker.reload_button().clicked.connect(self._reload_block)
        self._block_picker.add_blocks_requested.connect(self._add_blocks)
        self._block_picker.release_block_requested.connect(self._release_current_block)
        self._status_bus.status_changed.connect(self._on_status_changed)
        self._status_bus.status_changed.connect(self._on_any_status_refresh)
        self._busy_timer = QtCore.QTimer(self)
        self._busy_timer.timeout.connect(self._update_block_picker_busy)
        self._busy_timer.start(400)

    def _update_block_picker_busy(self) -> None:
        self._block_picker.set_session_busy(self._session_busy())

    def _populate_block_picker(self) -> None:
        self._block_picker.set_blocks(self._state.blocks, self._state.current_index)

    def _on_any_status_refresh(self, _tab_id: str, _status: StageStatus) -> None:
        for tab in self._tabs.values():
            if hasattr(tab, "on_filesystem_changed"):
                tab.on_filesystem_changed()

    def _session_busy(self) -> bool:
        for tab in self._tabs.values():
            worker = getattr(tab, "_worker", None)
            if worker is not None and worker.isRunning():
                return True
            batch = getattr(tab, "_batch_worker", None)
            if batch is not None and batch.isRunning():
                return True
        return False

    def _on_block_changed(self, index: int) -> None:
        if 0 <= index < len(self._state.blocks):
            self._state.current_index = index
        self._state.clear_cached_artefacts()
        block = self._state.current_block
        for tab in self._tabs.values():
            try:
                tab.set_block(block)
            except Exception as e:
                self.statusBar().showMessage(f"Error in tab {tab.tab_id}: {e}", 8000)
        self._status_bus.set_block(block)
        if block is None:
            self.statusBar().showMessage("No block loaded.")
        else:
            self.statusBar().showMessage(f"Active block: {block.display_label}")

    def _reload_block(self) -> None:
        block = self._state.current_block
        if block is not None:
            self._state.ensure_session().invalidate(block)
        self._on_block_changed(self._block_picker.current_index())

    def _add_blocks(self) -> None:
        if self._session_busy():
            QtWidgets.QMessageBox.warning(
                self,
                "Add block",
                "Wait for the current background job to finish.",
            )
            return
        animal = None
        if self._state.blocks:
            animal = self._state.blocks[0].animal_call
        loaded = {b.block_path for b in self._state.blocks}
        dlg = AddBlocksDialog(
            experiment_path=self._state.experiment_path,
            animal=animal,
            loaded_paths=loaded,
            parent=self,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_handles = dlg.selected_handles()
        if not new_handles:
            return
        if self._state.experiment_path is None and new_handles:
            self._state.experiment_path = new_handles[0].path_to_animal_folder
        added = self._state.add_blocks(new_handles)
        if added == 0:
            QtWidgets.QMessageBox.information(
                self, "Add block", "No new blocks were added (already in session)."
            )
            return
        first_new_idx = len(self._state.blocks) - added
        self._populate_block_picker()
        self._block_picker.set_blocks(self._state.blocks, first_new_idx)
        self.statusBar().showMessage(f"Added {added} block(s) to session.")

    def _release_current_block(self) -> None:
        if self._session_busy():
            QtWidgets.QMessageBox.warning(
                self,
                "Release block",
                "Wait for the current background or batch job to finish.",
            )
            return
        index = self._block_picker.current_index()
        if not (0 <= index < len(self._state.blocks)):
            return
        block = self._state.blocks[index]
        reply = QtWidgets.QMessageBox.question(
            self,
            "Release block",
            f"Release {block.display_label} from this session?\n\n"
            "On-disk analysis files are kept; only in-memory state is freed.",
            QtWidgets.QMessageBox.StandardButton.Ok
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Ok:
            return
        self._state.ensure_session().release(block)
        self._state.remove_block_at(index)
        self._populate_block_picker()
        new_index = min(index, max(0, len(self._state.blocks) - 1))
        if self._state.blocks:
            self._block_picker.set_blocks(self._state.blocks, new_index)
        else:
            self._block_picker.set_blocks([], 0)
            self._on_block_changed(-1)
        self.statusBar().showMessage(f"Released block {block.block_num} from session.")

    def _open_block_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select block folder")
        if not path:
            return
        inferred = infer_fields_from_block_folder(Path(path))
        if inferred is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Add block folder",
                "Could not infer experiment/animal/block from this folder.\n"
                "Expected .../<animal>/<date>/block_NNN or .../<animal>/block_NNN.",
            )
            return
        experiment_path, animal, block = inferred
        blocks = discover_blocks(experiment_path, animal, [block])
        if not blocks:
            QtWidgets.QMessageBox.warning(
                self,
                "Add block folder",
                f"No matching block discovered for inferred block id {block}.",
            )
            return
        if self._state.experiment_path is None:
            self._state.experiment_path = experiment_path
        added = self._state.add_blocks(blocks)
        if added == 0:
            QtWidgets.QMessageBox.information(
                self, "Add block folder", "That block is already in the session."
            )
            return
        self._populate_block_picker()
        idx = self._state.index_for_path(blocks[0].block_path)
        self._block_picker.set_blocks(
            self._state.blocks, idx if idx is not None else 0
        )
        self.statusBar().showMessage(f"Added block: {blocks[0].display_label}")

    def _on_status_changed(self, tab_id: str, status: StageStatus) -> None:
        tab = self._tabs.get(tab_id)
        if tab is None:
            return
        idx = self._tab_widget.indexOf(tab)
        if idx < 0:
            return
        icon = _make_status_icon(status)
        self._tab_widget.setTabIcon(idx, icon)
        self._tab_widget.setTabToolTip(
            idx,
            f"{tab.tab_label}: {_STATUS_DOTS[status][2]}",
        )

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About",
            "PETS Preprocessing GUI\n\n"
            "Wraps the preprocessing notebooks in a single PyQt6 dashboard.\n"
            "See development/plans/PREPROCESSING_GUI_AGENT_PLAN.md for the implementation plan.",
        )

    def _show_help_guide(self) -> None:
        text = (
            "<h3>Preprocessing GUI user guide</h3>"
            "<p>This app wraps notebook workflow stages into tabs. Work left-to-right: "
            "<b>Sync -> Verify -> Kerr -> Behavior -> Sync-free</b>. "
            "Tab status dots show stage state (grey=not started, yellow=partial, "
            "green=complete, red=stale).</p>"
            "<h4>Sync tab (Stage 1)</h4>"
            "<ul>"
            "<li><b>Prepare data (eye + arena)</b>: scans/validates eye and arena video metadata.</li>"
            "<li><b>Parse OE events</b>: loads or builds Open Ephys parsed event table.</li>"
            "<li><b>Extract brightness</b>: computes eye brightness vectors (auto-ROI first).</li>"
            "<li><b>Build arena grid</b>: creates the 60 Hz arena master timeline.</li>"
            "<li><b>Build simple sync</b>: maps eye frames to OE time and summarizes tick/fps.</li>"
            "<li><b>Open shift plot in browser</b>: opens the original Bokeh slider plot for manual alignment.</li>"
            "<li><b>Left/Right shift spinboxes</b>: type shifts found in browser; "
            "GUI values are inverse to Bokeh slider sign.</li>"
            "<li><b>Apply shifts</b>: applies index-based shift to frame_idx+brightness only.</li>"
            "<li><b>Frame insertion (advanced)</b>: manually insert duplicated frames by row position "
            "or OE sample if dropped frames need correction.</li>"
            "<li><b>Build final sync df</b>: merges both eyes onto arena grid using nearest-with-tolerance.</li>"
            "<li><b>Verify final df</b>: recomputes mapping checks and shows match percentages; "
            "native plot overlays left/right brightness plus LED events when available.</li>"
            "<li><b>Export final_sync_df.csv</b>: writes synchronized outputs to analysis folder.</li>"
            "</ul>"
            "<h4>Sync tab — step 5 (DLC + jitter + finalize)</h4>"
            "<ul>"
            "<li><b>Read DLC + fit ellipses</b>: reads DeepLabCut CSVs and builds le/re ellipse tables "
            "(requires exported final_sync_df.csv).</li>"
            "<li><b>Compute jitter report</b>: long-running drift analysis saved to analysis folder.</li>"
            "<li><b>Correct jitter & remove LED blinks</b>: applies jitter correction and LED-blink cleanup.</li>"
            "<li><b>Preview outliers (both eyes)</b>: plots top_correlation_dist with flagged peaks in native pyqtgraph.</li>"
            "<li><b>Apply removal (both eyes)</b>: NaNs outlier frames using the previewed indices.</li>"
            "<li><b>Finalize & export eye data</b>: writes left_eye_data.csv and right_eye_data.csv.</li>"
            "</ul>"
            "<h4>Batch (bottom of Sync tab)</h4>"
            "<p>When you launched the GUI with multiple blocks (e.g. <code>015,016,017</code>), "
            "use <b>Run for all blocks</b> to run a long step on every loaded block sequentially. "
            "Each block must already have the upstream outputs that step needs (e.g. DLC batch "
            "requires <code>final_sync_df.csv</code> per block). Use <b>Cancel batch</b> to stop "
            "after the current block finishes.</p>"
            "<h4>Other tabs (high-level)</h4>"
            "<ul>"
            "<li><b>Verify</b>: interactive ellipse verification and Kerr reference picking.</li>"
            "<li><b>Kerr</b>: calculates and exports Kerr angle CSV outputs.</li>"
            "<li><b>Behavior</b>: threshold-based behavioral state extraction from lizMov data.</li>"
            "<li><b>Sync-free</b>: optional alternative ellipse-to-Kerr mapping pipeline.</li>"
            "</ul>"
            "<p><b>Tip:</b> if a downstream tab is red (stale), re-run its upstream stage first.</p>"
        )
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Preprocessing GUI - Help")
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setTextFormat(QtCore.Qt.TextFormat.RichText)
        msg.setText(text)
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg.exec()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self._persist_defaults()
        except Exception:
            # Persistence is best-effort; never block close.
            pass
        super().closeEvent(event)

    def _persist_defaults(self) -> None:
        if self._state.experiment_path is not None:
            self._config.last_experiment_path = str(self._state.experiment_path)
        if self._state.blocks:
            animals = {b.animal_call for b in self._state.blocks}
            if len(animals) == 1:
                self._config.last_animal = next(iter(animals))
            self._config.last_blocks = sorted({b.block_num for b in self._state.blocks})
        behavior_tab = self._tabs.get("behavior")
        if behavior_tab is not None and hasattr(behavior_tab, "_persist_tab_config"):
            behavior_tab._persist_tab_config()
        save_config(self._config_path, self._config)


def _upstream_tabs(tab_id: str) -> list[str]:
    """Static dependency graph used by the status bus for STALE detection."""
    return {
        "sync": [],
        "verify": ["sync"],
        "kerr": ["sync", "verify"],
        "behavior": ["sync"],
        "syncfree": ["sync"],
        "explore": ["sync"],
    }.get(tab_id, [])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_exp = os.environ.get("PETS_PREPROC_EXPERIMENT_PATH")
    default_animal = os.environ.get("PETS_PREPROC_ANIMAL")
    default_blocks = os.environ.get("PETS_PREPROC_BLOCKS")
    default_output = os.environ.get("PETS_PREPROC_OUTPUT")

    parser = argparse.ArgumentParser(description="PETS Preprocessing GUI")
    parser.add_argument(
        "--experiment-path",
        type=Path,
        default=Path(default_exp) if default_exp else None,
        help="Experiment root containing <animal>/<yyyy_mm_dd>/block_NNN folders",
    )
    parser.add_argument(
        "--animal",
        type=str,
        default=default_animal,
        help="Animal subfolder name (e.g. PV_106)",
    )
    parser.add_argument(
        "--block",
        action="append",
        default=None,
        help=(
            "Block number to load (e.g. 015). May be passed multiple times "
            "for batch mode. Falls back to PETS_PREPROC_BLOCKS env var."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(default_output) if default_output else None,
        help="Output folder (holds preproc_gui_config.yaml + per-run logs)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit path to preproc_gui_config.yaml",
    )
    parser.add_argument(
        "--dialog",
        action="store_true",
        help="Show the startup dialog even if --experiment-path/--animal/--block are set",
    )
    args = parser.parse_args(argv)
    if args.block is None and default_blocks:
        args.block = [b.strip() for b in default_blocks.split(",") if b.strip()]
    return args


def _resolve_blocks_from_args(args: argparse.Namespace) -> list[BlockHandle]:
    if not args.experiment_path or not args.animal or not args.block:
        return []
    return discover_blocks(args.experiment_path, args.animal, args.block)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    output_folder: Path | None = args.output
    experiment_path: Path | None = args.experiment_path
    animal: str | None = args.animal
    blocks: list[str] | None = args.block

    need_dialog = (
        args.dialog
        or experiment_path is None
        or animal is None
        or not blocks
        or output_folder is None
    )
    if need_dialog:
        defaults = PreprocConfig()
        if output_folder is not None:
            try:
                defaults = load_config(args.config, output_folder)
            except Exception:
                pass
        if experiment_path is not None:
            defaults.last_experiment_path = str(experiment_path)
        if animal is not None:
            defaults.last_animal = animal
        if blocks:
            defaults.last_blocks = list(blocks)

        dlg = StartupDialog(defaults)
        if output_folder is not None:
            dlg._out_edit.setText(str(output_folder))
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return 0
        experiment_path = dlg.experiment_path
        animal = dlg.animal
        blocks = dlg.blocks
        output_folder = dlg.output_folder

    assert experiment_path is not None
    assert animal is not None
    assert blocks
    assert output_folder is not None

    ensure_config_template(output_folder)
    config = load_config(args.config, output_folder)
    config_path = args.config or (output_folder / "preproc_gui_config.yaml")

    discovered = discover_blocks(experiment_path, animal, blocks)

    state = GuiState(
        experiment_path=experiment_path,
        blocks=discovered,
        current_index=0,
        output_folder=output_folder,
    )
    state.ensure_session()

    win = PreprocessingGuiWindow(state, config, config_path)
    win.resize(1280, 860)
    win.show()

    if not discovered:
        QtWidgets.QMessageBox.warning(
            win,
            "No blocks found",
            f"No matching blocks found under {experiment_path / animal}.\n"
            f"Requested block ids: {blocks}",
        )

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
