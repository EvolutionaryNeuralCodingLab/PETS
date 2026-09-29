"""Block Annotator main window and application entry."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.annotation.block_annotator.annotation_panel import (
    AnnotationPanel,
)
from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
    find_blocks_with_sync,
    infer_metadata,
    load_block_session,
)
from eye_tracking_system_tools.annotation.block_annotator.config_io import (
    default_config_path,
    ensure_config_template,
    load_config,
    save_config,
)
from eye_tracking_system_tools.annotation.block_annotator.models import (
    BlockSession,
    first_timeline_index_with_frame,
)
from eye_tracking_system_tools.annotation.block_annotator.persistence import (
    load_annotations,
    save_annotations,
)
from eye_tracking_system_tools.annotation.block_annotator.playback_controller import (
    PlaybackController,
)
from eye_tracking_system_tools.annotation.block_annotator.trace_widget import TraceWidget
from eye_tracking_system_tools.annotation.block_annotator.video_widget import VideoPanel


def _ellipse_frame_col(df, preferred: str) -> str:
    if df is None:
        return preferred
    for name in (preferred, "eye_frame", "Eye_frame"):
        if name in df.columns:
            return name
    return preferred


class StartupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Block Annotator — Setup")
        self.output_folder: Path | None = None
        self.config_path: Path | None = None
        self.block_path: Path | None = None

        layout = QtWidgets.QFormLayout(self)
        self._out_edit = QtWidgets.QLineEdit()
        self._cfg_edit = QtWidgets.QLineEdit()
        self._block_edit = QtWidgets.QLineEdit()
        out_btn = QtWidgets.QPushButton("Browse…")
        cfg_btn = QtWidgets.QPushButton("Browse…")
        block_btn = QtWidgets.QPushButton("Browse…")

        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self._out_edit)
        out_row.addWidget(out_btn)
        cfg_row = QtWidgets.QHBoxLayout()
        cfg_row.addWidget(self._cfg_edit)
        cfg_row.addWidget(cfg_btn)
        block_row = QtWidgets.QHBoxLayout()
        block_row.addWidget(self._block_edit)
        block_row.addWidget(block_btn)

        layout.addRow("Output folder (required):", out_row)
        layout.addRow("Config YAML (optional):", cfg_row)
        layout.addRow("Block folder:", block_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addRow(buttons)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        out_btn.clicked.connect(self._pick_output)
        cfg_btn.clicked.connect(self._pick_config)
        block_btn.clicked.connect(self._pick_block)

    def _pick_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder")
        if path:
            self._out_edit.setText(path)
            tpl = Path(path) / "annotator_config.yaml"
            if not self._cfg_edit.text():
                self._cfg_edit.setPlaceholderText(str(tpl))

    def _pick_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Config file", filter="YAML (*.yaml *.yml)"
        )
        if path:
            self._cfg_edit.setText(path)

    def _pick_block(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Block folder")
        if path:
            self._block_edit.setText(path)

    def _accept(self) -> None:
        out = self._out_edit.text().strip()
        if not out:
            QtWidgets.QMessageBox.warning(self, "Setup", "Output folder is required.")
            return
        self.output_folder = Path(out)
        cfg = self._cfg_edit.text().strip()
        self.config_path = Path(cfg) if cfg else None
        block = self._block_edit.text().strip()
        if not block:
            QtWidgets.QMessageBox.warning(self, "Setup", "Block folder is required.")
            return
        self.block_path = Path(block)
        self.accept()


class BlockAnnotatorWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        output_folder: Path,
        config_path: Path | None,
        block_path: Path | None = None,
    ):
        super().__init__()
        self.setWindowTitle("PETS Block Annotator")
        self._output_folder = Path(output_folder)
        self._config_path = config_path
        self._session: BlockSession | None = None
        self._arena_index = 0

        ensure_config_template(self._output_folder)
        self._config = load_config(self._config_path, self._output_folder)
        if self._config_path is None:
            self._config_path = default_config_path(self._output_folder)

        self._playback = PlaybackController(self)
        self._build_ui()
        self._wire_signals()

        if block_path is not None:
            self._load_block_path(Path(block_path))
        else:
            self.statusBar().showMessage(
                "Use File → Open block to load data.", 0
            )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        transport_box = QtWidgets.QVBoxLayout()
        transport_row1 = QtWidgets.QHBoxLayout()
        transport_row2 = QtWidgets.QHBoxLayout()
        self._btn_play = QtWidgets.QPushButton("▶ Play")
        self._btn_reverse_play = QtWidgets.QPushButton("◀ Rev play")
        self._btn_step_back = QtWidgets.QPushButton("◀ 1")
        self._btn_step_fwd = QtWidgets.QPushButton("1 ▶")
        self._jump_step = QtWidgets.QSpinBox()
        self._jump_step.setRange(1, 1_000_000)
        self._jump_step.setValue(10)
        self._jump_step.setPrefix("jump ")
        self._btn_jump_back = QtWidgets.QPushButton("◀◀")
        self._btn_jump_fwd = QtWidgets.QPushButton("▶▶")
        self._speed = QtWidgets.QDoubleSpinBox()
        self._speed.setRange(0.25, 4.0)
        self._speed.setSingleStep(0.25)
        self._speed.setValue(1.0)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._hud = QtWidgets.QLabel("—")
        transport_row1.addWidget(self._btn_play)
        transport_row1.addWidget(self._btn_reverse_play)
        transport_row1.addWidget(self._btn_step_back)
        transport_row1.addWidget(self._btn_step_fwd)
        transport_row1.addWidget(QtWidgets.QLabel("Speed:"))
        transport_row1.addWidget(self._speed)
        transport_row1.addWidget(self._slider, stretch=1)
        transport_row1.addWidget(self._hud)
        transport_row2.addWidget(self._jump_step)
        transport_row2.addWidget(self._btn_jump_back)
        transport_row2.addWidget(self._btn_jump_fwd)
        transport_row2.addStretch(1)
        transport_box.addLayout(transport_row1)
        transport_box.addLayout(transport_row2)
        root.addLayout(transport_box)

        self._video_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )
        self._arena_panel = VideoPanel("Arena", min_width=320, min_height=240)
        self._left_panel = VideoPanel("Left eye", min_width=160, min_height=120)
        self._right_panel = VideoPanel("Right eye", min_width=160, min_height=120)
        self._video_splitter.addWidget(self._arena_panel)
        self._video_splitter.addWidget(self._left_panel)
        self._video_splitter.addWidget(self._right_panel)
        self._video_splitter.setStretchFactor(0, 3)
        self._video_splitter.setStretchFactor(1, 1)
        self._video_splitter.setStretchFactor(2, 1)
        self._video_splitter.setSizes([640, 280, 280])
        root.addWidget(self._video_splitter, stretch=2)

        controls = QtWidgets.QVBoxLayout()

        arena_row = QtWidgets.QHBoxLayout()
        self._arena_combo = QtWidgets.QComboBox()
        self._arena_max_w = QtWidgets.QSpinBox()
        self._arena_max_w.setRange(0, 4096)
        self._arena_max_w.setValue(960)
        self._arena_max_w.setSpecialValueText("auto")
        self._arena_max_w.setSuffix(" px")
        arena_row.addWidget(QtWidgets.QLabel("Arena video:"))
        arena_row.addWidget(self._arena_combo, stretch=1)
        arena_row.addWidget(QtWidgets.QLabel("max width"))
        arena_row.addWidget(self._arena_max_w)
        controls.addLayout(arena_row)

        eye_row = QtWidgets.QHBoxLayout()
        self._le_flip_h = QtWidgets.QCheckBox("L flip H")
        self._le_flip_v = QtWidgets.QCheckBox("L flip V")
        self._le_ann = QtWidgets.QCheckBox("L annotations")
        self._le_max_w = QtWidgets.QSpinBox()
        self._le_max_w.setRange(0, 2048)
        self._le_max_w.setValue(480)
        self._le_max_w.setSpecialValueText("auto")
        self._le_max_w.setSuffix(" px")
        self._re_flip_h = QtWidgets.QCheckBox("R flip H")
        self._re_flip_v = QtWidgets.QCheckBox("R flip V")
        self._re_ann = QtWidgets.QCheckBox("R annotations")
        self._re_max_w = QtWidgets.QSpinBox()
        self._re_max_w.setRange(0, 2048)
        self._re_max_w.setValue(480)
        self._re_max_w.setSpecialValueText("auto")
        self._re_max_w.setSuffix(" px")
        eye_row.addWidget(self._le_flip_h)
        eye_row.addWidget(self._le_flip_v)
        eye_row.addWidget(self._le_ann)
        eye_row.addWidget(QtWidgets.QLabel("L w"))
        eye_row.addWidget(self._le_max_w)
        eye_row.addSpacing(16)
        eye_row.addWidget(self._re_flip_h)
        eye_row.addWidget(self._re_flip_v)
        eye_row.addWidget(self._re_ann)
        eye_row.addWidget(QtWidgets.QLabel("R w"))
        eye_row.addWidget(self._re_max_w)
        eye_row.addStretch(1)
        controls.addLayout(eye_row)

        oe_row = QtWidgets.QHBoxLayout()
        self._show_oe_trace = QtWidgets.QCheckBox("Show OE trace (heavy — off by default)")
        self._show_oe_trace.setChecked(False)
        oe_row.addWidget(self._show_oe_trace)
        oe_row.addStretch(1)
        controls.addLayout(oe_row)

        root.addLayout(controls)

        self._trace = TraceWidget()
        self._trace.set_active(False)
        root.addWidget(self._trace, stretch=1)

        bottom = QtWidgets.QHBoxLayout()
        self._annotation = AnnotationPanel()
        bottom.addWidget(self._annotation, stretch=1)
        root.addLayout(bottom)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Open block…", self._menu_open_block)
        file_menu.addAction("Change output folder…", self._menu_output_folder)
        file_menu.addSeparator()
        file_menu.addAction("Add event type…", self._menu_add_event_type)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("Keyboard shortcuts…", self._show_keyboard_help)

        self.statusBar().showMessage("Ready")
        self._setup_shortcuts()

    def _wire_signals(self) -> None:
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_reverse_play.clicked.connect(self._toggle_reverse_play)
        self._btn_step_back.clicked.connect(self._step_back_one)
        self._btn_step_fwd.clicked.connect(self._step_forward_one)
        self._btn_jump_back.clicked.connect(self._step_back_jump)
        self._btn_jump_fwd.clicked.connect(self._step_forward_jump)
        self._speed.valueChanged.connect(self._playback.set_speed)
        self._slider.valueChanged.connect(self._on_slider)
        self._playback.index_changed.connect(self._on_index)
        self._playback.playing_changed.connect(self._on_playing_changed)
        self._arena_combo.currentIndexChanged.connect(self._on_arena_changed)
        for w in (
            self._le_flip_h,
            self._le_flip_v,
            self._le_ann,
            self._re_flip_h,
            self._re_flip_v,
            self._re_ann,
        ):
            w.toggled.connect(self._update_frame)
        for w in (self._arena_max_w, self._le_max_w, self._re_max_w):
            w.valueChanged.connect(self._apply_display_sizes)
        self._show_oe_trace.toggled.connect(self._on_oe_trace_toggled)
        self._annotation.seek_requested.connect(self._seek_ms)
        self._annotation.event_marked.connect(self._on_save_requested)
        self._apply_display_sizes()

    def _single_step_rows(self) -> int:
        return max(1, int(self._config.step_rows))

    def _jump_step_rows(self) -> int:
        return max(1, int(self._jump_step.value()))

    def _step_back_one(self) -> None:
        self._playback.step(-self._single_step_rows())

    def _step_forward_one(self) -> None:
        self._playback.step(self._single_step_rows())

    def _step_back_jump(self) -> None:
        self._playback.step(-self._jump_step_rows())

    def _step_forward_jump(self) -> None:
        self._playback.step(self._jump_step_rows())

    def _toggle_play(self) -> None:
        self._playback.toggle_play(forward=True)

    def _toggle_reverse_play(self) -> None:
        self._playback.toggle_play(forward=False)

    def _mark_event_shortcut(self) -> None:
        self._annotation._btn_mark.click()

    def _on_playing_changed(self, playing: bool) -> None:
        if playing and self._playback.forward:
            self._btn_play.setText("⏸ Pause")
            self._btn_reverse_play.setText("◀ Rev play")
        elif playing and not self._playback.forward:
            self._btn_play.setText("▶ Play")
            self._btn_reverse_play.setText("⏸ Rev pause")
        else:
            self._btn_play.setText("▶ Play")
            self._btn_reverse_play.setText("◀ Rev play")

    def _on_slider(self, value: int) -> None:
        self._playback.pause()
        self._playback.set_index(value)

    def _on_index(self, index: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._update_frame()
        self._annotation.set_timeline_index(index)

    def _seek_ms(self, ms: float) -> None:
        if self._session is None:
            return
        idx = int(np.searchsorted(self._session.ms_axis, ms, side="left"))
        idx = max(0, min(idx, self._session.n - 1))
        self._playback.pause()
        self._playback.set_index(idx)

    def _apply_display_sizes(self) -> None:
        self._arena_panel.set_display_limits(self._arena_max_w.value(), 0)
        self._left_panel.set_display_limits(self._le_max_w.value(), 0)
        self._right_panel.set_display_limits(self._re_max_w.value(), 0)

    def _apply_eye_display_options(self) -> None:
        self._left_panel.set_flip_horizontal(self._le_flip_h.isChecked())
        self._left_panel.set_flip_vertical(self._le_flip_v.isChecked())
        self._left_panel.set_show_annotations(self._le_ann.isChecked())
        self._right_panel.set_flip_horizontal(self._re_flip_h.isChecked())
        self._right_panel.set_flip_vertical(self._re_flip_v.isChecked())
        self._right_panel.set_show_annotations(self._re_ann.isChecked())

    def _on_oe_trace_toggled(self, checked: bool) -> None:
        self._trace.set_active(checked)
        if checked and self._session is not None:
            self._trace.set_recording(
                self._session.oe_rec,
                self._session.ms_axis,
                populate=True,
            )
            self._trace.set_playhead_ms(self._session.ms_at(self._playback.index))

    def _update_frame(self) -> None:
        if self._session is None:
            return
        i = self._playback.index
        arena_f, le_f, re_f = self._session.frame_ids_at(i)
        ms = self._session.ms_at(i)
        if self._show_oe_trace.isChecked():
            self._trace.set_playhead_ms(ms)
        hud = (
            f"i={i}/{self._session.n - 1} | {ms:.2f} ms | "
            f"A={arena_f} L={le_f} R={re_f}"
        )
        self._hud.setText(hud)
        self.statusBar().showMessage(
            f"{self._session.animal_call} block {self._session.block_num} | "
            f"{self._session.ms_min:.1f}–{self._session.ms_max:.1f} ms"
        )

        self._arena_panel.show_frame(arena_f, f"frame {arena_f}")
        self._apply_eye_display_options()
        self._left_panel.show_frame(le_f, f"frame {le_f}")
        self._right_panel.show_frame(re_f, f"frame {re_f}")

    def _current_arena_path(self) -> Path | None:
        if self._session is None or not self._session.arena_videos:
            return None
        idx = min(self._arena_index, len(self._session.arena_videos) - 1)
        return self._session.arena_videos[idx]

    def _on_arena_changed(self, index: int) -> None:
        self._arena_index = index
        if self._session and index < len(self._session.arena_videos):
            self._arena_panel.set_video(self._session.arena_videos[index])
            self._update_frame()

    def _load_block_path(self, block_path: Path) -> None:
        blocks = find_blocks_with_sync(block_path)
        if len(blocks) > 1:
            names = [str(b) for b in blocks]
            item, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Select block",
                "Multiple blocks found:",
                names,
                0,
                False,
            )
            if not ok:
                return
            block_path = Path(item)
        elif len(blocks) == 1:
            block_path = blocks[0]

        try:
            animal, exp_date, block_num, _ = infer_metadata(block_path)
        except ValueError:
            animal, ok = QtWidgets.QInputDialog.getText(
                self, "Metadata", "Animal call:"
            )
            if not ok or not animal:
                return
            block_num, ok = QtWidgets.QInputDialog.getText(
                self, "Metadata", "Block number:"
            )
            if not ok:
                return
            exp_date = None

        try:
            session = load_block_session(
                block_path,
                self._output_folder,
                self._config,
                animal_call=animal,
                experiment_date=exp_date,
                block_num=block_num,
            )
        except FileNotFoundError as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Load error", f"Failed to load block:\n{e}"
            )
            return

        if not session.arena_videos and not session.le_videos:
            QtWidgets.QMessageBox.warning(
                self,
                "Videos",
                "No arena or eye MP4 files found in this block.",
            )
        if session.oe_rec is None:
            QtWidgets.QMessageBox.information(
                self,
                "Open Ephys",
                "No OE recording found; trace panel will be empty.",
            )

        self._session = session
        events = load_annotations(self._output_folder, session)
        self._annotation.set_session(session, events)

        self._slider.setMaximum(max(0, session.n - 1))
        self._playback.set_fps(session.config.playback_fps)
        self._playback.set_timeline(session.n, session.ms_axis)
        self._trace.set_recording(
            session.oe_rec, session.ms_axis, populate=self._show_oe_trace.isChecked()
        )
        self._trace.set_active(self._show_oe_trace.isChecked())

        self._arena_combo.clear()
        for p in session.arena_videos:
            self._arena_combo.addItem(p.name, p)
        if session.arena_videos:
            self._arena_panel.set_video(session.arena_videos[0])
        if session.le_videos:
            self._left_panel.set_video(session.le_videos[0])
        if session.re_videos:
            self._right_panel.set_video(session.re_videos[0])
        self._left_panel.set_ellipse_data(
            session.le_ellipse_df, _ellipse_frame_col(session.le_ellipse_df, "L_eye_frame")
        )
        self._right_panel.set_ellipse_data(
            session.re_ellipse_df, _ellipse_frame_col(session.re_ellipse_df, "R_eye_frame")
        )

        start_i = first_timeline_index_with_frame(session.final_sync_df)
        self._playback.set_index(start_i)
        if start_i > 0:
            self.statusBar().showMessage(
                f"Timeline starts at row {start_i} (first row with a valid frame index).",
                10000,
            )
        self.setWindowTitle(
            f"PETS Block Annotator — {session.animal_call} block {session.block_num}"
        )

    def _on_save_requested(self, _=None) -> None:
        if self._session is None:
            return
        path = save_annotations(
            self._output_folder,
            self._session,
            self._annotation.events(),
        )
        QtWidgets.QMessageBox.information(
            self, "Saved", f"Annotations saved to:\n{path}"
        )

    def _menu_open_block(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Block folder")
        if path:
            self._load_block_path(Path(path))

    def _menu_output_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder")
        if path:
            self._output_folder = Path(path)
            ensure_config_template(self._output_folder)

    def _menu_add_event_type(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Add event type",
            "New event type name:",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            QtWidgets.QMessageBox.warning(
                self, "Add event type", "Name cannot be empty."
            )
            return
        if not self._config.add_event_type(name):
            QtWidgets.QMessageBox.warning(
                self,
                "Add event type",
                f'"{name}" already exists in the event type list.',
            )
            return
        save_config(self._config_path, self._config)
        self._annotation.reload_event_types(self._config.event_types, select=name)
        self.statusBar().showMessage(
            f'Added event type "{name}" and saved to {self._config_path.name}.',
            8000,
        )

    def _setup_shortcuts(self) -> None:
        def bind(key, slot):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(slot)

        bind(QtCore.Qt.Key.Key_Space, self._toggle_play)
        bind(QtCore.Qt.Key.Key_BracketLeft, self._step_back_one)
        bind(QtCore.Qt.Key.Key_BracketRight, self._step_forward_one)
        bind(QtCore.Qt.Key.Key_BraceLeft, self._step_back_jump)
        bind(QtCore.Qt.Key.Key_BraceRight, self._step_forward_jump)
        bind(QtCore.Qt.Key.Key_R, self._toggle_reverse_play)
        bind(QtCore.Qt.Key.Key_M, self._mark_event_shortcut)

    def _show_keyboard_help(self) -> None:
        jump = self._jump_step.value()
        text = f"""<h3>Block Annotator — keyboard shortcuts</h3>
<table>
<tr><td><b>Space</b></td><td>Play / pause (forward)</td></tr>
<tr><td><b>R</b></td><td>Reverse play / pause</td></tr>
<tr><td><b>[</b></td><td>Step back one row</td></tr>
<tr><td><b>]</b></td><td>Step forward one row</td></tr>
<tr><td><b>{{</b> (Shift+[)</td><td>Step back {jump} rows (jump size spinbox)</td></tr>
<tr><td><b>}}</b> (Shift+])</td><td>Step forward {jump} rows</td></tr>
<tr><td><b>M</b></td><td>Mark event (current type and range)</td></tr>
</table>
<p>Arrow keys are not bound; use <b>[</b> / <b>]</b> for single steps.</p>
"""
        QtWidgets.QMessageBox.information(self, "Keyboard shortcuts", text)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_Space:
            self._toggle_play()
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_BracketLeft:
            self._step_back_one()
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_BracketRight:
            self._step_forward_one()
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_BraceLeft:
            self._step_back_jump()
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_BraceRight:
            self._step_forward_jump()
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_R:
            self._toggle_reverse_play()
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_M:
            self._mark_event_shortcut()
            event.accept()
            return
        super().keyPressEvent(event)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_block = os.environ.get("PETS_ANNOTATOR_BLOCK")
    default_output = os.environ.get("PETS_ANNOTATOR_OUTPUT")
    parser = argparse.ArgumentParser(description="PETS Block Annotator")
    parser.add_argument(
        "--block",
        type=Path,
        default=Path(default_block) if default_block else None,
        help="Block folder with analysis/final_sync_df.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(default_output) if default_output else None,
        help="Annotation output folder",
    )
    parser.add_argument("--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument(
        "--dialog",
        action="store_true",
        help="Show startup dialog even when --block and --output are set",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    app = QtWidgets.QApplication(sys.argv)

    output_folder: Path | None = args.output
    config_path: Path | None = args.config
    block_path: Path | None = args.block

    if args.dialog or output_folder is None or block_path is None:
        dlg = StartupDialog()
        if output_folder is not None:
            dlg._out_edit.setText(str(output_folder))
        if config_path is not None:
            dlg._cfg_edit.setText(str(config_path))
        if block_path is not None:
            dlg._block_edit.setText(str(block_path))
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            sys.exit(0)
        output_folder = dlg.output_folder
        config_path = dlg.config_path
        block_path = dlg.block_path

    assert output_folder is not None and block_path is not None

    win = BlockAnnotatorWindow(output_folder, config_path, block_path)
    win.resize(1280, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
