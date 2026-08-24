"""Main window for event-centric saccade verification."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.analysis.block_registry import BlockSpec, load_registry
from eye_tracking_system_tools.analysis.param_tune import load_trace_bundle
from eye_tracking_system_tools.analysis.saccade_viewer.artifacts import (
    apply_saved_tags,
    save_verification_tags,
)
from eye_tracking_system_tools.analysis.saccade_viewer.event_clip import next_unset_index
from eye_tracking_system_tools.analysis.saccade_viewer.event_trace_panel import EventTracePanel
from eye_tracking_system_tools.analysis.saccade_viewer.event_video_panel import EventVideoPanel
from eye_tracking_system_tools.analysis.saccade_viewer.models import (
    EventBatch,
    PairingTag,
    VerificationEvent,
    VerificationStatus,
    count_by_status,
    normalize_event_batch,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_window import (
    VideoLoadWorker,
)


class SaccadeViewerWindow(QtWidgets.QMainWindow):
    """Verify a batch of events with synced video, traces, and good/bad tags."""

    closed = QtCore.pyqtSignal()
    def __init__(
        self,
        batch: EventBatch,
        *,
        pre_ms: float = 250.0,
        post_ms: float = 250.0,
        auto_advance: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._batch = batch
        self._pre_ms = float(pre_ms)
        self._post_ms = float(post_ms)
        self._auto_advance = bool(auto_advance)
        self._block_key = batch.block_keys[0] if batch.block_keys else ""
        self._event_index = 0
        self._block_events: list[VerificationEvent] = []
        self._session_cache: dict[str, object] = {}
        self._trace_cache: dict[str, tuple] = {}
        self._video_worker: VideoLoadWorker | None = None
        self._syncing_time = False

        self.setWindowTitle("Saccade Viewer — event verification")
        self.resize(1280, 760)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Top bar
        top = QtWidgets.QHBoxLayout()
        self._block_combo = QtWidgets.QComboBox()
        for key in batch.block_keys:
            self._block_combo.addItem(key, key)
        self._block_combo.currentIndexChanged.connect(self._on_block_changed)

        self._event_label = QtWidgets.QLabel("Event —")
        self._btn_prev = QtWidgets.QPushButton("◀ Prev")
        self._btn_next = QtWidgets.QPushButton("Next ▶")
        self._btn_prev.clicked.connect(lambda: self._step_event(-1))
        self._btn_next.clicked.connect(lambda: self._step_event(1))

        self._pre_spin = QtWidgets.QSpinBox()
        self._pre_spin.setRange(0, 5000)
        self._pre_spin.setValue(int(self._pre_ms))
        self._pre_spin.setSuffix(" ms")
        self._post_spin = QtWidgets.QSpinBox()
        self._post_spin.setRange(0, 5000)
        self._post_spin.setValue(int(self._post_ms))
        self._post_spin.setSuffix(" ms")
        self._pre_spin.valueChanged.connect(self._on_pre_post_changed)
        self._post_spin.valueChanged.connect(self._on_pre_post_changed)

        self._status_label = QtWidgets.QLabel("good 0 · bad 0 · unset 0")
        self._btn_help = QtWidgets.QPushButton("Help")
        self._btn_help.setToolTip("Keyboard shortcuts and tag file format")
        self._btn_help.clicked.connect(self._show_help_dialog)
        top.addWidget(QtWidgets.QLabel("Block:"))
        top.addWidget(self._block_combo, stretch=1)
        top.addWidget(self._event_label)
        top.addWidget(self._btn_prev)
        top.addWidget(self._btn_next)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Pre:"))
        top.addWidget(self._pre_spin)
        top.addWidget(QtWidgets.QLabel("Post:"))
        top.addWidget(self._post_spin)
        top.addStretch(1)
        top.addWidget(self._status_label)
        top.addWidget(self._btn_help)
        root.addLayout(top)

        # Main splitter: trace | video
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._trace = EventTracePanel()
        self._video = EventVideoPanel()
        splitter.addWidget(self._trace)
        splitter.addWidget(self._video)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, stretch=1)

        # Bottom: tags + pairing + comment + auto-advance
        bottom = QtWidgets.QVBoxLayout()

        tag_row = QtWidgets.QHBoxLayout()
        self._btn_good = QtWidgets.QPushButton("Good (G)")
        self._btn_bad = QtWidgets.QPushButton("Bad (B)")
        self._btn_clear = QtWidgets.QPushButton("Clear QC (U)")
        self._btn_good.clicked.connect(lambda: self._set_status("good"))
        self._btn_bad.clicked.connect(lambda: self._set_status("bad"))
        self._btn_clear.clicked.connect(lambda: self._set_status("unset"))
        tag_row.addWidget(self._btn_good)
        tag_row.addWidget(self._btn_bad)
        tag_row.addWidget(self._btn_clear)
        tag_row.addSpacing(16)
        self._btn_mono = QtWidgets.QPushButton("Monocular (M)")
        self._btn_conc = QtWidgets.QPushButton("Concurrent (C)")
        self._btn_clear_pair = QtWidgets.QPushButton("Clear pairing (P)")
        self._btn_mono.clicked.connect(lambda: self._set_pairing("monocular"))
        self._btn_conc.clicked.connect(lambda: self._set_pairing("concurrent"))
        self._btn_clear_pair.clicked.connect(lambda: self._set_pairing("unset"))
        tag_row.addWidget(self._btn_mono)
        tag_row.addWidget(self._btn_conc)
        tag_row.addWidget(self._btn_clear_pair)
        tag_row.addSpacing(16)
        self._auto_adv = QtWidgets.QCheckBox("Auto-advance after QC tag")
        self._auto_adv.setChecked(self._auto_advance)
        tag_row.addWidget(self._auto_adv)
        tag_row.addStretch(1)
        bottom.addLayout(tag_row)

        comment_row = QtWidgets.QHBoxLayout()
        comment_row.addWidget(QtWidgets.QLabel("Comment:"))
        self._comment = QtWidgets.QLineEdit()
        self._comment.setPlaceholderText("Free-text note (saved with tags)")
        self._comment.editingFinished.connect(self._save_comment)
        comment_row.addWidget(self._comment, stretch=1)
        bottom.addLayout(comment_row)

        root.addLayout(bottom)

        self._trace.set_pre_post_ms(self._pre_ms, self._post_ms)
        self._video.set_pre_post_ms(self._pre_ms, self._post_ms)
        self._trace.time_selected.connect(self._on_trace_time_selected)
        self._video.time_changed.connect(self._on_video_time_changed)

        self._setup_shortcuts()
        self._load_block(self._block_key)

    def _help_html(self) -> str:
        return """<h3>Saccade Viewer — keyboard shortcuts</h3>
<table>
<tr><td><b>← / →</b></td><td>Previous / next event</td></tr>
<tr><td><b>[ / ]</b></td><td>Step video ±1 sync row within clip</td></tr>
<tr><td><b>G</b></td><td>Mark good (QC)</td></tr>
<tr><td><b>B</b></td><td>Mark bad (QC)</td></tr>
<tr><td><b>U</b></td><td>Clear QC tag</td></tr>
<tr><td><b>M</b></td><td>Manual monocular pairing tag</td></tr>
<tr><td><b>C</b></td><td>Manual concurrent pairing tag</td></tr>
<tr><td><b>P</b></td><td>Clear manual pairing tag</td></tr>
<tr><td><b>N</b></td><td>Jump to next unset QC event in block</td></tr>
<tr><td><b>Space</b></td><td>Play / pause clip</td></tr>
<tr><td><b>Shift+Space</b></td><td>Play clip from the beginning</td></tr>
<tr><td><b>Home / End</b></td><td>Jump to clip start / end</td></tr>
</table>
<p>Tags autosave to <code>analysis/saccade_verification/tags.csv</code> per block.
Columns: <code>verification_status</code> (good/bad/unset),
<code>pairing_tag</code> (monocular/concurrent/unset), <code>notes</code>.</p>
"""

    def _show_help_dialog(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Saccade Viewer — keyboard shortcuts")
        dlg.resize(520, 420)
        layout = QtWidgets.QVBoxLayout(dlg)
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._help_html())
        layout.addWidget(browser)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn)
        layout.addLayout(row)
        dlg.exec()

    def _setup_shortcuts(self) -> None:
        def bind(key, slot):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(slot)

        bind(QtCore.Qt.Key.Key_G, lambda: self._set_status("good"))
        bind(QtCore.Qt.Key.Key_B, lambda: self._set_status("bad"))
        bind(QtCore.Qt.Key.Key_U, lambda: self._set_status("unset"))
        bind(QtCore.Qt.Key.Key_M, lambda: self._set_pairing("monocular"))
        bind(QtCore.Qt.Key.Key_C, lambda: self._set_pairing("concurrent"))
        bind(QtCore.Qt.Key.Key_P, lambda: self._set_pairing("unset"))
        bind(QtCore.Qt.Key.Key_N, self._jump_next_unset)
        bind(QtCore.Qt.Key.Key_Space, self._video.toggle_play)
        sc_replay = QtGui.QShortcut(QtGui.QKeySequence("Shift+Space"), self)
        sc_replay.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        sc_replay.activated.connect(self._video.play_from_start)
        bind(QtCore.Qt.Key.Key_Home, self._video.jump_clip_start)
        bind(QtCore.Qt.Key.Key_End, self._video.jump_clip_end)
        bind(QtCore.Qt.Key.Key_Left, lambda: self._step_event(-1))
        bind(QtCore.Qt.Key.Key_Right, lambda: self._step_event(1))
        bind(QtCore.Qt.Key.Key_BracketLeft, lambda: self._video.step_frame(-1))
        bind(QtCore.Qt.Key.Key_BracketRight, lambda: self._video.step_frame(1))

    def set_events(self, batch: EventBatch) -> None:
        """Phase 2 hook: replace the event batch without rebuilding the window."""
        self._batch = batch
        self._block_combo.blockSignals(True)
        self._block_combo.clear()
        for key in batch.block_keys:
            self._block_combo.addItem(key, key)
        self._block_combo.blockSignals(False)
        if batch.block_keys:
            self._load_block(batch.block_keys[0])

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._release_video()
        self.closed.emit()
        super().closeEvent(event)

    def _stop_video_worker(self) -> None:
        worker = self._video_worker
        if worker is None:
            return
        self._video_worker = None
        for sig in (worker.finished_ok, worker.failed):
            try:
                sig.disconnect()
            except TypeError:
                pass
        if worker.isRunning():
            worker.wait(5000)
        worker.deleteLater()

    def _release_video(self) -> None:
        self._video.panel._playback.pause()
        self._stop_video_worker()
        self._video.release_resources()

    def _on_block_changed(self, _index: int) -> None:
        key = self._block_combo.currentData()
        if key:
            self._load_block(str(key))

    def _load_block(self, block_key: str) -> None:
        self._block_key = block_key
        raw = self._batch.events_for_block(block_key)
        spec = self._batch.spec_for(block_key)
        if spec is None:
            return
        self._block_events = apply_saved_tags(raw, spec.block_path)
        self._event_index = 0
        self._refresh_counts()
        self._load_traces(spec)
        self._start_video_load(spec)
        self._show_current_event()

    def _load_traces(self, spec: BlockSpec) -> None:
        key = spec.block_key
        if key not in self._trace_cache:
            try:
                bundle = load_trace_bundle(spec, log=False)
                self._trace_cache[key] = (bundle.left, bundle.right)
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"Trace load warning: {exc}", 8000)
                self._trace_cache[key] = (None, None)
        left, right = self._trace_cache[key]
        self._trace.set_traces(left, right)

    def _start_video_load(self, spec: BlockSpec) -> None:
        self._release_video()
        progress = QtWidgets.QProgressDialog(
            f"Loading videos for {spec.block_key}…",
            None,
            0,
            0,
            self,
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()
        QtWidgets.QApplication.processEvents()

        block_path = spec.block_path
        out = spec.analysis_path

        def load_fn():
            from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
                load_block_session,
            )
            from eye_tracking_system_tools.annotation.block_annotator.models import (
                AnnotatorConfig,
            )

            return load_block_session(
                Path(block_path),
                Path(out),
                AnnotatorConfig(),
                animal_call=spec.animal,
                block_num=spec.block_num,
            )

        worker = VideoLoadWorker(load_fn, self)
        self._video_worker = worker

        def on_ok(session) -> None:
            progress.close()
            self._video_worker = None
            if self._block_key != spec.block_key:
                return
            self._session_cache[spec.block_key] = session
            self._video.bind_session(session)
            self._show_current_event()
            self.statusBar().showMessage(f"Loaded video for {spec.block_key}", 5000)

        def on_fail(msg: str) -> None:
            progress.close()
            self._video_worker = None
            QtWidgets.QMessageBox.warning(
                self,
                "Saccade Viewer",
                f"Could not load videos for {spec.block_key}:\n{msg}",
            )

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        worker.start()

    def _current_event(self) -> VerificationEvent | None:
        if not self._block_events:
            return None
        idx = max(0, min(self._event_index, len(self._block_events) - 1))
        return self._block_events[idx]

    def _show_current_event(self) -> None:
        ev = self._current_event()
        n = len(self._block_events)
        idx = self._event_index + 1 if n else 0
        self._event_label.setText(f"Event {idx}/{n}" if n else "Event —")
        self._comment.blockSignals(True)
        self._comment.setText(ev.notes if ev is not None else "")
        self._comment.blockSignals(False)
        self._trace.set_event(ev)
        self._video.set_event(ev)
        if ev is not None:
            self._trace.set_playhead_ms(ev.onset_ms - self._pre_ms)

    def _refresh_counts(self) -> None:
        c = count_by_status(self._block_events)
        self._status_label.setText(
            f"good {c['good']} · bad {c['bad']} · unset {c['unset']}"
        )

    def _step_event(self, delta: int) -> None:
        if not self._block_events:
            return
        self._event_index = (self._event_index + delta) % len(self._block_events)
        self._show_current_event()

    def _jump_next_unset(self) -> None:
        nxt = next_unset_index(self._block_events, self._event_index, forward=True)
        if nxt is not None:
            self._event_index = nxt
            self._show_current_event()

    def _replace_current_event(self, updated: VerificationEvent) -> None:
        self._block_events[self._event_index] = updated
        for i, master in enumerate(self._batch.events):
            if master.event_id == updated.event_id:
                self._batch.events[i] = updated
                break

    def _persist_block_tags(self, block_path: Path) -> None:
        save_verification_tags(block_path, self._block_events)

    def _set_status(self, status: VerificationStatus) -> None:
        ev = self._current_event()
        if ev is None:
            return
        updated = VerificationEvent(
            event_id=ev.event_id,
            animal=ev.animal,
            block=ev.block,
            block_key=ev.block_key,
            block_path=ev.block_path,
            eye=ev.eye,
            onset_ms=ev.onset_ms,
            off_ms=ev.off_ms,
            verification_status=status,
            pairing_tag=ev.pairing_tag,
            notes=self._comment.text().strip(),
            row_index=ev.row_index,
            extras=dict(ev.extras),
        )
        self._replace_current_event(updated)
        self._refresh_counts()
        self._trace.set_event(updated)
        self._persist_block_tags(ev.block_path)
        self.statusBar().showMessage(
            f"QC tag {ev.event_id} → {status} (saved)",
            3000,
        )
        if self._auto_adv.isChecked() and status in {"good", "bad"}:
            self._step_event(1)

    def _set_pairing(self, pairing: PairingTag) -> None:
        ev = self._current_event()
        if ev is None:
            return
        updated = VerificationEvent(
            event_id=ev.event_id,
            animal=ev.animal,
            block=ev.block,
            block_key=ev.block_key,
            block_path=ev.block_path,
            eye=ev.eye,
            onset_ms=ev.onset_ms,
            off_ms=ev.off_ms,
            verification_status=ev.verification_status,
            pairing_tag=pairing,
            notes=self._comment.text().strip(),
            row_index=ev.row_index,
            extras=dict(ev.extras),
        )
        self._replace_current_event(updated)
        self._trace.set_event(updated)
        self._persist_block_tags(ev.block_path)
        self.statusBar().showMessage(
            f"Pairing tag {ev.event_id} → {pairing} (saved)",
            3000,
        )

    def _save_comment(self) -> None:
        ev = self._current_event()
        if ev is None:
            return
        text = self._comment.text().strip()
        if text == ev.notes:
            return
        updated = VerificationEvent(
            event_id=ev.event_id,
            animal=ev.animal,
            block=ev.block,
            block_key=ev.block_key,
            block_path=ev.block_path,
            eye=ev.eye,
            onset_ms=ev.onset_ms,
            off_ms=ev.off_ms,
            verification_status=ev.verification_status,
            pairing_tag=ev.pairing_tag,
            notes=text,
            row_index=ev.row_index,
            extras=dict(ev.extras),
        )
        self._replace_current_event(updated)
        self._persist_block_tags(ev.block_path)
        self.statusBar().showMessage("Comment saved", 2000)

    def _on_pre_post_changed(self) -> None:
        self._pre_ms = float(self._pre_spin.value())
        self._post_ms = float(self._post_spin.value())
        self._trace.set_pre_post_ms(self._pre_ms, self._post_ms)
        self._video.set_pre_post_ms(self._pre_ms, self._post_ms)

    def _on_trace_time_selected(self, ms: float) -> None:
        if self._syncing_time:
            return
        self._syncing_time = True
        try:
            self._video.seek_ms(ms, emit=False)
            self._trace.set_playhead_ms(ms)
        finally:
            self._syncing_time = False

    def _on_video_time_changed(self, ms: float) -> None:
        if self._syncing_time:
            return
        self._syncing_time = True
        try:
            self._trace.set_playhead_ms(ms)
        finally:
            self._syncing_time = False


def build_viewer_from_dataframe(
    df,
    *,
    registry_path: Path | str | None = None,
    specs: list[BlockSpec] | None = None,
    pre_ms: float = 250.0,
    post_ms: float = 250.0,
    auto_advance: bool = False,
) -> SaccadeViewerWindow:
    if specs is None and registry_path is not None:
        specs = load_registry(registry_path)
    batch = normalize_event_batch(df, specs)
    return SaccadeViewerWindow(
        batch,
        pre_ms=pre_ms,
        post_ms=post_ms,
        auto_advance=auto_advance,
    )
