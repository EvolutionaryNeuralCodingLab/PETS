"""Event list, mark/edit/delete, range controls."""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.block_annotator.models import (
    AnnotatorConfig,
    AnnotationEvent,
    BlockSession,
)


class AnnotationPanel(QtWidgets.QWidget):
    event_marked = QtCore.pyqtSignal(object)
    seek_requested = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session: BlockSession | None = None
        self._events: list[AnnotationEvent] = []
        self._current_index = 0

        layout = QtWidgets.QVBoxLayout(self)

        type_row = QtWidgets.QHBoxLayout()
        self._type_combo = QtWidgets.QComboBox()
        self._half_width = QtWidgets.QDoubleSpinBox()
        self._half_width.setRange(0.0, 60000.0)
        self._half_width.setValue(100.0)
        self._half_width.setSuffix(" ms ±")
        type_row.addWidget(QtWidgets.QLabel("Type:"))
        type_row.addWidget(self._type_combo)
        type_row.addWidget(self._half_width)
        layout.addLayout(type_row)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_mark = QtWidgets.QPushButton("Mark event")
        self._btn_delete = QtWidgets.QPushButton("Delete")
        self._btn_save = QtWidgets.QPushButton("Save JSON")
        btn_row.addWidget(self._btn_mark)
        btn_row.addWidget(self._btn_delete)
        btn_row.addWidget(self._btn_save)
        layout.addLayout(btn_row)

        self._list = QtWidgets.QListWidget()
        layout.addWidget(self._list, stretch=1)

        edit = QtWidgets.QFormLayout()
        self._edit_note = QtWidgets.QLineEdit()
        self._edit_start = QtWidgets.QDoubleSpinBox()
        self._edit_end = QtWidgets.QDoubleSpinBox()
        self._edit_start.setRange(-1e9, 1e9)
        self._edit_end.setRange(-1e9, 1e9)
        for w in (self._edit_start, self._edit_end):
            w.setDecimals(2)
            w.setSuffix(" ms")
        edit.addRow("Note:", self._edit_note)
        edit.addRow("Start:", self._edit_start)
        edit.addRow("End:", self._edit_end)
        layout.addLayout(edit)

        self._btn_mark.clicked.connect(self._mark_event)
        self._btn_delete.clicked.connect(self._delete_selected)
        self._btn_save.clicked.connect(self._emit_save)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._edit_note.editingFinished.connect(self._apply_edit_to_selected)
        self._edit_start.editingFinished.connect(self._apply_edit_to_selected)
        self._edit_end.editingFinished.connect(self._apply_edit_to_selected)

    def set_session(self, session: BlockSession, events: list[AnnotationEvent]) -> None:
        self._session = session
        self._events = list(events)
        self.reload_event_types(session.config.event_types)
        self._half_width.setValue(session.config.default_range_half_width_ms)
        self._refresh_list()

    def reload_event_types(
        self, event_types: list[str], *, select: str | None = None
    ) -> None:
        """Refresh the type dropdown; optionally select a type by name."""
        previous = select or self._type_combo.currentText()
        self._type_combo.blockSignals(True)
        self._type_combo.clear()
        for t in event_types:
            self._type_combo.addItem(t)
        if previous:
            idx = self._type_combo.findText(previous)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
        self._type_combo.blockSignals(False)

    def set_timeline_index(self, index: int) -> None:
        self._current_index = int(index)

    def events(self) -> list[AnnotationEvent]:
        return list(self._events)

    def _emit_save(self) -> None:
        self.event_marked.emit(None)

    def _mark_event(self) -> None:
        if self._session is None:
            return
        hw = float(self._half_width.value())
        tp = self._session.ms_at(self._current_index)
        ev = AnnotationEvent(
            event_type=self._type_combo.currentText(),
            timepoint_ms=tp,
            start_ms=tp - hw,
            end_ms=tp + hw,
            range_half_width_ms=hw,
            row_index=self._current_index,
        )
        arena, le, re = self._session.frame_ids_at(self._current_index)
        ev.arena_frame = arena
        ev.l_eye_frame = le
        ev.r_eye_frame = re
        self._events.append(ev)
        self._refresh_list()
        self._list.setCurrentRow(len(self._events) - 1)

    def _delete_selected(self) -> None:
        row = self._list.currentRow()
        if 0 <= row < len(self._events):
            del self._events[row]
            self._refresh_list()

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._events):
            return
        ev = self._events[row]
        self._edit_note.setText(ev.note)
        self._edit_start.setValue(ev.start_ms)
        self._edit_end.setValue(ev.end_ms)
        self.seek_requested.emit(ev.timepoint_ms)

    def _apply_edit_to_selected(self) -> None:
        row = self._list.currentRow()
        if row < 0 or row >= len(self._events):
            return
        ev = self._events[row]
        ev.note = self._edit_note.text()
        ev.start_ms = float(self._edit_start.value())
        ev.end_ms = float(self._edit_end.value())
        ev.timepoint_ms = 0.5 * (ev.start_ms + ev.end_ms)
        ev.range_half_width_ms = 0.5 * (ev.end_ms - ev.start_ms)
        self._refresh_list()
        self._list.setCurrentRow(row)

    def _refresh_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for ev in self._events:
            self._list.addItem(
                f"{ev.event_type} @ {ev.timepoint_ms:.1f} ms "
                f"[{ev.start_ms:.0f}–{ev.end_ms:.0f}]"
            )
        self._list.blockSignals(False)
