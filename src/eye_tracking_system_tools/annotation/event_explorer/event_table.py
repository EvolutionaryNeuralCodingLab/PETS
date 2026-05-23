"""Event catalog table with checkbox selection, filters, column visibility, sort."""

from __future__ import annotations

from datetime import datetime

from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.event_explorer.models import (
    BlockStatus,
    EventRecord,
)

SELECT_KEY = "__select__"

COLUMN_SPECS: list[tuple[str, str, callable]] = [
    ("event_type", "Event type", lambda r: r.event_type),
    ("animal_call", "Animal", lambda r: r.animal_call),
    ("experiment_date", "Date", lambda r: r.experiment_date or ""),
    ("block_num", "Block", lambda r: r.block_num),
    ("timepoint_ms", "Timepoint (ms)", lambda r: f"{r.timepoint_ms:.2f}"),
    ("start_ms", "Start (ms)", lambda r: f"{r.start_ms:.2f}"),
    ("end_ms", "End (ms)", lambda r: f"{r.end_ms:.2f}"),
    ("note", "Note", lambda r: r.note),
    ("arena_frame", "Arena frame", lambda r: _fmt_int(r.arena_frame)),
    ("l_eye_frame", "L eye frame", lambda r: _fmt_int(r.l_eye_frame)),
    ("r_eye_frame", "R eye frame", lambda r: _fmt_int(r.r_eye_frame)),
    ("row_index", "Row index", lambda r: _fmt_int(r.row_index)),
    ("has_le", "L eye", lambda r: _flag(r.block_status, BlockStatus.HAS_LE)),
    ("has_re", "R eye", lambda r: _flag(r.block_status, BlockStatus.HAS_RE)),
    ("has_oe", "OE", lambda r: _flag(r.block_status, BlockStatus.HAS_OE)),
    ("stale_sync", "Stale sync", lambda r: _flag(r.block_status, BlockStatus.STALE_SYNC)),
    ("annotation_path", "Annotation file", lambda r: str(r.annotation_path)),
]


def _fmt_int(v) -> str:
    return "" if v is None else str(v)


def _flag(status: BlockStatus, bit: BlockStatus) -> str:
    return "yes" if status & bit else ""


class EventTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._records: list[EventRecord] = []
        self._visible_keys = [SELECT_KEY] + [k for k, _, _ in COLUMN_SPECS]
        self._checked_ids: set[str] = set()
        self._updating = False

    def set_records(self, records: list[EventRecord]) -> None:
        self.beginResetModel()
        self._records = list(records)
        valid = {r.event_id for r in self._records}
        self._checked_ids &= valid
        self.endResetModel()

    def record_at(self, row: int) -> EventRecord | None:
        if 0 <= row < len(self._records):
            return self._records[row]
        return None

    def set_visible_columns(self, keys: list[str]) -> None:
        self.beginResetModel()
        keys = [SELECT_KEY] + [k for k in keys if k != SELECT_KEY]
        self._visible_keys = keys
        self.endResetModel()

    def visible_column_keys(self) -> list[str]:
        return list(self._visible_keys)

    def is_checked(self, event_id: str) -> bool:
        return event_id in self._checked_ids

    def set_checked(self, event_id: str, checked: bool) -> None:
        if checked:
            self._checked_ids.add(event_id)
        else:
            self._checked_ids.discard(event_id)

    def set_checked_ids(self, ids: set[str]) -> None:
        self._checked_ids = set(ids)

    def checked_ids(self) -> set[str]:
        return set(self._checked_ids)

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self._records)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return len(self._visible_keys)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        base = super().flags(index)
        if self._visible_keys[index.column()] == SELECT_KEY:
            return (
                base
                | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                | QtCore.Qt.ItemFlag.ItemIsEnabled
            )
        return base | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        rec = self._records[index.row()]
        key = self._visible_keys[index.column()]

        if key == SELECT_KEY:
            if role == QtCore.Qt.ItemDataRole.CheckStateRole:
                return (
                    QtCore.Qt.CheckState.Checked
                    if rec.event_id in self._checked_ids
                    else QtCore.Qt.CheckState.Unchecked
                )
            return None

        spec = next(s for s in COLUMN_SPECS if s[0] == key)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return spec[2](rec)
        if role == QtCore.Qt.ItemDataRole.UserRole:
            return rec
        return None

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.CheckStateRole:
            return False
        if self._visible_keys[index.column()] != SELECT_KEY:
            return False
        rec = self._records[index.row()]
        checked = int(value) == int(QtCore.Qt.CheckState.Checked)
        self.set_checked(rec.event_id, checked)
        self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.CheckStateRole])
        return True

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            key = self._visible_keys[section]
            if key == SELECT_KEY:
                return "✓"
            return next(s[1] for s in COLUMN_SPECS if s[0] == key)
        return str(section + 1)

    def sort(self, column: int, order=QtCore.Qt.SortOrder.AscendingOrder) -> None:
        key = self._visible_keys[column]
        if key == SELECT_KEY:
            return

        spec = next(s for s in COLUMN_SPECS if s[0] == key)

        def sort_key(rec: EventRecord):
            v = spec[2](rec)
            try:
                return float(v)
            except (TypeError, ValueError):
                return str(v).lower()

        self.layoutAboutToBeChanged.emit()
        self._records.sort(
            key=sort_key,
            reverse=order == QtCore.Qt.SortOrder.DescendingOrder,
        )
        self.layoutChanged.emit()


class EventFilterProxy(QtCore.QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._event_types: set[str] = set()
        self._animals: set[str] = set()
        self._blocks: set[str] = set()
        self._date_from: str | None = None
        self._date_to: str | None = None
        self._ms_min: float | None = None
        self._ms_max: float | None = None
        self._note_text: str = ""
        self._require_le = False
        self._require_re = False
        self._require_oe = False

    def set_filters(
        self,
        *,
        event_types: set[str] | None = None,
        animals: set[str] | None = None,
        blocks: set[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        ms_min: float | None = None,
        ms_max: float | None = None,
        note_text: str = "",
        require_le: bool = False,
        require_re: bool = False,
        require_oe: bool = False,
    ) -> None:
        self._event_types = event_types or set()
        self._animals = animals or set()
        self._blocks = blocks or set()
        self._date_from = date_from
        self._date_to = date_to
        self._ms_min = ms_min
        self._ms_max = ms_max
        self._note_text = note_text.strip().lower()
        self._require_le = require_le
        self._require_re = require_re
        self._require_oe = require_oe
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QtCore.QModelIndex) -> bool:
        model: EventTableModel = self.sourceModel()  # type: ignore
        rec = model.record_at(source_row)
        if rec is None:
            return False
        if self._event_types and rec.event_type not in self._event_types:
            return False
        if self._animals and rec.animal_call not in self._animals:
            return False
        if self._blocks and rec.block_num not in self._blocks:
            return False
        if self._note_text and self._note_text not in rec.note.lower():
            return False
        if self._ms_min is not None and rec.timepoint_ms < self._ms_min:
            return False
        if self._ms_max is not None and rec.timepoint_ms > self._ms_max:
            return False
        if self._require_le and not (rec.block_status & BlockStatus.HAS_LE):
            return False
        if self._require_re and not (rec.block_status & BlockStatus.HAS_RE):
            return False
        if self._require_oe and not (rec.block_status & BlockStatus.HAS_OE):
            return False
        if rec.experiment_date and (self._date_from or self._date_to):
            try:
                d = datetime.strptime(rec.experiment_date, "%Y_%m_%d")
                if self._date_from:
                    if d < datetime.strptime(self._date_from, "%Y_%m_%d"):
                        return False
                if self._date_to:
                    if d > datetime.strptime(self._date_to, "%Y_%m_%d"):
                        return False
            except ValueError:
                pass
        return True


class EventTableWidget(QtWidgets.QWidget):
    selection_changed = QtCore.pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = EventTableModel(self)
        self._proxy = EventFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        btn_row = QtWidgets.QHBoxLayout()
        self._select_all_btn = QtWidgets.QPushButton("Select all (filtered)")
        self._clear_sel_btn = QtWidgets.QPushButton("Clear selection")
        btn_row.addWidget(self._select_all_btn)
        btn_row.addWidget(self._clear_sel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._table = QtWidgets.QTableView()
        self._table.setModel(self._proxy)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 36)
        layout.addWidget(self._table)

        self._model.dataChanged.connect(self._on_check_changed)
        self._select_all_btn.clicked.connect(self.select_all_filtered)
        self._clear_sel_btn.clicked.connect(self.clear_selection)

    def set_records(self, records: list[EventRecord]) -> None:
        self._model.set_records(records)

    def selected_records(self) -> list[EventRecord]:
        ids = self._model.checked_ids()
        return [r for r in self._model._records if r.event_id in ids]

    def select_all_filtered(self) -> None:
        for row in range(self._proxy.rowCount()):
            src = self._proxy.mapToSource(self._proxy.index(row, 0))
            rec = self._model.record_at(src.row())
            if rec:
                self._model.set_checked(rec.event_id, True)
        if self._model.rowCount() > 0:
            self._model.dataChanged.emit(
                self._model.index(0, 0),
                self._model.index(
                    self._model.rowCount() - 1, self._model.columnCount() - 1
                ),
                [QtCore.Qt.ItemDataRole.CheckStateRole],
            )
        self._emit_selection()

    def clear_selection(self) -> None:
        self._model.set_checked_ids(set())
        if self._model.rowCount() > 0:
            self._model.dataChanged.emit(
                self._model.index(0, 0),
                self._model.index(
                    self._model.rowCount() - 1, self._model.columnCount() - 1
                ),
                [QtCore.Qt.ItemDataRole.CheckStateRole],
            )
        self._emit_selection()

    def _on_check_changed(self, top_left, bottom_right, roles) -> None:
        if QtCore.Qt.ItemDataRole.CheckStateRole in roles:
            self._emit_selection()

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self.selected_records())

    def show_column_menu(self, pos) -> None:
        menu = QtWidgets.QMenu(self)
        visible = set(self._model.visible_column_keys())
        for key, label, _ in COLUMN_SPECS:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key in visible)
            action.setData(key)

        chosen = menu.exec(pos)
        if chosen is None:
            return
        key = chosen.data()
        visible_now = set(self._model.visible_column_keys())
        if chosen.isChecked():
            visible_now.add(key)
        else:
            visible_now.discard(key)
        ordered = [SELECT_KEY] + [k for k, _, _ in COLUMN_SPECS if k in visible_now]
        self._model.set_visible_columns(ordered)

    def filter_proxy(self) -> EventFilterProxy:
        return self._proxy

    def table_view(self) -> QtWidgets.QTableView:
        return self._table
