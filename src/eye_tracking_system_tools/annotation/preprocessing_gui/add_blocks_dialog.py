"""Add one or more blocks to the preprocessing GUI session."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.block_picker import (
    discover_blocks,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle


class AddBlocksDialog(QtWidgets.QDialog):
    """Browse block folders and/or pick from the experiment tree."""

    def __init__(
        self,
        *,
        experiment_path: Path | None,
        animal: str | None,
        loaded_paths: set[Path],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Add block(s)")
        self.resize(640, 480)
        self._experiment_path = experiment_path
        self._animal = animal
        self._loaded_paths = {p.resolve() for p in loaded_paths}
        self._pending: list[BlockHandle] = []

        layout = QtWidgets.QVBoxLayout(self)

        browse_row = QtWidgets.QHBoxLayout()
        self._btn_browse = QtWidgets.QPushButton("Browse block folder…")
        browse_row.addWidget(self._btn_browse)
        browse_row.addStretch(1)
        layout.addLayout(browse_row)

        self._pending_list = QtWidgets.QListWidget()
        self._pending_list.setMaximumHeight(100)
        layout.addWidget(QtWidgets.QLabel("Queued from browse:"))
        layout.addWidget(self._pending_list)

        layout.addWidget(QtWidgets.QLabel("Or select from experiment:"))
        self._filter_edit = QtWidgets.QLineEdit()
        self._filter_edit.setPlaceholderText("Filter blocks (e.g. 011)")
        layout.addWidget(self._filter_edit)

        self._discover_list = QtWidgets.QListWidget()
        self._discover_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        layout.addWidget(self._discover_list, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        self._btn_browse.clicked.connect(self._browse_folder)
        self._filter_edit.textChanged.connect(self._refresh_discover_list)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

        self._refresh_discover_list()

    def selected_handles(self) -> list[BlockHandle]:
        return list(self._pending)

    def _browse_folder(self) -> None:
        from eye_tracking_system_tools.annotation.preprocessing_gui.block_picker import (
            infer_fields_from_block_folder,
        )

        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select block folder"
        )
        if not path:
            return
        inferred = infer_fields_from_block_folder(Path(path))
        if inferred is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Add block",
                "Could not infer experiment/animal/block from this folder.",
            )
            return
        experiment_path, animal, block_num = inferred
        blocks = discover_blocks(experiment_path, animal, [block_num])
        if not blocks:
            QtWidgets.QMessageBox.warning(self, "Add block", "No block discovered.")
            return
        handle = blocks[0]
        key = Path(handle.block_path).resolve()
        if key in self._loaded_paths or any(
            Path(h.block_path).resolve() == key for h in self._pending
        ):
            QtWidgets.QMessageBox.information(
                self, "Add block", "That block is already in the session."
            )
            return
        self._pending.append(handle)
        self._pending_list.addItem(handle.display_label)

    def _refresh_discover_list(self) -> None:
        self._discover_list.clear()
        if self._experiment_path is None or not self._animal:
            item = QtWidgets.QListWidgetItem(
                "(No experiment context — use Browse block folder)"
            )
            item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self._discover_list.addItem(item)
            return
        filt = self._filter_edit.text().strip().lower()
        blocks = discover_blocks(self._experiment_path, self._animal, None)
        for handle in blocks:
            if filt and filt not in handle.block_num.lower() and filt not in handle.display_label.lower():
                continue
            key = Path(handle.block_path).resolve()
            if key in self._loaded_paths:
                continue
            item = QtWidgets.QListWidgetItem(handle.display_label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, handle)
            self._discover_list.addItem(item)

    def _accept(self) -> None:
        for item in self._discover_list.selectedItems():
            handle = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if handle is None:
                continue
            key = Path(handle.block_path).resolve()
            if key in self._loaded_paths:
                continue
            if any(Path(h.block_path).resolve() == key for h in self._pending):
                continue
            self._pending.append(handle)
        if not self._pending:
            QtWidgets.QMessageBox.warning(
                self, "Add block", "Select or browse at least one new block."
            )
            return
        self.accept()
