"""Remap broken block_path to a valid block folder."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets


class RemapBlockDialog(QtWidgets.QDialog):
    def __init__(
        self,
        original: Path,
        animal_call: str,
        block_num: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Remap block folder")
        self._chosen: Path | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                f"Block path not found:\n{original}\n\n"
                f"Select the correct folder for {animal_call} block {block_num}:"
            )
        )
        self._path_edit = QtWidgets.QLineEdit(str(original))
        browse = QtWidgets.QPushButton("Browse…")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._path_edit)
        row.addWidget(browse)
        layout.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)
        browse.clicked.connect(self._browse)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

    def _browse(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Block folder")
        if path:
            self._path_edit.setText(path)

    def _accept(self) -> None:
        p = Path(self._path_edit.text().strip())
        sync = p / "analysis" / "final_sync_df.csv"
        alt = p / "analysis" / "blocksync_df.csv"
        if not sync.exists() and not alt.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Remap",
                "Selected folder has no analysis/final_sync_df.csv",
            )
            return
        self._chosen = p.resolve()
        self.accept()

    @property
    def chosen_path(self) -> Path | None:
        return self._chosen
