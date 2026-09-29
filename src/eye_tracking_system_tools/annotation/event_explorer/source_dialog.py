"""Unified Add sources dialog: files, folders, recursive scan."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets

from eye_tracking_system_tools.annotation.event_explorer.catalog import (
    discover_annotation_files,
)


class AddSourcesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add annotation sources")
        self._paths: list[Path] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Add Block Annotator *_annotations.json files and/or folders to scan."
            )
        )

        btn_files = QtWidgets.QPushButton("Add JSON files…")
        btn_folder = QtWidgets.QPushButton("Add folder (non-recursive)…")
        btn_folder_rec = QtWidgets.QPushButton("Add folder (recursive)…")
        layout.addWidget(btn_files)
        layout.addWidget(btn_folder)
        layout.addWidget(btn_folder_rec)

        self._list = QtWidgets.QListWidget()
        layout.addWidget(self._list)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)

        self._json_files: list[Path] = []
        self._scan_dirs: list[tuple[Path, bool]] = []

        btn_files.clicked.connect(self._add_files)
        btn_folder.clicked.connect(lambda: self._add_folder(False))
        btn_folder_rec.clicked.connect(lambda: self._add_folder(True))
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)

    def _add_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Annotation JSON files",
            filter="Annotations (*_annotations.json);;JSON (*.json)",
        )
        for p in paths:
            self._json_files.append(Path(p))
            self._list.addItem(f"FILE: {p}")

    def _add_folder(self, recursive: bool) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder to scan")
        if path:
            self._scan_dirs.append((Path(path), recursive))
            tag = "RECURSIVE" if recursive else "FLAT"
            self._list.addItem(f"{tag}: {path}")

    def _accept(self) -> None:
        self._paths = discover_annotation_files(
            json_paths=self._json_files,
            scan_dirs=[d for d, _ in self._scan_dirs],
            recursive=any(r for _, r in self._scan_dirs) or True,
        )
        if not self._paths and not self._json_files and not self._scan_dirs:
            QtWidgets.QMessageBox.information(self, "Add sources", "Nothing selected.")
            return
        if not self._paths:
            QtWidgets.QMessageBox.warning(
                self,
                "Add sources",
                "No *_annotations.json files found.",
            )
            return
        self.accept()

    @property
    def discovered_paths(self) -> list[Path]:
        return list(self._paths)
