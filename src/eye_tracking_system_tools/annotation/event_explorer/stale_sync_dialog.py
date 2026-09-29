"""Per-block confirmation when final_sync_df is newer than eye CSV."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets


class StaleSyncDialog(QtWidgets.QDialog):
    def __init__(
        self,
        block_path: Path,
        sync_path: Path,
        eye_paths: list[Path],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stale eye CSV warning")
        eye_list = "\n".join(str(p) for p in eye_paths) or "(none)"
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                f"Block: {block_path}\n\n"
                f"`final_sync_df.csv` is newer than the selected eye CSV file(s).\n"
                f"Eye data may be out of sync with the master timeline.\n\n"
                f"Sync: {sync_path}\n"
                f"Eye CSV(s):\n{eye_list}\n\n"
                "Continue loading this block?"
            )
        )
        buttons = QtWidgets.QDialogButtonBox()
        yes_btn = buttons.addButton("Yes, continue", QtWidgets.QDialogButtonBox.ButtonRole.YesRole)
        no_btn = buttons.addButton("No, skip block", QtWidgets.QDialogButtonBox.ButtonRole.NoRole)
        layout.addWidget(buttons)
        yes_btn.clicked.connect(self.accept)
        no_btn.clicked.connect(self.reject)
