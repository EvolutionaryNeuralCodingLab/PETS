"""Stage 4 -- Accelerometer-based behavior annotation.

Phase 0 skeleton. The tab disables itself with a tooltip when ``lizMov.mat``
is absent (per the user's decision on open-item #5).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab


def _liz_mov_paths(block: BlockHandle) -> list[Path]:
    """Locations where ``lizMov.mat`` is expected to be found."""
    return [
        block.block_path / "IMU" / "lizMov.mat",
        block.block_path / "lizMov.mat",
        block.block_path / "analysis" / "lizMov.mat",
    ]


def has_liz_mov(block: BlockHandle) -> bool:
    return any(p.exists() for p in _liz_mov_paths(block))


class BehaviorTab(BaseTab):
    tab_id = "behavior"
    tab_label = "Behavior"

    def build_ui(self) -> None:
        super().build_ui()
        self._lizmov_banner = QtWidgets.QLabel(
            "<b>lizMov.mat not found.</b> Run the MATLAB getLizMovement "
            "function for this block first."
        )
        self._lizmov_banner.setStyleSheet(
            "QLabel { background: #fff3cd; color: #856404; padding: 6px; "
            "border: 1px solid #ffeeba; border-radius: 3px; }"
        )
        self._lizmov_banner.hide()
        self.layout().insertWidget(0, self._lizmov_banner)

    def set_block(self, block: BlockHandle | None) -> None:
        super().set_block(block)
        if block is None:
            self._lizmov_banner.hide()
            self.setEnabled(True)
            self.setToolTip("")
            return
        present = has_liz_mov(block)
        self._lizmov_banner.setVisible(not present)
        self.setEnabled(present)
        if present:
            self.setToolTip("")
        else:
            self.setToolTip(
                "Behavior tab is disabled because lizMov.mat is missing for "
                "this block. Run the MATLAB getLizMovement function first."
            )

    def status_signature(self, block: BlockHandle) -> list[Path]:
        return [
            block.analysis_path / f"block_{block.block_num}_behavior_state.csv",
        ]
