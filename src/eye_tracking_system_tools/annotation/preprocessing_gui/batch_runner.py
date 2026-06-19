"""Sequential multi-block batch runner for long Sync-tab steps."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6 import QtCore

from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle


class SequentialBatchWorker(QtCore.QThread):
    """Run the same block-level job on every loaded block, one at a time."""

    progress = QtCore.pyqtSignal(str)
    block_done = QtCore.pyqtSignal(str, bool, str)  # display_label, ok, detail
    finished_all = QtCore.pyqtSignal()
    cancelled = QtCore.pyqtSignal()

    def __init__(
        self,
        blocks: list[BlockHandle],
        run_one: Callable[[BlockHandle], str],
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self._blocks = list(blocks)
        self._run_one = run_one
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        n = len(self._blocks)
        for i, block in enumerate(self._blocks):
            if self._cancel_requested:
                self.cancelled.emit()
                return
            label = block.display_label
            self.progress.emit(f"[{i + 1}/{n}] {label} …")
            try:
                detail = self._run_one(block)
                self.block_done.emit(label, True, detail)
            except Exception as exc:
                self.block_done.emit(label, False, str(exc))
        if not self._cancel_requested:
            self.finished_all.emit()
