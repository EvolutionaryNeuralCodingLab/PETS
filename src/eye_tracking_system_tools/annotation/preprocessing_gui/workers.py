"""Background workers for long-running preprocessing GUI tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6 import QtCore


class CallableWorker(QtCore.QThread):
    """Run a callable off the UI thread and report success/failure."""

    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, work_fn: Callable[[], Any], parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._work_fn = work_fn

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._work_fn())
        except Exception as exc:
            self.failed.emit(str(exc))
