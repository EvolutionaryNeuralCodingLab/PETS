"""Floating window hosting synced explore video."""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
    ExploreVideoPanel,
)


class VideoLoadWorker(QtCore.QThread):
    """Load BlockSession off the UI thread; bind on the caller afterward."""

    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, load_fn, parent=None):
        super().__init__(parent)
        self._load_fn = load_fn

    def run(self) -> None:
        try:
            result = self._load_fn()
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ExploreVideoWindow(QtWidgets.QMainWindow):
    """Resizable top-level window for L / Arena / R video + transport."""

    closed = QtCore.pyqtSignal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        title: str = "Explore — synced video",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 520)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self.panel = ExploreVideoPanel(self)
        self.setCentralWidget(self.panel)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.panel.clear()
        self.closed.emit()
        super().closeEvent(event)
