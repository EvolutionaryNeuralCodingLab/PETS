"""Floating window hosting synced explore video."""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
    ExploreVideoPanel,
)


class ExploreVideoWindow(QtWidgets.QMainWindow):
    """Resizable top-level window for L / Arena / R video + transport."""

    closed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Explore — synced video")
        self.resize(1100, 520)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self.panel = ExploreVideoPanel(self)
        self.setCentralWidget(self.panel)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.panel.clear()
        self.closed.emit()
        super().closeEvent(event)
