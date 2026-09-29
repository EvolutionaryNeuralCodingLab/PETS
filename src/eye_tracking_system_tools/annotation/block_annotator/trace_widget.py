"""pyqtgraph Open Ephys trace view with playhead."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

from eye_tracking_system_tools.annotation.block_annotator.oe_streams import (
    OEStream,
    list_oe_streams,
    load_overview_trace,
)


class TraceWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        self._stream_combo = QtWidgets.QComboBox()
        self._downsample = QtWidgets.QSpinBox()
        self._downsample.setRange(1, 10000)
        self._downsample.setValue(50)
        self._downsample.setPrefix("downsample ")
        row.addWidget(QtWidgets.QLabel("OE stream:"))
        row.addWidget(self._stream_combo, stretch=1)
        row.addWidget(self._downsample)
        layout.addLayout(row)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "time", units="ms")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._curve = self._plot.plot([], [], pen=pg.mkPen("c", width=1))
        self._playhead = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("y", width=2)
        )
        self._plot.addItem(self._playhead)
        layout.addWidget(self._plot, stretch=1)

        self._oe_rec = None
        self._ms_axis: np.ndarray | None = None
        self._streams: list[OEStream] = []
        self._active = False

        self._stream_combo.currentIndexChanged.connect(self._reload_trace)
        self._downsample.valueChanged.connect(self._reload_trace)

    def set_active(self, active: bool) -> None:
        """When False, hides the panel and skips expensive trace loading."""
        self._active = bool(active)
        self.setVisible(self._active)
        if self._active:
            self._reload_trace()

    def set_recording(
        self, oe_rec, ms_axis: np.ndarray, *, populate: bool = True
    ) -> None:
        self._oe_rec = oe_rec
        self._ms_axis = np.asarray(ms_axis, dtype=np.float64)
        if not populate:
            return
        self._streams = list_oe_streams(oe_rec)
        self._stream_combo.blockSignals(True)
        self._stream_combo.clear()
        if not self._streams:
            self._stream_combo.addItem("(no OE streams)")
            self._curve.setData([], [])
        else:
            for s in self._streams:
                self._stream_combo.addItem(s.label, s)
        self._stream_combo.blockSignals(False)
        if self._active:
            self._reload_trace()
        else:
            self._curve.setData([], [])

    def _current_stream(self) -> OEStream | None:
        if not self._streams:
            return None
        data = self._stream_combo.currentData()
        return data if isinstance(data, OEStream) else self._streams[0]

    def _reload_trace(self) -> None:
        if not self._active:
            return
        stream = self._current_stream()
        if stream is None or self._ms_axis is None or len(self._ms_axis) == 0:
            self._curve.setData([], [])
            return
        result = load_overview_trace(
            self._oe_rec,
            stream,
            self._ms_axis,
            downsample=int(self._downsample.value()),
        )
        if result is None:
            self._curve.setData([], [])
            return
        t, y = result
        self._curve.setData(t, y)

    def set_playhead_ms(self, ms: float) -> None:
        if not self._active:
            return
        self._playhead.setPos(float(ms))
