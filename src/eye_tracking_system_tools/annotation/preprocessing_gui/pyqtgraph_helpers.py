"""Native pyqtgraph plots used by the Preprocessing GUI."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtWidgets
import pyqtgraph as pg


class FinalSyncSanityPlot(QtWidgets.QWidget):
    """Small wrapper around a pyqtgraph PlotWidget for final sync sanity."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", "OE time (s)")
        self.plot_widget.setLabel("left", "Brightness (a.u.)")
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen("k"))
        self.plot_widget.getAxis("left").setPen(pg.mkPen("k"))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen("k"))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen("k"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.plot_widget)

    def set_data(self, final_df, *, fs_hz: float, led_samples=None) -> None:
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        x_s = np.asarray(final_df["Arena_TTL"], dtype=float) / float(fs_hz)
        y_l = np.asarray(final_df["L_values"], dtype=float)
        y_r = np.asarray(final_df["R_values"], dtype=float)
        self.plot_widget.plot(x_s, y_l, pen=pg.mkPen("#1f77b4", width=1.5), name="Left")
        self.plot_widget.plot(x_s, y_r, pen=pg.mkPen("#d62728", width=1.5), name="Right")
        if led_samples is not None:
            led_arr = np.asarray(led_samples, dtype=float)
            led_arr = led_arr[np.isfinite(led_arr)]
            for s in led_arr:
                self.plot_widget.addItem(
                    pg.InfiniteLine(
                        pos=float(s) / float(fs_hz),
                        angle=90,
                        pen=pg.mkPen((44, 160, 44, 140), width=1),
                    )
                )


class JitterDriftPlot(QtWidgets.QWidget):
    """Plot top_correlation_dist with optional flagged peak indices."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "Frame index")
        self.plot_widget.setLabel("left", "top_correlation_dist")
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen("k"))
        self.plot_widget.getAxis("left").setPen(pg.mkPen("k"))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen("k"))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen("k"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.plot_widget)

    def set_drift(self, drift_values, peak_indices=None) -> None:
        self.plot_widget.clear()
        y = np.asarray(drift_values, dtype=float)
        x = np.arange(len(y))
        self.plot_widget.plot(x, y, pen=pg.mkPen("#1f77b4", width=1.2))
        if peak_indices is not None:
            peaks = np.asarray(peak_indices, dtype=int)
            peaks = peaks[(peaks >= 0) & (peaks < len(y))]
            if peaks.size:
                self.plot_widget.plot(
                    peaks,
                    y[peaks],
                    pen=None,
                    symbol="o",
                    symbolBrush=pg.mkBrush("#d62728"),
                    symbolSize=7,
                )
