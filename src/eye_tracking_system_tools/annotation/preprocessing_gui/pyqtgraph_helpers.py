"""Native pyqtgraph plots used by the Preprocessing GUI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets


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


class EyeBrightnessPreviewPlot(QtWidgets.QWidget):
    """Per-frame eye LED brightness traces (from ``eye_brightness_values_dict.pkl``)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", "Video frame index")
        self.plot_widget.setLabel("left", "Mean brightness (a.u.)")
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen("k"))
        self.plot_widget.getAxis("left").setPen(pg.mkPen("k"))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen("k"))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen("k"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.plot_widget)

    def set_traces(self, left_values, right_values) -> None:
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        left = np.asarray(left_values, dtype=float)
        right = np.asarray(right_values, dtype=float)
        x_l = np.arange(len(left))
        x_r = np.arange(len(right))
        self.plot_widget.plot(
            x_l, left, pen=pg.mkPen("#1f77b4", width=1.2), name="Left eye"
        )
        self.plot_widget.plot(
            x_r, right, pen=pg.mkPen("#d62728", width=1.2), name="Right eye"
                )


class BehaviorThresholdPlot(QtWidgets.QWidget):
    """Rolling average movement with draggable threshold and active/quiet shading."""

    threshold_changed = QtCore.pyqtSignal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", "Time (ms)")
        self.plot_widget.setLabel("left", "Average movement")
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen("k"))
        self.plot_widget.getAxis("left").setPen(pg.mkPen("k"))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen("k"))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen("k"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.plot_widget)

        self._df: pd.DataFrame | None = None
        self._step_ms = 1000
        self._region_items: list[pg.LinearRegionItem] = []
        self._threshold_line = pg.InfiniteLine(
            pos=0.3,
            angle=0,
            movable=True,
            pen=pg.mkPen("#c0392b", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self._threshold_line)
        self._threshold_line.sigPositionChanged.connect(self._on_line_dragged)

    def set_rolling_data(self, df: pd.DataFrame, *, step_ms: int) -> None:
        self._df = df.copy()
        self._step_ms = int(step_ms)
        self._redraw_trace()
        self._update_regions(float(self._threshold_line.value()))

    def set_threshold(self, value: float, *, emit: bool = True) -> None:
        self._threshold_line.blockSignals(True)
        self._threshold_line.setPos(float(value))
        self._threshold_line.blockSignals(False)
        self._update_regions(float(value))
        if emit:
            self.threshold_changed.emit(float(value))

    def _on_line_dragged(self) -> None:
        value = float(self._threshold_line.value())
        self._update_regions(value)
        self.threshold_changed.emit(value)

    def _clear_regions(self) -> None:
        for item in self._region_items:
            self.plot_widget.removeItem(item)
        self._region_items.clear()

    def _update_regions(self, threshold: float) -> None:
        self._clear_regions()
        if self._df is None or self._df.empty:
            return
        active = self._df["average_movAll"].to_numpy(dtype=float) > float(threshold)
        labels = np.where(active, "active", "quiet")
        starts = self._df["window_start"].to_numpy(dtype=float)
        step = float(self._step_ms)
        colors = {"active": (255, 193, 7, 55), "quiet": (52, 152, 219, 40)}
        i = 0
        while i < len(labels):
            label = labels[i]
            start = starts[i]
            j = i + 1
            while j < len(labels) and labels[j] == label:
                j += 1
            end = starts[j - 1] + step
            region = pg.LinearRegionItem(
                values=(start, end),
                movable=False,
                brush=pg.mkBrush(*colors[label]),
                pen=pg.mkPen(None),
            )
            region.setZValue(-10)
            self.plot_widget.addItem(region)
            self._region_items.append(region)
            i = j

    def _redraw_trace(self) -> None:
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self._region_items.clear()
        if self._df is None or self._df.empty:
            self.plot_widget.addItem(self._threshold_line)
            return
        x = self._df["window_start"].to_numpy(dtype=float)
        y = self._df["average_movAll"].to_numpy(dtype=float)
        self.plot_widget.plot(
            x, y, pen=pg.mkPen("#1f77b4", width=1.5), name="Average movement"
        )
        self.plot_widget.addItem(self._threshold_line)
        self._auto_range_view()

    def _auto_range_view(self) -> None:
        vb = self.plot_widget.getViewBox()
        vb.enableAutoRange(pg.ViewBox.XYAxes)
        vb.autoRange(padding=0.05)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._df is not None and not self._df.empty:
            QtCore.QTimer.singleShot(0, self._auto_range_view)


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
