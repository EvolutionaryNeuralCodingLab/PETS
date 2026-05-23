"""pyqtgraph plot panel: single-trial multi-stream and multi-trial average."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.event_explorer.models import EventSnippet
from eye_tracking_system_tools.annotation.event_explorer.snippet_extractor import (
    STREAM_EP,
    STREAM_LABELS,
    STREAM_L_DEG,
    STREAM_L_PUPIL,
    STREAM_R_DEG,
    STREAM_R_PUPIL,
    normalize_trials,
    resample_to_grid,
)

# Light plot theme (avoid IDE dark theme inheritance)
pg.setConfigOptions(antialias=True, background="w", foreground="k")

STREAM_COLORS = {
    STREAM_L_PUPIL: "#1f77b4",
    STREAM_R_PUPIL: "#ff7f0e",
    STREAM_L_DEG: "#2ca02c",
    STREAM_R_DEG: "#d62728",
    STREAM_EP: "#9467bd",
}

TRIAL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

NORMALIZATION_MODES = [
    ("none", "None"),
    ("zscore", "Per-trial z-score"),
    ("baseline_divide", "Baseline divide (−100…0 ms)"),
    ("minmax", "Min-max [0, 1]"),
]


class PlotPanel(QtWidgets.QWidget):
    preview_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        status_row = QtWidgets.QHBoxLayout()
        self._status_icon = QtWidgets.QLabel()
        self._status_icon.setFixedWidth(22)
        self._status_text = QtWidgets.QLabel("Ready")
        self._status_spinner = QtWidgets.QProgressBar()
        self._status_spinner.setRange(0, 0)
        self._status_spinner.setFixedWidth(80)
        self._status_spinner.hide()
        status_row.addWidget(self._status_icon)
        status_row.addWidget(self._status_text)
        status_row.addWidget(self._status_spinner)
        status_row.addStretch()
        self._refresh_btn = QtWidgets.QPushButton("Refresh preview")
        self._zoom_all_btn = QtWidgets.QPushButton("Zoom to all")
        status_row.addWidget(self._refresh_btn)
        status_row.addWidget(self._zoom_all_btn)
        layout.addLayout(status_row)

        ctrl = QtWidgets.QHBoxLayout()
        self._half_window = QtWidgets.QDoubleSpinBox()
        self._half_window.setRange(1, 60000)
        self._half_window.setValue(100)
        self._half_window.setSuffix(" ms")
        ctrl.addWidget(QtWidgets.QLabel("±"))
        ctrl.addWidget(self._half_window)

        self._chk_lpupil = QtWidgets.QCheckBox("L pupil")
        self._chk_rpupil = QtWidgets.QCheckBox("R pupil")
        self._chk_ldeg = QtWidgets.QCheckBox("L deg")
        self._chk_rdeg = QtWidgets.QCheckBox("R deg")
        self._chk_ep = QtWidgets.QCheckBox("EP trace")
        self._chk_ep.setChecked(False)
        self._chk_lpupil.setChecked(True)
        self._chk_rpupil.setChecked(True)
        self._chk_ldeg.setChecked(True)
        self._chk_rdeg.setChecked(True)
        for w in (self._chk_lpupil, self._chk_rpupil, self._chk_ldeg, self._chk_rdeg, self._chk_ep):
            ctrl.addWidget(w)

        self._oe_channels = QtWidgets.QLineEdit("1")
        self._oe_channels.setPlaceholderText("HS channels e.g. 1 or 1,2,3")
        self._oe_channels.setEnabled(False)
        ctrl.addWidget(QtWidgets.QLabel("HS ch:"))
        ctrl.addWidget(self._oe_channels)

        self._avg_group = QtWidgets.QButtonGroup(self)
        self._avg_lpupil = QtWidgets.QRadioButton("L pupil")
        self._avg_rpupil = QtWidgets.QRadioButton("R pupil")
        self._avg_ldeg = QtWidgets.QRadioButton("L deg")
        self._avg_rdeg = QtWidgets.QRadioButton("R deg")
        self._avg_ep = QtWidgets.QRadioButton("EP")
        self._avg_lpupil.setChecked(True)
        for rb in (self._avg_lpupil, self._avg_rpupil, self._avg_ldeg, self._avg_rdeg, self._avg_ep):
            self._avg_group.addButton(rb)
            ctrl.addWidget(rb)
            rb.hide()

        ctrl.addWidget(QtWidgets.QLabel("Normalize:"))
        self._norm_combo = QtWidgets.QComboBox()
        for _, label in NORMALIZATION_MODES:
            self._norm_combo.addItem(label)
        ctrl.addWidget(self._norm_combo)

        layout.addLayout(ctrl)

        self._plot = pg.PlotWidget()
        self._apply_light_theme()
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.addLegend(offset=(10, 10))
        self._zero_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color="#888888", style=QtCore.Qt.PenStyle.DotLine),
        )
        self._plot.addItem(self._zero_line)
        layout.addWidget(self._plot, stretch=1)

        self._chk_ep.toggled.connect(self._oe_channels.setEnabled)
        self._refresh_btn.clicked.connect(self.preview_requested.emit)
        self._zoom_all_btn.clicked.connect(self.zoom_to_all)
        self._last_snippets: list[EventSnippet] = []
        self._mode = "empty"
        self._set_status_ready()

    def _apply_light_theme(self) -> None:
        self._plot.setBackground("w")
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(color="k"))
            axis.setTextPen(pg.mkPen(color="k"))

    def _set_status_ready(self, message: str = "Ready") -> None:
        self._status_spinner.hide()
        self._status_icon.setPixmap(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton
            ).pixmap(18, 18)
        )
        self._status_text.setText(message)
        self._status_text.setStyleSheet("color: #22863a;")

    def _set_status_loading(self, message: str = "Loading traces…") -> None:
        self._status_spinner.show()
        self._status_icon.setPixmap(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_BrowserReload
            ).pixmap(18, 18)
        )
        self._status_text.setText(message)
        self._status_text.setStyleSheet("color: #555;")
        QtWidgets.QApplication.processEvents()

    def _set_status_error(self, message: str) -> None:
        self._status_spinner.hide()
        self._status_icon.setPixmap(
            self.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning
            ).pixmap(18, 18)
        )
        self._status_text.setText(message)
        self._status_text.setStyleSheet("color: #b31d28;")

    def half_window_ms(self) -> float:
        return float(self._half_window.value())

    def oe_channel_list(self) -> list[int]:
        text = self._oe_channels.text().strip()
        if not text:
            return [1]
        text = text.strip("[]")
        return [int(x.strip()) for x in text.replace(" ", "").split(",") if x.strip()]

    def enabled_single_streams(self) -> list[str]:
        out = []
        if self._chk_lpupil.isChecked():
            out.append(STREAM_L_PUPIL)
        if self._chk_rpupil.isChecked():
            out.append(STREAM_R_PUPIL)
        if self._chk_ldeg.isChecked():
            out.append(STREAM_L_DEG)
        if self._chk_rdeg.isChecked():
            out.append(STREAM_R_DEG)
        if self._chk_ep.isChecked():
            out.append(STREAM_EP)
        return out

    def average_stream(self) -> str:
        if self._avg_rpupil.isChecked():
            return STREAM_R_PUPIL
        if self._avg_ldeg.isChecked():
            return STREAM_L_DEG
        if self._avg_rdeg.isChecked():
            return STREAM_R_DEG
        if self._avg_ep.isChecked():
            return STREAM_EP
        return STREAM_L_PUPIL

    def normalization_mode(self) -> str:
        idx = self._norm_combo.currentIndex()
        return NORMALIZATION_MODES[idx][0]

    def set_selection_mode(self, n_selected: int) -> None:
        single = n_selected == 1
        multi = n_selected >= 2
        for w in (
            self._chk_lpupil,
            self._chk_rpupil,
            self._chk_ldeg,
            self._chk_rdeg,
            self._chk_ep,
            self._oe_channels,
        ):
            w.setVisible(single)
        for rb in (
            self._avg_lpupil,
            self._avg_rpupil,
            self._avg_ldeg,
            self._avg_rdeg,
            self._avg_ep,
        ):
            rb.setVisible(multi)
        self._mode = "single" if single else ("average" if multi else "empty")

    def clear_plot(self) -> None:
        self._plot.clear()
        self._apply_light_theme()
        self._plot.addItem(self._zero_line)
        self._plot.addLegend(offset=(10, 10))
        self._last_snippets = []

    def _normalize_snippet(self, snip: EventSnippet, mode: str) -> EventSnippet:
        if mode == "none" or not mode:
            return snip
        y = normalize_trials([snip.values], [snip.time_rel_ms], mode)[0]
        return EventSnippet(
            event_id=snip.event_id,
            stream_id=snip.stream_id,
            time_rel_ms=snip.time_rel_ms,
            values=y,
            source=snip.source,
            meta=snip.meta,
        )

    def _stream_color(self, stream_id: str, index: int) -> str:
        if stream_id in STREAM_COLORS:
            return STREAM_COLORS[stream_id]
        return TRIAL_COLORS[index % len(TRIAL_COLORS)]

    def plot_single(self, snippets: list[EventSnippet], normalization: str = "none") -> None:
        self.clear_plot()
        if not snippets:
            self._set_status_error("No trace data for selection")
            return
        norm_snips = [self._normalize_snippet(s, normalization) for s in snippets]
        self._last_snippets = norm_snips
        for i, snip in enumerate(norm_snips):
            label = STREAM_LABELS.get(snip.stream_id, snip.stream_id)
            if snip.stream_id == STREAM_EP:
                label += f" ch{snip.meta.get('channel', '')}"
            color = self._stream_color(snip.stream_id, i)
            self._plot.plot(
                snip.time_rel_ms,
                snip.values,
                pen=pg.mkPen(color=color, width=1.5),
                name=label,
            )
        self.zoom_to_all()
        self._set_status_ready(f"Showing {len(norm_snips)} trace(s)")

    def plot_average(
        self,
        snippets: list[EventSnippet],
        normalization: str,
    ) -> None:
        self.clear_plot()
        if not snippets:
            self._set_status_error("No trace data for selection")
            return
        grid, stacked = resample_to_grid(
            snippets, half_window_ms=self.half_window_ms()
        )
        trials = [stacked[i] for i in range(stacked.shape[0])]
        times = [grid] * len(trials)
        trials = normalize_trials(trials, times, normalization)
        stacked = np.vstack(trials)
        mean = np.nanmean(stacked, axis=0)
        sem = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(max(1, stacked.shape[0]))

        n = stacked.shape[0]
        for i in range(n):
            self._plot.plot(
                grid,
                stacked[i],
                pen=pg.mkPen(color=TRIAL_COLORS[i % len(TRIAL_COLORS)], width=1),
                name=f"trial {i + 1}",
            )
        self._plot.plot(
            grid,
            mean,
            pen=pg.mkPen(color="k", width=3),
            name=f"mean (N={n})",
        )
        lower = mean - sem
        upper = mean + sem
        fill = pg.FillBetweenItem(
            pg.PlotCurveItem(grid, lower),
            pg.PlotCurveItem(grid, upper),
            brush=pg.mkBrush(0, 0, 0, 60),
        )
        self._plot.addItem(fill)
        self._last_snippets = snippets
        self.zoom_to_all()
        label = STREAM_LABELS.get(snippets[0].stream_id, snippets[0].stream_id)
        self._set_status_ready(f"Average: {label}, N={n}")

    def zoom_to_all(self) -> None:
        """Set view range to fit all plotted data (maximum zoom that shows everything)."""
        items = []
        pi = self._plot.getPlotItem()
        for item in pi.listDataItems():
            if hasattr(item, "xData") and item.xData is not None:
                items.append(item)
        if not items:
            return
        x_min = y_min = float("inf")
        x_max = y_max = float("-inf")
        for item in items:
            x = np.asarray(item.xData, dtype=np.float64)
            y = np.asarray(item.yData, dtype=np.float64)
            if x.size == 0:
                continue
            x_min = min(x_min, float(np.nanmin(x)))
            x_max = max(x_max, float(np.nanmax(x)))
            y_min = min(y_min, float(np.nanmin(y)))
            y_max = max(y_max, float(np.nanmax(y)))
        if not np.isfinite(x_min):
            return
        x_pad = max((x_max - x_min) * 0.03, 1.0)
        y_pad = max((y_max - y_min) * 0.08, 1e-6) if y_max > y_min else 1.0
        self._plot.setXRange(x_min - x_pad, x_max + x_pad, padding=0)
        self._plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0)

    def last_average_arrays(
        self,
        snippets: list[EventSnippet],
        normalization: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        grid, stacked = resample_to_grid(
            snippets, half_window_ms=self.half_window_ms()
        )
        trials = [stacked[i] for i in range(stacked.shape[0])]
        times = [grid] * len(trials)
        trials_n = normalize_trials(trials, times, normalization)
        stacked = np.vstack(trials_n)
        mean = np.nanmean(stacked, axis=0)
        sem = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(max(1, stacked.shape[0]))
        return grid, stacked, mean, sem
