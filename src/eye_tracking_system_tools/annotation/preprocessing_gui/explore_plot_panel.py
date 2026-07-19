"""Stacked linked-X pyqtgraph panel for full-block Data Exploration."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.block_annotator.oe_streams import OEStream
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_series import (
    COLOR_LEFT,
    COLOR_RIGHT,
    DEFAULT_EYE_ENABLED,
    EP_COLORS,
    ExploreCatalog,
    EyeDataVersion,
    available_eye_metric_ids,
    ep_stream_key,
    load_ep_overview,
)

pg.setConfigOptions(antialias=True, background="w", foreground="k")


def _make_check_list(
    title: str,
    *,
    min_height: int = 40,
    max_height: int = 72,
) -> tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
    box = QtWidgets.QGroupBox(title)
    outer = QtWidgets.QVBoxLayout(box)
    outer.setContentsMargins(4, 4, 4, 4)
    outer.setSpacing(2)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setMinimumHeight(min_height)
    scroll.setMaximumHeight(max_height)
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    lay.setContentsMargins(2, 2, 2, 2)
    lay.setSpacing(1)
    lay.addStretch(1)
    scroll.setWidget(host)
    outer.addWidget(scroll)
    box.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Preferred,
        QtWidgets.QSizePolicy.Policy.Maximum,
    )
    return box, lay


class _YLimitControls(QtWidgets.QWidget):
    """Compact ymin/ymax inputs beside a plot's Y axis."""

    limits_applied = QtCore.pyqtSignal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(2)
        self._ymin = QtWidgets.QDoubleSpinBox()
        self._ymax = QtWidgets.QDoubleSpinBox()
        for spin in (self._ymin, self._ymax):
            spin.setDecimals(3)
            spin.setRange(-1e12, 1e12)
            spin.setMaximumWidth(78)
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._ymin.setToolTip("Y min")
        self._ymax.setToolTip("Y max")
        btn = QtWidgets.QPushButton("Y")
        btn.setFixedWidth(28)
        btn.setToolTip("Apply Y limits")
        btn.clicked.connect(self._emit)
        lay.addWidget(QtWidgets.QLabel("max"))
        lay.addWidget(self._ymax)
        lay.addWidget(QtWidgets.QLabel("min"))
        lay.addWidget(self._ymin)
        lay.addWidget(btn)
        lay.addStretch(1)
        self.setFixedWidth(86)

    def set_limits(self, ymin: float, ymax: float) -> None:
        self._ymin.blockSignals(True)
        self._ymax.blockSignals(True)
        self._ymin.setValue(float(ymin))
        self._ymax.setValue(float(ymax))
        self._ymin.blockSignals(False)
        self._ymax.blockSignals(False)

    def _emit(self) -> None:
        ymin = float(self._ymin.value())
        ymax = float(self._ymax.value())
        if ymax < ymin:
            ymin, ymax = ymax, ymin
            self.set_limits(ymin, ymax)
        self.limits_applied.emit(ymin, ymax)


class ExplorePlotPanel(QtWidgets.QWidget):
    """Paired eye metrics + multi-EP stacked plots with shared X (ms)."""

    time_selected = QtCore.pyqtSignal(float)
    time_preview = QtCore.pyqtSignal(float)  # playhead drag (no video seek)
    refresh_requested = QtCore.pyqtSignal()
    eye_version_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._catalog: ExploreCatalog | None = None
        self._playhead_ms = 0.0
        self._plot_rows: dict[str, pg.PlotWidget] = {}
        self._playheads: dict[str, pg.InfiniteLine] = {}
        self._y_controls: dict[str, _YLimitControls] = {}
        self._y_limits: dict[str, tuple[float, float]] = {}
        self._eye_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._ep_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._ep_streams_by_key: dict[str, OEStream] = {}
        self._updating = False
        self._suppress_version_signal = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._chrome = QtWidgets.QWidget()
        chrome_lay = QtWidgets.QVBoxLayout(self._chrome)
        chrome_lay.setContentsMargins(0, 0, 0, 0)
        chrome_lay.setSpacing(4)

        version_row = QtWidgets.QHBoxLayout()
        version_row.addWidget(QtWidgets.QLabel("Eye data:"))
        self._version_combo = QtWidgets.QComboBox()
        self._version_combo.setMinimumWidth(160)
        self._version_combo.setEnabled(False)
        version_row.addWidget(self._version_combo, stretch=1)
        chrome_lay.addLayout(version_row)

        pickers = QtWidgets.QHBoxLayout()
        pickers.setSpacing(4)
        eye_box, self._eye_list_layout = _make_check_list("Eye (L+R)")
        ep_box, self._ep_list_layout = _make_check_list("EP")
        pickers.addWidget(eye_box, stretch=1)
        pickers.addWidget(ep_box, stretch=1)

        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(4)
        tools.addWidget(QtWidgets.QLabel("EP↓"))
        self._downsample = QtWidgets.QSpinBox()
        self._downsample.setRange(1, 10000)
        self._downsample.setValue(50)
        self._downsample.setMaximumWidth(64)
        self._downsample.setEnabled(False)
        tools.addWidget(self._downsample)
        self._box_zoom_btn = QtWidgets.QPushButton("Box zoom")
        self._box_zoom_btn.setCheckable(True)
        self._box_zoom_btn.setChecked(True)
        self._box_zoom_btn.setToolTip(
            "When on: drag a rectangle to zoom. Drag the playhead (or turn off) to set time.\n"
            "When off: left-drag pans; click sets the playhead."
        )
        self._zoom_all_btn = QtWidgets.QPushButton("Zoom all")
        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        tools.addWidget(self._box_zoom_btn)
        tools.addWidget(self._zoom_all_btn)
        tools.addWidget(self._refresh_btn)
        tools.addStretch(1)
        pickers_col = QtWidgets.QVBoxLayout()
        pickers_col.setSpacing(2)
        pickers_col.addLayout(pickers)
        pickers_col.addLayout(tools)
        chrome_lay.addLayout(pickers_col)

        x_row = QtWidgets.QHBoxLayout()
        x_row.setSpacing(4)
        x_row.addWidget(QtWidgets.QLabel("t (ms)"))
        self._center_ms = QtWidgets.QDoubleSpinBox()
        self._center_ms.setDecimals(1)
        self._center_ms.setRange(-1e12, 1e12)
        self._center_ms.setMaximumWidth(110)
        self._center_ms.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        x_row.addWidget(self._center_ms)
        x_row.addWidget(QtWidgets.QLabel("window (ms)"))
        self._window_ms = QtWidgets.QDoubleSpinBox()
        self._window_ms.setDecimals(1)
        self._window_ms.setRange(1.0, 1e12)
        self._window_ms.setValue(2000.0)
        self._window_ms.setMaximumWidth(100)
        self._window_ms.setToolTip("Visible X span centered on t.")
        self._window_ms.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        x_row.addWidget(self._window_ms)
        self._btn_apply_x = QtWidgets.QPushButton("Apply X")
        self._btn_apply_x.setToolTip(
            "Move playhead to t and set linked X range to t ± window/2."
        )
        x_row.addWidget(self._btn_apply_x)
        legend = QtWidgets.QLabel(
            '<span style="color:#1f77b4;">● L</span> '
            '<span style="color:#d62728;">● R</span>'
        )
        x_row.addWidget(legend)
        x_row.addStretch(1)
        chrome_lay.addLayout(x_row)

        self._chrome.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        layout.addWidget(self._chrome)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._plots_host = QtWidgets.QWidget()
        self._plots_layout = QtWidgets.QVBoxLayout(self._plots_host)
        self._plots_layout.setContentsMargins(0, 0, 0, 0)
        self._plots_layout.setSpacing(4)
        self._scroll.setWidget(self._plots_host)
        layout.addWidget(self._scroll, stretch=1)

        self._empty_label = QtWidgets.QLabel("Load analysis to explore block traces.")
        self._empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._plots_layout.addWidget(self._empty_label)

        self._version_combo.currentIndexChanged.connect(self._on_version_combo_changed)
        self._downsample.valueChanged.connect(self._on_ep_controls_changed)
        self._box_zoom_btn.toggled.connect(self._apply_mouse_mode)
        self._zoom_all_btn.clicked.connect(self.zoom_to_all)
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)
        self._btn_apply_x.clicked.connect(self._apply_x_window)

    @property
    def chrome_widget(self) -> QtWidgets.QWidget:
        return self._chrome

    def set_eye_versions(
        self,
        versions: list[EyeDataVersion],
        *,
        selected_tag: str | None = None,
    ) -> None:
        """Populate the eye-data version combo without emitting change."""
        self._suppress_version_signal = True
        try:
            self._version_combo.clear()
            if not versions:
                self._version_combo.setEnabled(False)
                self._version_combo.addItem("(no eye_data CSVs)", "")
                return
            self._version_combo.setEnabled(True)
            for version in versions:
                self._version_combo.addItem(version.label, version.tag)
            if selected_tag is not None:
                idx = self._version_combo.findData(selected_tag)
                if idx >= 0:
                    self._version_combo.setCurrentIndex(idx)
        finally:
            self._suppress_version_signal = False

    def current_eye_version_tag(self) -> str:
        data = self._version_combo.currentData()
        return str(data) if data is not None else ""

    def set_catalog(self, catalog: ExploreCatalog | None) -> None:
        self._catalog = catalog
        prev_eye = set(self.enabled_eye_metric_ids())
        prev_ep = set(self.enabled_ep_stream_keys())
        self._rebuild_eye_picker(prefer_checked=prev_eye or None)
        self._rebuild_ep_picker(prefer_checked=prev_ep or None)
        if catalog is not None and len(catalog.ms_axis):
            self._center_ms.setValue(float(self._playhead_ms or catalog.ms_axis[0]))
        self._redraw()

    def playhead_ms(self) -> float:
        return float(self._playhead_ms)

    def set_playhead_ms(self, ms: float, *, emit: bool = False) -> None:
        self._playhead_ms = float(ms)
        self._center_ms.blockSignals(True)
        self._center_ms.setValue(self._playhead_ms)
        self._center_ms.blockSignals(False)
        for line in self._playheads.values():
            line.blockSignals(True)
            line.setPos(self._playhead_ms)
            line.blockSignals(False)
        if emit:
            self.time_selected.emit(self._playhead_ms)

    def enabled_eye_metric_ids(self) -> list[str]:
        return [
            mid
            for mid, chk in self._eye_checks.items()
            if chk.isChecked() and chk.isEnabled()
        ]

    def enabled_ep_stream_keys(self) -> list[str]:
        return [
            key
            for key, chk in self._ep_checks.items()
            if chk.isChecked() and chk.isEnabled()
        ]

    def zoom_to_all(self) -> None:
        for row_id, plot in self._plot_rows.items():
            plot.enableAutoRange(axis=pg.ViewBox.XYAxes)
            plot.autoRange(padding=0.05)
            self._y_limits.pop(row_id, None)
            self._sync_y_controls_from_plot(row_id)

    def _apply_x_window(self) -> None:
        center = float(self._center_ms.value())
        window = max(1.0, float(self._window_ms.value()))
        half = window / 2.0
        self.set_playhead_ms(center, emit=True)
        x0, x1 = center - half, center + half
        for plot in self._plot_rows.values():
            plot.setXRange(x0, x1, padding=0.0)

    def _apply_mouse_mode(self, *_args) -> None:
        mode = (
            pg.ViewBox.RectMode
            if self._box_zoom_btn.isChecked()
            else pg.ViewBox.PanMode
        )
        for plot in self._plot_rows.values():
            plot.getViewBox().setMouseMode(mode)

    def _clear_layout_items(
        self, layout: QtWidgets.QVBoxLayout, *, keep_stretch: bool = True
    ) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if keep_stretch:
            layout.addStretch(1)

    def _clear_plots(self) -> None:
        while self._plots_layout.count():
            item = self._plots_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._plot_rows.clear()
        self._playheads.clear()
        self._y_controls.clear()

    def _rebuild_eye_picker(self, prefer_checked: set[str] | None = None) -> None:
        self._clear_layout_items(self._eye_list_layout)
        self._eye_checks.clear()
        if self._catalog is None:
            lbl = QtWidgets.QLabel("(no eye data)")
            lbl.setStyleSheet("color: #888;")
            self._eye_list_layout.insertWidget(0, lbl)
            return
        metrics = available_eye_metric_ids(self._catalog)
        if not metrics:
            lbl = QtWidgets.QLabel("(no eye metrics)")
            lbl.setStyleSheet("color: #888;")
            self._eye_list_layout.insertWidget(0, lbl)
            return
        for mid in metrics:
            pair = self._catalog.eye_metrics[mid]
            chk = QtWidgets.QCheckBox(pair.label)
            if prefer_checked is not None:
                chk.setChecked(mid in prefer_checked)
            else:
                chk.setChecked(mid in DEFAULT_EYE_ENABLED)
            chk.toggled.connect(self._on_selection_changed)
            self._eye_checks[mid] = chk
            self._eye_list_layout.insertWidget(self._eye_list_layout.count() - 1, chk)

    def _rebuild_ep_picker(self, prefer_checked: set[str] | None = None) -> None:
        self._clear_layout_items(self._ep_list_layout)
        self._ep_checks.clear()
        self._ep_streams_by_key.clear()
        has_oe = (
            self._catalog is not None
            and self._catalog.oe_rec is not None
            and bool(self._catalog.oe_streams)
        )
        self._downsample.setEnabled(has_oe)
        if not has_oe:
            lbl = QtWidgets.QLabel("(no OE)")
            lbl.setStyleSheet("color: #888;")
            self._ep_list_layout.insertWidget(0, lbl)
            return
        assert self._catalog is not None
        for stream in self._catalog.oe_streams:
            key = ep_stream_key(stream)
            self._ep_streams_by_key[key] = stream
            chk = QtWidgets.QCheckBox(stream.label)
            chk.setChecked(bool(prefer_checked and key in prefer_checked))
            chk.toggled.connect(self._on_selection_changed)
            self._ep_checks[key] = chk
            self._ep_list_layout.insertWidget(self._ep_list_layout.count() - 1, chk)

    def _on_version_combo_changed(self, _index: int) -> None:
        if self._suppress_version_signal or self._updating:
            return
        self.eye_version_changed.emit(self.current_eye_version_tag())

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        self._redraw()

    def _on_ep_controls_changed(self) -> None:
        if self._updating or not self.enabled_ep_stream_keys():
            return
        self._redraw()

    def _make_plot_row(self, row_id: str, label: str) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("left", label)
        plot.setMinimumHeight(120)
        plot.addLegend(offset=(10, 10))
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen("k"))
            axis.setTextPen(pg.mkPen("k"))
        plot.setLabel("bottom", "Time (ms)")
        plot.getViewBox().setMouseMode(
            pg.ViewBox.RectMode
            if self._box_zoom_btn.isChecked()
            else pg.ViewBox.PanMode
        )
        plot.scene().sigMouseClicked.connect(
            lambda event, p=plot: self._on_plot_clicked(event, p)
        )
        playhead = pg.InfiniteLine(
            pos=self._playhead_ms,
            angle=90,
            movable=True,
            pen=pg.mkPen("#333333", width=2),
        )
        playhead.sigPositionChanged.connect(self._on_playhead_dragged)
        playhead.sigPositionChangeFinished.connect(self._on_playhead_drag_finished)
        plot.addItem(playhead)
        self._playheads[row_id] = playhead

        yctl = _YLimitControls()
        yctl.limits_applied.connect(
            lambda ymin, ymax, rid=row_id: self._on_y_limits(rid, ymin, ymax)
        )
        self._y_controls[row_id] = yctl

        row = QtWidgets.QWidget()
        row_lay = QtWidgets.QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(0)
        row_lay.addWidget(yctl)
        row_lay.addWidget(plot, stretch=1)
        self._plots_layout.addWidget(row)
        self._plot_rows[row_id] = plot
        return plot

    def _on_y_limits(self, row_id: str, ymin: float, ymax: float) -> None:
        self._y_limits[row_id] = (ymin, ymax)
        plot = self._plot_rows.get(row_id)
        if plot is not None:
            plot.setYRange(ymin, ymax, padding=0.0)

    def _sync_y_controls_from_plot(self, row_id: str) -> None:
        plot = self._plot_rows.get(row_id)
        yctl = self._y_controls.get(row_id)
        if plot is None or yctl is None:
            return
        if row_id in self._y_limits:
            ymin, ymax = self._y_limits[row_id]
        else:
            (ymin, ymax), _ = plot.viewRange()
        yctl.set_limits(float(ymin), float(ymax))

    def _link_x_axes(self) -> None:
        plots = list(self._plot_rows.values())
        if len(plots) < 2:
            return
        master = plots[0]
        for other in plots[1:]:
            other.setXLink(master)

    def _on_playhead_dragged(self) -> None:
        if self._updating:
            return
        sender = self.sender()
        if not isinstance(sender, pg.InfiniteLine):
            return
        ms = float(sender.value())
        self.set_playhead_ms(ms, emit=False)
        self.time_preview.emit(ms)

    def _on_playhead_drag_finished(self) -> None:
        if self._updating:
            return
        sender = self.sender()
        if not isinstance(sender, pg.InfiniteLine):
            return
        self.set_playhead_ms(float(sender.value()), emit=True)

    def _on_plot_clicked(self, event, plot: pg.PlotWidget) -> None:
        if self._box_zoom_btn.isChecked():
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if plot.sceneBoundingRect().contains(event.scenePos()):
            mouse_point = plot.getViewBox().mapSceneToView(event.scenePos())
            self.set_playhead_ms(float(mouse_point.x()), emit=True)

    def _add_eye_plot(self, metric_id: str) -> None:
        assert self._catalog is not None
        pair = self._catalog.eye_metrics.get(metric_id)
        if pair is None or not pair.available:
            return
        plot = self._make_plot_row(metric_id, pair.label)
        if pair.left is not None and pair.left.available:
            plot.plot(
                pair.left.time_ms,
                pair.left.values,
                pen=pg.mkPen(COLOR_LEFT, width=1.2),
                name="Left",
            )
        if pair.right is not None and pair.right.available:
            plot.plot(
                pair.right.time_ms,
                pair.right.values,
                pen=pg.mkPen(COLOR_RIGHT, width=1.2),
                name="Right",
            )

    def _add_ep_plot(self, key: str, color_index: int) -> None:
        assert self._catalog is not None
        stream = self._ep_streams_by_key.get(key)
        if stream is None:
            return
        ep = load_ep_overview(
            self._catalog,
            stream,
            downsample=int(self._downsample.value()),
        )
        if ep is None or not ep.available:
            return
        color = EP_COLORS[color_index % len(EP_COLORS)]
        row_id = f"ep:{key}"
        plot = self._make_plot_row(row_id, ep.label)
        plot.plot(
            ep.time_ms,
            ep.values,
            pen=pg.mkPen(color, width=1.2),
            name=ep.label,
        )

    def _redraw(self) -> None:
        self._updating = True
        try:
            self._clear_plots()
            eye_ids = self.enabled_eye_metric_ids()
            ep_keys = self.enabled_ep_stream_keys()
            if not eye_ids and not ep_keys:
                self._empty_label = QtWidgets.QLabel(
                    "No traces selected. Pick eye metrics and/or EP streams above."
                )
                self._empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self._plots_layout.addWidget(self._empty_label)
                return

            for mid in eye_ids:
                self._add_eye_plot(mid)
            for i, key in enumerate(ep_keys):
                self._add_ep_plot(key, i)

            if not self._plot_rows:
                self._empty_label = QtWidgets.QLabel(
                    "Selected traces could not be loaded."
                )
                self._empty_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self._plots_layout.addWidget(self._empty_label)
                return

            self._link_x_axes()
            for row_id, plot in self._plot_rows.items():
                plot.enableAutoRange(axis=pg.ViewBox.XYAxes)
                plot.autoRange(padding=0.05)
                if row_id in self._y_limits:
                    ymin, ymax = self._y_limits[row_id]
                    plot.setYRange(ymin, ymax, padding=0.0)
                self._sync_y_controls_from_plot(row_id)
            self.set_playhead_ms(self._playhead_ms, emit=False)
        finally:
            self._updating = False
