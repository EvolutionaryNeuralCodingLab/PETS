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
    min_height: int = 80,
    max_height: int = 160,
) -> tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
    box = QtWidgets.QGroupBox(title)
    outer = QtWidgets.QVBoxLayout(box)
    outer.setContentsMargins(6, 6, 6, 6)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setMinimumHeight(min_height)
    scroll.setMaximumHeight(max_height)
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    lay.setContentsMargins(2, 2, 2, 2)
    lay.setSpacing(2)
    lay.addStretch(1)
    scroll.setWidget(host)
    outer.addWidget(scroll)
    return box, lay


class ExplorePlotPanel(QtWidgets.QWidget):
    """Paired eye metrics + multi-EP stacked plots with shared X (ms)."""

    time_selected = QtCore.pyqtSignal(float)
    refresh_requested = QtCore.pyqtSignal()
    eye_version_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._catalog: ExploreCatalog | None = None
        self._playhead_ms = 0.0
        self._plot_rows: dict[str, pg.PlotWidget] = {}
        self._playheads: dict[str, pg.InfiniteLine] = {}
        self._eye_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._ep_checks: dict[str, QtWidgets.QCheckBox] = {}
        self._ep_streams_by_key: dict[str, OEStream] = {}
        self._updating = False
        self._suppress_version_signal = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        version_row = QtWidgets.QHBoxLayout()
        version_row.addWidget(QtWidgets.QLabel("Eye data version:"))
        self._version_combo = QtWidgets.QComboBox()
        self._version_combo.setMinimumWidth(220)
        self._version_combo.setEnabled(False)
        version_row.addWidget(self._version_combo, stretch=1)
        version_row.addStretch(1)
        layout.addLayout(version_row)

        pickers = QtWidgets.QHBoxLayout()
        eye_box, self._eye_list_layout = _make_check_list("Eye traces (L+R)")
        ep_box, self._ep_list_layout = _make_check_list(
            "Electrophysiology", min_height=80, max_height=200
        )
        pickers.addWidget(eye_box, stretch=1)
        pickers.addWidget(ep_box, stretch=1)

        ep_opts = QtWidgets.QVBoxLayout()
        ep_opts.addWidget(QtWidgets.QLabel("EP downsample"))
        self._downsample = QtWidgets.QSpinBox()
        self._downsample.setRange(1, 10000)
        self._downsample.setValue(50)
        self._downsample.setEnabled(False)
        ep_opts.addWidget(self._downsample)
        ep_opts.addStretch(1)
        self._box_zoom_btn = QtWidgets.QPushButton("Box zoom")
        self._box_zoom_btn.setCheckable(True)
        self._box_zoom_btn.setChecked(True)
        self._box_zoom_btn.setToolTip(
            "When on: drag a rectangle to zoom. Drag the playhead (or turn off) to set time.\n"
            "When off: left-drag pans; click sets the playhead."
        )
        self._zoom_all_btn = QtWidgets.QPushButton("Zoom to all")
        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        ep_opts.addWidget(self._box_zoom_btn)
        ep_opts.addWidget(self._zoom_all_btn)
        ep_opts.addWidget(self._refresh_btn)
        pickers.addLayout(ep_opts)
        layout.addLayout(pickers)

        legend = QtWidgets.QLabel(
            '<span style="color:#1f77b4;">● Left</span> &nbsp; '
            '<span style="color:#d62728;">● Right</span>'
        )
        layout.addWidget(legend)

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
        # Preserve checked eye metrics / EP streams across version switches.
        prev_eye = set(self.enabled_eye_metric_ids())
        prev_ep = set(self.enabled_ep_stream_keys())
        self._rebuild_eye_picker(prefer_checked=prev_eye or None)
        self._rebuild_ep_picker(prefer_checked=prev_ep or None)
        self._redraw()

    def playhead_ms(self) -> float:
        return float(self._playhead_ms)

    def set_playhead_ms(self, ms: float, *, emit: bool = False) -> None:
        self._playhead_ms = float(ms)
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
        for plot in self._plot_rows.values():
            plot.enableAutoRange(axis=pg.ViewBox.XYAxes)
            plot.autoRange(padding=0.05)

    def _apply_mouse_mode(self, *_args) -> None:
        """Box zoom (RectMode) vs pan (PanMode) on all plot rows."""
        mode = (
            pg.ViewBox.RectMode
            if self._box_zoom_btn.isChecked()
            else pg.ViewBox.PanMode
        )
        for plot in self._plot_rows.values():
            plot.getViewBox().setMouseMode(mode)

    def _clear_layout_items(self, layout: QtWidgets.QVBoxLayout, *, keep_stretch: bool = True) -> None:
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
            lbl = QtWidgets.QLabel("(no eye metrics available)")
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
            lbl = QtWidgets.QLabel("(no OE streams)")
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
        plot.addItem(playhead)
        self._playheads[row_id] = playhead
        return plot

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
        self.set_playhead_ms(float(sender.value()), emit=True)

    def _on_plot_clicked(self, event, plot: pg.PlotWidget) -> None:
        # In box-zoom mode, left-drag is for the rectangle; playhead via InfiniteLine.
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
        self._plot_rows[metric_id] = plot
        self._plots_layout.addWidget(plot)

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
        self._plot_rows[row_id] = plot
        self._plots_layout.addWidget(plot)

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
            self.zoom_to_all()
            self.set_playhead_ms(self._playhead_ms, emit=False)
        finally:
            self._updating = False
