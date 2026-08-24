"""Pyqtgraph velocity + threshold panel for the Saccades preprocessing tab.

Plots only a time window (default 100 s) for responsiveness. Full traces stay
in the parent tab; this panel receives the already-sliced window DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

pg.setConfigOptions(antialias=True, background="w", foreground="k")

L_COLOR = "#1f77b4"
R_COLOR = "#d62728"
THR_COLOR = "#222222"
RAW_BRUSH = pg.mkBrush(200, 200, 200, 60)
CONCURRENT_BRUSH = pg.mkBrush(46, 139, 87, 80)
MONOCULAR_BRUSH = pg.mkBrush(210, 105, 30, 70)
BAD_EVENT_BRUSH = pg.mkBrush(176, 48, 48, 110)
BAD_SPAN_BRUSH = pg.mkBrush(176, 48, 48, 35)
ZOOM_BRUSH = pg.mkBrush(100, 100, 255, 40)
BAD_SELECT_BRUSH = pg.mkBrush(176, 48, 48, 50)
MAX_DRAW = 12_000
DEFAULT_WINDOW_MS = 100_000.0  # 100 seconds


def _downsample(t: np.ndarray, y: np.ndarray, max_n: int = MAX_DRAW):
    n = len(t)
    if n <= max_n:
        return t, y
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return t[idx], y[idx]


def _disable_si_prefix(plot: pg.PlotWidget, axis: str = "bottom") -> None:
    """Keep axis labels in raw units (avoid pyqtgraph 'kms' for large ms values)."""
    ax = plot.getAxis(axis)
    ax.enableAutoSIPrefix(False)


class SaccadeVelocityPanel(QtWidgets.QWidget):
    """Both-eye speed traces with a draggable horizontal threshold."""

    threshold_changed = QtCore.pyqtSignal(float)
    mode_changed = QtCore.pyqtSignal(str)  # "deg" | "px"
    window_nav_requested = QtCore.pyqtSignal(str)  # "prev" | "next" | "fit"
    time_selected = QtCore.pyqtSignal(float)  # absolute ms — seek video
    time_preview = QtCore.pyqtSignal(float)  # absolute ms — playhead drag
    bad_span_tag_requested = QtCore.pyqtSignal(float, float)  # abs ms t0, t1
    bad_span_untag_requested = QtCore.pyqtSignal(float, float)
    bad_spans_clear_requested = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = "deg"
        self._threshold = 0.8
        self._t_l = np.array([])
        self._t_r = np.array([])
        self._spd_l = np.array([])
        self._spd_r = np.array([])
        self._t0 = 0.0  # absolute ms origin of current window (for overlays)
        self._playhead_ms = 0.0  # absolute ms
        self._updating = False
        self._raw_regions: list[pg.LinearRegionItem] = []
        self._event_regions: list[pg.LinearRegionItem] = []
        self._bad_span_regions: list[pg.LinearRegionItem] = []
        self._select_region: pg.LinearRegionItem | None = None
        self._region_mode: str | None = None  # None | "zoom" | "bad"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QtWidgets.QHBoxLayout()
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._rb_deg = QtWidgets.QRadioButton("Degrees")
        self._rb_px = QtWidgets.QRadioButton("Pixels")
        self._rb_deg.setChecked(True)
        self._rb_deg.setToolTip("Angular Euclidean speed √(Δk_φ² + Δk_θ²) in deg/frame")
        self._rb_px.setToolTip("Pixel Euclidean speed √(Δx² + Δy²) in px/frame")
        self._mode_group.addButton(self._rb_deg)
        self._mode_group.addButton(self._rb_px)
        toolbar.addWidget(self._rb_deg)
        toolbar.addWidget(self._rb_px)
        toolbar.addSpacing(12)

        self._btn_prev = QtWidgets.QPushButton("◀ Prev 100 s")
        self._btn_prev.setToolTip("Load the previous 100-second window")
        self._btn_next = QtWidgets.QPushButton("Next 100 s ▶")
        self._btn_next.setToolTip("Load the next 100-second window")
        self._window_label = QtWidgets.QLabel("Window: —")
        toolbar.addWidget(self._btn_prev)
        toolbar.addWidget(self._btn_next)
        toolbar.addWidget(self._window_label)
        toolbar.addStretch(1)

        self._btn_zoom_region = QtWidgets.QPushButton("Zoom to region")
        self._btn_zoom_region.setCheckable(True)
        self._btn_zoom_region.setToolTip(
            "Drag a horizontal region on the plot, then click Apply zoom "
            "(or uncheck to cancel)."
        )
        self._btn_apply_zoom = QtWidgets.QPushButton("Apply zoom")
        self._btn_apply_zoom.setEnabled(False)
        self._btn_apply_zoom.setToolTip("Zoom X axis to the selected region")
        self._btn_fit = QtWidgets.QPushButton("Fit window")
        self._btn_fit.setToolTip("Reset X/Y view to the full current 100 s window")
        self._btn_autoscale = QtWidgets.QPushButton("Auto-zoom Y")
        self._btn_autoscale.setToolTip("Rescale Y to the 0.5–99.5 percentile of visible speed")
        for b in (
            self._btn_zoom_region,
            self._btn_apply_zoom,
            self._btn_fit,
            self._btn_autoscale,
        ):
            toolbar.addWidget(b)
        layout.addLayout(toolbar)

        bad_bar = QtWidgets.QHBoxLayout()
        self._btn_mark_bad = QtWidgets.QPushButton("Mark bad region")
        self._btn_mark_bad.setCheckable(True)
        self._btn_mark_bad.setToolTip(
            "Drag a time region, then Tag as bad to set bad_detections=True "
            "on overlapping saccades (downstream: query bad_detections==False)."
        )
        self._btn_tag_bad = QtWidgets.QPushButton("Tag as bad")
        self._btn_tag_bad.setEnabled(False)
        self._btn_tag_bad.setToolTip(
            "Flag saccades overlapping the selected region as bad_detections"
        )
        self._btn_untag_bad = QtWidgets.QPushButton("Untag region")
        self._btn_untag_bad.setEnabled(False)
        self._btn_untag_bad.setToolTip(
            "Clear bad_detections for saccades overlapping the selected region"
        )
        self._btn_clear_bad = QtWidgets.QPushButton("Clear all bad tags")
        self._btn_clear_bad.setToolTip("Set bad_detections=False on every event")
        self._bad_label = QtWidgets.QLabel("bad_detections: 0")
        for b in (
            self._btn_mark_bad,
            self._btn_tag_bad,
            self._btn_untag_bad,
            self._btn_clear_bad,
        ):
            bad_bar.addWidget(b)
        bad_bar.addWidget(self._bad_label)
        bad_bar.addStretch(1)
        layout.addLayout(bad_bar)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time (ms)")
        self._plot.setLabel("left", "Speed (deg/frame)")
        _disable_si_prefix(self._plot, "bottom")
        _disable_si_prefix(self._plot, "left")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.addLegend(offset=(10, 10))
        # Rect-drag zoom with middle button; left pan remains default via ViewBox
        vb = self._plot.getViewBox()
        vb.setMouseMode(pg.ViewBox.PanMode)
        self._plot.setToolTip(
            "Scroll to zoom · drag to pan · click / drag vertical playhead to seek · "
            "middle-drag for rect zoom · or use Zoom to region"
        )
        layout.addWidget(self._plot, stretch=1)

        self._curve_l = self._plot.plot(pen=pg.mkPen(L_COLOR, width=1), name="L")
        self._curve_r = self._plot.plot(pen=pg.mkPen(R_COLOR, width=1), name="R")
        self._thr_line = pg.InfiniteLine(
            pos=self._threshold,
            angle=0,
            movable=True,
            pen=pg.mkPen(THR_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
            label="thr={value:.4g}",
            labelOpts={"position": 0.95, "color": THR_COLOR},
        )
        self._plot.addItem(self._thr_line)
        self._playhead_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#333333", width=2),
        )
        self._playhead_line.setZValue(20)
        self._plot.addItem(self._playhead_line)

        self._rb_deg.toggled.connect(self._on_mode_toggled)
        self._thr_line.sigPositionChanged.connect(self._on_thr_dragged)
        self._playhead_line.sigPositionChanged.connect(self._on_playhead_dragged)
        self._playhead_line.sigPositionChangeFinished.connect(
            self._on_playhead_drag_finished
        )
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
        self._btn_autoscale.clicked.connect(self._autoscale_y)
        self._btn_fit.clicked.connect(self._fit_window)
        self._btn_zoom_region.toggled.connect(self._on_zoom_region_toggled)
        self._btn_apply_zoom.clicked.connect(self._apply_zoom_region)
        self._btn_mark_bad.toggled.connect(self._on_mark_bad_toggled)
        self._btn_tag_bad.clicked.connect(self._emit_tag_bad)
        self._btn_untag_bad.clicked.connect(self._emit_untag_bad)
        self._btn_clear_bad.clicked.connect(self.bad_spans_clear_requested.emit)
        self._btn_prev.clicked.connect(lambda: self.window_nav_requested.emit("prev"))
        self._btn_next.clicked.connect(lambda: self.window_nav_requested.emit("next"))

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def threshold(self) -> float:
        return float(self._threshold)

    def playhead_ms(self) -> float:
        return float(self._playhead_ms)

    def set_playhead_ms(self, ms: float, *, emit: bool = False) -> None:
        """Set playhead in absolute ms; X position is relative to the window origin."""
        self._playhead_ms = float(ms)
        self._updating = True
        self._playhead_line.setPos(self._playhead_ms - self._t0)
        self._updating = False
        if emit:
            self.time_selected.emit(self._playhead_ms)

    def set_nav_enabled(self, *, prev: bool, next_: bool) -> None:
        self._btn_prev.setEnabled(prev)
        self._btn_next.setEnabled(next_)

    def set_window_label(self, text: str) -> None:
        self._window_label.setText(text)

    def set_threshold(self, value: float, *, emit: bool = False) -> None:
        self._threshold = float(value)
        self._updating = True
        self._thr_line.setValue(self._threshold)
        self._updating = False
        if emit:
            self.threshold_changed.emit(self._threshold)

    def set_traces(
        self,
        left: pd.DataFrame | None,
        right: pd.DataFrame | None,
        *,
        mode: str | None = None,
        window_t0_ms: float | None = None,
    ) -> None:
        """Plot one window. Times are absolute ``ms_axis``; X is relative to window start."""
        if mode is not None:
            self._mode = mode
            self._rb_deg.blockSignals(True)
            self._rb_px.blockSignals(True)
            self._rb_deg.setChecked(mode == "deg")
            self._rb_px.setChecked(mode == "px")
            self._rb_deg.blockSignals(False)
            self._rb_px.blockSignals(False)

        speed_col = "angular_speed_r" if self._mode == "deg" else "speed_r"
        unit = "deg/frame" if self._mode == "deg" else "px/frame"
        self._plot.setLabel("left", f"Speed ({unit})")
        _disable_si_prefix(self._plot, "left")

        def _extract(df: pd.DataFrame | None):
            if df is None or df.empty or "ms_axis" not in df.columns:
                return np.array([]), np.array([])
            if speed_col not in df.columns:
                return np.array([]), np.array([])
            t = df["ms_axis"].to_numpy(dtype=float)
            y = df[speed_col].to_numpy(dtype=float)
            return t, y

        self._t_l, self._spd_l = _extract(left)
        self._t_r, self._spd_r = _extract(right)
        if window_t0_ms is not None:
            self._t0 = float(window_t0_ms)
        else:
            t0_candidates = [t[0] for t in (self._t_l, self._t_r) if len(t)]
            self._t0 = float(min(t0_candidates)) if t0_candidates else 0.0

        self._clear_select_region()
        self._btn_zoom_region.blockSignals(True)
        self._btn_mark_bad.blockSignals(True)
        self._btn_zoom_region.setChecked(False)
        self._btn_mark_bad.setChecked(False)
        self._btn_zoom_region.blockSignals(False)
        self._btn_mark_bad.blockSignals(False)
        self._redraw_curves()
        self._fit_window()
        self.clear_event_overlays()
        self.clear_raw_overlays()
        self.clear_bad_span_overlays()
        # Keep absolute playhead; re-map relative X after window origin changes.
        self.set_playhead_ms(self._playhead_ms, emit=False)

    def _redraw_curves(self) -> None:
        if len(self._t_l):
            t, y = _downsample(self._t_l - self._t0, self._spd_l)
            self._curve_l.setData(t, y)
        else:
            self._curve_l.setData([], [])
        if len(self._t_r):
            t, y = _downsample(self._t_r - self._t0, self._spd_r)
            self._curve_r.setData(t, y)
        else:
            self._curve_r.setData([], [])

    def _autoscale_y(self) -> None:
        parts = [a[np.isfinite(a)] for a in (self._spd_l, self._spd_r) if len(a)]
        if not parts:
            return
        v = np.concatenate(parts)
        if v.size == 0:
            return
        lo = float(np.nanpercentile(v, 0.5))
        hi = float(np.nanpercentile(v, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        pad = 0.08 * (hi - lo)
        self._plot.setYRange(lo - pad, hi + pad, padding=0)

    def _fit_window(self) -> None:
        tmax = 0.0
        for t in (self._t_l, self._t_r):
            if len(t):
                tmax = max(tmax, float(np.nanmax(t) - self._t0))
        if tmax <= 0:
            tmax = 1.0
        self._plot.setXRange(0.0, tmax, padding=0.02)
        self._autoscale_y()

    def _on_zoom_region_toggled(self, checked: bool) -> None:
        if checked:
            self._btn_mark_bad.blockSignals(True)
            self._btn_mark_bad.setChecked(False)
            self._btn_mark_bad.blockSignals(False)
            self._enter_select_region("zoom")
            return
        if self._region_mode == "zoom":
            self._clear_select_region()

    def _on_mark_bad_toggled(self, checked: bool) -> None:
        if checked:
            self._btn_zoom_region.blockSignals(True)
            self._btn_zoom_region.setChecked(False)
            self._btn_zoom_region.blockSignals(False)
            self._enter_select_region("bad")
            return
        if self._region_mode == "bad":
            self._clear_select_region()

    def _enter_select_region(self, mode: str) -> None:
        self._clear_select_region()
        self._region_mode = mode
        x0, x1 = self._plot.viewRange()[0]
        mid = 0.5 * (x0 + x1)
        half = 0.15 * max(x1 - x0, 1.0)
        brush = ZOOM_BRUSH if mode == "zoom" else BAD_SELECT_BRUSH
        self._select_region = pg.LinearRegionItem(
            values=(mid - half, mid + half),
            movable=True,
            brush=brush,
        )
        self._select_region.setZValue(10)
        self._plot.addItem(self._select_region)
        self._btn_apply_zoom.setEnabled(mode == "zoom")
        self._btn_tag_bad.setEnabled(mode == "bad")
        self._btn_untag_bad.setEnabled(mode == "bad")

    def _clear_select_region(self) -> None:
        if self._select_region is not None:
            self._plot.removeItem(self._select_region)
            self._select_region = None
        self._region_mode = None
        self._btn_apply_zoom.setEnabled(False)
        self._btn_tag_bad.setEnabled(False)
        self._btn_untag_bad.setEnabled(False)

    def _selected_abs_span(self) -> tuple[float, float] | None:
        if self._select_region is None:
            return None
        x0, x1 = self._select_region.getRegion()
        if x1 < x0:
            x0, x1 = x1, x0
        if x1 - x0 < 1.0:
            return None
        return float(x0) + self._t0, float(x1) + self._t0

    def _apply_zoom_region(self) -> None:
        span = self._selected_abs_span()
        if span is None:
            return
        t0, t1 = span
        self._plot.setXRange(t0 - self._t0, t1 - self._t0, padding=0)
        self._btn_zoom_region.setChecked(False)

    def _emit_tag_bad(self) -> None:
        span = self._selected_abs_span()
        if span is None:
            return
        self.bad_span_tag_requested.emit(*span)

    def _emit_untag_bad(self) -> None:
        span = self._selected_abs_span()
        if span is None:
            return
        self.bad_span_untag_requested.emit(*span)

    def set_bad_count(self, n_bad: int, n_total: int | None = None) -> None:
        if n_total is None:
            self._bad_label.setText(f"bad_detections: {int(n_bad)}")
        else:
            self._bad_label.setText(
                f"bad_detections: {int(n_bad)} / {int(n_total)}"
            )

    def clear_raw_overlays(self) -> None:
        for r in self._raw_regions:
            self._plot.removeItem(r)
        self._raw_regions.clear()

    def clear_event_overlays(self) -> None:
        for r in self._event_regions:
            self._plot.removeItem(r)
        self._event_regions.clear()

    def clear_bad_span_overlays(self) -> None:
        for r in self._bad_span_regions:
            self._plot.removeItem(r)
        self._bad_span_regions.clear()

    def set_bad_spans(self, spans: list[tuple[float, float]]) -> None:
        """Draw persistent bad time sections (absolute ms)."""
        self.clear_bad_span_overlays()
        t_end = self._t0
        for t in (self._t_l, self._t_r):
            if len(t):
                t_end = max(t_end, float(np.nanmax(t)))
        for on, off in spans or []:
            if off < self._t0 or on > t_end:
                continue
            region = pg.LinearRegionItem(
                values=(max(on, self._t0) - self._t0, min(off, t_end) - self._t0),
                movable=False,
                brush=BAD_SPAN_BRUSH,
                pen=pg.mkPen(None),
            )
            region.setZValue(-15)
            self._plot.addItem(region)
            self._bad_span_regions.append(region)

    def set_raw_runs(self, runs: list[tuple[float, float]]) -> None:
        """``runs`` are absolute ms onsets/offsets; only those overlapping the window are drawn."""
        self.clear_raw_overlays()
        t_end = self._t0
        for t in (self._t_l, self._t_r):
            if len(t):
                t_end = max(t_end, float(np.nanmax(t)))
        for on, off in runs:
            if off < self._t0 or on > t_end:
                continue
            region = pg.LinearRegionItem(
                values=(max(on, self._t0) - self._t0, min(off, t_end) - self._t0),
                movable=False,
                brush=RAW_BRUSH,
                pen=pg.mkPen(None),
            )
            region.setZValue(-20)
            self._plot.addItem(region)
            self._raw_regions.append(region)

    def set_event_overlays(
        self,
        events: pd.DataFrame,
        *,
        concurrency_col: str = "concurrency",
    ) -> None:
        self.clear_event_overlays()
        if events is None or events.empty:
            return
        on_col = "saccade_on_ms"
        off_col = "saccade_off_ms"
        if on_col not in events.columns or off_col not in events.columns:
            return
        t_end = self._t0
        for t in (self._t_l, self._t_r):
            if len(t):
                t_end = max(t_end, float(np.nanmax(t)))
        # Cap overlays for responsiveness
        n_drawn = 0
        max_regions = 800
        for _, row in events.iterrows():
            on = float(row[on_col])
            off = float(row[off_col])
            if not (np.isfinite(on) and np.isfinite(off)):
                continue
            if off < self._t0 or on > t_end:
                continue
            is_bad = False
            if "bad_detections" in events.columns:
                is_bad = bool(row["bad_detections"])
            if is_bad:
                brush = BAD_EVENT_BRUSH
            else:
                conc = (
                    str(row[concurrency_col])
                    if concurrency_col in events.columns
                    else "monocular"
                )
                brush = CONCURRENT_BRUSH if conc == "concurrent" else MONOCULAR_BRUSH
            region = pg.LinearRegionItem(
                values=(max(on, self._t0) - self._t0, min(off, t_end) - self._t0),
                movable=False,
                brush=brush,
                pen=pg.mkPen(None),
            )
            region.setZValue(-10)
            self._plot.addItem(region)
            self._event_regions.append(region)
            n_drawn += 1
            if n_drawn >= max_regions:
                break

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self._mode = "deg" if self._rb_deg.isChecked() else "px"
        self.mode_changed.emit(self._mode)

    def _on_thr_dragged(self) -> None:
        if self._updating:
            return
        self._threshold = float(self._thr_line.value())
        self.threshold_changed.emit(self._threshold)

    def _on_playhead_dragged(self) -> None:
        if self._updating:
            return
        rel = float(self._playhead_line.value())
        self._playhead_ms = rel + self._t0
        self.time_preview.emit(self._playhead_ms)

    def _on_playhead_drag_finished(self) -> None:
        if self._updating:
            return
        rel = float(self._playhead_line.value())
        self.set_playhead_ms(rel + self._t0, emit=True)

    def _on_plot_clicked(self, event) -> None:
        if self._region_mode is not None:
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self._plot.sceneBoundingRect().contains(event.scenePos()):
            mouse_point = self._plot.getViewBox().mapSceneToView(event.scenePos())
            self.set_playhead_ms(float(mouse_point.x()) + self._t0, emit=True)
