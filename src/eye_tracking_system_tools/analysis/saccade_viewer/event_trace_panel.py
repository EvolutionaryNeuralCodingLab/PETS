"""Event-window velocity and optional phi/theta trace panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.analysis.param_tune import prepare_traces
from eye_tracking_system_tools.analysis.saccade_viewer.models import VerificationEvent

pg.setConfigOptions(antialias=True, background="w", foreground="k")

L_COLOR = "#1f77b4"
R_COLOR = "#d62728"
ONSET_COLOR = "#333333"
SACCADE_BRUSH = pg.mkBrush(46, 139, 87, 50)
MAX_DRAW = 8_000


def _downsample(t: np.ndarray, y: np.ndarray, max_n: int = MAX_DRAW):
    n = len(t)
    if n <= max_n:
        return t, y
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return t[idx], y[idx]


def _disable_si(plot: pg.PlotWidget, axis: str = "bottom") -> None:
    plot.getAxis(axis).enableAutoSIPrefix(False)


class EventTracePanel(QtWidgets.QWidget):
    """Both-eye angular speed around a single event; optional φ/θ subplots."""

    time_selected = QtCore.pyqtSignal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._left: pd.DataFrame | None = None
        self._right: pd.DataFrame | None = None
        self._event: VerificationEvent | None = None
        self._pre_ms = 250.0
        self._post_ms = 250.0
        self._playhead_ms = 0.0
        self._show_angles = False
        self._saccade_region: pg.LinearRegionItem | None = None
        self._angle_saccade_region: pg.LinearRegionItem | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QtWidgets.QHBoxLayout()
        self._toggle_angles = QtWidgets.QCheckBox("Show φ / θ")
        self._toggle_angles.setToolTip(
            "Split the trace area to show raw k_phi and k_theta for each eye"
        )
        self._toggle_angles.toggled.connect(self._on_toggle_angles)
        toolbar.addWidget(self._toggle_angles)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self._plot_stack = QtWidgets.QWidget()
        stack_layout = QtWidgets.QVBoxLayout(self._plot_stack)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.setSpacing(2)

        self._speed_plot = pg.PlotWidget()
        self._speed_plot.setLabel("bottom", "Time (ms)")
        self._speed_plot.setLabel("left", "Speed (deg/frame)")
        _disable_si(self._speed_plot, "bottom")
        _disable_si(self._speed_plot, "left")
        self._speed_plot.showGrid(x=True, y=True, alpha=0.25)
        self._speed_plot.addLegend(offset=(10, 10))

        self._angle_plot = pg.PlotWidget()
        self._angle_plot.setLabel("bottom", "Time (ms)")
        self._angle_plot.setLabel("left", "Angle (deg)")
        _disable_si(self._angle_plot, "bottom")
        _disable_si(self._angle_plot, "left")
        self._angle_plot.showGrid(x=True, y=True, alpha=0.25)
        self._angle_plot.addLegend(offset=(10, 10))
        self._angle_plot.setXLink(self._speed_plot)
        self._angle_plot.hide()

        stack_layout.addWidget(self._speed_plot, stretch=1)
        stack_layout.addWidget(self._angle_plot, stretch=1)
        layout.addWidget(self._plot_stack, stretch=1)

        self._curve_l = self._speed_plot.plot(pen=pg.mkPen(L_COLOR, width=1.5), name="L spd")
        self._curve_r = self._speed_plot.plot(pen=pg.mkPen(R_COLOR, width=1.5), name="R spd")
        self._onset_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen(ONSET_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._playhead_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#888888", width=1.5),
        )
        self._speed_plot.addItem(self._onset_line)
        self._speed_plot.addItem(self._playhead_line)
        self._playhead_line.sigPositionChangeFinished.connect(self._on_playhead_finished)
        self._speed_plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self._l_phi = self._angle_plot.plot(
            pen=pg.mkPen(L_COLOR, width=1.2), name="L φ"
        )
        self._l_theta = self._angle_plot.plot(
            pen=pg.mkPen(L_COLOR, width=1.2, style=QtCore.Qt.PenStyle.DashLine),
            name="L θ",
        )
        self._r_phi = self._angle_plot.plot(
            pen=pg.mkPen(R_COLOR, width=1.2), name="R φ"
        )
        self._r_theta = self._angle_plot.plot(
            pen=pg.mkPen(R_COLOR, width=1.2, style=QtCore.Qt.PenStyle.DashLine),
            name="R θ",
        )
        self._angle_onset_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen(ONSET_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._angle_playhead_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#888888", width=1.5),
        )
        self._angle_plot.addItem(self._angle_onset_line)
        self._angle_plot.addItem(self._angle_playhead_line)

        self._meta = QtWidgets.QLabel("—")
        self._meta.setWordWrap(True)
        layout.addWidget(self._meta)

    def _on_toggle_angles(self, checked: bool) -> None:
        self._show_angles = bool(checked)
        self._angle_plot.setVisible(self._show_angles)
        self._redraw()

    def set_traces(self, left: pd.DataFrame | None, right: pd.DataFrame | None) -> None:
        self._left = prepare_traces(left) if left is not None else None
        self._right = prepare_traces(right) if right is not None else None
        self._redraw()

    def set_pre_post_ms(self, pre_ms: float, post_ms: float) -> None:
        self._pre_ms = max(0.0, float(pre_ms))
        self._post_ms = max(0.0, float(post_ms))
        self._redraw()

    def set_event(self, event: VerificationEvent | None) -> None:
        self._event = event
        self._redraw()

    def set_playhead_ms(self, ms: float) -> None:
        self._playhead_ms = float(ms)
        if self._event is None:
            return
        rel = self._playhead_ms - (self._event.onset_ms - self._pre_ms)
        for line in (self._playhead_line, self._angle_playhead_line):
            line.blockSignals(True)
            line.setPos(rel)
            line.blockSignals(False)

    def _window_abs(self) -> tuple[float, float] | None:
        if self._event is None:
            return None
        t0 = self._event.onset_ms - self._pre_ms
        t1 = self._event.onset_ms + self._post_ms
        return t0, t1

    def _slice_col(
        self,
        df: pd.DataFrame | None,
        col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if df is None or df.empty or "ms_axis" not in df.columns or col not in df.columns:
            return np.array([]), np.array([])
        win = self._window_abs()
        if win is None:
            return np.array([]), np.array([])
        t0, t1 = win
        ms = df["ms_axis"].to_numpy(dtype=float)
        y = df[col].to_numpy(dtype=float)
        mask = np.isfinite(ms) & (ms >= t0) & (ms <= t1)
        t = ms[mask]
        y = y[mask]
        return _downsample(t, y)

    def _slice_speed(self, df: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
        if df is None or df.empty or "ms_axis" not in df.columns:
            return np.array([]), np.array([])
        win = self._window_abs()
        if win is None:
            return np.array([]), np.array([])
        t0, t1 = win
        ms = df["ms_axis"].to_numpy(dtype=float)
        spd = (
            df["angular_speed_r"].to_numpy(dtype=float)
            if "angular_speed_r" in df.columns
            else np.full(len(ms), np.nan)
        )
        mask = np.isfinite(ms) & (ms >= t0) & (ms <= t1)
        return _downsample(ms[mask], spd[mask])

    def _clear_saccade_regions(self) -> None:
        for plot, attr in (
            (self._speed_plot, "_saccade_region"),
            (self._angle_plot, "_angle_saccade_region"),
        ):
            region = getattr(self, attr)
            if region is not None:
                plot.removeItem(region)
                setattr(self, attr, None)

    def _add_saccade_region(self, plot: pg.PlotWidget, rel_onset: float, rel_off: float):
        region = pg.LinearRegionItem(
            values=[rel_onset, rel_off],
            movable=False,
            brush=SACCADE_BRUSH,
        )
        region.setZValue(-5)
        plot.addItem(region)
        return region

    def _autoscale_y(self, plot: pg.PlotWidget, arrays: list[np.ndarray]) -> None:
        parts = [y for y in arrays if len(y)]
        if not parts:
            return
        vals = np.concatenate(parts)
        lo = float(np.nanpercentile(vals, 1))
        hi = float(np.nanpercentile(vals, 99))
        if hi <= lo:
            hi = lo + 1.0
        plot.setYRange(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    def _redraw(self) -> None:
        self._clear_saccade_regions()

        if self._event is None:
            self._curve_l.setData([], [])
            self._curve_r.setData([], [])
            for c in (self._l_phi, self._l_theta, self._r_phi, self._r_theta):
                c.setData([], [])
            self._meta.setText("—")
            return

        win = self._window_abs()
        assert win is not None
        t0, t1 = win
        span = t1 - t0

        t_l, y_l = self._slice_speed(self._left)
        t_r, y_r = self._slice_speed(self._right)
        self._curve_l.setData(t_l - t0, y_l)
        self._curve_r.setData(t_r - t0, y_r)

        rel_onset = self._event.onset_ms - t0
        self._onset_line.setPos(rel_onset)
        self._angle_onset_line.setPos(rel_onset)

        if self._event.off_ms > self._event.onset_ms:
            rel_off = self._event.off_ms - t0
            self._saccade_region = self._add_saccade_region(
                self._speed_plot, rel_onset, rel_off
            )
            if self._show_angles:
                self._angle_saccade_region = self._add_saccade_region(
                    self._angle_plot, rel_onset, rel_off
                )

        self._speed_plot.setXRange(0, span, padding=0.02)
        self._autoscale_y(self._speed_plot, [y_l, y_r])

        if self._show_angles:
            t_l_phi, y_l_phi = self._slice_col(self._left, "k_phi")
            t_l_theta, y_l_theta = self._slice_col(self._left, "k_theta")
            t_r_phi, y_r_phi = self._slice_col(self._right, "k_phi")
            t_r_theta, y_r_theta = self._slice_col(self._right, "k_theta")
            self._l_phi.setData(t_l_phi - t0, y_l_phi)
            self._l_theta.setData(t_l_theta - t0, y_l_theta)
            self._r_phi.setData(t_r_phi - t0, y_r_phi)
            self._r_theta.setData(t_r_theta - t0, y_r_theta)
            self._autoscale_y(
                self._angle_plot,
                [y_l_phi, y_l_theta, y_r_phi, y_r_theta],
            )

        amp = self._event.extras.get("net_angular_disp", "—")
        det = self._event.extras.get("concurrency", "—")
        pairing = self._event.pairing_tag
        self._meta.setText(
            f"<b>{self._event.block_key}</b> · eye {self._event.eye or '?'} · "
            f"onset {self._event.onset_ms:.1f} ms · off {self._event.off_ms:.1f} ms · "
            f"amp {amp} · detected {det} · "
            f"QC <b>{self._event.verification_status}</b> · "
            f"manual pairing <b>{pairing}</b>"
        )
        self.set_playhead_ms(self._event.onset_ms - self._pre_ms)

    def _abs_from_rel(self, rel_x: float) -> float | None:
        if self._event is None:
            return None
        return float(self._event.onset_ms - self._pre_ms + rel_x)

    def _on_playhead_finished(self) -> None:
        ms = self._abs_from_rel(float(self._playhead_line.value()))
        if ms is not None:
            self.time_selected.emit(ms)

    def _on_plot_clicked(self, event) -> None:
        if self._event is None:
            return
        pos = self._speed_plot.plotItem.vb.mapSceneToView(event.scenePos())
        ms = self._abs_from_rel(float(pos.x()))
        if ms is not None:
            self.time_selected.emit(ms)
