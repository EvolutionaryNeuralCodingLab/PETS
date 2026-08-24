"""Interactive Fig 2f ROI picker for building verification event subsets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.analysis.figures_2f_2h_2i import (
    Figure2fDisplayData,
    Figure2fPoints,
    select_figure_2f_roi,
)
from eye_tracking_system_tools.analysis.paper_export import FigureBuildResult
from eye_tracking_system_tools.analysis.saccade_viewer.launch import (
    _block_until_window_closed,
    _ensure_qapplication,
    launch_saccade_viewer,
)
from eye_tracking_system_tools.analysis.saccade_viewer.selectors.common import (
    enrich_events_for_viewer,
    unique_events_by_identity,
)
from eye_tracking_system_tools.analysis.saccade_viewer.selectors.figure_2f_context import (
    Figure2fRoiContext,
    prepare_figure_2f_from_selector,
    prepare_figure_2f_roi_context,
)


def _figure_2f_cmap_lut() -> np.ndarray:
    """Turbo colormap with white at zero (matches export_figure_2f)."""
    import matplotlib.pyplot as plt

    turbo = plt.get_cmap("turbo", 256)
    colors = turbo(np.linspace(0, 1, 256))
    colors[0] = np.array([1, 1, 1, 1])
    return (colors * 255).astype(np.uint8)


def _make_snipping_roi(
    plot: pg.PlotWidget,
    pos: tuple[float, float],
    size: tuple[float, float],
) -> pg.RectROI:
    """RectROI with four corner handles + edge handles (snipping-tool style)."""
    roi = pg.RectROI(
        pos,
        size,
        pen=pg.mkPen("y", width=2),
        hoverPen=pg.mkPen("w", width=2),
        handlePen=pg.mkPen("y", width=2),
        handleHoverPen=pg.mkPen("w", width=2),
        movable=True,
        resizable=True,
        sideScalers=True,
    )
    roi.handleSize = 9
    # Default RectROI only exposes the bottom-right corner; add the other three.
    roi.addScaleHandle([0, 0], [1, 1])
    roi.addScaleHandle([0, 1], [1, 0])
    roi.addScaleHandle([1, 0], [0, 1])
    plot.addItem(roi)
    return roi


class _HistogramRoiPanel(QtWidgets.QGroupBox):
    """Single macro/micro 2f histogram with a snipping-style rectangular ROI."""

    roi_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        title: str,
        hist: dict[str, np.ndarray],
        rng: tuple[float, float],
        tick_list: list[float],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self._rng = rng
        xedges = hist["xedges"]
        yedges = hist["yedges"]
        counts = hist["norm_counts"]

        layout = QtWidgets.QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Right max V [deg/ms]")
        self.plot.setLabel("left", "Left max V [deg/ms]")
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=False, y=False)
        layout.addWidget(self.plot)

        lut = _figure_2f_cmap_lut()
        vmax = float(np.nanmax(counts)) if counts.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        self._img = pg.ImageItem(axisOrder="row-major")
        image = counts.T.astype(float)
        self._img.setImage(image)
        self._img.setLookupTable(lut)
        self._img.setLevels([0.0, vmax])
        x0, x1 = float(xedges[0]), float(xedges[-1])
        y0, y1 = float(yedges[0]), float(yedges[-1])
        self._img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self.plot.addItem(self._img)

        diag = pg.PlotDataItem(
            [rng[0], rng[1]],
            [rng[0], rng[1]],
            pen=pg.mkPen(color=(128, 128, 128), width=1, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.plot.addItem(diag)

        span = (rng[1] - rng[0]) * 0.25
        cx = rng[0] + span * 0.5
        self._roi = _make_snipping_roi(self.plot, (cx, cx), (span, span))
        self.plot.setXRange(rng[0], rng[1], padding=0.02)
        self.plot.setYRange(rng[0], rng[1], padding=0.02)
        ticks = [(float(v), str(v)) for v in tick_list]
        ax = self.plot.getAxis("bottom")
        ax.setTicks([ticks])
        ay = self.plot.getAxis("left")
        ay.setTicks([ticks])
        self._roi.sigRegionChanged.connect(self.roi_changed.emit)

    def roi_bounds(self) -> tuple[float, float, float, float]:
        pos = self._roi.pos()
        size = self._roi.size()
        x0 = float(pos.x())
        y0 = float(pos.y())
        x1 = x0 + float(size.x())
        y1 = y0 + float(size.y())
        return x0, x1, y0, y1


class Figure2fRoiPickerWindow(QtWidgets.QMainWindow):
    """
    Interactive Fig 2f histogram picker.

    Draw/adjust rectangular ROIs on macro or micro panels, accumulate selections,
    then launch the saccade viewer on the pooled events.
    """

    closed = QtCore.pyqtSignal()

    def __init__(
        self,
        context: Figure2fRoiContext,
        *,
        registry_path: Path | str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._context = context
        self._collected = context.collected
        self._tables = context.tables
        self.registry_path = Path(registry_path) if registry_path else None
        self.selected_events = pd.DataFrame()
        self._pool: list[pd.DataFrame] = []

        display = context.display
        macro_hist = display.macro
        micro_hist = display.micro

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        root.addWidget(
            QtWidgets.QLabel(
                "Drag corner/edge handles on the yellow ROI (Macro or Micro), then "
                "<b>Add selection</b>. Selections accumulate; <b>Launch viewer</b> "
                "opens verification on the pool."
            )
        )
        root.addWidget(
            QtWidgets.QLabel(
                f"Dataset: {len(context.block_keys)} block(s), "
                f"{len(self._collected.points)} events, "
                f"sample_mode={context.sample_mode}, "
                f"histogram from {display.source}"
            )
        )

        panels = QtWidgets.QHBoxLayout()
        self._macro_panel = _HistogramRoiPanel(
            "Macro — active for Add",
            macro_hist,
            display.macro_range,
            display.macro_tick_list,
        )
        self._micro_panel = _HistogramRoiPanel(
            "Micro",
            micro_hist,
            display.micro_range,
            display.micro_tick_list,
        )
        panels.addWidget(self._macro_panel)
        panels.addWidget(self._micro_panel)
        root.addLayout(panels)

        self._active = QtWidgets.QButtonGroup(self)
        self._rb_macro = QtWidgets.QRadioButton("Use Macro panel")
        self._rb_micro = QtWidgets.QRadioButton("Use Micro panel")
        self._rb_macro.setChecked(True)
        self._active.addButton(self._rb_macro, 0)
        self._active.addButton(self._rb_micro, 1)
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(self._rb_macro)
        sel_row.addWidget(self._rb_micro)
        sel_row.addStretch()
        root.addLayout(sel_row)

        self._status = QtWidgets.QLabel("")
        self._pool_label = QtWidgets.QLabel("Pool: 0 events")
        btn_add = QtWidgets.QPushButton("Add selection")
        btn_clear = QtWidgets.QPushButton("Clear pool")
        btn_launch = QtWidgets.QPushButton("Launch viewer")
        btn_close = QtWidgets.QPushButton("Close")
        btn_add.clicked.connect(self._add_selection)
        btn_clear.clicked.connect(self._clear_pool)
        btn_launch.clicked.connect(self._launch_viewer)
        btn_close.clicked.connect(self.close)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_clear)
        btn_row.addWidget(btn_launch)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)
        root.addWidget(self._pool_label)
        root.addWidget(self._status)

        self._macro_panel.roi_changed.connect(self._update_preview)
        self._micro_panel.roi_changed.connect(self._update_preview)
        self._active.buttonClicked.connect(lambda _: self._update_active_label())

        title_mode = "legacy" if context.sample_mode == "contra_window" else "strict"
        self.setWindowTitle(f"Fig 2f ROI picker ({title_mode}) — verification subset")
        self.resize(960, 520)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._update_active_label()
        self._update_preview()

    def _active_panel(self) -> _HistogramRoiPanel:
        return self._micro_panel if self._rb_micro.isChecked() else self._macro_panel

    def _update_active_label(self) -> None:
        macro_title = "Macro — active for Add" if self._rb_macro.isChecked() else "Macro"
        micro_title = "Micro — active for Add" if self._rb_micro.isChecked() else "Micro"
        self._macro_panel.setTitle(macro_title)
        self._micro_panel.setTitle(micro_title)

    def _update_preview(self) -> None:
        panel = self._active_panel()
        x0, x1, y0, y1 = panel.roi_bounds()
        n = len(select_figure_2f_roi(self._collected, x0, x1, y0, y1))
        self._status.setText(
            f"Active ROI: right [{min(x0, x1):.4g}, {max(x0, x1):.4g}] deg/ms, "
            f"left [{min(y0, y1):.4g}, {max(y0, y1):.4g}] deg/ms → {n} events"
        )

    def _add_selection(self) -> None:
        panel = self._active_panel()
        x0, x1, y0, y1 = panel.roi_bounds()
        picked = select_figure_2f_roi(self._collected, x0, x1, y0, y1)
        if picked.empty:
            self._status.setText("ROI is empty — widen or move the rectangle.")
            return
        self._pool.append(picked)
        merged = unique_events_by_identity(pd.concat(self._pool, ignore_index=True))
        self.selected_events = enrich_events_for_viewer(self._tables, merged)
        self._pool_label.setText(
            f"Pool: {len(self.selected_events)} unique events "
            f"({len(self._pool)} ROI addition(s))"
        )
        self._status.setText(f"Added {len(picked)} events ({len(self.selected_events)} in pool).")

    def _clear_pool(self) -> None:
        self._pool.clear()
        self.selected_events = pd.DataFrame()
        self._pool_label.setText("Pool: 0 events")
        self._status.setText("Pool cleared.")
        self._update_preview()

    def _launch_viewer(self) -> None:
        if self.selected_events.empty:
            self._status.setText("Pool is empty — add at least one ROI selection first.")
            return
        launch_saccade_viewer(
            self.selected_events,
            registry_path=self.registry_path,
            block=False,
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


def _open_roi_picker(
    context: Figure2fRoiContext,
    *,
    registry_path: Path | str | None = None,
    block: bool = True,
) -> Figure2fRoiPickerWindow | None:
    app = _ensure_qapplication()
    win = Figure2fRoiPickerWindow(context, registry_path=registry_path)
    win.show()
    win.raise_()
    win.activateWindow()
    if block:
        _block_until_window_closed(win, app)
    return win


def launch_figure_2f_roi_picker(
    tables: Any,
    *,
    registry_path: Path | str | None = None,
    block_keys: list[str] | None = None,
    figure_2f_cfg: dict | None = None,
    sample_mode: str | None = None,
    build_pickle: Path | str | None = None,
    ensure_traces: bool = True,
    block: bool = True,
) -> Figure2fRoiPickerWindow | None:
    """
    Open the interactive Fig 2f ROI picker (low-level API).

    Prefer :func:`launch_figure_2f_roi_picker_from_selector` after a Fig 2f Build.
    """
    from eye_tracking_system_tools.analysis.pipeline import EventTables

    if not isinstance(tables, EventTables):
        raise TypeError("tables must be an EventTables instance")
    keys = block_keys or [b.spec.block_key for b in tables.blocks]
    context = prepare_figure_2f_roi_context(
        tables,
        block_keys=keys,
        params_overrides={"figure_2f": figure_2f_cfg} if figure_2f_cfg else None,
        sample_mode=sample_mode,
        build_pickle=build_pickle,
        ensure_traces=ensure_traces,
    )
    return _open_roi_picker(context, registry_path=registry_path, block=block)


def launch_figure_2f_roi_picker_from_selector(
    selector: Any,
    build_result: FigureBuildResult | None = None,
    *,
    registry_path: Path | str | None = None,
    sample_mode: str | None = None,
    require_build: bool = True,
    block: bool = True,
) -> Figure2fRoiPickerWindow | None:
    """
    Open the ROI picker using the same blocks, filters, params, and export pickle
    as a ``PaperFigureSelector('2f')`` **Build**.

    ``sample_mode``:
        ``None`` — use the selector Legacy/Strict toggle (or the Built config);
        ``contra_window`` — legacy paper 2f (±``contra_sample_ms`` around onset);
        ``event_span`` — strict contra peak over [on, off] ms.
    """
    context = prepare_figure_2f_from_selector(
        selector,
        build_result,
        sample_mode=sample_mode,
        require_build=require_build,
    )
    reg = registry_path
    if reg is None and hasattr(selector, "ctx"):
        reg = getattr(selector.ctx, "registry_path", None)
    return _open_roi_picker(context, registry_path=reg, block=block)


__all__ = [
    "Figure2fRoiPickerWindow",
    "launch_figure_2f_roi_picker",
    "launch_figure_2f_roi_picker_from_selector",
]
