"""Helper dialog: DLC likelihood histogram with interactive threshold preview."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.preprocessing.dlc_csv_io import (
    likelihood_threshold_stats,
    load_dlc_likelihood_values_many,
)

_REMOVED_BRUSH = pg.mkBrush(220, 80, 80, 180)
_KEPT_BRUSH = pg.mkBrush(70, 130, 180, 200)
_THRESHOLD_PEN = pg.mkPen("#222222", width=2, style=QtCore.Qt.PenStyle.DashLine)


class LikelihoodThresholdDialog(QtWidgets.QDialog):
    """Histogram of DLC likelihoods with a threshold that previews data loss."""

    def __init__(
        self,
        csv_paths: list[Path | str],
        *,
        initial_threshold: float = 0.95,
        n_bins: int = 50,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("DLC likelihood distribution")
        self.resize(720, 480)
        self._n_bins = max(5, int(n_bins))
        self._paths = [Path(p) for p in csv_paths]
        self._values = load_dlc_likelihood_values_many(self._paths)
        self._edges: np.ndarray | None = None
        self._bar_removed: pg.BarGraphItem | None = None
        self._bar_kept: pg.BarGraphItem | None = None
        self._threshold_line: pg.InfiniteLine | None = None

        layout = QtWidgets.QVBoxLayout(self)

        names = ", ".join(p.name for p in self._paths) or "(none)"
        layout.addWidget(QtWidgets.QLabel(f"Files: {names}"))

        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.setLabel("bottom", "Likelihood")
        self._plot.setLabel("left", "Count")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.addLegend(offset=(10, 10))
        layout.addWidget(self._plot, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Threshold:"))
        self._threshold = QtWidgets.QDoubleSpinBox()
        self._threshold.setRange(0.0, 1.0)
        self._threshold.setDecimals(3)
        self._threshold.setSingleStep(0.01)
        self._threshold.setValue(float(np.clip(initial_threshold, 0.0, 1.0)))
        controls.addWidget(self._threshold)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._stats_label = QtWidgets.QLabel("")
        self._stats_label.setWordWrap(True)
        layout.addWidget(self._stats_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            "Apply threshold"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._threshold.valueChanged.connect(self._refresh_plot)
        self._build_histogram()
        self._refresh_plot()

    def threshold(self) -> float:
        return float(self._threshold.value())

    def _build_histogram(self) -> None:
        self._plot.clear()
        self._plot.addLegend(offset=(10, 10))
        self._bar_removed = None
        self._bar_kept = None
        self._threshold_line = None
        self._edges = None

        if self._values.size == 0:
            self._stats_label.setText("No Pupil/edge likelihood values found in the selected CSVs.")
            return

        lo = float(np.min(self._values))
        hi = float(np.max(self._values))
        if hi <= lo:
            lo = min(lo, 0.0)
            hi = max(hi, 1.0)
        # Keep bins on a stable [0, 1]-ish span when data is likelihood-like.
        bin_lo = min(0.0, lo)
        bin_hi = max(1.0, hi)
        self._edges = np.linspace(bin_lo, bin_hi, self._n_bins + 1)
        centers = 0.5 * (self._edges[:-1] + self._edges[1:])
        widths = np.diff(self._edges) * 0.92

        self._bar_removed = pg.BarGraphItem(
            x=centers,
            height=np.zeros_like(centers),
            width=widths,
            brush=_REMOVED_BRUSH,
            pen=pg.mkPen(None),
            name="Below threshold (removed)",
        )
        self._bar_kept = pg.BarGraphItem(
            x=centers,
            height=np.zeros_like(centers),
            width=widths,
            y0=np.zeros_like(centers),
            brush=_KEPT_BRUSH,
            pen=pg.mkPen(None),
            name="Above threshold (kept)",
        )
        self._plot.addItem(self._bar_removed)
        self._plot.addItem(self._bar_kept)

        self._threshold_line = pg.InfiniteLine(
            pos=self.threshold(),
            angle=90,
            pen=_THRESHOLD_PEN,
            movable=True,
            bounds=[0.0, 1.0],
            label="thr={value:0.3f}",
            labelOpts={"position": 0.95, "color": "#222222", "fill": (255, 255, 255, 160)},
        )
        self._threshold_line.sigPositionChanged.connect(self._on_line_moved)
        self._plot.addItem(self._threshold_line)
        self._plot.setXRange(bin_lo, bin_hi, padding=0.02)

    def _on_line_moved(self, line: pg.InfiniteLine) -> None:
        value = float(np.clip(line.value(), 0.0, 1.0))
        self._threshold.blockSignals(True)
        self._threshold.setValue(value)
        self._threshold.blockSignals(False)
        self._refresh_plot(update_line=False)

    def _refresh_plot(self, *_args, update_line: bool = True) -> None:
        thr = self.threshold()
        stats = likelihood_threshold_stats(self._values, thr)
        if stats["n_total"] == 0:
            return

        self._stats_label.setText(
            f"{stats['n_total']:,} likelihood samples — "
            f"kept {stats['n_kept']:,} ({100.0 * float(stats['frac_kept']):.1f}%), "
            f"removed {stats['n_removed']:,} ({100.0 * float(stats['frac_removed']):.1f}%) "
            f"at threshold {thr:.3f} (keep if likelihood > threshold)."
        )

        if self._edges is None or self._bar_removed is None or self._bar_kept is None:
            return

        below = self._values[self._values <= thr]
        above = self._values[self._values > thr]
        h_removed, _ = np.histogram(below, bins=self._edges)
        h_kept, _ = np.histogram(above, bins=self._edges)
        centers = 0.5 * (self._edges[:-1] + self._edges[1:])
        widths = np.diff(self._edges) * 0.92
        self._bar_removed.setOpts(x=centers, height=h_removed, width=widths, y0=0)
        self._bar_kept.setOpts(x=centers, height=h_kept, width=widths, y0=h_removed)

        if update_line and self._threshold_line is not None:
            self._threshold_line.blockSignals(True)
            self._threshold_line.setValue(thr)
            self._threshold_line.blockSignals(False)

        ymax = float(np.max(h_removed + h_kept)) if (h_removed.size or h_kept.size) else 1.0
        self._plot.setYRange(0.0, max(1.0, ymax * 1.05), padding=0.0)
