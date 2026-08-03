"""Dialog: preview Kerr phi/theta histograms for a tentative reference point."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.analysis.eye_movement_span import percentile_axis_span
from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
    KerrAnglePreview,
    preview_kerr_angles,
)

_PHI_BRUSH = pg.mkBrush(70, 130, 180, 200)
_THETA_BRUSH = pg.mkBrush(220, 120, 60, 200)


def _finite_pair(preview: KerrAnglePreview) -> tuple[np.ndarray, np.ndarray]:
    phi = np.asarray(preview.phi, dtype=float)
    theta = np.asarray(preview.theta, dtype=float)
    mask = np.isfinite(phi) & np.isfinite(theta)
    return phi[mask], theta[mask]


def _span_text(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: (no finite samples)"
    full = percentile_axis_span(values, lo=0.0, hi=100.0)
    clip = percentile_axis_span(values, lo=5.0, hi=95.0)
    assert full is not None and clip is not None
    return (
        f"{name}: full [{full.p_lo:.2f}, {full.p_hi:.2f}]° "
        f"(span {full.span:.2f}°) · "
        f"p5–p95 [{clip.p_lo:.2f}, {clip.p_hi:.2f}]° "
        f"(span {clip.span:.2f}°)"
    )


def _add_histogram(plot: pg.PlotWidget, values: np.ndarray, *, brush, n_bins: int) -> None:
    plot.clear()
    plot.showGrid(x=True, y=True, alpha=0.25)
    if values.size == 0:
        return
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi <= lo:
        pad = max(0.5, abs(lo) * 0.05 + 0.5)
        lo -= pad
        hi += pad
    edges = np.linspace(lo, hi, max(5, int(n_bins)) + 1)
    counts, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges) * 0.92
    plot.addItem(
        pg.BarGraphItem(
            x=centers,
            height=counts,
            width=widths,
            brush=brush,
            pen=pg.mkPen(None),
        )
    )
    plot.setXRange(lo, hi, padding=0.02)
    ymax = float(np.max(counts)) if counts.size else 1.0
    plot.setYRange(0.0, max(1.0, ymax * 1.05), padding=0.0)


def _add_heatmap(
    plot: pg.PlotWidget,
    phi: np.ndarray,
    theta: np.ndarray,
    *,
    n_bins: int,
) -> None:
    plot.clear()
    plot.showGrid(x=True, y=True, alpha=0.25)
    plot.setLabel("bottom", "φ (deg)")
    plot.setLabel("left", "θ (deg)")
    if phi.size == 0:
        return

    n = max(5, int(n_bins))
    phi_lo, phi_hi = float(np.min(phi)), float(np.max(phi))
    th_lo, th_hi = float(np.min(theta)), float(np.max(theta))
    if phi_hi <= phi_lo:
        pad = max(0.5, abs(phi_lo) * 0.05 + 0.5)
        phi_lo, phi_hi = phi_lo - pad, phi_hi + pad
    if th_hi <= th_lo:
        pad = max(0.5, abs(th_lo) * 0.05 + 0.5)
        th_lo, th_hi = th_lo - pad, th_hi + pad

    counts, xedges, yedges = np.histogram2d(
        phi,
        theta,
        bins=n,
        range=[[phi_lo, phi_hi], [th_lo, th_hi]],
    )
    # ImageItem rows = y (theta), cols = x (phi); transpose histogram2d output.
    img = pg.ImageItem(image=counts.T)
    try:
        cmap = pg.colormap.get("viridis")
        img.setLookupTable(cmap.getLookupTable(nPts=256))
    except Exception:
        pass
    img.setLevels([0.0, float(np.max(counts)) if counts.size else 1.0])
    img.setRect(
        QtCore.QRectF(
            float(xedges[0]),
            float(yedges[0]),
            float(xedges[-1] - xedges[0]),
            float(yedges[-1] - yedges[0]),
        )
    )
    plot.addItem(img)
    plot.setXRange(phi_lo, phi_hi, padding=0.02)
    plot.setYRange(th_lo, th_hi, padding=0.02)
    plot.setAspectLocked(False)


class KerrAnglesSpanDialog(QtWidgets.QDialog):
    """Show 1-D / 2-D Kerr angle histograms; allow tweaking the Kerr reference."""

    def __init__(
        self,
        preview: KerrAnglePreview,
        *,
        eye: str,
        eye_df: pd.DataFrame | None = None,
        n_bins: int = 60,
        n_bins_2d: int = 50,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        eye_lc = eye.lower()
        self._eye = eye_lc
        self._eye_label = "Left" if eye_lc == "left" else "Right"
        self._eye_df = None if eye_df is None else eye_df.copy()
        self._n_bins = max(5, int(n_bins))
        self._n_bins_2d = max(5, int(n_bins_2d))
        self._preview = preview

        self.setWindowTitle(f"{self._eye_label} eye — Kerr angles span preview")
        self.resize(980, 720)

        layout = QtWidgets.QVBoxLayout(self)

        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(QtWidgets.QLabel("Kerr ref x:"))
        self._spin_ref_x = QtWidgets.QDoubleSpinBox()
        self._spin_ref_x.setDecimals(1)
        self._spin_ref_x.setRange(-5000.0, 5000.0)
        self._spin_ref_x.setSingleStep(1.0)
        self._spin_ref_x.setKeyboardTracking(False)
        self._spin_ref_x.setValue(float(preview.ref_x))
        ref_row.addWidget(self._spin_ref_x)

        ref_row.addWidget(QtWidgets.QLabel("y:"))
        self._spin_ref_y = QtWidgets.QDoubleSpinBox()
        self._spin_ref_y.setDecimals(1)
        self._spin_ref_y.setRange(-5000.0, 5000.0)
        self._spin_ref_y.setSingleStep(1.0)
        self._spin_ref_y.setKeyboardTracking(False)
        self._spin_ref_y.setValue(float(preview.ref_y))
        ref_row.addWidget(self._spin_ref_y)

        self._btn_recompute = QtWidgets.QPushButton("Recompute")
        self._btn_recompute.setToolTip(
            "Re-run Kerr angles with the edited reference and refresh histograms"
        )
        self._btn_recompute.setEnabled(self._eye_df is not None)
        ref_row.addWidget(self._btn_recompute)
        ref_row.addStretch(1)
        layout.addLayout(ref_row)

        if self._eye_df is None:
            layout.addWidget(
                QtWidgets.QLabel(
                    "Recompute disabled — eye data was not passed into this dialog."
                )
            )

        self._summary = QtWidgets.QLabel("")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._phi_span = QtWidgets.QLabel("")
        self._phi_span.setWordWrap(True)
        layout.addWidget(self._phi_span)

        self._theta_span = QtWidgets.QLabel("")
        self._theta_span.setWordWrap(True)
        layout.addWidget(self._theta_span)

        self._warn = QtWidgets.QLabel("")
        self._warn.setWordWrap(True)
        self._warn.setStyleSheet("color: #664d03;")
        self._warn.hide()
        layout.addWidget(self._warn)

        hist_row = QtWidgets.QHBoxLayout()
        self._phi_plot = pg.PlotWidget()
        self._phi_plot.setBackground("w")
        self._phi_plot.setLabel("bottom", "φ (deg)")
        self._phi_plot.setLabel("left", "Count")
        self._phi_plot.setTitle("Phi")
        hist_row.addWidget(self._phi_plot, stretch=1)

        self._theta_plot = pg.PlotWidget()
        self._theta_plot.setBackground("w")
        self._theta_plot.setLabel("bottom", "θ (deg)")
        self._theta_plot.setLabel("left", "Count")
        self._theta_plot.setTitle("Theta")
        hist_row.addWidget(self._theta_plot, stretch=1)
        layout.addLayout(hist_row, stretch=1)

        self._heat_plot = pg.PlotWidget()
        self._heat_plot.setBackground("w")
        self._heat_plot.setTitle("Angle space (φ × θ)")
        layout.addWidget(self._heat_plot, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)

        self._btn_recompute.clicked.connect(self._on_recompute)
        self._refresh_from_preview(preview)

    def ref_xy(self) -> tuple[int, int]:
        """Current Kerr reference (rounded to integer pixels for the verifier)."""
        return (
            int(round(self._spin_ref_x.value())),
            int(round(self._spin_ref_y.value())),
        )

    def _refresh_from_preview(self, preview: KerrAnglePreview) -> None:
        self._preview = preview
        phi, theta = _finite_pair(preview)
        self._summary.setText(
            f"{self._eye_label} eye · ref = ({preview.ref_x:.1f}, {preview.ref_y:.1f}) · "
            f"f_z = {preview.f_z:.2f} · "
            f"{preview.n_finite:,} / {preview.n_input:,} finite angle samples"
        )
        self._phi_span.setText(_span_text("φ (phi)", phi))
        self._theta_span.setText(_span_text("θ (theta)", theta))
        if preview.n_finite == 0:
            self._warn.setText(
                "No finite φ/θ samples — the reference may be far from the pupil "
                "cloud, or ellipse axes may be invalid."
            )
            self._warn.show()
        else:
            self._warn.hide()

        _add_histogram(self._phi_plot, phi, brush=_PHI_BRUSH, n_bins=self._n_bins)
        _add_histogram(self._theta_plot, theta, brush=_THETA_BRUSH, n_bins=self._n_bins)
        _add_heatmap(self._heat_plot, phi, theta, n_bins=self._n_bins_2d)

    def _on_recompute(self) -> None:
        if self._eye_df is None:
            return
        ref_x = float(self._spin_ref_x.value())
        ref_y = float(self._spin_ref_y.value())
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            preview = preview_kerr_angles(self._eye_df, ref_x, ref_y)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(
                self,
                "Recompute",
                f"Kerr preview failed:\n{exc}",
            )
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._refresh_from_preview(preview)
