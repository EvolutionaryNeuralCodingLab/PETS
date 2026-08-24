"""Dialog: preview Kerr phi/theta histograms for a tentative reference point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.analysis.eye_movement_span import percentile_axis_span
from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
    numpy_rgb_to_qpixmap,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker import (
    QtRoiPickerDialog,
)
from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
    KerrAnglePreview,
    preview_kerr_angles,
)

_PHI_BRUSH = pg.mkBrush(70, 130, 180, 200)
_THETA_BRUSH = pg.mkBrush(220, 120, 60, 200)
_POINT_PEN = QtGui.QPen(QtGui.QColor(255, 220, 40), 2)
_POINT_BRUSH = QtGui.QBrush(QtGui.QColor(255, 80, 40, 200))
_POINT_HOVER_BRUSH = QtGui.QBrush(QtGui.QColor(40, 200, 90, 230))
_POINT_MATCH_BRUSH = QtGui.QBrush(QtGui.QColor(40, 120, 255, 230))
_POINT_LOCK_BRUSH = QtGui.QBrush(QtGui.QColor(255, 40, 200, 230))
_ROI_PEN = QtGui.QPen(QtGui.QColor(80, 200, 255), 2)


@dataclass(frozen=True)
class KerrRefSpanMetrics:
    """Full-range / p5–p95 spans and MAD for φ/θ at one Kerr reference."""

    ref_x: float
    ref_y: float
    f_z: float
    n_finite: int
    n_input: int
    phi_full: float | None
    phi_p5_95: float | None
    theta_full: float | None
    theta_p5_95: float | None
    phi_mad: float | None
    theta_mad: float | None


MatchMode = Literal["span", "mad", "span_mad"]


def _finite_pair(preview: KerrAnglePreview) -> tuple[np.ndarray, np.ndarray]:
    phi = np.asarray(preview.phi, dtype=float)
    theta = np.asarray(preview.theta, dtype=float)
    mask = np.isfinite(phi) & np.isfinite(theta)
    return phi[mask], theta[mask]


def axis_mad(values: np.ndarray) -> float | None:
    """Mean absolute deviation from the mean of finite samples."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    mu = float(np.mean(arr))
    return float(np.mean(np.abs(arr - mu)))


def _axis_spans(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    full = percentile_axis_span(values, lo=0.0, hi=100.0)
    clip = percentile_axis_span(values, lo=5.0, hi=95.0)
    assert full is not None and clip is not None
    return float(full.span), float(clip.span)


def span_metrics_from_preview(preview: KerrAnglePreview) -> KerrRefSpanMetrics:
    """Derive full / p5–p95 spans and MAD for φ/θ from an existing Kerr preview."""
    phi, theta = _finite_pair(preview)
    phi_full, phi_clip = _axis_spans(phi)
    theta_full, theta_clip = _axis_spans(theta)
    return KerrRefSpanMetrics(
        ref_x=float(preview.ref_x),
        ref_y=float(preview.ref_y),
        f_z=float(preview.f_z),
        n_finite=int(preview.n_finite),
        n_input=int(preview.n_input),
        phi_full=phi_full,
        phi_p5_95=phi_clip,
        theta_full=theta_full,
        theta_p5_95=theta_clip,
        phi_mad=axis_mad(phi),
        theta_mad=axis_mad(theta),
    )


def compute_kerr_ref_span(
    eye_df: pd.DataFrame,
    ref_x: float,
    ref_y: float,
) -> KerrRefSpanMetrics:
    """Run Kerr preview at ``(ref_x, ref_y)`` and return span + MAD metrics."""
    preview = preview_kerr_angles(eye_df, ref_x, ref_y)
    return span_metrics_from_preview(preview)


def grid_points_in_roi(
    roi: tuple[int, int, int, int],
    n_x: int,
    n_y: int,
) -> list[tuple[float, float]]:
    """
    Equally spaced Kerr-reference candidates inside an ``(x, y, w, h)`` ROI.

    For ``n >= 2`` uses inclusive linspace endpoints; for ``n == 1`` uses the
    ROI center along that axis.
    """
    x, y, w, h = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    if w < 1 or h < 1:
        raise ValueError("ROI width and height must be >= 1.")
    nx = max(1, int(n_x))
    ny = max(1, int(n_y))
    x0, x1 = float(x), float(x + w - 1)
    y0, y1 = float(y), float(y + h - 1)
    xs = [0.5 * (x0 + x1)] if nx == 1 else list(np.linspace(x0, x1, nx))
    ys = [0.5 * (y0 + y1)] if ny == 1 else list(np.linspace(y0, y1, ny))
    return [(float(px), float(py)) for py in ys for px in xs]


def compute_kerr_ref_span_grid(
    eye_df: pd.DataFrame,
    roi: tuple[int, int, int, int],
    n_x: int,
    n_y: int,
) -> list[KerrRefSpanMetrics]:
    """Compute span + MAD metrics for every grid point inside ``roi``."""
    points = grid_points_in_roi(roi, n_x, n_y)
    return [compute_kerr_ref_span(eye_df, px, py) for px, py in points]


def span_feature_vector(
    metrics: KerrRefSpanMetrics,
    *,
    mode: MatchMode = "span",
) -> np.ndarray | None:
    """Feature vector for opposite-eye matching; ``None`` if incomplete for ``mode``."""
    span_vals = (
        metrics.phi_full,
        metrics.phi_p5_95,
        metrics.theta_full,
        metrics.theta_p5_95,
    )
    mad_vals = (metrics.phi_mad, metrics.theta_mad)
    if mode == "span":
        vals = span_vals
    elif mode == "mad":
        vals = mad_vals
    else:
        vals = span_vals + mad_vals
    if any(v is None for v in vals):
        return None
    return np.asarray(vals, dtype=float)


def span_match_distance(
    candidate: KerrRefSpanMetrics,
    target: KerrRefSpanMetrics,
    *,
    mode: MatchMode = "span",
) -> float | None:
    """L2 distance between match feature vectors; ``None`` if either is incomplete."""
    a = span_feature_vector(candidate, mode=mode)
    b = span_feature_vector(target, mode=mode)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def index_closest_span_match(
    candidates: list[KerrRefSpanMetrics],
    target: KerrRefSpanMetrics,
    *,
    mode: MatchMode = "span",
) -> tuple[int, float] | None:
    """Index and distance of the candidate that best matches ``target`` under ``mode``."""
    best_i: int | None = None
    best_d: float | None = None
    for i, cand in enumerate(candidates):
        d = span_match_distance(cand, target, mode=mode)
        if d is None:
            continue
        if best_d is None or d < best_d:
            best_i = i
            best_d = d
    if best_i is None or best_d is None:
        return None
    return best_i, best_d


def _fmt_span(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _metrics_span_text(
    label: str,
    metrics: KerrRefSpanMetrics,
    *,
    include_mad: bool = True,
) -> str:
    text = (
        f"{label}: φ full {_fmt_span(metrics.phi_full)}° · "
        f"φ p5–p95 {_fmt_span(metrics.phi_p5_95)}° · "
        f"θ full {_fmt_span(metrics.theta_full)}° · "
        f"θ p5–p95 {_fmt_span(metrics.theta_p5_95)}°"
    )
    if include_mad:
        text += (
            f" · φ MAD {_fmt_span(metrics.phi_mad)}° · "
            f"θ MAD {_fmt_span(metrics.theta_mad)}°"
        )
    return text


def _span_text(name: str, values: np.ndarray, *, include_mad: bool = True) -> str:
    if values.size == 0:
        return f"{name}: (no finite samples)"
    full = percentile_axis_span(values, lo=0.0, hi=100.0)
    clip = percentile_axis_span(values, lo=5.0, hi=95.0)
    assert full is not None and clip is not None
    text = (
        f"{name}: full [{full.p_lo:.2f}, {full.p_hi:.2f}]° "
        f"(span {full.span:.2f}°) · "
        f"p5–p95 [{clip.p_lo:.2f}, {clip.p_hi:.2f}]° "
        f"(span {clip.span:.2f}°)"
    )
    if include_mad:
        mad = axis_mad(values)
        text += f" · MAD {_fmt_span(mad)}°"
    return text


def _match_mode_label(mode: MatchMode) -> str:
    if mode == "mad":
        return "φ/θ MAD"
    if mode == "span_mad":
        return "φ/θ full + p5–p95 + MAD"
    return "φ/θ full + p5–p95"


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


class _GridPointsView(QtWidgets.QGraphicsView):
    """Still frame with ROI + candidate points; hover or click-lock selection."""

    point_hovered = QtCore.pyqtSignal(int)
    point_locked = QtCore.pyqtSignal(int)  # metric index, or -1 when unlocked

    def __init__(
        self,
        frame_rgb: np.ndarray,
        roi: tuple[int, int, int, int],
        metrics: list[KerrRefSpanMetrics],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._metrics = list(metrics)
        self._point_items: list[QtWidgets.QGraphicsEllipseItem] = []
        self._hover_idx = -1
        self._match_idx = -1
        self._locked = False
        self._locked_idx = -1
        self._hit_radius = 10.0

        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))

        pix = numpy_rgb_to_qpixmap(np.asarray(frame_rgb))
        self._pixmap_item = scene.addPixmap(pix)
        self.setSceneRect(self._pixmap_item.boundingRect())

        x, y, w, h = roi
        roi_item = scene.addRect(
            float(x),
            float(y),
            float(w),
            float(h),
            _ROI_PEN,
            QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush),
        )
        roi_item.setZValue(5)

        r = 5.0
        for m in self._metrics:
            item = scene.addEllipse(
                float(m.ref_x) - r,
                float(m.ref_y) - r,
                2.0 * r,
                2.0 * r,
                _POINT_PEN,
                _POINT_BRUSH,
            )
            item.setZValue(10)
            self._point_items.append(item)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _hit_index_at(self, scene_pos: QtCore.QPointF) -> int:
        best_i = -1
        best_d2 = self._hit_radius * self._hit_radius
        for i, m in enumerate(self._metrics):
            dx = float(m.ref_x) - float(scene_pos.x())
            dy = float(m.ref_y) - float(scene_pos.y())
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._locked:
            super().mouseMoveEvent(event)
            return
        best_i = self._hit_index_at(self.mapToScene(event.pos()))
        if best_i != self._hover_idx:
            self._hover_idx = best_i
            self._refresh_point_brushes()
            self.point_hovered.emit(best_i)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            idx = self._hit_index_at(self.mapToScene(event.pos()))
            if idx >= 0:
                self.lock_point(idx)
            elif self._locked:
                self.unlock_point()
                # Resume hover for the click location.
                self._hover_idx = -1
                best_i = self._hit_index_at(self.mapToScene(event.pos()))
                if best_i != self._hover_idx:
                    self._hover_idx = best_i
                    self._refresh_point_brushes()
                    if best_i >= 0:
                        self.point_hovered.emit(best_i)
        super().mousePressEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if not self._locked and self._hover_idx != -1:
            self._hover_idx = -1
            self._refresh_point_brushes()
            self.point_hovered.emit(-1)
        super().leaveEvent(event)

    def lock_point(self, idx: int) -> None:
        idx = int(idx)
        if idx < 0 or idx >= len(self._metrics):
            return
        self._locked = True
        self._locked_idx = idx
        self._hover_idx = -1
        self._refresh_point_brushes()
        self.point_locked.emit(idx)

    def unlock_point(self) -> None:
        if not self._locked and self._locked_idx < 0:
            return
        self._locked = False
        self._locked_idx = -1
        self._refresh_point_brushes()
        self.point_locked.emit(-1)

    @property
    def is_locked(self) -> bool:
        return bool(self._locked)

    def set_match_index(self, idx: int) -> None:
        self._match_idx = int(idx)
        self._refresh_point_brushes()

    def _brush_for_index(self, idx: int) -> QtGui.QBrush:
        if self._locked and idx == self._locked_idx:
            return _POINT_LOCK_BRUSH
        if not self._locked and idx == self._hover_idx:
            return _POINT_HOVER_BRUSH
        if idx == self._match_idx:
            return _POINT_MATCH_BRUSH
        return _POINT_BRUSH

    def _refresh_point_brushes(self) -> None:
        for i, item in enumerate(self._point_items):
            item.setBrush(self._brush_for_index(i))


class _NumericSortItem(QtWidgets.QTableWidgetItem):
    """Table cell that sorts by a numeric ``UserRole`` value (missing → +inf)."""

    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        a = self.data(QtCore.Qt.ItemDataRole.UserRole)
        b = other.data(QtCore.Qt.ItemDataRole.UserRole)
        af = float("inf") if a is None else float(a)
        bf = float("inf") if b is None else float(b)
        return af < bf


def _combined_span(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) + float(b)


class KerrRefSpanGridDialog(QtWidgets.QDialog):
    """Show grid candidates on the eye image; hover updates the span table."""

    ref_chosen = QtCore.pyqtSignal(float, float)

    _METRIC_IDX_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 32

    # (combo label, column key) — column keys map into _column_keys order.
    _SORT_OPTIONS = (
        ("φ full", "phi_full"),
        ("θ full", "theta_full"),
        ("φ+θ full (combined)", "combined_full"),
        ("φ p5–p95", "phi_p5_95"),
        ("θ p5–p95", "theta_p5_95"),
        ("φ+θ p5–p95 (combined)", "combined_p5_95"),
        ("φ MAD", "phi_mad"),
        ("θ MAD", "theta_mad"),
        ("φ+θ MAD (combined)", "combined_mad"),
        ("Δ opposite", "delta_opposite"),
    )

    def __init__(
        self,
        frame_rgb: np.ndarray,
        roi: tuple[int, int, int, int],
        metrics: list[KerrRefSpanMetrics],
        *,
        eye: str,
        n_x: int,
        n_y: int,
        target_metrics: KerrRefSpanMetrics | None = None,
        target_eye_label: str | None = None,
        include_mad: bool = True,
        match_mode: MatchMode = "span_mad",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        eye_lc = eye.lower()
        eye_label = "Left" if eye_lc == "left" else "Right"
        self._metrics = list(metrics)
        self._target_metrics = target_metrics
        self._include_mad = bool(include_mad)
        self._match_mode: MatchMode = match_mode
        self._selected_idx = -1
        self.setWindowTitle(
            f"{eye_label} eye — Kerr ref span grid ({n_x}×{n_y})"
        )
        self.resize(1100, 680)

        layout = QtWidgets.QVBoxLayout(self)
        tip = (
            f"ROI=({roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}) · "
            f"{len(metrics)} candidate(s). Hover to preview; click a point to lock "
            "(click empty image to unlock hover)."
        )
        if target_metrics is not None:
            other = target_eye_label or "opposite"
            tip += (
                f" Blue = closest match to {other}-eye "
                f"({_match_mode_label(self._match_mode)})."
            )
        layout.addWidget(QtWidgets.QLabel(tip))

        if target_metrics is not None:
            other = target_eye_label or "Opposite"
            layout.addWidget(
                QtWidgets.QLabel(
                    _metrics_span_text(
                        f"{other} eye target",
                        target_metrics,
                        include_mad=self._include_mad,
                    )
                )
            )

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._view = _GridPointsView(frame_rgb, roi, metrics)
        split.addWidget(self._view)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.addWidget(QtWidgets.QLabel("Span table (rows = candidates)"))

        self._column_keys = [
            "ref_x",
            "ref_y",
            "phi_full",
            "phi_p5_95",
            "theta_full",
            "theta_p5_95",
            "combined_full",
            "combined_p5_95",
        ]
        headers = [
            "ref_x",
            "ref_y",
            "φ full (°)",
            "φ p5–p95 (°)",
            "θ full (°)",
            "θ p5–p95 (°)",
            "φ+θ full (°)",
            "φ+θ p5–p95 (°)",
        ]
        if self._include_mad:
            self._column_keys.extend(["phi_mad", "theta_mad", "combined_mad"])
            headers.extend(["φ MAD (°)", "θ MAD (°)", "φ+θ MAD (°)"])
        self._column_keys.append("n_finite")
        headers.append("n finite")
        if target_metrics is not None:
            self._column_keys.append("delta_opposite")
            headers.append("Δ opposite")

        self._col_index = {key: i for i, key in enumerate(self._column_keys)}
        self._table = QtWidgets.QTableWidget(len(metrics), len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSortingEnabled(False)
        for row, m in enumerate(metrics):
            combined_full = _combined_span(m.phi_full, m.theta_full)
            combined_p95 = _combined_span(m.phi_p5_95, m.theta_p5_95)
            combined_mad = _combined_span(m.phi_mad, m.theta_mad)
            delta = (
                None
                if target_metrics is None
                else span_match_distance(
                    m, target_metrics, mode=self._match_mode
                )
            )
            cells: list[tuple[str, float | None]] = [
                (f"{m.ref_x:.1f}", float(m.ref_x)),
                (f"{m.ref_y:.1f}", float(m.ref_y)),
                (_fmt_span(m.phi_full), m.phi_full),
                (_fmt_span(m.phi_p5_95), m.phi_p5_95),
                (_fmt_span(m.theta_full), m.theta_full),
                (_fmt_span(m.theta_p5_95), m.theta_p5_95),
                (_fmt_span(combined_full), combined_full),
                (_fmt_span(combined_p95), combined_p95),
            ]
            if self._include_mad:
                cells.extend(
                    [
                        (_fmt_span(m.phi_mad), m.phi_mad),
                        (_fmt_span(m.theta_mad), m.theta_mad),
                        (_fmt_span(combined_mad), combined_mad),
                    ]
                )
            cells.append((str(m.n_finite), float(m.n_finite)))
            if target_metrics is not None:
                cells.append(("—" if delta is None else f"{delta:.2f}", delta))
            for col, (text, sort_val) in enumerate(cells):
                item = _NumericSortItem(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, sort_val)
                item.setData(self._METRIC_IDX_ROLE, int(row))
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)
        right_lay.addWidget(self._table, stretch=1)

        sort_row = QtWidgets.QHBoxLayout()
        sort_row.addWidget(QtWidgets.QLabel("Sort by:"))
        self._combo_sort = QtWidgets.QComboBox()
        for label, key in self._SORT_OPTIONS:
            if key == "delta_opposite" and target_metrics is None:
                continue
            if key in {"phi_mad", "theta_mad", "combined_mad"} and not self._include_mad:
                continue
            self._combo_sort.addItem(label, key)
        sort_row.addWidget(self._combo_sort)
        self._chk_sort_desc = QtWidgets.QCheckBox("Descending")
        self._chk_sort_desc.setChecked(True)
        self._chk_sort_desc.setToolTip(
            "When checked, largest spans / MAD (or Δ) appear first"
        )
        sort_row.addWidget(self._chk_sort_desc)
        self._btn_sort = QtWidgets.QPushButton("Apply sort")
        self._btn_sort.setToolTip(
            "Sort table rows by the selected column "
            "(header clicks also sort numerically)"
        )
        sort_row.addWidget(self._btn_sort)
        sort_row.addStretch(1)
        right_lay.addLayout(sort_row)

        action_row = QtWidgets.QHBoxLayout()
        self._btn_match = QtWidgets.QPushButton("Pick closest match to opposite eye")
        self._btn_match.setToolTip(
            "Select the grid point closest (L2) to the opposite eye under the "
            f"current match mode ({_match_mode_label(self._match_mode)})"
        )
        self._btn_match.setEnabled(target_metrics is not None)
        action_row.addWidget(self._btn_match)
        self._btn_use_ref = QtWidgets.QPushButton("Use selected as Kerr ref")
        self._btn_use_ref.setToolTip(
            "Apply the selected grid point as this eye's Kerr reference and "
            "recompute histograms in the parent dialog"
        )
        self._btn_use_ref.setEnabled(False)
        action_row.addWidget(self._btn_use_ref)
        action_row.addStretch(1)
        right_lay.addLayout(action_row)

        self._match_status = QtWidgets.QLabel("")
        self._match_status.setWordWrap(True)
        right_lay.addWidget(self._match_status)

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        buttons.rejected.connect(self.reject)
        close_btn = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        layout.addWidget(buttons)

        self._view.point_hovered.connect(self._on_point_hovered)
        self._view.point_locked.connect(self._on_point_locked)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._btn_match.clicked.connect(self._on_pick_closest_match)
        self._btn_use_ref.clicked.connect(self._on_use_selected_ref)
        self._btn_sort.clicked.connect(self._on_apply_sort)
        self._combo_sort.currentIndexChanged.connect(self._on_apply_sort)
        self._chk_sort_desc.toggled.connect(self._on_apply_sort)

        self._on_apply_sort()
        if target_metrics is not None:
            self._on_pick_closest_match()

    def _metric_index_at_visual_row(self, visual_row: int) -> int:
        item = self._table.item(visual_row, 0)
        if item is None:
            return -1
        raw = item.data(self._METRIC_IDX_ROLE)
        return -1 if raw is None else int(raw)

    def _visual_row_for_metric(self, metric_idx: int) -> int:
        for row in range(self._table.rowCount()):
            if self._metric_index_at_visual_row(row) == metric_idx:
                return row
        return -1

    def _on_apply_sort(self, *_args) -> None:
        key = self._combo_sort.currentData()
        if key is None:
            return
        col = self._col_index.get(str(key))
        if col is None:
            return
        order = (
            QtCore.Qt.SortOrder.DescendingOrder
            if self._chk_sort_desc.isChecked()
            else QtCore.Qt.SortOrder.AscendingOrder
        )
        self._table.sortItems(col, order)
        if self._selected_idx >= 0:
            self._select_index(self._selected_idx)

    def _select_index(self, idx: int, *, from_hover: bool = False) -> None:
        if idx < 0 or idx >= len(self._metrics):
            if not from_hover:
                self._table.clearSelection()
                self._selected_idx = -1
                self._btn_use_ref.setEnabled(False)
            return
        self._selected_idx = idx
        self._btn_use_ref.setEnabled(True)
        visual = self._visual_row_for_metric(idx)
        if visual < 0:
            return
        self._table.blockSignals(True)
        self._table.selectRow(visual)
        self._table.blockSignals(False)
        item = self._table.item(visual, 0)
        if item is not None:
            self._table.scrollToItem(item)

    def _on_point_hovered(self, idx: int) -> None:
        if self._view.is_locked or idx < 0:
            return
        self._select_index(idx, from_hover=True)

    def _on_point_locked(self, idx: int) -> None:
        if idx < 0:
            return
        self._select_index(idx)

    def _on_table_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._selected_idx = -1
            self._btn_use_ref.setEnabled(False)
            return
        metric_idx = self._metric_index_at_visual_row(rows[0].row())
        if metric_idx < 0:
            self._selected_idx = -1
            self._btn_use_ref.setEnabled(False)
            return
        self._selected_idx = metric_idx
        self._btn_use_ref.setEnabled(True)

    def _on_pick_closest_match(self) -> None:
        if self._target_metrics is None:
            self._match_status.setText("Opposite-eye spans are not available.")
            return
        hit = index_closest_span_match(
            self._metrics, self._target_metrics, mode=self._match_mode
        )
        if hit is None:
            self._match_status.setText(
                "No candidate has a complete metric vector to compare."
            )
            return
        idx, dist = hit
        self._view.set_match_index(idx)
        self._view.lock_point(idx)
        self._select_index(idx)
        m = self._metrics[idx]
        self._match_status.setText(
            f"Closest match: ref=({m.ref_x:.1f}, {m.ref_y:.1f}) · "
            f"Δ={dist:.2f}° (L2 over {_match_mode_label(self._match_mode)})"
        )

    def _on_use_selected_ref(self) -> None:
        if self._selected_idx < 0 or self._selected_idx >= len(self._metrics):
            return
        m = self._metrics[self._selected_idx]
        self.ref_chosen.emit(float(m.ref_x), float(m.ref_y))


class KerrAnglesSpanDialog(QtWidgets.QDialog):
    """Show 1-D / 2-D Kerr angle histograms; allow tweaking the Kerr reference."""

    def __init__(
        self,
        preview: KerrAnglePreview,
        *,
        eye: str,
        eye_df: pd.DataFrame | None = None,
        frame_rgb: np.ndarray | None = None,
        other_preview: KerrAnglePreview | None = None,
        other_eye: str | None = None,
        n_bins: int = 60,
        n_bins_2d: int = 50,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        eye_lc = eye.lower()
        self._eye = eye_lc
        self._eye_label = "Left" if eye_lc == "left" else "Right"
        self._eye_df = None if eye_df is None else eye_df.copy()
        self._frame_rgb = None if frame_rgb is None else np.asarray(frame_rgb).copy()
        self._n_bins = max(5, int(n_bins))
        self._n_bins_2d = max(5, int(n_bins_2d))
        self._preview = preview
        self._other_preview = other_preview
        self._other_eye = None if other_eye is None else str(other_eye).lower()
        if self._other_eye == "left":
            self._other_eye_label = "Left"
        elif self._other_eye == "right":
            self._other_eye_label = "Right"
        else:
            self._other_eye_label = "Opposite"
        self._other_metrics = (
            None
            if other_preview is None
            else span_metrics_from_preview(other_preview)
        )
        self._grid_dialog: KerrRefSpanGridDialog | None = None
        self._showing_other = False

        self.setWindowTitle(f"{self._eye_label} eye — Kerr angles span preview")
        self.resize(980, 760)

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

        grid_row = QtWidgets.QHBoxLayout()
        grid_row.addWidget(QtWidgets.QLabel("ROI grid:"))
        self._spin_grid_x = QtWidgets.QSpinBox()
        self._spin_grid_x.setRange(1, 15)
        self._spin_grid_x.setValue(3)
        self._spin_grid_x.setToolTip("Number of Kerr-reference candidates along X")
        grid_row.addWidget(self._spin_grid_x)
        grid_row.addWidget(QtWidgets.QLabel("×"))
        self._spin_grid_y = QtWidgets.QSpinBox()
        self._spin_grid_y.setRange(1, 15)
        self._spin_grid_y.setValue(3)
        self._spin_grid_y.setToolTip("Number of Kerr-reference candidates along Y")
        grid_row.addWidget(self._spin_grid_y)
        self._btn_roi_grid = QtWidgets.QPushButton("ROI span sweep…")
        self._btn_roi_grid.setToolTip(
            "Draw a rectangular ROI on the eye frame, evaluate full and p5–p95 "
            "φ/θ spans on an evenly spaced Kerr-ref grid, and open a hover plot"
        )
        can_sweep = self._eye_df is not None and self._frame_rgb is not None
        self._btn_roi_grid.setEnabled(can_sweep)
        grid_row.addWidget(self._btn_roi_grid)
        grid_row.addStretch(1)
        layout.addLayout(grid_row)

        mad_row = QtWidgets.QHBoxLayout()
        self._chk_include_mad = QtWidgets.QCheckBox("Include MAD (mean abs. deviation)")
        self._chk_include_mad.setChecked(True)
        self._chk_include_mad.setToolTip(
            "Show φ/θ MAD alongside span values, and include MAD columns in the "
            "ROI grid table / sort options"
        )
        mad_row.addWidget(self._chk_include_mad)
        mad_row.addWidget(QtWidgets.QLabel("Match opposite eye on:"))
        self._combo_match_mode = QtWidgets.QComboBox()
        self._combo_match_mode.addItem("Spans (full + p5–p95)", "span")
        self._combo_match_mode.addItem("MAD only", "mad")
        self._combo_match_mode.addItem("Spans + MAD", "span_mad")
        self._combo_match_mode.setCurrentIndex(2)
        self._combo_match_mode.setToolTip(
            "Feature set used when picking the grid point closest to the opposite eye"
        )
        self._combo_match_mode.setEnabled(self._other_metrics is not None)
        mad_row.addWidget(self._combo_match_mode)
        mad_row.addStretch(1)
        layout.addLayout(mad_row)

        if self._eye_df is None:
            layout.addWidget(
                QtWidgets.QLabel(
                    "Recompute / ROI sweep disabled — eye data was not passed "
                    "into this dialog."
                )
            )
        elif self._frame_rgb is None:
            layout.addWidget(
                QtWidgets.QLabel(
                    "ROI span sweep disabled — no eye frame image was passed "
                    "into this dialog."
                )
            )

        other_box = QtWidgets.QGroupBox("Opposite eye")
        other_lay = QtWidgets.QVBoxLayout(other_box)
        self._other_summary = QtWidgets.QLabel("")
        self._other_summary.setWordWrap(True)
        other_lay.addWidget(self._other_summary)
        self._other_span = QtWidgets.QLabel("")
        self._other_span.setWordWrap(True)
        other_lay.addWidget(self._other_span)
        self._chk_show_other = QtWidgets.QCheckBox(
            f"Display {self._other_eye_label.lower()}-eye histograms / heatmap"
        )
        self._chk_show_other.setToolTip(
            "When checked, the plots below show the opposite eye's Kerr angles "
            "at its saved reference (read-only comparison)"
        )
        self._chk_show_other.setEnabled(self._other_preview is not None)
        other_lay.addWidget(self._chk_show_other)
        layout.addWidget(other_box)
        self._refresh_other_eye_labels()

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
        self._btn_roi_grid.clicked.connect(self._on_roi_span_sweep)
        self._chk_show_other.toggled.connect(self._on_toggle_show_other)
        self._chk_include_mad.toggled.connect(self._on_mad_display_toggled)
        self._refresh_from_preview(preview)

    def ref_xy(self) -> tuple[int, int]:
        """Current Kerr reference (rounded to integer pixels for the verifier)."""
        return (
            int(round(self._spin_ref_x.value())),
            int(round(self._spin_ref_y.value())),
        )

    def _include_mad(self) -> bool:
        return bool(self._chk_include_mad.isChecked())

    def _match_mode(self) -> MatchMode:
        data = self._combo_match_mode.currentData()
        mode = str(data) if data is not None else "span_mad"
        if mode not in ("span", "mad", "span_mad"):
            return "span_mad"
        return mode  # type: ignore[return-value]

    def _refresh_other_eye_labels(self) -> None:
        if self._other_preview is None or self._other_metrics is None:
            self._other_summary.setText(
                "Opposite-eye Kerr reference is not set — set it on the other "
                "verifier to compare / match spans."
            )
            self._other_span.setText("")
            return
        p = self._other_preview
        self._other_summary.setText(
            f"{self._other_eye_label} eye · "
            f"ref = ({p.ref_x:.1f}, {p.ref_y:.1f}) · "
            f"f_z = {p.f_z:.2f} · "
            f"{p.n_finite:,} / {p.n_input:,} finite angle samples"
        )
        self._other_span.setText(
            _metrics_span_text(
                f"{self._other_eye_label} spans",
                self._other_metrics,
                include_mad=self._include_mad(),
            )
        )

    def _displayed_preview(self) -> KerrAnglePreview:
        if self._showing_other and self._other_preview is not None:
            return self._other_preview
        return self._preview

    def _refresh_plots_from_preview(self, preview: KerrAnglePreview) -> None:
        phi, theta = _finite_pair(preview)
        which = (
            self._other_eye_label
            if (self._showing_other and self._other_preview is not None)
            else self._eye_label
        )
        self._phi_plot.setTitle(f"Phi — {which}")
        self._theta_plot.setTitle(f"Theta — {which}")
        self._heat_plot.setTitle(f"Angle space (φ × θ) — {which}")
        _add_histogram(self._phi_plot, phi, brush=_PHI_BRUSH, n_bins=self._n_bins)
        _add_histogram(self._theta_plot, theta, brush=_THETA_BRUSH, n_bins=self._n_bins)
        _add_heatmap(self._heat_plot, phi, theta, n_bins=self._n_bins_2d)

    def _refresh_from_preview(self, preview: KerrAnglePreview) -> None:
        self._preview = preview
        phi, theta = _finite_pair(preview)
        self._summary.setText(
            f"{self._eye_label} eye · ref = ({preview.ref_x:.1f}, {preview.ref_y:.1f}) · "
            f"f_z = {preview.f_z:.2f} · "
            f"{preview.n_finite:,} / {preview.n_input:,} finite angle samples"
        )
        include_mad = self._include_mad()
        self._phi_span.setText(_span_text("φ (phi)", phi, include_mad=include_mad))
        self._theta_span.setText(_span_text("θ (theta)", theta, include_mad=include_mad))
        if preview.n_finite == 0:
            self._warn.setText(
                "No finite φ/θ samples — the reference may be far from the pupil "
                "cloud, or ellipse axes may be invalid."
            )
            self._warn.show()
        else:
            self._warn.hide()
        self._refresh_other_eye_labels()
        self._refresh_plots_from_preview(self._displayed_preview())

    def _on_mad_display_toggled(self, _checked: bool = False) -> None:
        self._refresh_from_preview(self._preview)

    def _on_toggle_show_other(self, checked: bool) -> None:
        self._showing_other = bool(checked) and self._other_preview is not None
        self._refresh_plots_from_preview(self._displayed_preview())

    def _apply_grid_ref(self, ref_x: float, ref_y: float) -> None:
        self._spin_ref_x.setValue(float(ref_x))
        self._spin_ref_y.setValue(float(ref_y))
        if self._chk_show_other.isChecked():
            self._chk_show_other.setChecked(False)
        self._on_recompute()

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

    def _on_roi_span_sweep(self) -> None:
        if self._eye_df is None or self._frame_rgb is None:
            return

        frame_bgr = cv2.cvtColor(self._frame_rgb, cv2.COLOR_RGB2BGR)
        picker = QtRoiPickerDialog(
            frame_bgr,
            title=f"{self._eye_label} eye — Kerr ref ROI",
            instruction=(
                "Drag a rectangle over the region to sample Kerr reference "
                "candidates. Coordinates match Kerr/raw image space."
            ),
            parent=self,
        )
        if picker.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        roi = picker.selected_roi()
        if roi is None:
            return

        n_x = int(self._spin_grid_x.value())
        n_y = int(self._spin_grid_y.value())
        n_pts = n_x * n_y
        progress = QtWidgets.QProgressDialog(
            f"Computing Kerr spans for {n_pts} reference(s)…",
            "Cancel",
            0,
            n_pts,
            self,
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        metrics: list[KerrRefSpanMetrics] = []
        try:
            points = grid_points_in_roi(roi, n_x, n_y)
            for i, (px, py) in enumerate(points):
                if progress.wasCanceled():
                    return
                metrics.append(compute_kerr_ref_span(self._eye_df, px, py))
                progress.setValue(i + 1)
                QtWidgets.QApplication.processEvents()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "ROI span sweep",
                f"Grid span computation failed:\n{exc}",
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            progress.close()

        if len(metrics) != n_pts:
            return

        self._grid_dialog = KerrRefSpanGridDialog(
            self._frame_rgb,
            roi,
            metrics,
            eye=self._eye,
            n_x=n_x,
            n_y=n_y,
            target_metrics=self._other_metrics,
            target_eye_label=self._other_eye_label,
            include_mad=self._include_mad(),
            match_mode=self._match_mode(),
            parent=self,
        )
        self._grid_dialog.ref_chosen.connect(self._apply_grid_ref)
        self._grid_dialog.show()
