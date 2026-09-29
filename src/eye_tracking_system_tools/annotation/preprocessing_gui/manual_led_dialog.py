"""Manual LED_driver replacement from eye brightness blinks (Prepare-stage tool)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker import (
    load_eye_brightness_lists,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.ttl_mapping import (
    load_led_manual_replacement,
    save_led_manual_replacement,
)


BLINK_INTERVAL_S = 60.0
BLINK_OFF_DURATION_S = 0.034
DEFAULT_FPS = 60.0


def brightness_length(values) -> int:
    """Return number of video frames represented by a brightness vector."""
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        return 0
    if len(arr) == 0:
        return 0
    # Legacy [(frame_idx, values), ...] — use max frame index + 1 if present
    first = arr[0]
    if isinstance(first, (list, tuple)) and len(first) >= 1:
        try:
            frames = [int(x[0]) for x in arr]
            return max(frames) + 1 if frames else 0
        except (TypeError, ValueError, IndexError):
            pass
    return int(len(arr))


def generate_blink_frame_series(
    first_frame: int,
    n_frames: int,
    *,
    fps: float = DEFAULT_FPS,
    interval_s: float = BLINK_INTERVAL_S,
) -> np.ndarray:
    """
    Video-frame indices of LED blinks starting at ``first_frame``, every ``interval_s``.

    ``K = floor(duration_s / interval_s) + 1`` with duration from ``n_frames / fps``.
    Frames that fall outside ``[0, n_frames)`` are dropped.
    """
    first_frame = int(first_frame)
    n_frames = int(n_frames)
    if n_frames <= 0:
        return np.array([], dtype=np.int64)
    fps = float(fps) if fps and fps > 0 else DEFAULT_FPS
    interval_s = float(interval_s) if interval_s and interval_s > 0 else BLINK_INTERVAL_S
    duration_s = n_frames / fps
    k_max = int(np.floor(duration_s / interval_s)) + 1
    step = int(round(interval_s * fps))
    if step <= 0:
        step = int(round(BLINK_INTERVAL_S * DEFAULT_FPS))
    frames = first_frame + np.arange(k_max, dtype=np.int64) * step
    frames = frames[(frames >= 0) & (frames < n_frames)]
    return frames.astype(np.int64, copy=False)


def eye_ttl_samples(oe_events: pd.DataFrame, *, prefer_left: bool = True) -> np.ndarray:
    """Sorted rising-edge OE samples for the preferred eye TTL column."""
    primary = "L_eye_TTL" if prefer_left else "R_eye_TTL"
    secondary = "R_eye_TTL" if prefer_left else "L_eye_TTL"
    for col in (primary, secondary):
        if col in oe_events.columns:
            samples = oe_events[col].dropna().to_numpy(dtype=np.int64)
            if len(samples) > 0:
                samples = np.sort(samples)
                return samples
    raise ValueError(
        "oe_events must contain L_eye_TTL or R_eye_TTL columns to map blink frames to OE samples."
    )


def frames_to_oe_samples(
    frames: np.ndarray,
    eye_ttl: np.ndarray,
) -> np.ndarray:
    """Map video frame indices to OE samples via the Nth eye-TTL rising edge."""
    frames = np.asarray(frames, dtype=np.int64)
    eye_ttl = np.asarray(eye_ttl, dtype=np.int64)
    if len(eye_ttl) == 0:
        return np.array([], dtype=np.int64)
    out: list[int] = []
    for f in frames:
        if f < 0 or f >= len(eye_ttl):
            break
        out.append(int(eye_ttl[int(f)]))
    return np.asarray(out, dtype=np.int64)


def synthesize_led_fall_samples(
    on_samples: np.ndarray,
    sample_rate: float,
    *,
    off_duration_s: float = BLINK_OFF_DURATION_S,
) -> np.ndarray:
    """OFF sample = ON − off_duration (matches ``_get_led_off_oe_samples``)."""
    on_samples = np.asarray(on_samples, dtype=np.int64)
    delta = int(round(float(off_duration_s) * float(sample_rate)))
    return on_samples - delta


def build_led_replacement_payload(
    first_frame: int,
    n_frames: int,
    eye_ttl: np.ndarray,
    sample_rate: float,
    *,
    fps: float = DEFAULT_FPS,
    interval_s: float = BLINK_INTERVAL_S,
) -> dict[str, Any]:
    """Pure helper: frame series → OE on/fall samples + sidecar fields."""
    frames = generate_blink_frame_series(
        first_frame, n_frames, fps=fps, interval_s=interval_s
    )
    on_samples = frames_to_oe_samples(frames, eye_ttl)
    if len(on_samples) == 0:
        raise ValueError(
            "No blink frames mapped to eye TTL samples. "
            "Check first-frame placement and that eye TTLs were parsed."
        )
    fall_samples = synthesize_led_fall_samples(on_samples, sample_rate)
    return {
        "first_frame": int(first_frame),
        "interval_s": float(interval_s),
        "fps": float(fps),
        "blink_frames": [int(x) for x in frames[: len(on_samples)]],
        "on_samples": [int(x) for x in on_samples],
        "fall_samples": [int(x) for x in fall_samples],
        "source": "manual_brightness",
    }


def inject_led_into_oe_events(
    oe_events: pd.DataFrame,
    on_samples: list[int] | np.ndarray,
    fall_samples: list[int] | np.ndarray,
) -> pd.DataFrame:
    """
    Insert/replace ``LED_driver`` and ``LED_driver_fall`` columns in ``oe_events``.

    Extends the DataFrame length if needed so all samples fit; other columns get NaN
    in newly appended rows.
    """
    on_samples = np.asarray(on_samples, dtype=np.int64)
    fall_samples = np.asarray(fall_samples, dtype=np.int64)
    n = max(len(on_samples), len(fall_samples))
    if n == 0:
        raise ValueError("Cannot inject empty LED sample lists.")

    out = oe_events.copy()
    if len(out) < n:
        extra = pd.DataFrame(index=range(len(out), n), columns=out.columns)
        out = pd.concat([out, extra], axis=0)

    # Clear existing LED columns then write
    for col in ("LED_driver", "LED_driver_fall", "LED_driver_frame"):
        if col in out.columns:
            out[col] = np.nan

    led = np.full(len(out), np.nan)
    led_fall = np.full(len(out), np.nan)
    led_frame = np.full(len(out), np.nan)
    for i, s in enumerate(on_samples):
        led[i] = int(s)
        led_frame[i] = i
    for i, s in enumerate(fall_samples):
        led_fall[i] = int(s)

    out["LED_driver"] = led
    out["LED_driver_fall"] = led_fall
    out["LED_driver_frame"] = led_frame
    return out


def apply_led_manual_replacement_to_blocksync(
    blocksync,
    payload: dict[str, Any],
    *,
    write_parsed_csv: bool = True,
) -> None:
    """Inject synthetic LED columns into ``blocksync.oe_events`` and optionally disk."""
    oe = getattr(blocksync, "oe_events", None)
    if oe is None or not isinstance(oe, pd.DataFrame) or oe.empty:
        raise ValueError("blocksync.oe_events is missing; Parse OE events first.")

    blocksync.oe_events = inject_led_into_oe_events(
        oe,
        payload["on_samples"],
        payload["fall_samples"],
    )

    oe_dirname = getattr(blocksync, "oe_dirname", None)
    if oe_dirname:
        save_led_manual_replacement(
            blocksync.block_path,
            oe_dirname,
            first_frame=int(payload["first_frame"]),
            interval_s=float(payload["interval_s"]),
            fps=float(payload["fps"]),
            on_samples=list(payload["on_samples"]),
            fall_samples=list(payload["fall_samples"]),
            source=str(payload.get("source", "manual_brightness")),
        )
        if write_parsed_csv:
            parsed = (
                Path(blocksync.block_path)
                / "oe_files"
                / oe_dirname
                / "parsed_events.csv"
            )
            blocksync.oe_events.to_csv(parsed)


def try_apply_saved_led_replacement(blocksync) -> bool:
    """
    If a LED replacement sidecar exists and OE lacks LED columns, apply it.

    Returns True when injection ran.
    """
    from eye_tracking_system_tools.annotation.preprocessing_gui.ttl_mapping import (
        oe_events_have_led_driver,
    )

    if oe_events_have_led_driver(getattr(blocksync, "oe_events", None)):
        return False
    oe_dirname = getattr(blocksync, "oe_dirname", None)
    payload = load_led_manual_replacement(blocksync.block_path, oe_dirname)
    if payload is None:
        return False
    apply_led_manual_replacement_to_blocksync(blocksync, payload, write_parsed_csv=True)
    return True


class BrightnessBlinkPlot(QtWidgets.QWidget):
    """Brightness traces with a draggable primary blink and series markers."""

    primary_frame_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.addLegend()
        self.plot_widget.setLabel("bottom", "Video frame index")
        self.plot_widget.setLabel("left", "Mean brightness (a.u.)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        layout.addWidget(self.plot_widget)

        self._primary_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=True,
            pen=pg.mkPen("#c0392b", width=2.5),
            hoverPen=pg.mkPen("#e74c3c", width=3),
        )
        self._primary_line.setZValue(20)
        self._series_lines: list[pg.InfiniteLine] = []
        self._primary_line.sigPositionChanged.connect(self._on_primary_dragged)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_click)
        self._n_frames = 0

    def set_traces(self, left_values, right_values) -> None:
        self._clear_series()
        self.plot_widget.clear()
        self.plot_widget.addLegend()
        left = np.asarray(left_values, dtype=float)
        right = np.asarray(right_values, dtype=float)
        self._n_frames = int(min(len(left), len(right))) if len(left) and len(right) else max(
            len(left), len(right)
        )
        if len(left):
            self.plot_widget.plot(
                np.arange(len(left)),
                left,
                pen=pg.mkPen("#1f77b4", width=1.2),
                name="Left eye",
            )
        if len(right):
            self.plot_widget.plot(
                np.arange(len(right)),
                right,
                pen=pg.mkPen("#d62728", width=1.2),
                name="Right eye",
            )
        self.plot_widget.addItem(self._primary_line)
        self._primary_line.setVisible(False)

    def _clear_series(self) -> None:
        for item in self._series_lines:
            self.plot_widget.removeItem(item)
        self._series_lines.clear()

    def set_blink_series(self, frames: np.ndarray, *, primary_frame: int) -> None:
        self._clear_series()
        frames = np.asarray(frames, dtype=np.int64)
        for f in frames:
            if int(f) == int(primary_frame):
                continue
            line = pg.InfiniteLine(
                pos=float(f),
                angle=90,
                movable=False,
                pen=pg.mkPen("#27ae60", width=1.2, style=QtCore.Qt.PenStyle.DashLine),
            )
            line.setZValue(10)
            self.plot_widget.addItem(line)
            self._series_lines.append(line)
        self._primary_line.blockSignals(True)
        self._primary_line.setPos(float(primary_frame))
        self._primary_line.setVisible(True)
        self._primary_line.blockSignals(False)

    def primary_frame(self) -> int | None:
        if not self._primary_line.isVisible():
            return None
        return int(round(float(self._primary_line.value())))

    def _on_primary_dragged(self) -> None:
        frame = int(round(float(self._primary_line.value())))
        if self._n_frames > 0:
            frame = max(0, min(frame, self._n_frames - 1))
            self._primary_line.blockSignals(True)
            self._primary_line.setPos(float(frame))
            self._primary_line.blockSignals(False)
        self.primary_frame_changed.emit(frame)

    def _on_click(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if not self.plot_widget.sceneBoundingRect().contains(event.scenePos()):
            return
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
        frame = int(round(float(mouse_point.x())))
        if self._n_frames > 0:
            frame = max(0, min(frame, self._n_frames - 1))
        self._primary_line.setVisible(True)
        self._primary_line.blockSignals(True)
        self._primary_line.setPos(float(frame))
        self._primary_line.blockSignals(False)
        self.primary_frame_changed.emit(frame)
        event.accept()


class ManualLedDialog(QtWidgets.QDialog):
    """Place a single blink on brightness traces; expand to a 60 s LED_driver series."""

    def __init__(
        self,
        blocksync,
        *,
        fps: float = DEFAULT_FPS,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._blocksync = blocksync
        self._fps = float(fps)
        self._payload: dict[str, Any] | None = None

        left, right = load_eye_brightness_lists(blocksync)
        self._left = left
        self._right = right
        self._n_frames = min(brightness_length(left), brightness_length(right))
        if self._n_frames <= 0:
            raise ValueError("Brightness vectors are empty.")

        oe = getattr(blocksync, "oe_events", None)
        if oe is None or not isinstance(oe, pd.DataFrame):
            raise ValueError("Parse OE events before placing a manual LED series.")
        self._eye_ttl = eye_ttl_samples(oe, prefer_left=True)

        self.setWindowTitle(f"Manual LED blink replacement — block {blocksync.block_num}")
        self.resize(960, 640)
        self._build_ui()
        self._restore_from_sidecar()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Click a brightness dip to place the first LED blink. "
            "The tool generates events every 60 s. Drag the red line to fine-tune "
            "(the green series moves with it). Zoom/pan the plot as needed."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self._plot = BrightnessBlinkPlot()
        self._plot.setMinimumHeight(360)
        self._plot.set_traces(self._left, self._right)
        self._plot.primary_frame_changed.connect(self._on_primary_changed)
        root.addWidget(self._plot, stretch=1)

        form = QtWidgets.QFormLayout()
        self._first_frame_spin = QtWidgets.QSpinBox()
        self._first_frame_spin.setRange(0, max(0, self._n_frames - 1))
        self._fps_spin = QtWidgets.QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 240.0)
        self._fps_spin.setValue(self._fps)
        self._summary = QtWidgets.QLabel("No blink placed yet.")
        self._summary.setWordWrap(True)
        form.addRow("First blink frame:", self._first_frame_spin)
        form.addRow("Assumed fps:", self._fps_spin)
        form.addRow(self._summary)
        root.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self._btn_apply = QtWidgets.QPushButton("Apply LED replacement")
        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        btns.addWidget(self._btn_apply)
        btns.addWidget(self._btn_cancel)
        root.addLayout(btns)

        self._first_frame_spin.valueChanged.connect(self._on_spin_changed)
        self._fps_spin.valueChanged.connect(self._refresh_series)
        self._btn_apply.clicked.connect(self._on_accept)
        self._btn_cancel.clicked.connect(self.reject)

    def _restore_from_sidecar(self) -> None:
        oe_dirname = getattr(self._blocksync, "oe_dirname", None)
        payload = load_led_manual_replacement(self._blocksync.block_path, oe_dirname)
        if payload is None:
            return
        first = int(payload.get("first_frame", 0))
        fps = float(payload.get("fps", self._fps))
        self._fps_spin.blockSignals(True)
        self._fps_spin.setValue(fps)
        self._fps_spin.blockSignals(False)
        self._first_frame_spin.blockSignals(True)
        self._first_frame_spin.setValue(max(0, min(first, self._n_frames - 1)))
        self._first_frame_spin.blockSignals(False)
        self._refresh_series()

    def _on_primary_changed(self, frame: int) -> None:
        self._first_frame_spin.blockSignals(True)
        self._first_frame_spin.setValue(int(frame))
        self._first_frame_spin.blockSignals(False)
        self._refresh_series()

    def _on_spin_changed(self, frame: int) -> None:
        self._plot._primary_line.setVisible(True)
        self._plot._primary_line.blockSignals(True)
        self._plot._primary_line.setPos(float(frame))
        self._plot._primary_line.blockSignals(False)
        self._refresh_series()

    def _refresh_series(self) -> None:
        first = int(self._first_frame_spin.value())
        fps = float(self._fps_spin.value())
        frames = generate_blink_frame_series(
            first, self._n_frames, fps=fps, interval_s=BLINK_INTERVAL_S
        )
        # Only keep frames that map into eye TTL
        mappable = frames[frames < len(self._eye_ttl)]
        self._plot.set_blink_series(mappable, primary_frame=first)
        self._summary.setText(
            f"{len(mappable)} blink(s) · interval {BLINK_INTERVAL_S:.0f} s · "
            f"eye TTL edges available: {len(self._eye_ttl)}"
        )

    def _on_accept(self) -> None:
        try:
            first = int(self._first_frame_spin.value())
            fps = float(self._fps_spin.value())
            payload = build_led_replacement_payload(
                first,
                self._n_frames,
                self._eye_ttl,
                float(self._blocksync.sample_rate),
                fps=fps,
                interval_s=BLINK_INTERVAL_S,
            )
            apply_led_manual_replacement_to_blocksync(self._blocksync, payload)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Manual LED replacement", str(e))
            return
        self._payload = payload
        self.accept()

    def payload(self) -> dict[str, Any] | None:
        return self._payload
