"""Saccades tab — velocity threshold, detect, finalize (windowed traces)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PyQt6 import QtWidgets

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.export_meta import load_params_yaml
from eye_tracking_system_tools.analysis.eye_trace_io import load_block_eyes
from eye_tracking_system_tools.analysis.param_tune import prepare_traces, raw_threshold_runs
from eye_tracking_system_tools.analysis.saccade_export import (
    DetectResult,
    deg_per_frame_from_deg_per_ms,
    deg_per_ms_from_deg_per_frame,
    detect_block_saccades,
    has_finalized_saccades,
    infer_frame_ms,
    read_finalized_saccades,
    saccades_dir,
    write_finalized_saccades,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    SACCADES_ARTIFACT_PROFILE,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.saccade_plot_panel import (
    DEFAULT_WINDOW_MS,
    SaccadeVelocityPanel,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.workers import CallableWorker


def _repo_params_path() -> Path:
    return Path(__file__).resolve().parents[5] / "configs" / "analysis_params.yaml"


def _slice_window(
    df: pd.DataFrame | None, t0: float, t1: float
) -> pd.DataFrame | None:
    if df is None or df.empty or "ms_axis" not in df.columns:
        return df
    ms = df["ms_axis"]
    return df.loc[(ms >= t0) & (ms < t1)].copy()


class SaccadesTab(BaseTab):
    tab_id = "saccades"
    tab_label = "Saccades"

    def __init__(self, state, config, parent=None):
        self._worker: CallableWorker | None = None
        self._left_full: pd.DataFrame | None = None
        self._right_full: pd.DataFrame | None = None
        self._frame_ms = 1000.0 / 60.0
        self._result: DetectResult | None = None
        self._params: dict[str, Any] = {}
        self._suppress_thr = False
        self._window_ms = DEFAULT_WINDOW_MS
        self._window_idx = 0
        self._t_min = 0.0
        self._t_max = 0.0
        super().__init__(state, config, parent)

    def artifact_profile(self):
        return SACCADES_ARTIFACT_PROFILE

    def status_signature(self, block: BlockHandle) -> list[Path]:
        d = saccades_dir(block.block_path)
        return [
            d / "saccade_events.csv",
            d / "detection_params.yaml",
        ]

    # ------------------------------------------------------------------ UI
    def build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self._build_artifact_panel())

        self._info = QtWidgets.QLabel(
            "Load Kerr eye CSVs, tune the speed threshold on a 100 s window, "
            "run detection on the full block, then finalize. "
            "Pixel size lives on the Calibration tab."
        )
        self._info.setWordWrap(True)
        root.addWidget(self._info)

        vel = QtWidgets.QGroupBox("1. Velocity trace & threshold (100 s windows)")
        vel_lay = QtWidgets.QVBoxLayout(vel)
        load_row = QtWidgets.QHBoxLayout()
        self._btn_load_traces = QtWidgets.QPushButton("Load eye traces")
        self._btn_load_traces.setToolTip(
            "Load Kerr eye CSVs, prepare speed columns, show the first 100 s"
        )
        self._trace_status = QtWidgets.QLabel("")
        load_row.addWidget(self._btn_load_traces)
        load_row.addWidget(self._trace_status, stretch=1)
        vel_lay.addLayout(load_row)

        self._plot = SaccadeVelocityPanel()
        self._plot.setMinimumHeight(280)
        vel_lay.addWidget(self._plot)

        params_form = QtWidgets.QFormLayout()
        self._thr_deg_ms = QtWidgets.QDoubleSpinBox()
        self._thr_deg_ms.setDecimals(5)
        self._thr_deg_ms.setRange(1e-6, 10.0)
        self._thr_deg_ms.setSingleStep(0.001)
        self._thr_deg_ms.setValue(0.048)
        self._thr_deg_ms.setToolTip(
            "Detector threshold in deg/ms; converted to deg/frame using the "
            "inferred frame period"
        )
        self._thr_frame_label = QtWidgets.QLabel("≈ 0.8 deg/frame")
        thr_row = QtWidgets.QHBoxLayout()
        thr_row.addWidget(self._thr_deg_ms)
        thr_row.addWidget(self._thr_frame_label)
        params_form.addRow("Speed threshold (deg/ms):", thr_row)

        self._dir_delta = QtWidgets.QDoubleSpinBox()
        self._dir_delta.setRange(1.0, 180.0)
        self._dir_delta.setValue(90.0)
        self._dir_delta.setToolTip("Split a speed run when instantaneous direction jumps by more than this")
        params_form.addRow("Directional delta (deg):", self._dir_delta)

        self._min_samples = QtWidgets.QSpinBox()
        self._min_samples.setRange(1, 50)
        self._min_samples.setValue(2)
        params_form.addRow("Min subsaccade samples:", self._min_samples)

        self._min_disp = QtWidgets.QDoubleSpinBox()
        self._min_disp.setDecimals(3)
        self._min_disp.setRange(0.0, 50.0)
        self._min_disp.setValue(0.5)
        params_form.addRow("Min net disp (deg):", self._min_disp)

        self._sync_ms = QtWidgets.QDoubleSpinBox()
        self._sync_ms.setDecimals(1)
        self._sync_ms.setRange(1.0, 500.0)
        self._sync_ms.setValue(34.0)
        self._sync_ms.setToolTip("L/R onsets within this window are concurrent pairs")
        params_form.addRow("Binocular sync window (ms):", self._sync_ms)

        self._chk_raw = QtWidgets.QCheckBox("Shade raw above-threshold runs (this window)")
        self._chk_raw.setChecked(True)
        params_form.addRow("", self._chk_raw)
        vel_lay.addLayout(params_form)

        self._frame_ms_label = QtWidgets.QLabel("Frame period: —")
        vel_lay.addWidget(self._frame_ms_label)
        root.addWidget(vel)

        det = QtWidgets.QGroupBox("2. Detect & statistics (full block)")
        det_lay = QtWidgets.QHBoxLayout(det)
        left_col = QtWidgets.QVBoxLayout()
        self._btn_detect = QtWidgets.QPushButton("Run saccade detection")
        self._btn_detect.setToolTip(
            "Run the angular detector + binocular pairing on the full block "
            "(not just the visible window)"
        )
        left_col.addWidget(self._btn_detect)
        self._stats = QtWidgets.QPlainTextEdit()
        self._stats.setReadOnly(True)
        self._stats.setMaximumHeight(160)
        left_col.addWidget(self._stats)
        det_lay.addLayout(left_col, stretch=2)

        hist_col = QtWidgets.QVBoxLayout()
        hist_col.addWidget(QtWidgets.QLabel("Amplitude histogram (net_angular_disp)"))
        self._hist_plot = self._make_hist_widget()
        hist_col.addWidget(self._hist_plot)
        det_lay.addLayout(hist_col, stretch=1)
        root.addWidget(det)

        fin = QtWidgets.QGroupBox("3. Finalize")
        fin_lay = QtWidgets.QHBoxLayout(fin)
        self._btn_finalize = QtWidgets.QPushButton("Save saccades to analysis/saccades/")
        self._btn_finalize.setEnabled(False)
        self._finalize_status = QtWidgets.QLabel("")
        fin_lay.addWidget(self._btn_finalize)
        fin_lay.addWidget(self._finalize_status, stretch=1)
        root.addWidget(fin)

        self._status = QtWidgets.QLabel("")
        root.addWidget(self._status)
        root.addStretch(1)

        self._load_default_params()
        self._wire()

    def _make_hist_widget(self) -> QtWidgets.QWidget:
        import pyqtgraph as pg

        w = pg.PlotWidget()
        w.setMinimumWidth(220)
        w.setMaximumHeight(180)
        w.setLabel("bottom", "Amplitude (deg)")
        w.setLabel("left", "Count")
        ax = w.getAxis("bottom")
        ax.enableAutoSIPrefix(False)
        self._hist_widget = w
        return w

    def _wire(self) -> None:
        self._btn_load_traces.clicked.connect(self._load_traces)
        self._btn_detect.clicked.connect(self._run_detection)
        self._btn_finalize.clicked.connect(self._finalize)
        self._plot.threshold_changed.connect(self._on_plot_threshold)
        self._plot.mode_changed.connect(self._on_mode_changed)
        self._plot.window_nav_requested.connect(self._on_window_nav)
        self._thr_deg_ms.valueChanged.connect(self._on_thr_spin_changed)
        self._chk_raw.toggled.connect(lambda _: self._refresh_raw_runs())

    def _load_default_params(self) -> None:
        path = _repo_params_path()
        try:
            self._params = load_params_yaml(path) if path.is_file() else {}
        except Exception:
            self._params = {}
        s = self._params.get("saccade", {})
        b = self._params.get("binocular", {})
        thr_frame = float(s.get("speed_threshold_deg_per_frame", 0.8))
        self._dir_delta.setValue(float(s.get("directional_delta_threshold_deg", 90.0)))
        self._min_samples.setValue(int(s.get("min_subsaccade_samples", 2)))
        self._min_disp.setValue(float(s.get("min_net_disp_deg", 0.5)))
        self._sync_ms.setValue(float(b.get("sync_diff_ms", 34.0)))
        thr_ms = deg_per_ms_from_deg_per_frame(thr_frame, self._frame_ms)
        self._suppress_thr = True
        self._thr_deg_ms.setValue(thr_ms)
        self._suppress_thr = False
        self._sync_threshold_widgets(from_spin=True)

    # ------------------------------------------------------------------ block
    def set_block(self, block: BlockHandle | None) -> None:
        self._left_full = None
        self._right_full = None
        self._result = None
        self._window_idx = 0
        self._t_min = 0.0
        self._t_max = 0.0
        self._btn_finalize.setEnabled(False)
        self._finalize_status.setText("")
        self._stats.clear()
        self._clear_hist()
        if hasattr(self, "_plot"):
            self._plot.set_traces(None, None)
            self._plot.set_window_label("Window: —")
            self._plot.set_nav_enabled(prev=False, next_=False)
        if block is None:
            self._info.setText("No block loaded.")
            self._trace_status.setText("")
            self._status.setText("")
        else:
            self._info.setText(
                f"Active block: {block.display_label}. "
                "Pixel calibration is on the Calibration tab."
            )
            if has_finalized_saccades(block.block_path):
                self._status.setText(
                    f"Finalized saccades already on disk under {saccades_dir(block.block_path)}"
                )
            else:
                self._status.setText(
                    "Load eye traces (first 100 s), tune threshold, run detection, then finalize."
                )
        super().set_block(block)

    def _after_load_artifacts(self, report) -> None:
        if self._block is not None and has_finalized_saccades(self._block.block_path):
            try:
                fin = read_finalized_saccades(self._block.block_path)
                self._apply_saved_params(fin.params)
                self._status.setText(
                    f"Loaded finalized params from disk ({fin.source}). "
                    "Load traces and re-detect to preview, or keep as-is."
                )
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Could not read finalized saccades: {exc}")

    def _apply_saved_params(self, params: dict[str, Any]) -> None:
        s = params.get("saccade", {})
        b = params.get("binocular", {})
        if "frame_ms" in params:
            self._frame_ms = float(params["frame_ms"])
        thr_frame = float(s.get("speed_threshold_deg_per_frame", 0.8))
        thr_ms = float(
            params.get(
                "speed_threshold_deg_per_ms",
                deg_per_ms_from_deg_per_frame(thr_frame, self._frame_ms),
            )
        )
        self._suppress_thr = True
        self._thr_deg_ms.setValue(thr_ms)
        self._dir_delta.setValue(float(s.get("directional_delta_threshold_deg", 90.0)))
        self._min_samples.setValue(int(s.get("min_subsaccade_samples", 2)))
        self._min_disp.setValue(float(s.get("min_net_disp_deg", 0.5)))
        self._sync_ms.setValue(float(b.get("sync_diff_ms", 34.0)))
        self._suppress_thr = False
        self._sync_threshold_widgets(from_spin=True)
        self._frame_ms_label.setText(f"Frame period: {self._frame_ms:.4g} ms")

    def _spec(self) -> BlockSpec:
        if self._block is None:
            raise RuntimeError("No block loaded.")
        return BlockSpec(
            animal=self._block.animal_call,
            block_path=self._block.block_path,
            block_num=str(self._block.block_num).zfill(3),
        )

    # ------------------------------------------------------------------ windowing
    def _n_windows(self) -> int:
        span = max(0.0, self._t_max - self._t_min)
        if span <= 0:
            return 1
        return max(1, int(np.ceil(span / self._window_ms)))

    def _window_bounds(self) -> tuple[float, float]:
        t0 = self._t_min + self._window_idx * self._window_ms
        t1 = t0 + self._window_ms
        return t0, t1

    def _refresh_window_view(self) -> None:
        if self._left_full is None and self._right_full is None:
            return
        t0, t1 = self._window_bounds()
        left = _slice_window(self._left_full, t0, t1)
        right = _slice_window(self._right_full, t0, t1)
        self._plot.set_traces(left, right, mode=self._plot.mode, window_t0_ms=t0)
        n = self._n_windows()
        self._plot.set_window_label(
            f"Window {self._window_idx + 1}/{n}: "
            f"{t0 / 1000:.1f}–{min(t1, self._t_max) / 1000:.1f} s"
        )
        self._plot.set_nav_enabled(
            prev=self._window_idx > 0,
            next_=self._window_idx < n - 1,
        )
        self._sync_threshold_widgets(from_spin=True)
        self._refresh_raw_runs()
        if self._result is not None:
            self._plot.set_event_overlays(self._result.all_saccades)

    def _on_window_nav(self, direction: str) -> None:
        n = self._n_windows()
        if direction == "prev" and self._window_idx > 0:
            self._window_idx -= 1
        elif direction == "next" and self._window_idx < n - 1:
            self._window_idx += 1
        else:
            return
        self._refresh_window_view()

    # ------------------------------------------------------------------ traces
    def _load_traces(self) -> None:
        if self._block is None:
            return
        self._status.setText("Loading eye traces…")
        self._set_busy(True)
        spec = self._spec()

        def work():
            loaded = load_block_eyes(spec)
            left = prepare_traces(loaded.left)
            right = prepare_traces(loaded.right)
            return left, right, loaded

        self._start_worker(work, self._on_traces_loaded, self._on_worker_fail)

    def _on_traces_loaded(self, payload) -> None:
        self._set_busy(False)
        left, right, loaded = payload
        self._left_full = left
        self._right_full = right
        self._frame_ms = infer_frame_ms(left, right)
        self._frame_ms_label.setText(f"Frame period: {self._frame_ms:.4g} ms")

        t_parts = []
        for df in (left, right):
            if df is not None and not df.empty and "ms_axis" in df.columns:
                ms = df["ms_axis"].to_numpy(dtype=float)
                ms = ms[np.isfinite(ms)]
                if ms.size:
                    t_parts.append(ms)
        if t_parts:
            all_t = np.concatenate(t_parts)
            self._t_min = float(np.nanmin(all_t))
            self._t_max = float(np.nanmax(all_t))
        else:
            self._t_min = 0.0
            self._t_max = self._window_ms

        self._window_idx = 0
        meta = (
            f"L={Path(loaded.left_csv.path).name} ({loaded.left_csv.rule}); "
            f"R={Path(loaded.right_csv.path).name} ({loaded.right_csv.rule}); "
            f"span={(self._t_max - self._t_min) / 1000:.1f} s → "
            f"{self._n_windows()}×100 s windows"
        )
        self._trace_status.setText(meta)
        self._refresh_window_view()
        self._status.setText(
            f"Loaded traces; showing first 100 s "
            f"({self._n_windows()} windows total)."
        )

    # ------------------------------------------------------------------ threshold sync
    def _current_thr_frame(self) -> float:
        if self._plot.mode == "px":
            return float(self._plot.threshold)
        return deg_per_frame_from_deg_per_ms(
            float(self._thr_deg_ms.value()), self._frame_ms
        )

    def _sync_threshold_widgets(self, *, from_spin: bool) -> None:
        if self._plot.mode == "px":
            self._thr_frame_label.setText(
                f"detector ≈ {deg_per_frame_from_deg_per_ms(float(self._thr_deg_ms.value()), self._frame_ms):.4g} "
                f"deg/frame (plot thr is px/frame)"
            )
            return
        thr_ms = float(self._thr_deg_ms.value())
        thr_frame = deg_per_frame_from_deg_per_ms(thr_ms, self._frame_ms)
        self._thr_frame_label.setText(f"≈ {thr_frame:.4g} deg/frame")
        if from_spin:
            self._suppress_thr = True
            self._plot.set_threshold(thr_frame)
            self._suppress_thr = False

    def _on_thr_spin_changed(self, _v: float) -> None:
        if self._suppress_thr:
            return
        self._sync_threshold_widgets(from_spin=True)
        self._refresh_raw_runs()

    def _on_plot_threshold(self, value: float) -> None:
        if self._suppress_thr:
            return
        if self._plot.mode == "px":
            self._thr_frame_label.setText(
                f"plot thr={value:.4g} px/frame; detector stays at "
                f"{deg_per_frame_from_deg_per_ms(float(self._thr_deg_ms.value()), self._frame_ms):.4g} deg/frame"
            )
            self._refresh_raw_runs()
            return
        thr_ms = deg_per_ms_from_deg_per_frame(float(value), self._frame_ms)
        self._suppress_thr = True
        self._thr_deg_ms.setValue(thr_ms)
        self._suppress_thr = False
        self._thr_frame_label.setText(f"≈ {float(value):.4g} deg/frame")
        self._refresh_raw_runs()

    def _on_mode_changed(self, mode: str) -> None:
        if self._left_full is None:
            return
        self._refresh_window_view()
        if mode == "px":
            t0, t1 = self._window_bounds()
            parts = []
            for df in (
                _slice_window(self._left_full, t0, t1),
                _slice_window(self._right_full, t0, t1),
            ):
                if df is not None and "speed_r" in df.columns:
                    v = df["speed_r"].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    if v.size:
                        parts.append(v)
            if parts:
                seed = float(np.nanpercentile(np.concatenate(parts), 95))
                self._plot.set_threshold(max(seed, 0.5))

    def _refresh_raw_runs(self) -> None:
        if not self._chk_raw.isChecked() or self._left_full is None:
            self._plot.clear_raw_overlays()
            return
        t0, t1 = self._window_bounds()
        mode = self._plot.mode
        thr = float(self._plot.threshold)
        runs: list[tuple[float, float]] = []
        for df in (
            _slice_window(self._left_full, t0, t1),
            _slice_window(self._right_full, t0, t1),
        ):
            if df is None:
                continue
            runs.extend(raw_threshold_runs(df, mode=mode, threshold=thr))
        self._plot.set_raw_runs(runs)

    # ------------------------------------------------------------------ detect
    def _gather_params(self) -> dict[str, Any]:
        # Always use deg/ms spin → deg/frame for the angular detector
        thr_frame = deg_per_frame_from_deg_per_ms(
            float(self._thr_deg_ms.value()), self._frame_ms
        )
        return {
            "saccade": {
                "speed_threshold_deg_per_frame": float(thr_frame),
                "directional_delta_threshold_deg": float(self._dir_delta.value()),
                "min_subsaccade_samples": int(self._min_samples.value()),
                "min_net_disp_deg": float(self._min_disp.value()),
                "speed_profile": True,
            },
            "binocular": {
                "sync_diff_ms": float(self._sync_ms.value()),
            },
        }

    def _run_detection(self) -> None:
        if self._block is None:
            return
        if self._left_full is None or self._right_full is None:
            QtWidgets.QMessageBox.information(
                self, "Load traces first", "Click 'Load eye traces' before detecting."
            )
            return
        params = self._gather_params()
        self._params = params
        self._status.setText("Running saccade detection on full block…")
        self._set_busy(True)
        spec = self._spec()

        def work():
            return detect_block_saccades(
                spec.block_path,
                spec.animal,
                spec.block_num,
                params,
                keep_traces=True,
                spec=spec,
            )

        self._start_worker(work, self._on_detect_done, self._on_worker_fail)

    def _on_detect_done(self, result: DetectResult) -> None:
        self._set_busy(False)
        self._result = result
        self._frame_ms = result.frame_ms
        self._frame_ms_label.setText(f"Frame period: {self._frame_ms:.4g} ms")
        self._plot.set_event_overlays(result.all_saccades)
        self._show_stats(result.summary)
        self._show_hist(result.summary)
        self._btn_finalize.setEnabled(True)
        s = result.summary
        self._status.setText(
            f"Detected {s.get('n_events', 0)} events "
            f"({s.get('n_concurrent_pairs', 0)} concurrent pairs, "
            f"{s.get('n_monocular', 0)} monocular). Overlays shown in current window."
        )

    def _show_stats(self, summary: dict[str, Any]) -> None:
        isi_mean = summary.get("isi_mean_ms", float("nan"))
        isi_std = summary.get("isi_std_ms", float("nan"))
        lines = [
            f"Events: {summary.get('n_events', 0)}",
            f"Concurrent pairs: {summary.get('n_concurrent_pairs', 0)}",
            f"Monocular events: {summary.get('n_monocular', 0)}",
            f"ISI mean ± std: {isi_mean:.2f} ± {isi_std:.2f} ms "
            f"(n={summary.get('isi_n', 0)})",
            f"Amplitude mean / median: "
            f"{summary.get('amp_mean_deg', float('nan')):.3f} / "
            f"{summary.get('amp_median_deg', float('nan')):.3f} deg",
            f"Threshold: {summary.get('speed_threshold_deg_per_frame')} deg/frame",
        ]
        self._stats.setPlainText("\n".join(lines))

    def _clear_hist(self) -> None:
        if getattr(self, "_hist_widget", None) is None:
            return
        self._hist_widget.clear()

    def _show_hist(self, summary: dict[str, Any]) -> None:
        import pyqtgraph as pg

        self._clear_hist()
        counts = summary.get("amplitude_hist_counts") or []
        edges = summary.get("amplitude_hist_edges_deg") or []
        if not counts or len(edges) < 2:
            return
        widths = np.diff(np.asarray(edges, dtype=float))
        x = np.asarray(edges[:-1], dtype=float)
        y = np.asarray(counts, dtype=float)
        bar = pg.BarGraphItem(x=x, height=y, width=widths * 0.9, brush="#4c78a8")
        self._hist_widget.addItem(bar)

    # ------------------------------------------------------------------ finalize
    def _finalize(self) -> None:
        if self._block is None or self._result is None:
            return
        if has_finalized_saccades(self._block.block_path):
            ans = QtWidgets.QMessageBox.question(
                self,
                "Overwrite finalized saccades?",
                f"Replace existing files under {saccades_dir(self._block.block_path)}?",
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            paths = write_finalized_saccades(
                self._block.block_path, self._result, overwrite=True
            )
            self._finalize_status.setText(f"Wrote {paths['dir']}")
            self._status.setText(f"Finalized saccades → {paths['dir']}")
            self._notify_status()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Finalize failed", str(exc))

    # ------------------------------------------------------------------ workers
    def _start_worker(self, work_fn, on_ok, on_fail) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.warning(
                self, "Busy", "Another task is still running."
            )
            return
        self._worker = CallableWorker(work_fn, parent=self)
        self._worker.finished_ok.connect(on_ok)
        self._worker.failed.connect(on_fail)
        self._worker.start()

    def _on_worker_fail(self, msg: str) -> None:
        self._set_busy(False)
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self._status.setText(f"Error: {msg}")

    def _set_busy(self, busy: bool) -> None:
        for w in (self._btn_load_traces, self._btn_detect, self._btn_finalize):
            w.setEnabled(not busy)
        if not busy and self._result is not None:
            self._btn_finalize.setEnabled(True)

    def _notify_status(self) -> None:
        parent = self.window()
        bus = getattr(parent, "_status_bus", None)
        if bus is not None:
            bus.refresh_all()
