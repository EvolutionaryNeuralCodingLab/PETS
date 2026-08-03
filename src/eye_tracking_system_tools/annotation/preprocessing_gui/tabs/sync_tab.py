"""Stage 1 -- synchronization workflow (deterministic + DLC/jitter stages)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    SYNC_ARTIFACT_PROFILE,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.bokeh_launcher import (
    open_shift_plot,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.manual_ttl_dialog import (
    ManualTtlDialog,
    parse_open_ephys_with_manual_override,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.ttl_mapping import (
    effective_channeldict,
    list_mapping_sources,
    load_ttl_sidecar,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker import (
    extract_brightness_with_roi_fallback,
    jitter_report_needs_computation,
    pick_jitter_rois_for_block,
    show_eye_brightness_preview,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.pyqtgraph_helpers import (
    FinalSyncSanityPlot,
    JitterDriftPlot,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.batch_runner import (
    SequentialBatchWorker,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.workers import CallableWorker
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.dlc_csv_io import (
    default_dlc_csv,
    list_dlc_csvs,
)
from eye_tracking_system_tools.preprocessing.noise_epochs import (
    CATEGORY_LED_BLINK,
    append_epochs,
    apply_noise_epochs_to_block_csvs,
    list_categories,
    read_noise_epochs,
)
from eye_tracking_system_tools.preprocessing.block_sync_core import (
    drop_pandas_index_artifact_columns,
    load_eye_tracking_df_csv,
)
from eye_tracking_system_tools.preprocessing.notebook_helpers import (
    build_arena_grid_df,
    build_final_sync_df_merge_nearest,
    describe_eye_tick,
    export_eye_data_2d,
    export_final_sync_df,
    find_jittery_frames,
    insert_dup_by_oe_sample,
    insert_dup_by_pos,
    load_final_sync_df,
    shift_eye_df_by_index,
    simple_sync_build,
    verify_final_df_against_sources,
)


class SyncTab(BaseTab):
    tab_id = "sync"
    tab_label = "Sync"

    def __init__(self, state, config, parent=None):
        self._verify_result: dict | None = None
        self._worker: CallableWorker | None = None
        self._batch_worker: SequentialBatchWorker | None = None
        self._vid_inds_left: np.ndarray | None = None
        self._vid_inds_right: np.ndarray | None = None
        super().__init__(state, config, parent)

    def artifact_profile(self):
        return SYNC_ARTIFACT_PROFILE

    def build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(self._build_artifact_panel())
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(split, stretch=1)

        self._steps = QtWidgets.QListWidget()
        self._steps.addItems(
            [
                "1. Setup + Prepare",
                "2. Arena grid + simple sync",
                "3. Shift correction + insertion",
                "4. Final merge + verify + export",
                "5. DLC + jitter + finalize",
            ]
        )
        self._steps.setMaximumWidth(260)
        split.addWidget(self._steps)

        self._stack = QtWidgets.QStackedWidget()
        split.addWidget(self._stack)
        split.setStretchFactor(1, 1)

        self._stack.addWidget(self._build_prepare_panel())
        self._stack.addWidget(self._build_simple_sync_panel())
        self._stack.addWidget(self._build_shift_panel())
        self._stack.addWidget(self._build_verify_panel())
        self._stack.addWidget(self._build_dlc_jitter_panel())

        self._steps.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._steps.setCurrentRow(0)

        self._status_label = QtWidgets.QLabel("No block loaded.")
        root.addWidget(self._status_label)
        root.addWidget(self._build_batch_panel())

    def _build_batch_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Batch — all loaded blocks")
        layout = QtWidgets.QVBoxLayout(box)

        row = QtWidgets.QHBoxLayout()
        self._batch_op = QtWidgets.QComboBox()
        self._batch_op.addItems(
            [
                "Extract brightness",
                "Read DLC + fit ellipses",
                "Compute jitter report",
                "Correct jitter & catalog LED blinks",
            ]
        )
        self._btn_batch_run = QtWidgets.QPushButton("Run for all blocks")
        self._btn_batch_cancel = QtWidgets.QPushButton("Cancel batch")
        self._btn_batch_cancel.setEnabled(False)
        row.addWidget(self._batch_op, stretch=1)
        row.addWidget(self._btn_batch_run)
        row.addWidget(self._btn_batch_cancel)
        layout.addLayout(row)

        self._batch_log = QtWidgets.QPlainTextEdit()
        self._batch_log.setReadOnly(True)
        self._batch_log.setMaximumHeight(140)
        self._batch_log.setPlaceholderText("Batch progress appears here…")
        layout.addWidget(self._batch_log)

        self._btn_batch_run.clicked.connect(self._run_batch)
        self._btn_batch_cancel.clicked.connect(self._cancel_batch)
        return box

    def _build_prepare_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        self._block_info = QtWidgets.QLabel("No block")
        layout.addWidget(self._block_info)

        btns = QtWidgets.QHBoxLayout()
        self._btn_prepare = QtWidgets.QPushButton("Prepare data (eye + arena)")
        self._btn_parse_oe = QtWidgets.QPushButton("Parse OE events")
        self._btn_manual_ttl = QtWidgets.QPushButton("Manual TTL mapping…")
        self._btn_extract_brightness = QtWidgets.QPushButton("Extract brightness")
        self._btn_preview_brightness = QtWidgets.QPushButton("Preview eye brightness traces")
        self._btn_manual_roi = QtWidgets.QPushButton(
            "Re-run eye brightness with manual ROIs"
        )
        btns.addWidget(self._btn_prepare)
        btns.addWidget(self._btn_parse_oe)
        btns.addWidget(self._btn_manual_ttl)
        btns.addWidget(self._btn_extract_brightness)
        layout.addLayout(btns)

        btns2 = QtWidgets.QHBoxLayout()
        btns2.addWidget(self._btn_preview_brightness)
        btns2.addWidget(self._btn_manual_roi)
        btns2.addStretch(1)
        layout.addLayout(btns2)
        layout.addStretch(1)

        self._btn_prepare.clicked.connect(self._run_prepare_data)
        self._btn_parse_oe.clicked.connect(self._run_parse_oe)
        self._btn_manual_ttl.clicked.connect(self._run_manual_ttl_override)
        self._btn_extract_brightness.clicked.connect(self._run_extract_brightness)
        self._btn_preview_brightness.clicked.connect(self._run_preview_brightness)
        self._btn_manual_roi.clicked.connect(self._run_manual_roi_override)
        return w

    def _build_simple_sync_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        p = QtWidgets.QFormLayout()
        self._arena_target_fps = QtWidgets.QDoubleSpinBox()
        self._arena_target_fps.setRange(1.0, 240.0)
        self._arena_target_fps.setValue(float(self._config.arena_target_fps))
        self._arena_tol_hz = QtWidgets.QDoubleSpinBox()
        self._arena_tol_hz.setRange(0.1, 30.0)
        self._arena_tol_hz.setValue(float(self._config.arena_fps_tol_hz))
        p.addRow("Target fps:", self._arena_target_fps)
        p.addRow("Arena fps tol (Hz):", self._arena_tol_hz)
        layout.addLayout(p)

        btns = QtWidgets.QHBoxLayout()
        self._btn_build_arena_grid = QtWidgets.QPushButton("Build arena grid")
        self._btn_build_simple_sync = QtWidgets.QPushButton("Build simple sync")
        btns.addWidget(self._btn_build_arena_grid)
        btns.addWidget(self._btn_build_simple_sync)
        layout.addLayout(btns)

        self._simple_sync_summary = QtWidgets.QPlainTextEdit()
        self._simple_sync_summary.setReadOnly(True)
        self._simple_sync_summary.setMaximumHeight(130)
        layout.addWidget(self._simple_sync_summary)
        layout.addStretch(1)

        self._btn_build_arena_grid.clicked.connect(self._run_build_arena_grid)
        self._btn_build_simple_sync.clicked.connect(self._run_simple_sync_build)
        return w

    def _build_shift_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        info = QtWidgets.QLabel(
            "Use browser Bokeh slider plot to discover shifts. "
            "GUI shift values are the inverse of Bokeh slider values."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        row = QtWidgets.QHBoxLayout()
        self._btn_open_shift = QtWidgets.QPushButton("Open shift plot in browser")
        row.addWidget(self._btn_open_shift)
        layout.addLayout(row)

        form = QtWidgets.QFormLayout()
        self._left_shift = QtWidgets.QSpinBox()
        self._left_shift.setRange(-2000, 2000)
        self._right_shift = QtWidgets.QSpinBox()
        self._right_shift.setRange(-2000, 2000)
        self._left_shift_ms = QtWidgets.QLabel("~0 ms")
        self._right_shift_ms = QtWidgets.QLabel("~0 ms")
        left_wrap = QtWidgets.QWidget()
        lr = QtWidgets.QHBoxLayout(left_wrap)
        lr.setContentsMargins(0, 0, 0, 0)
        lr.addWidget(self._left_shift)
        lr.addWidget(self._left_shift_ms)
        right_wrap = QtWidgets.QWidget()
        rr = QtWidgets.QHBoxLayout(right_wrap)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.addWidget(self._right_shift)
        rr.addWidget(self._right_shift_ms)
        form.addRow("Left shift (ticks):", left_wrap)
        form.addRow("Right shift (ticks):", right_wrap)
        layout.addLayout(form)

        row2 = QtWidgets.QHBoxLayout()
        self._btn_apply_shifts = QtWidgets.QPushButton("Apply shifts")
        self._btn_preview_applied = QtWidgets.QPushButton("Preview applied in browser")
        row2.addWidget(self._btn_apply_shifts)
        row2.addWidget(self._btn_preview_applied)
        layout.addLayout(row2)

        self._advanced_box = QtWidgets.QGroupBox("Frame insertion (advanced)")
        self._advanced_box.setCheckable(True)
        self._advanced_box.setChecked(False)
        adv = QtWidgets.QFormLayout(self._advanced_box)
        self._insert_eye = QtWidgets.QComboBox()
        self._insert_eye.addItems(["left", "right"])
        self._insert_mode = QtWidgets.QComboBox()
        self._insert_mode.addItems(["pos", "oe_sample"])
        self._insert_dup = QtWidgets.QComboBox()
        self._insert_dup.addItems(["prev", "current"])
        self._insert_positions = QtWidgets.QLineEdit()
        self._insert_positions.setPlaceholderText("e.g. 100, 102, 104")
        self._btn_apply_insertions = QtWidgets.QPushButton("Apply insertions")
        adv.addRow("Eye:", self._insert_eye)
        adv.addRow("Position mode:", self._insert_mode)
        adv.addRow("Duplicate:", self._insert_dup)
        adv.addRow("Positions:", self._insert_positions)
        adv.addRow(self._btn_apply_insertions)
        layout.addWidget(self._advanced_box)
        layout.addStretch(1)

        self._btn_open_shift.clicked.connect(self._run_open_shift_plot)
        self._btn_apply_shifts.clicked.connect(self._run_apply_shifts)
        self._btn_preview_applied.clicked.connect(self._run_preview_applied)
        self._btn_apply_insertions.clicked.connect(self._run_apply_insertions)
        self._left_shift.valueChanged.connect(self._refresh_shift_ms_labels)
        self._right_shift.valueChanged.connect(self._refresh_shift_ms_labels)
        return w

    def _build_verify_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        form = QtWidgets.QFormLayout()
        self._final_tol_frac = QtWidgets.QDoubleSpinBox()
        self._final_tol_frac.setRange(0.1, 2.0)
        self._final_tol_frac.setSingleStep(0.05)
        self._final_tol_frac.setValue(float(self._config.final_sync_tol_frac))
        form.addRow("Final merge tol_frac:", self._final_tol_frac)
        layout.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        self._btn_build_final = QtWidgets.QPushButton("Build final sync df")
        self._btn_verify_final = QtWidgets.QPushButton("Verify final df")
        self._btn_export_final = QtWidgets.QPushButton("Export final_sync_df.csv")
        self._btn_export_final.setEnabled(False)
        row.addWidget(self._btn_build_final)
        row.addWidget(self._btn_verify_final)
        row.addWidget(self._btn_export_final)
        layout.addLayout(row)

        self._sanity_plot = FinalSyncSanityPlot()
        self._sanity_plot.setMinimumHeight(240)
        layout.addWidget(self._sanity_plot)

        self._verify_text = QtWidgets.QPlainTextEdit()
        self._verify_text.setReadOnly(True)
        self._verify_text.setMaximumHeight(160)
        layout.addWidget(self._verify_text)

        self._btn_build_final.clicked.connect(self._run_build_final_df)
        self._btn_verify_final.clicked.connect(self._run_verify_final_df)
        self._btn_export_final.clicked.connect(self._run_export_final_df)
        return w

    def _build_dlc_jitter_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        note = QtWidgets.QLabel(
            "Requires exported final_sync_df.csv. Long steps run in the background; "
            "the UI stays responsive. If analysis/pupil_perimeters.yaml exists "
            "(from Verify), Pupil keypoints outside those bounds are excluded before "
            "ellipse fitting. Correct jitter catalogs LED blinks into "
            "noise_epochs_{left,right}.csv (does not NaN eye data); use "
            "Apply selected noise… to NaN geometry only when you confirm."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        dlc_row = QtWidgets.QHBoxLayout()
        self._dlc_threshold = QtWidgets.QDoubleSpinBox()
        self._dlc_threshold.setRange(0.0, 1.0)
        self._dlc_threshold.setSingleStep(0.01)
        self._dlc_threshold.setDecimals(3)
        self._dlc_threshold.setValue(
            float(getattr(self._config, "dlc_threshold_to_use", 0.95))
        )
        self._btn_read_dlc = QtWidgets.QPushButton("Read DLC + fit ellipses")
        self._dlc_overwrite = QtWidgets.QCheckBox("Overwrite existing")
        self._dlc_overwrite.setChecked(False)
        self._dlc_overwrite.setToolTip(
            "When checked, recompute ellipses even if le_df.csv / re_df.csv already exist."
        )
        self._dlc_le_combo = QtWidgets.QComboBox()
        self._dlc_le_combo.setMinimumWidth(180)
        self._dlc_re_combo = QtWidgets.QComboBox()
        self._dlc_re_combo.setMinimumWidth(180)
        dlc_row.addWidget(QtWidgets.QLabel("DLC threshold:"))
        dlc_row.addWidget(self._dlc_threshold)
        self._btn_likelihood_hist = QtWidgets.QPushButton("Likelihood histogram…")
        self._btn_likelihood_hist.setToolTip(
            "Inspect likelihood distribution in the selected DLC CSVs and preview "
            "how much data a threshold would discard."
        )
        dlc_row.addWidget(self._btn_likelihood_hist)
        dlc_row.addWidget(QtWidgets.QLabel("LE DLC:"))
        dlc_row.addWidget(self._dlc_le_combo)
        dlc_row.addWidget(QtWidgets.QLabel("RE DLC:"))
        dlc_row.addWidget(self._dlc_re_combo)
        dlc_row.addWidget(self._dlc_overwrite)
        dlc_row.addWidget(self._btn_read_dlc)
        dlc_row.addStretch(1)
        layout.addLayout(dlc_row)

        jitter_row = QtWidgets.QHBoxLayout()
        self._btn_jitter_report = QtWidgets.QPushButton("Compute jitter report")
        self._jitter_overwrite = QtWidgets.QCheckBox("Overwrite existing report")
        self._jitter_overwrite.setChecked(False)
        self._jitter_overwrite.setToolTip(
            "When checked, recompute jitter even if analysis/jitter_report_dict.pkl exists "
            "(re-prompts for eye ROIs)."
        )
        self._btn_correct_jitter = QtWidgets.QPushButton(
            "Correct jitter & catalog LED blinks"
        )
        self._btn_correct_jitter.setToolTip(
            "Runs correct_jitter + find_led_blink_frames, then appends led_blink "
            "epochs. Does not NaN eye CSVs."
        )
        jitter_row.addWidget(self._btn_jitter_report)
        jitter_row.addWidget(self._jitter_overwrite)
        jitter_row.addWidget(self._btn_correct_jitter)
        layout.addLayout(jitter_row)

        noise_row = QtWidgets.QHBoxLayout()
        self._led_catalog_status = QtWidgets.QLabel(
            "LED blinks: (run Correct jitter to catalog)"
        )
        self._led_catalog_status.setWordWrap(True)
        self._btn_apply_noise = QtWidgets.QPushButton("Apply selected noise to eye data…")
        self._btn_apply_noise.setToolTip(
            "NaN geometry in le/re_df and/or left/right_eye_data for chosen "
            "noise-epoch categories (confirmed write)."
        )
        noise_row.addWidget(self._led_catalog_status, stretch=1)
        noise_row.addWidget(self._btn_apply_noise)
        layout.addLayout(noise_row)

        params = QtWidgets.QFormLayout()
        self._jitter_max_distance = QtWidgets.QSpinBox()
        self._jitter_max_distance.setRange(1, 500)
        self._jitter_max_distance.setValue(
            int(getattr(self._config, "jitter_max_distance", 60))
        )
        self._jitter_diff_threshold = QtWidgets.QSpinBox()
        self._jitter_diff_threshold.setRange(1, 100)
        self._jitter_diff_threshold.setValue(
            int(getattr(self._config, "jitter_diff_threshold", 5))
        )
        self._jitter_gap = QtWidgets.QSpinBox()
        self._jitter_gap.setRange(0, 200)
        self._jitter_gap.setValue(int(getattr(self._config, "jitter_gap_to_bridge", 24)))
        params.addRow("max_distance:", self._jitter_max_distance)
        params.addRow("diff_threshold:", self._jitter_diff_threshold)
        params.addRow("gap_to_bridge:", self._jitter_gap)
        layout.addLayout(params)

        preview_row = QtWidgets.QHBoxLayout()
        self._btn_preview_jitter = QtWidgets.QPushButton("Preview outliers (both eyes)")
        self._btn_apply_jitter = QtWidgets.QPushButton("Apply removal (both eyes)")
        self._btn_apply_jitter.setEnabled(False)
        preview_row.addWidget(self._btn_preview_jitter)
        preview_row.addWidget(self._btn_apply_jitter)
        layout.addLayout(preview_row)

        plots = QtWidgets.QHBoxLayout()
        left_box = QtWidgets.QGroupBox("Left eye drift")
        left_lay = QtWidgets.QVBoxLayout(left_box)
        self._jitter_plot_left = JitterDriftPlot()
        self._jitter_plot_left.setMinimumHeight(180)
        left_lay.addWidget(self._jitter_plot_left)
        right_box = QtWidgets.QGroupBox("Right eye drift")
        right_lay = QtWidgets.QVBoxLayout(right_box)
        self._jitter_plot_right = JitterDriftPlot()
        self._jitter_plot_right.setMinimumHeight(180)
        right_lay.addWidget(self._jitter_plot_right)
        plots.addWidget(left_box)
        plots.addWidget(right_box)
        layout.addLayout(plots)

        self._btn_finalize_eye = QtWidgets.QPushButton("Finalize & export eye data (left/right_eye_data.csv)")
        layout.addWidget(self._btn_finalize_eye)
        layout.addStretch(1)

        self._btn_read_dlc.clicked.connect(self._run_read_dlc)
        self._btn_likelihood_hist.clicked.connect(self._open_likelihood_histogram)
        self._btn_jitter_report.clicked.connect(self._run_jitter_report)
        self._btn_correct_jitter.clicked.connect(self._run_correct_jitter)
        self._btn_preview_jitter.clicked.connect(self._run_preview_jitter)
        self._btn_apply_jitter.clicked.connect(self._run_apply_jitter_removal)
        self._btn_apply_noise.clicked.connect(self._run_apply_selected_noise)
        self._btn_finalize_eye.clicked.connect(self._run_finalize_eye_data)
        return w

    def set_block(self, block: BlockHandle | None) -> None:
        self._verify_result = None
        self._vid_inds_left = None
        self._vid_inds_right = None
        self._btn_export_final.setEnabled(False)
        self._btn_apply_jitter.setEnabled(False)
        if block is None:
            self._block_info.setText("No block loaded")
            self._status("No block loaded.")
            self._refresh_dlc_combos(None)
            self._refresh_led_catalog_status(None)
        else:
            self._block_info.setText(f"Active block: {block.display_label}\n{block.block_path}")
            self._status("Ready.")
            self._refresh_dlc_combos(block)
            self._refresh_led_catalog_status(block)
        super().set_block(block)

    def status_signature(self, block: BlockHandle) -> list[Path]:
        ap = block.analysis_path
        return [ap / "final_sync_df.csv", ap / "left_eye_data.csv", ap / "right_eye_data.csv"]

    def _status(self, text: str) -> None:
        self._status_label.setText(text)

    def _populate_dlc_combo(self, combo: QtWidgets.QComboBox, eye_path: Path) -> bool:
        combo.clear()
        candidates = list_dlc_csvs(eye_path)
        if not candidates:
            combo.addItem("(no DLC csv)", None)
            combo.setEnabled(False)
            return False
        for path in candidates:
            combo.addItem(path.name, str(path))
        default_path = default_dlc_csv(candidates)
        default_index = combo.findData(str(default_path))
        if default_index >= 0:
            combo.setCurrentIndex(default_index)
        combo.setEnabled(True)
        return True

    def _refresh_dlc_combos(self, block: BlockHandle | None) -> None:
        if block is None:
            self._dlc_le_combo.clear()
            self._dlc_re_combo.clear()
            self._dlc_le_combo.setEnabled(False)
            self._dlc_re_combo.setEnabled(False)
            self._btn_read_dlc.setEnabled(False)
            return
        b = self._session.get(block)
        le_ok = self._populate_dlc_combo(self._dlc_le_combo, Path(b.l_e_path))
        re_ok = self._populate_dlc_combo(self._dlc_re_combo, Path(b.r_e_path))
        self._btn_read_dlc.setEnabled(le_ok and re_ok)

    @staticmethod
    def _format_dlc_fit_report(report: dict) -> str:
        lines = ["DLC ellipse fit summary:"]
        for eye_key, label in (("left", "Left eye"), ("right", "Right eye")):
            stats = report.get(eye_key, {})
            dlc_name = Path(stats.get("dlc_csv", "")).name or "?"
            peri_n = stats.get("n_keypoints_masked_by_perimeter", 0)
            peri_bit = (
                f", perimeter_masked={peri_n}"
                if peri_n
                else ""
            )
            lines.append(
                f"{label} ({dlc_name}): "
                f"yield {stats.get('yield_pct', float('nan')):.1f}% "
                f"({stats.get('n_fitted', 0)}/{stats.get('n_frames', 0)} frames), "
                f"mean likelihood all={stats.get('mean_likelihood_all', float('nan')):.3f}, "
                f"used={stats.get('mean_likelihood_used', float('nan')):.3f}, "
                f"fit_failed={stats.get('n_fit_failed', 0)}{peri_bit}"
            )
        return "\n".join(lines)

    def _show_dlc_fit_report(self, report: dict | None) -> None:
        if not report:
            return
        summary = self._format_dlc_fit_report(report)
        self._status(summary.splitlines()[0])
        QtWidgets.QMessageBox.information(self, "DLC ellipse fit", summary)

    @staticmethod
    def _sanitize_blocksync_artifacts(blocksync: BlockSync) -> None:
        from eye_tracking_system_tools.annotation.preprocessing_gui.block_session import (
            BlockSyncSession,
        )

        BlockSyncSession.sanitize(blocksync)

    def _ensure_eye_tracking_dfs(self, blocksync: BlockSync) -> None:
        """DLC/jitter steps need le_df/re_df on the BlockSync object, not only on disk."""
        self._require_final_sync_on_disk(blocksync)
        if (
            getattr(blocksync, "le_df", None) is not None
            and getattr(blocksync, "re_df", None) is not None
            and "center_x" in blocksync.le_df.columns
            and "Arena_TTL" in blocksync.le_df.columns
        ):
            return
        ap = Path(blocksync.analysis_path)
        le_path = ap / "le_df.csv"
        re_path = ap / "re_df.csv"
        if le_path.exists() and re_path.exists():
            blocksync.le_df = load_eye_tracking_df_csv(le_path)
            blocksync.re_df = load_eye_tracking_df_csv(re_path)
            return
        raise RuntimeError(
            "le_df.csv / re_df.csv are missing. Run 'Read DLC + fit ellipses' in step 5 first."
        )

    @staticmethod
    def _ensure_jitter_dicts(blocksync: BlockSync) -> None:
        if getattr(blocksync, "le_jitter_dict", None) is None or getattr(
            blocksync, "re_jitter_dict", None
        ) is None:
            pkl = Path(blocksync.analysis_path) / "jitter_report_dict.pkl"
            if pkl.exists():
                blocksync.get_jitter_reports(
                    export=False,
                    overwrite=False,
                    remove_led_blinks=False,
                    sort_on_loading=True,
                )
        for eye, jd in (("left", getattr(blocksync, "le_jitter_dict", None)), (
            "right",
            getattr(blocksync, "re_jitter_dict", None),
        )):
            if jd is None:
                raise RuntimeError(
                    "Jitter report is not loaded. Click 'Compute jitter report' first."
                )
            if "x_displacement" not in jd or "top_correlation_dist" not in jd:
                raise RuntimeError(
                    f"The saved jitter report for the {eye} eye looks incomplete. "
                    "Delete analysis/jitter_report_dict.pkl and recompute."
                )

    def _run_guarded(self, fn, ok_msg: str) -> None:
        try:
            fn()
        except Exception as e:
            self._status(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Sync tab", str(e))
            return
        self._status(ok_msg)

    def _run_prepare_data(self) -> None:
        def _do():
            b = self._require_blocksync()
            b.handle_eye_videos()
            b.handle_arena_files()
        self._run_guarded(_do, "Prepared eye/arena video metadata.")

    def _run_parse_oe(self) -> None:
        def _do():
            b = self._require_blocksync()
            sidecar = load_ttl_sidecar(b.block_path, b.oe_dirname)
            if sidecar is not None:
                parse_open_ephys_with_manual_override(
                    b,
                    sidecar["manual_line_map"],
                    sidecar["arena_window"],
                    overwrite=True,
                )
            else:
                try:
                    b.parse_open_ephys_events(overwrite=False, interactive_on_fail=False)
                except Exception as auto_err:
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Parse OE events",
                        f"Automatic parse failed:\n{auto_err}\n\n"
                        "Open manual TTL mapping dialog?",
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                        raise auto_err
                    if not self._open_manual_ttl_dialog(b, overwrite=True):
                        raise RuntimeError("Manual TTL mapping cancelled.")
            self._sanitize_blocksync_artifacts(b)
            self._session.remember_channeldict(effective_channeldict(b))

        self._run_guarded(_do, "Parsed Open Ephys events.")

    def _events_csv_path(self, blocksync: BlockSync) -> Path:
        return (
            blocksync.block_path
            / "oe_files"
            / blocksync.oe_dirname
            / "events.csv"
        )

    def _open_manual_ttl_dialog(
        self, blocksync: BlockSync, *, overwrite: bool
    ) -> bool:
        events_csv = self._events_csv_path(blocksync)
        if not events_csv.is_file():
            blocksync.oe_events_to_csv(align_to_zero=True)
        mapping_sources: list[tuple[str, dict[str, int]]] = []
        if self._block is not None:
            mapping_sources = list_mapping_sources(
                self._session,
                self._state.blocks,
                self._block,
            )
        dlg = ManualTtlDialog(
            blocksync,
            events_csv,
            mapping_sources=mapping_sources,
            parent=self,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        manual_line_map, arena_window = dlg.payload()
        if manual_line_map is None or arena_window is None:
            return False
        parse_open_ephys_with_manual_override(
            blocksync,
            manual_line_map,
            arena_window,
            overwrite=overwrite,
        )
        self._session.remember_channeldict(effective_channeldict(blocksync))
        return True

    def _run_manual_ttl_override(self) -> None:
        def _do():
            b = self._require_blocksync()
            if not self._open_manual_ttl_dialog(b, overwrite=True):
                raise RuntimeError("Manual TTL mapping cancelled.")
            self._sanitize_blocksync_artifacts(b)

        self._run_guarded(_do, "Applied manual TTL mapping and re-parsed events.")

    def _run_extract_brightness(self) -> None:
        def _do():
            b = self._require_blocksync()
            pkl = Path(b.analysis_path) / "eye_brightness_values_dict.pkl"
            if pkl.is_file():
                b.get_eye_brightness_vectors(use_auto_roi=True, create_if_missing=False)
                return
            extract_brightness_with_roi_fallback(
                b,
                use_auto_roi=True,
                parent=self,
            )

        self._run_guarded(_do, "Extracted brightness vectors.")

    def _run_preview_brightness(self) -> None:
        try:
            b = self._require_blocksync()
            show_eye_brightness_preview(b, parent=self)
            self._status("Displayed eye brightness preview.")
        except Exception as e:
            self._status(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Eye brightness preview", str(e))

    def _run_manual_roi_override(self) -> None:
        b = self._require_blocksync()
        pkl = Path(b.analysis_path) / "eye_brightness_values_dict.pkl"
        if pkl.is_file():
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Overwrite eye brightness file?",
                "This will re-run eye brightness generation with manual ROIs and "
                f"overwrite:\n{pkl}\n\n"
                "Only continue if the current ROI/traces look wrong.",
                QtWidgets.QMessageBox.StandardButton.Ok
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Ok:
                self._status("Manual ROI re-run cancelled.")
                return

        def _do():
            extract_brightness_with_roi_fallback(
                b,
                use_auto_roi=False,
                force=True,
                parent=self,
            )

        self._run_guarded(
            _do,
            "Re-ran eye brightness generation with manual ROIs.",
        )

    def _run_build_arena_grid(self) -> None:
        def _do():
            b = self._require_blocksync()
            self._sanitize_blocksync_artifacts(b)
            grid, info = build_arena_grid_df(
                b,
                target_fps=float(self._arena_target_fps.value()),
                arena_fps_tol_hz=float(self._arena_tol_hz.value()),
            )
            self._state.arena_grid_df = grid
            self._simple_sync_summary.setPlainText(
                f"Arena grid rows: {len(grid)}\n"
                f"Inferred fps: {info.inferred_arena_fps:.4f}"
            )
        self._run_guarded(_do, "Built arena grid.")

    def _run_simple_sync_build(self) -> None:
        def _do():
            b = self._require_blocksync()
            self._sanitize_blocksync_artifacts(b)
            df_l, df_r = simple_sync_build(b, export=True)
            self._state.df_left_simple_sync = df_l
            self._state.df_right_simple_sync = df_r
            left_tick = describe_eye_tick(df_l)
            right_tick = describe_eye_tick(df_r)
            self._simple_sync_summary.setPlainText(
                f"Left rows: {len(df_l)} | tick_ms~ {left_tick:.4f}\n"
                f"Right rows: {len(df_r)} | tick_ms~ {right_tick:.4f}\n"
                f"Left fps_est~ {1000.0 / left_tick:.4f}\n"
                f"Right fps_est~ {1000.0 / right_tick:.4f}"
            )
            self._refresh_shift_ms_labels()
        self._run_guarded(_do, "Built simple sync dataframes.")

    def _require_simple_sync(self) -> tuple:
        if self._state.df_left_simple_sync is None or self._state.df_right_simple_sync is None:
            raise RuntimeError("Build simple sync first.")
        return self._state.df_left_simple_sync, self._state.df_right_simple_sync

    def _run_open_shift_plot(self) -> None:
        def _do():
            b = self._require_blocksync()
            df_l, df_r = self._require_simple_sync()
            open_shift_plot(b, df_l, df_r, show_led=True)
        self._run_guarded(_do, "Opened shift plot in browser.")

    def _refresh_shift_ms_labels(self) -> None:
        if self._state.df_left_simple_sync is None or self._state.df_right_simple_sync is None:
            self._left_shift_ms.setText("~? ms")
            self._right_shift_ms.setText("~? ms")
            return
        l_tick = describe_eye_tick(self._state.df_left_simple_sync)
        r_tick = describe_eye_tick(self._state.df_right_simple_sync)
        self._left_shift_ms.setText(f"~{self._left_shift.value() * l_tick:.3f} ms")
        self._right_shift_ms.setText(f"~{self._right_shift.value() * r_tick:.3f} ms")

    def _run_apply_shifts(self) -> None:
        def _do():
            df_l, df_r = self._require_simple_sync()
            # Per notebook convention: GUI input is inverse to plot slider values.
            self._state.df_left_simple_sync = shift_eye_df_by_index(df_l, -int(self._left_shift.value()))
            self._state.df_right_simple_sync = shift_eye_df_by_index(df_r, -int(self._right_shift.value()))
            self._refresh_shift_ms_labels()
        self._run_guarded(_do, "Applied index shifts to simple-sync dataframes.")

    def _run_preview_applied(self) -> None:
        def _do():
            b = self._require_blocksync()
            df_l, df_r = self._require_simple_sync()
            open_shift_plot(b, df_l, df_r, show_led=True)
        self._run_guarded(_do, "Opened preview for shifted traces.")

    def _run_apply_insertions(self) -> None:
        def _do():
            df_l, df_r = self._require_simple_sync()
            raw = self._insert_positions.text().strip()
            if not raw:
                raise RuntimeError("Provide insertion positions.")
            positions = [int(x.strip()) for x in raw.split(",") if x.strip()]
            eye = self._insert_eye.currentText()
            mode = self._insert_mode.currentText()
            dup = self._insert_dup.currentText()
            if eye == "left":
                if mode == "pos":
                    self._state.df_left_simple_sync = insert_dup_by_pos(df_l, positions, duplicate=dup)
                else:
                    self._state.df_left_simple_sync = insert_dup_by_oe_sample(df_l, positions, duplicate=dup)
            else:
                if mode == "pos":
                    self._state.df_right_simple_sync = insert_dup_by_pos(df_r, positions, duplicate=dup)
                else:
                    self._state.df_right_simple_sync = insert_dup_by_oe_sample(df_r, positions, duplicate=dup)
        self._run_guarded(_do, "Applied manual frame insertions.")

    def _run_build_final_df(self) -> None:
        def _do():
            b = self._require_blocksync()
            df_l, df_r = self._require_simple_sync()
            final_df = build_final_sync_df_merge_nearest(
                b,
                df_l,
                df_r,
                target_fps=float(self._arena_target_fps.value()),
                tol_frac=float(self._final_tol_frac.value()),
                export_csv=False,
            )
            self._state.final_sync_df = final_df
            self._sanity_plot.set_data(
                final_df,
                fs_hz=float(b.sample_rate),
                led_samples=self._led_samples_from_blocksync(b),
            )
            self._btn_export_final.setEnabled(False)
            self._verify_result = None
            self._verify_text.clear()
        self._run_guarded(_do, "Built final sync dataframe (in memory).")

    def _run_verify_final_df(self) -> None:
        def _do():
            b = self._require_blocksync()
            df_l, df_r = self._require_simple_sync()
            if self._state.final_sync_df is None:
                raise RuntimeError("Build final sync df first.")
            res = verify_final_df_against_sources(
                b,
                self._state.final_sync_df,
                df_l,
                df_r,
                target_fps=float(self._arena_target_fps.value()),
                tol_frac=float(self._final_tol_frac.value()),
            )
            self._verify_result = res
            lines = [f"{k}: {v}" for k, v in res.items()]
            self._verify_text.setPlainText("\n".join(lines))
            self._btn_export_final.setEnabled(True)
            self._sanity_plot.set_data(
                self._state.final_sync_df,
                fs_hz=float(b.sample_rate),
                led_samples=self._led_samples_from_blocksync(b),
            )
        self._run_guarded(_do, "Verified final sync mapping against sources.")

    def _run_export_final_df(self) -> None:
        def _do():
            b = self._require_blocksync()
            if self._state.final_sync_df is None:
                raise RuntimeError("Build final sync df first.")
            export_final_sync_df(b, self._state.final_sync_df, overwrite=True)
        self._run_guarded(_do, "Exported final sync CSV files.")

    def _phase2_action_buttons(self) -> list[QtWidgets.QPushButton]:
        return [
            self._btn_read_dlc,
            self._btn_jitter_report,
            self._btn_correct_jitter,
            self._btn_preview_jitter,
            self._btn_apply_jitter,
            self._btn_finalize_eye,
        ]

    def _sync_action_buttons(self) -> list[QtWidgets.QPushButton]:
        return [
            self._btn_prepare,
            self._btn_parse_oe,
            self._btn_manual_ttl,
            self._btn_extract_brightness,
            self._btn_preview_brightness,
            self._btn_manual_roi,
            self._btn_build_arena_grid,
            self._btn_build_simple_sync,
            self._btn_open_shift,
            self._btn_apply_shifts,
            self._btn_preview_applied,
            self._btn_apply_insertions,
            self._btn_build_final,
            self._btn_verify_final,
            self._btn_export_final,
            *self._phase2_action_buttons(),
        ]

    def _set_sync_actions_busy(self, busy: bool, label: str = "") -> None:
        for btn in self._sync_action_buttons():
            btn.setEnabled(not busy)
        if busy and label:
            self._status(label)

    def _set_phase2_busy(self, busy: bool, label: str = "") -> None:
        self._set_sync_actions_busy(busy, label)

    def _require_final_sync_on_disk(self, blocksync: BlockSync) -> None:
        path = Path(blocksync.analysis_path) / "final_sync_df.csv"
        if not path.exists():
            raise RuntimeError(
                "final_sync_df.csv is missing. Complete step 4 and export first."
            )
        load_final_sync_df(blocksync, verbose=False)

    def _run_async(self, work_fn, ok_msg: str, busy_label: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Sync tab", "A background job is already running."
            )
            return

        self._set_phase2_busy(True, busy_label)
        worker = CallableWorker(work_fn, self)

        def on_ok(_result=None) -> None:
            self._set_phase2_busy(False)
            self._status(ok_msg)
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str) -> None:
            self._set_phase2_busy(False)
            self._status(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Sync tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _run_read_dlc(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Sync tab", "A background job is already running."
            )
            return

        le_path = self._dlc_le_combo.currentData()
        re_path = self._dlc_re_combo.currentData()
        if not le_path or not re_path:
            QtWidgets.QMessageBox.warning(
                self, "Sync tab", "Select DLC CSV files for both eyes."
            )
            return

        overwrite = self._dlc_overwrite.isChecked()

        def work():
            b = self._require_blocksync()
            self._require_final_sync_on_disk(b)
            b.read_dlc_data(
                threshold_to_use=float(self._dlc_threshold.value()),
                export=True,
                overwrite=overwrite,
                le_dlc_path=le_path,
                re_dlc_path=re_path,
            )
            return b.dlc_ellipse_fit_report

        self._set_phase2_busy(True, "Running DLC + ellipse fitting…")
        worker = CallableWorker(work, self)

        def on_ok(report) -> None:
            self._set_phase2_busy(False)
            if report is None:
                self._status(
                    "Loaded existing le_df/re_df (overwrite unchecked). "
                    "Check 'Overwrite existing' to recompute."
                )
            else:
                peri_bits = []
                for side in ("left", "right"):
                    side_rep = (report or {}).get(side) or {}
                    n_mask = side_rep.get("n_keypoints_masked_by_perimeter")
                    if n_mask:
                        peri_bits.append(f"{side[0].upper()}={n_mask} pts")
                peri_note = (
                    f" Perimeter mask: {', '.join(peri_bits)}."
                    if peri_bits
                    else ""
                )
                self._status(f"Read DLC and fitted ellipses.{peri_note}")
                self._show_dlc_fit_report(report)
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str) -> None:
            self._set_phase2_busy(False)
            self._status(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Sync tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _run_jitter_report(self) -> None:
        try:
            b = self._require_blocksync()
            self._require_final_sync_on_disk(b)
            overwrite = self._jitter_overwrite.isChecked()
            roi_dict = None
            if jitter_report_needs_computation(b, overwrite=overwrite):
                roi_dict = pick_jitter_rois_for_block(b, parent=self)
                if roi_dict is None:
                    self._status("Jitter ROI selection cancelled.")
                    return
        except Exception as e:
            self._status(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Sync tab", str(e))
            return

        captured_roi = roi_dict
        captured_overwrite = self._jitter_overwrite.isChecked()

        def work():
            blk = self._require_blocksync()
            self._require_final_sync_on_disk(blk)
            blk.get_jitter_reports(
                export=True,
                overwrite=captured_overwrite,
                remove_led_blinks=False,
                sort_on_loading=True,
                roi_dict=captured_roi,
            )

        msg = (
            "Recomputed jitter reports (overwrite)."
            if captured_overwrite
            else "Computed jitter reports."
        )
        self._run_async(work, msg, "Computing jitter report…")

    def _open_likelihood_histogram(self) -> None:
        le_path = self._dlc_le_combo.currentData()
        re_path = self._dlc_re_combo.currentData()
        paths = [p for p in (le_path, re_path) if p]
        if not paths:
            QtWidgets.QMessageBox.warning(
                self, "Sync tab", "Select at least one DLC CSV first."
            )
            return
        from eye_tracking_system_tools.annotation.preprocessing_gui.likelihood_threshold_dialog import (
            LikelihoodThresholdDialog,
        )

        dlg = LikelihoodThresholdDialog(
            paths,
            initial_threshold=float(self._dlc_threshold.value()),
            parent=self,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._dlc_threshold.setValue(float(dlg.threshold()))
            self._status(
                f"DLC threshold set to {self._dlc_threshold.value():.3f} "
                f"from likelihood histogram."
            )

    def _refresh_led_catalog_status(self, block: BlockHandle | None) -> None:
        if not hasattr(self, "_led_catalog_status"):
            return
        if block is None:
            self._led_catalog_status.setText("LED blinks: (no block)")
            return
        parts: list[str] = []
        for eye, label in (("left", "L"), ("right", "R")):
            ep = read_noise_epochs(block.block_path, eye)
            led = ep[ep["category"] == CATEGORY_LED_BLINK] if not ep.empty else ep
            if led is None or led.empty:
                parts.append(f"{label}=0 frames / 0 epochs")
            else:
                n_frames = int(
                    (led["end_frame"].astype(int) - led["start_frame"].astype(int) + 1).sum()
                )
                parts.append(f"{label}={n_frames} frames / {len(led)} epochs")
        self._led_catalog_status.setText(
            "LED blinks: " + ", ".join(parts) + " (catalogued, not applied)"
        )

    @staticmethod
    def _catalog_led_blinks(blocksync: BlockSync) -> dict[str, tuple[int, int]]:
        """Append led_blink epochs for both eyes; return {eye: (n_frames, n_epochs)}."""
        out: dict[str, tuple[int, int]] = {}
        block_path = Path(blocksync.block_path)
        for eye, attr in (("left", "led_blink_frames_l"), ("right", "led_blink_frames_r")):
            frames = getattr(blocksync, attr, None)
            if frames is None:
                frames = []
            _, n_frames, n_epochs = append_epochs(
                block_path,
                eye,
                category=CATEGORY_LED_BLINK,
                frames=frames,
                replace_category=True,
            )
            out[eye] = (n_frames, n_epochs)
        return out

    def _run_correct_jitter(self) -> None:
        def work():
            b = self._require_blocksync()
            self._ensure_jitter_dicts(b)
            self._ensure_eye_tracking_dfs(b)
            b.correct_jitter()
            b.find_led_blink_frames(plot=False)
            return self._catalog_led_blinks(b)

        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Sync tab", "A background job is already running."
            )
            return

        self._set_phase2_busy(True, "Correcting jitter and cataloguing LED blinks…")
        worker = CallableWorker(work, self)

        def on_ok(result=None) -> None:
            self._set_phase2_busy(False)
            msg = "Corrected jitter; LED blinks catalogued (not applied to eye data)."
            if isinstance(result, dict):
                l_n, l_e = result.get("left", (0, 0))
                r_n, r_e = result.get("right", (0, 0))
                msg = (
                    f"Corrected jitter. LED blinks catalogued: "
                    f"L={l_n} frames / {l_e} epochs, R={r_n} frames / {r_e} epochs "
                    "(not applied)."
                )
            self._status(msg)
            if self._block is not None:
                self._refresh_led_catalog_status(self._block)
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str) -> None:
            self._set_phase2_busy(False)
            self._status(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Sync tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _run_apply_selected_noise(self) -> None:
        if self._block is None:
            QtWidgets.QMessageBox.information(self, "Sync tab", "Load a block first.")
            return
        cats = list_categories(self._block.block_path)
        if not cats:
            QtWidgets.QMessageBox.information(
                self,
                "Apply noise",
                "No noise-epoch categories found. Commit perimeter bad points "
                "or run Correct jitter & catalog LED blinks first.",
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Apply selected noise to eye data")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(
            QtWidgets.QLabel(
                "NaN ellipse geometry for frames covered by the selected categories.\n"
                "This writes le_df/re_df and left/right_eye_data.csv when present."
            )
        )
        checks: dict[str, QtWidgets.QCheckBox] = {}
        for cat in cats:
            cb = QtWidgets.QCheckBox(cat)
            checks[cat] = cb
            lay.addWidget(cb)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        selected = [c for c, cb in checks.items() if cb.isChecked()]
        if not selected:
            QtWidgets.QMessageBox.information(
                self, "Apply noise", "No categories selected."
            )
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm apply",
            f"NaN geometry for categories: {', '.join(selected)}?\n"
            "This modifies on-disk eye CSVs.",
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            report = apply_noise_epochs_to_block_csvs(
                self._block.block_path, categories=selected
            )
            # Refresh in-memory dfs if loaded
            try:
                b = self._require_blocksync()
                le = Path(b.analysis_path) / "le_df.csv"
                re = Path(b.analysis_path) / "re_df.csv"
                if le.is_file():
                    b.le_df = load_eye_tracking_df_csv(le)
                if re.is_file():
                    b.re_df = load_eye_tracking_df_csv(re)
            except Exception:
                pass
            bits = []
            for eye, info in report.get("eyes", {}).items():
                files = info.get("files") or {}
                if files:
                    bits.append(
                        f"{eye}: "
                        + ", ".join(f"{k}={v}" for k, v in files.items())
                    )
            detail = "; ".join(bits) if bits else "no matching rows / files"
            self._status(f"Applied noise categories {selected}: {detail}")
            QtWidgets.QMessageBox.information(
                self,
                "Noise applied",
                f"Categories: {', '.join(selected)}\n{detail}",
            )
        except Exception as e:
            self._status(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Sync tab", str(e))

    def _run_preview_jitter(self) -> None:
        try:
            b = self._require_blocksync()
            self._ensure_jitter_dicts(b)
            self._ensure_eye_tracking_dfs(b)
            md = int(self._jitter_max_distance.value())
            dt = int(self._jitter_diff_threshold.value())
            gap = int(self._jitter_gap.value())
            _, vid_l = find_jittery_frames(b, "left", md, dt, gap_to_bridge=gap)
            _, vid_r = find_jittery_frames(b, "right", md, dt, gap_to_bridge=gap)
            self._vid_inds_left = np.asarray(vid_l, dtype=int)
            self._vid_inds_right = np.asarray(vid_r, dtype=int)
            ldf = pd.DataFrame.from_dict(b.le_jitter_dict)
            rdf = pd.DataFrame.from_dict(b.re_jitter_dict)
            self._jitter_plot_left.set_drift(
                ldf["top_correlation_dist"].to_numpy(),
                self._vid_inds_left,
            )
            self._jitter_plot_right.set_drift(
                rdf["top_correlation_dist"].to_numpy(),
                self._vid_inds_right,
            )
            self._btn_apply_jitter.setEnabled(True)
            self._status(
                f"Preview ready: left peaks={len(self._vid_inds_left)}, "
                f"right peaks={len(self._vid_inds_right)}."
            )
        except Exception as e:
            self._status(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Sync tab", str(e))

    def _run_apply_jitter_removal(self) -> None:
        if self._vid_inds_left is None or self._vid_inds_right is None:
            QtWidgets.QMessageBox.warning(
                self, "Sync tab", "Preview outliers before applying removal."
            )
            return

        def work():
            b = self._require_blocksync()
            self._ensure_eye_tracking_dfs(b)
            b.remove_eye_datapoints_based_on_video_frames(
                "left", indices_to_nan=self._vid_inds_left
            )
            b.remove_eye_datapoints_based_on_video_frames(
                "right", indices_to_nan=self._vid_inds_right
            )

        self._run_async(work, "Applied jitter outlier removal for both eyes.", "Applying outlier removal…")

    def _run_finalize_eye_data(self) -> None:
        def work():
            b = self._require_blocksync()
            self._ensure_eye_tracking_dfs(b)
            b.create_eye_data()
            export_eye_data_2d(b)

        self._run_async(
            work,
            "Created and exported left/right_eye_data.csv.",
            "Finalizing eye data export…",
        )

    def _append_batch_log(self, line: str) -> None:
        self._batch_log.appendPlainText(line)

    def _batch_blocks(self) -> list[BlockHandle]:
        blocks = list(self._state.blocks)
        if len(blocks) < 2:
            raise RuntimeError(
                "Load at least two blocks at startup (comma-separated block list) "
                "to use batch mode."
            )
        return blocks

    def _run_batch(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self, "Sync tab", "Wait for the current background job to finish."
            )
            return
        if self._batch_worker is not None and self._batch_worker.isRunning():
            return
        try:
            blocks = self._batch_blocks()
        except RuntimeError as e:
            QtWidgets.QMessageBox.warning(self, "Sync tab", str(e))
            return

        op = self._batch_op.currentText()
        self._batch_log.clear()
        self._append_batch_log(f"Starting batch: {op} ({len(blocks)} blocks)")
        self._btn_batch_run.setEnabled(False)
        self._btn_batch_cancel.setEnabled(True)
        self._set_sync_actions_busy(True)

        if op == "Extract brightness":
            run_one = self._batch_extract_brightness
        elif op == "Read DLC + fit ellipses":
            run_one = self._batch_read_dlc
        elif op == "Compute jitter report":
            run_one = self._batch_jitter_report
        else:
            run_one = self._batch_correct_jitter

        self._batch_worker = SequentialBatchWorker(blocks, run_one, self)
        self._batch_worker.progress.connect(self._append_batch_log)
        self._batch_worker.block_done.connect(self._on_batch_block_done)
        self._batch_worker.finished_all.connect(self._on_batch_finished)
        self._batch_worker.cancelled.connect(self._on_batch_cancelled)
        self._batch_worker.start()

    def _cancel_batch(self) -> None:
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._append_batch_log("Cancel requested…")
            self._batch_worker.request_cancel()

    def _on_batch_block_done(self, label: str, ok: bool, detail: str) -> None:
        prefix = "OK" if ok else "FAIL"
        self._append_batch_log(f"  {prefix} {label}: {detail}")

    def _on_batch_finished(self) -> None:
        self._append_batch_log("Batch finished.")
        self._finish_batch_ui()

    def _on_batch_cancelled(self) -> None:
        self._append_batch_log("Batch cancelled.")
        self._finish_batch_ui()

    def _finish_batch_ui(self) -> None:
        self._btn_batch_run.setEnabled(True)
        self._btn_batch_cancel.setEnabled(False)
        self._set_sync_actions_busy(False)
        if self._batch_worker is not None:
            self._batch_worker.deleteLater()
            self._batch_worker = None
        self._status("Batch complete.")

    def _batch_extract_brightness(self, handle: BlockHandle) -> str:
        b = self._session.get(handle)
        pkl = Path(b.analysis_path) / "eye_brightness_values_dict.pkl"
        if pkl.is_file():
            b.get_eye_brightness_vectors(use_auto_roi=True, create_if_missing=False)
        else:
            extract_brightness_with_roi_fallback(
                b,
                use_auto_roi=True,
                parent=self,
            )
        return "brightness vectors ready"

    def _batch_read_dlc(self, handle: BlockHandle) -> str:
        b = self._session.get(handle)
        path = Path(b.analysis_path) / "final_sync_df.csv"
        if not path.exists():
            raise RuntimeError("final_sync_df.csv missing — run sync through step 4 first")
        load_final_sync_df(b, verbose=False)
        b.read_dlc_data(
            threshold_to_use=float(self._dlc_threshold.value()),
            export=True,
            overwrite=self._dlc_overwrite.isChecked(),
        )
        return "le_df.csv / re_df.csv written"

    def _batch_jitter_report(self, handle: BlockHandle) -> str:
        b = self._session.get(handle)
        path = Path(b.analysis_path) / "final_sync_df.csv"
        if not path.exists():
            raise RuntimeError("final_sync_df.csv missing")
        load_final_sync_df(b, verbose=False)
        overwrite = self._jitter_overwrite.isChecked()
        roi_dict = None
        if jitter_report_needs_computation(b, overwrite=overwrite):
            roi_dict = pick_jitter_rois_for_block(b, parent=self)
            if roi_dict is None:
                raise RuntimeError("Jitter ROI selection cancelled.")
        b.get_jitter_reports(
            export=True,
            overwrite=overwrite,
            remove_led_blinks=False,
            sort_on_loading=True,
            roi_dict=roi_dict,
        )
        return "jitter_report_dict.pkl written"

    def _batch_correct_jitter(self, handle: BlockHandle) -> str:
        b = self._session.get(handle)
        load_final_sync_df(b, verbose=False)
        le_path = Path(b.analysis_path) / "le_df.csv"
        re_path = Path(b.analysis_path) / "re_df.csv"
        if not le_path.exists() or not re_path.exists():
            raise RuntimeError("le_df/re_df missing — run Read DLC first")
        b.le_df = load_eye_tracking_df_csv(le_path)
        b.re_df = load_eye_tracking_df_csv(re_path)
        b.get_jitter_reports(
            export=False,
            overwrite=False,
            remove_led_blinks=False,
            sort_on_loading=True,
        )
        if b.le_jitter_dict is None or b.re_jitter_dict is None:
            raise RuntimeError("jitter report missing — run Compute jitter report first")
        b.correct_jitter()
        b.find_led_blink_frames(plot=False)
        catalog = SyncTab._catalog_led_blinks(b)
        l_n, l_e = catalog.get("left", (0, 0))
        r_n, r_e = catalog.get("right", (0, 0))
        return (
            f"jitter corrected; LED blinks catalogued "
            f"L={l_n}/{l_e}, R={r_n}/{r_e} (not applied)"
        )

    @staticmethod
    def _led_samples_from_blocksync(blocksync: BlockSync):
        oe = getattr(blocksync, "oe_events", None)
        if oe is None or "LED_driver" not in oe.columns:
            return None
        return oe["LED_driver"].dropna().to_numpy(dtype=float)
