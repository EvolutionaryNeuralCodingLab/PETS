"""Stage 5 -- Sync-free eye ellipse pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6 import QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    syncfree_artifact_profile,
    syncfree_status_signature_paths,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.ellipse_verifier import (
    EllipseVerifierWidget,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.workers import CallableWorker
from eye_tracking_system_tools.preprocessing.sync_free_eye_io import (
    analysis_syncfree_timeline_path,
    default_syncfree_paths,
    discover_eye_video_paths,
    finalize_syncfree_eye,
    maybe_load_kerr_refs,
    resolve_syncfree_working_csv,
    run_syncfree_ellipses_for_eye,
    syncfree_mapped_is_stale,
    video_path_for_eye,
    write_syncfree_draft,
)


class SyncFreeTab(BaseTab):
    tab_id = "syncfree"
    tab_label = "Sync-free"

    def __init__(self, state, config, parent=None):
        self._left_verifier: EllipseVerifierWidget | None = None
        self._right_verifier: EllipseVerifierWidget | None = None
        self._worker: CallableWorker | None = None
        super().__init__(state, config, parent)

    def artifact_profile(self):
        return syncfree_artifact_profile(self._artifact_tag_value())

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_artifact_panel())

        self._stale_banner = QtWidgets.QLabel()
        self._stale_banner.setWordWrap(True)
        self._stale_banner.setStyleSheet(
            "background-color: #fff3cd; color: #664d03; padding: 8px; border-radius: 4px;"
        )
        self._stale_banner.hide()
        layout.addWidget(self._stale_banner)

        self._missing_sync_banner = QtWidgets.QLabel(
            "<b>final_sync_df.csv is missing.</b> You can still finalize per-video "
            "<code>*_eye_data.csv</code> files. Timeline mapping requires the Sync tab."
        )
        self._missing_sync_banner.setWordWrap(True)
        self._missing_sync_banner.setStyleSheet(
            "background-color: #f8f9fa; color: #495057; padding: 8px; border-radius: 4px;"
        )
        self._missing_sync_banner.hide()
        layout.addWidget(self._missing_sync_banner)

        self._info = QtWidgets.QLabel(
            "Optional DLC → ellipse → verify pipeline per eye video. "
            "<b>Finalize</b> writes one <code>*_eye_data.csv</code> per eye (ellipse + Kerr), "
            "<code>*_kerr_refs.csv</code>, and <code>*_meta.json</code>."
        )
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        params = QtWidgets.QFormLayout()
        self._artifact_tag = QtWidgets.QLineEdit(str(self._config.syncfree_artifact_tag))
        self._artifact_tag.setPlaceholderText("v1")
        self._artifact_tag.textChanged.connect(lambda _: self._refresh_artifact_ui())
        self._uncertainty_thr = QtWidgets.QDoubleSpinBox()
        self._uncertainty_thr.setRange(0.0, 1.0)
        self._uncertainty_thr.setDecimals(3)
        self._uncertainty_thr.setSingleStep(0.01)
        self._uncertainty_thr.setValue(float(self._config.syncfree_uncertainty_thr))
        params.addRow("artifact_tag:", self._artifact_tag)
        params.addRow("uncertainty_thr:", self._uncertainty_thr)
        layout.addLayout(params)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_ellipses = QtWidgets.QPushButton("Run ellipses (both eyes)")
        self._btn_save_draft = QtWidgets.QPushButton("Save draft corrections")
        self._btn_save_draft.setEnabled(False)
        self._btn_finalize = QtWidgets.QPushButton("Finalize (both eyes)")
        self._btn_finalize.setEnabled(False)
        btn_row.addWidget(self._btn_ellipses)
        btn_row.addWidget(self._btn_save_draft)
        btn_row.addWidget(self._btn_finalize)
        layout.addLayout(btn_row)

        self._chk_map_timeline = QtWidgets.QCheckBox(
            "Also write OE timeline CSVs under analysis/ "
            "(requires final_sync_df.csv)"
        )
        self._chk_map_timeline.setChecked(True)
        layout.addWidget(self._chk_map_timeline)

        self._verifier_host = QtWidgets.QWidget()
        self._verifier_layout = QtWidgets.QHBoxLayout(self._verifier_host)
        self._verifier_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._verifier_host, stretch=1)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

        self._btn_ellipses.clicked.connect(self._run_ellipses)
        self._btn_save_draft.clicked.connect(self._save_draft_corrections)
        self._btn_finalize.clicked.connect(self._run_finalize)

    def status_signature(self, block: BlockHandle) -> list[Path]:
        return syncfree_status_signature_paths(block, self._config)

    def set_block(self, block: BlockHandle | None) -> None:
        self._clear_verifiers()
        self._btn_save_draft.setEnabled(False)
        self._btn_finalize.setEnabled(False)
        if block is None:
            self._info.setText("No block loaded.")
            self._stale_banner.hide()
            self._missing_sync_banner.hide()
            self._status.setText("")
        else:
            self._info.setText(f"Active block: {block.display_label}")
            final_sync = block.analysis_path / "final_sync_df.csv"
            self._missing_sync_banner.setVisible(not final_sync.is_file())
            self._chk_map_timeline.setEnabled(final_sync.is_file())
            if not final_sync.is_file():
                self._chk_map_timeline.setChecked(False)
            self._update_stale_banner(block)
            self._status.setText(
                "Use 'Load prev analysis' when sync-free outputs exist on disk."
            )
        super().set_block(block)

    def _after_load_artifacts(self, report) -> None:
        if self._block is None:
            return
        try:
            self._try_load_verifiers()
            self._status.setText("Loaded sync-free artifacts from disk.")
        except Exception as e:
            self._info.setText(f"Cannot load sync-free data: {e}")
            self._status.setText("")

    def _artifact_tag_value(self) -> str:
        tag = self._artifact_tag.text().strip()
        return tag or str(self._config.syncfree_artifact_tag)

    def _uncertainty_value(self) -> float:
        return float(self._uncertainty_thr.value())

    def _require_blocksync(self):
        bs = super()._require_blocksync()
        self._session.ensure_eye_videos(bs)
        return bs

    def _video_for_eye(self, eye: str) -> Path:
        return video_path_for_eye(self._require_blocksync(), eye)  # type: ignore[arg-type]

    def _paths_for_eye(self, eye: str) -> dict[str, Path]:
        return default_syncfree_paths(self._video_for_eye(eye), eye, self._artifact_tag_value())

    def _update_stale_banner(self, block: BlockHandle) -> None:
        tag = self._artifact_tag_value()
        stale = any(
            syncfree_mapped_is_stale(block, eye, tag) for eye in ("left", "right")
        )
        if stale:
            self._stale_banner.setText(
                "Timeline CSVs under analysis/ are older than final_sync_df.csv — "
                "re-run <b>Finalize</b> with timeline mapping enabled."
            )
            self._stale_banner.show()
        else:
            self._stale_banner.hide()

    def _clear_verifiers(self) -> None:
        while self._verifier_layout.count():
            item = self._verifier_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._left_verifier = None
        self._right_verifier = None

    def _load_kerr_ref_for_eye(self, eye: str) -> tuple[int, int] | None:
        paths = self._paths_for_eye(eye)
        if paths["kerr_refs"].is_file():
            row = pd.read_csv(paths["kerr_refs"]).iloc[0]
            return int(row["kerr_ref_x"]), int(row["kerr_ref_y"])
        try:
            return maybe_load_kerr_refs(self._require_blocksync(), eye)  # type: ignore[arg-type]
        except FileNotFoundError:
            return None

    def _try_load_verifiers(self) -> None:
        blocksync = self._require_blocksync()
        tag = self._artifact_tag_value()
        sources: dict[str, Path] = {}
        for eye in ("left", "right"):
            video = video_path_for_eye(blocksync, eye)
            src = resolve_syncfree_working_csv(video, eye, tag)
            if src is None:
                self._clear_verifiers()
                self._btn_save_draft.setEnabled(False)
                self._btn_finalize.setEnabled(False)
                return
            sources[eye] = src

        left_df = pd.read_csv(sources["left"])
        right_df = pd.read_csv(sources["right"])
        self._clear_verifiers()
        block_path = Path(blocksync.block_path)
        self._left_verifier = EllipseVerifierWidget(
            left_df,
            video_path_for_eye(blocksync, "left"),
            "left",
            ref_point_xy=self._load_kerr_ref_for_eye("left"),
            parent=self,
            block_path=block_path,
        )
        self._right_verifier = EllipseVerifierWidget(
            right_df,
            video_path_for_eye(blocksync, "right"),
            "right",
            ref_point_xy=self._load_kerr_ref_for_eye("right"),
            parent=self,
            block_path=block_path,
        )
        self._verifier_layout.addWidget(self._left_verifier)
        self._verifier_layout.addWidget(self._right_verifier)
        self._btn_save_draft.setEnabled(True)
        self._btn_finalize.setEnabled(True)

    def _set_busy(self, busy: bool) -> None:
        for btn in (self._btn_ellipses, self._btn_save_draft, self._btn_finalize):
            btn.setEnabled(not busy)
        if not busy and self._left_verifier is not None:
            self._btn_save_draft.setEnabled(True)
            self._btn_finalize.setEnabled(True)

    def _run_ellipses(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        def work():
            blocksync = self._require_blocksync()
            thr = self._uncertainty_value()
            tag = self._artifact_tag_value()
            for eye in ("left", "right"):
                video = video_path_for_eye(blocksync, eye)
                df_ell, meta = run_syncfree_ellipses_for_eye(blocksync, eye, thr)  # type: ignore[arg-type]
                write_syncfree_draft(df_ell, meta, video, eye, tag)

        self._set_busy(True)
        worker = CallableWorker(work, self)

        def on_ok(_=None):
            self._set_busy(False)
            self._try_load_verifiers()
            self._status.setText(
                f"Draft ellipses ready for both eyes (tag={self._artifact_tag_value()}). "
                "Verify, then Finalize."
            )
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str):
            self._set_busy(False)
            self._status.setText(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Sync-free tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()

    def _save_draft_corrections(self) -> None:
        if self._left_verifier is None or self._right_verifier is None:
            return
        try:
            tag = self._artifact_tag_value()
            for eye, verifier in (
                ("left", self._left_verifier),
                ("right", self._right_verifier),
            ):
                paths = self._paths_for_eye(eye)
                verifier.df().to_csv(paths["draft"], index=False)
                ref = verifier.ref_xy()
                if ref is not None:
                    rx, ry = int(ref[0]), int(ref[1])
                    pd.DataFrame(
                        [{"eye": eye, "kerr_ref_x": rx, "kerr_ref_y": ry}]
                    ).to_csv(paths["kerr_refs"], index=False)
            self._status.setText("Draft corrections saved (not finalized yet).")
        except Exception as e:
            self._status.setText(f"Error: {e}")
            QtWidgets.QMessageBox.warning(self, "Sync-free tab", str(e))

    def _run_finalize(self) -> None:
        if self._left_verifier is None or self._right_verifier is None:
            return
        map_timeline = self._chk_map_timeline.isChecked()
        if map_timeline and self._block is not None:
            if not (self._block.analysis_path / "final_sync_df.csv").is_file():
                QtWidgets.QMessageBox.warning(
                    self,
                    "Sync-free tab",
                    "final_sync_df.csv is missing — uncheck timeline mapping or "
                    "complete the Sync tab first.",
                )
                return

        for verifier in (self._left_verifier, self._right_verifier):
            if verifier.ref_xy() is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Sync-free tab",
                    "Pick a Kerr reference point on each eye (click the video) "
                    "before finalizing.",
                )
                return

        if self._worker is not None and self._worker.isRunning():
            return

        left_df = self._left_verifier.df()
        right_df = self._right_verifier.df()
        left_ref = self._left_verifier.ref_xy()
        right_ref = self._right_verifier.ref_xy()
        assert left_ref is not None and right_ref is not None
        tag = self._artifact_tag_value()

        def work():
            blocksync = self._require_blocksync()
            finalize_syncfree_eye(
                blocksync,
                "left",
                left_df,
                left_ref,
                tag=tag,
                map_to_timeline=map_timeline,
            )
            finalize_syncfree_eye(
                blocksync,
                "right",
                right_df,
                right_ref,
                tag=tag,
                map_to_timeline=map_timeline,
            )

        self._set_busy(True)
        worker = CallableWorker(work, self)

        def on_ok(_=None):
            self._set_busy(False)
            if self._block is not None:
                self._update_stale_banner(self._block)
            self._try_load_verifiers()
            parts = [
                f"{self._artifact_tag_value()} eye_data + kerr_refs + meta for both eyes"
            ]
            if map_timeline:
                parts.append("timeline CSVs under analysis/")
            self._status.setText("Finalized: " + "; ".join(parts) + ".")
            self._worker = None
            worker.deleteLater()

        def on_fail(msg: str):
            self._set_busy(False)
            self._status.setText(f"Error: {msg}")
            QtWidgets.QMessageBox.warning(self, "Sync-free tab", msg)
            self._worker = None
            worker.deleteLater()

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        self._worker = worker
        worker.start()
