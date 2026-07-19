"""Data Exploration tab — full-block traces + floating synced video."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
    discover_block_videos,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    EXPLORE_ARTIFACT_PROFILE,
    LoadReport,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_plot_panel import (
    ExplorePlotPanel,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_series import (
    EYE_VERSION_BASE,
    build_explore_catalog,
    discover_eye_data_versions,
    eye_csvs_are_stale,
    load_eye_data_version,
    prefer_eye_data_version,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_cache import (
    ExploreVideoCache,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_cache_dialog import (
    ExploreVideoCacheDialog,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
    ExploreVideoPanel,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_window import (
    ExploreVideoWindow,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.preprocessing.block_sync_core import load_final_sync_df


class _VideoLoadWorker(QtCore.QThread):
    """Load BlockSession + bind videos off the UI thread where possible."""

    finished_ok = QtCore.pyqtSignal(object)  # BlockSession-like result via panel API
    failed = QtCore.pyqtSignal(str)

    def __init__(self, load_fn, parent=None):
        super().__init__(parent)
        self._load_fn = load_fn

    def run(self) -> None:
        try:
            result = self._load_fn()
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ExploreTab(BaseTab):
    tab_id = "explore"
    tab_label = "Data Exploration"

    def __init__(self, state, config, parent=None):
        self._plot: ExplorePlotPanel | None = None
        self._video_window: ExploreVideoWindow | None = None
        self._video_cache = ExploreVideoCache()
        self._pending_le_df = None
        self._pending_re_df = None
        self._time_label: QtWidgets.QLabel | None = None
        self._cache_status: QtWidgets.QLabel | None = None
        self._btn_open_video: QtWidgets.QPushButton | None = None
        self._btn_cache_videos: QtWidgets.QPushButton | None = None
        self._stale_banner: QtWidgets.QLabel | None = None
        self._info: QtWidgets.QLabel | None = None
        self._syncing_time = False
        self._video_load_worker: _VideoLoadWorker | None = None
        super().__init__(state, config, parent)

    @property
    def _video(self) -> ExploreVideoPanel | None:
        if self._video_window is None:
            return None
        return self._video_window.panel

    def artifact_profile(self):
        return EXPLORE_ARTIFACT_PROFILE

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)

        self._stale_banner = QtWidgets.QLabel()
        self._stale_banner.setWordWrap(True)
        self._stale_banner.setStyleSheet(
            "background-color: #fff3cd; color: #664d03; padding: 6px; border-radius: 4px;"
        )
        self._stale_banner.hide()
        layout.addWidget(self._stale_banner)

        self._info = QtWidgets.QLabel(
            "Load prev analysis, then open the video window for synced L/Arena/R."
        )
        self._info.setWordWrap(True)
        self._info.setMaximumHeight(36)
        layout.addWidget(self._info)

        self._plot = ExplorePlotPanel()
        # Half-width: load-prev | plot chrome (selectors + tools)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        artifact = self._build_artifact_panel()
        chrome = self._plot.chrome_widget
        self._plot.layout().removeWidget(chrome)
        chrome.setParent(None)
        top.addWidget(artifact, stretch=1)
        top.addWidget(chrome, stretch=1)
        layout.addLayout(top)

        layout.addWidget(self._plot, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        self._btn_open_video = QtWidgets.QPushButton("Open video window")
        self._btn_open_video.setToolTip(
            "Open a floating window with Left / Arena / Right video synced to the playhead."
        )
        controls.addWidget(self._btn_open_video)
        self._btn_cache_videos = QtWidgets.QPushButton("Cache videos locally")
        self._btn_cache_videos.setToolTip(
            "Copy selected remote video files into a temporary local folder. "
            "Copies are deleted when you leave this tab or switch blocks."
        )
        self._btn_cache_videos.setEnabled(False)
        controls.addWidget(self._btn_cache_videos)
        self._cache_status = QtWidgets.QLabel("Using remote paths")
        self._cache_status.setStyleSheet("color: #555;")
        self._cache_status.setMinimumWidth(160)
        controls.addWidget(self._cache_status)
        controls.addStretch(1)
        self._time_label = QtWidgets.QLabel("t =            — ms")
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self._time_label.setFont(mono)
        self._time_label.setMinimumWidth(180)
        self._time_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        controls.addWidget(self._time_label)
        layout.addLayout(controls)

        self._plot.time_selected.connect(self._on_plot_time_selected)
        self._plot.time_preview.connect(self._set_time_label)
        self._plot.refresh_requested.connect(self._reload_catalog_into_plot)
        self._plot.eye_version_changed.connect(self._on_eye_version_changed)
        self._btn_open_video.clicked.connect(self._open_video_window)
        self._btn_cache_videos.clicked.connect(self._cache_videos_locally)

    def status_signature(self, block: BlockHandle) -> list[Path]:
        return [block.analysis_path / "final_sync_df.csv"]

    def set_block(self, block: BlockHandle | None) -> None:
        self._release_video_resources()
        if self._plot is not None:
            self._plot.set_catalog(None)
            self._plot.set_eye_versions([])
        if self._time_label is not None:
            self._time_label.setText("t =            — ms")
        self._pending_le_df = None
        self._pending_re_df = None
        if self._btn_cache_videos is not None:
            self._btn_cache_videos.setEnabled(False)
        if block is None:
            if self._info is not None:
                self._info.setText("No block loaded.")
            if self._stale_banner is not None:
                self._stale_banner.hide()
        else:
            if self._info is not None:
                self._info.setText(
                    f"Active block: {block.display_label}. "
                    "Use Load prev analysis when final_sync_df.csv exists."
                )
            self._update_stale_banner(block)
            if self._plot is not None:
                versions = discover_eye_data_versions(block.analysis_path)
                preferred = prefer_eye_data_version(versions)
                self._plot.set_eye_versions(
                    versions,
                    selected_tag=preferred.tag if preferred else EYE_VERSION_BASE,
                )
            if self._btn_cache_videos is not None:
                self._btn_cache_videos.setEnabled(True)
        super().set_block(block)

    def on_tab_deactivated(self) -> None:
        self._release_video_resources()

    def closeEvent(self, event) -> None:
        self._release_video_resources()
        super().closeEvent(event)

    def _release_video_resources(self) -> None:
        if self._video_load_worker is not None and self._video_load_worker.isRunning():
            self._video_load_worker.wait(100)
        self._video_load_worker = None
        self._close_video_window()
        self._video_cache.clear()
        self._update_cache_status()

    def _close_video_window(self) -> None:
        win = self._video_window
        if win is None:
            return
        self._video_window = None
        win.panel.clear()
        win.close()
        win.deleteLater()

    def _on_video_window_closed(self) -> None:
        win = self._video_window
        self._video_window = None
        if win is not None:
            win.deleteLater()

    def _open_video_window(self) -> None:
        if self._block is None:
            QtWidgets.QMessageBox.information(
                self, "Data Exploration", "Load a block first."
            )
            return
        if self._video_window is not None:
            self._video_window.show()
            self._video_window.raise_()
            self._video_window.activateWindow()
            return

        win = ExploreVideoWindow(self.window())
        win.closed.connect(self._on_video_window_closed)
        win.panel.time_changed.connect(self._on_video_time_changed)
        self._video_window = win
        win.show()
        win.raise_()
        # Defer heavy load so the window paints first; run load off UI where possible.
        QtCore.QTimer.singleShot(0, self._start_video_load)

    def _start_video_load(self) -> None:
        if self._block is None or self._video is None:
            return
        if self._video_load_worker is not None and self._video_load_worker.isRunning():
            return

        block = self._block
        out = self._state.output_folder or block.analysis_path
        le_df = self._pending_le_df
        re_df = self._pending_re_df
        panel = self._video

        progress = QtWidgets.QProgressDialog(
            "Opening videos…", None, 0, 0, self
        )
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()
        QtWidgets.QApplication.processEvents()

        def load_fn():
            # load_block_session is the heavy part; bind on UI thread afterward
            from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
                load_block_session,
            )
            from eye_tracking_system_tools.annotation.block_annotator.models import (
                AnnotatorConfig,
            )

            session = load_block_session(
                Path(block.block_path),
                Path(out),
                AnnotatorConfig(),
                animal_call=block.animal_call,
                experiment_date=block.experiment_date,
                block_num=block.block_num,
            )
            if le_df is not None:
                session.le_ellipse_df = le_df
            if re_df is not None:
                session.re_ellipse_df = re_df
            return session

        worker = _VideoLoadWorker(load_fn, self)
        self._video_load_worker = worker

        def on_ok(session) -> None:
            progress.close()
            self._video_load_worker = None
            if self._video is None:
                return
            try:
                self._video._bind_session(session)
                if self._video_cache.active:
                    self._apply_cache_to_open_panel()
                if self._plot is not None:
                    self._video.seek_ms(self._plot.playhead_ms(), emit=False)
                self._update_cache_status()
            except Exception as exc:  # noqa: BLE001
                if self._info is not None:
                    self._info.setText(
                        (self._info.text() + f"\nVideo bind warning: {exc}").strip()
                    )

        def on_fail(msg: str) -> None:
            progress.close()
            self._video_load_worker = None
            if self._info is not None:
                self._info.setText(
                    (self._info.text() + f"\nVideo load warning: {msg}").strip()
                )
            QtWidgets.QMessageBox.warning(
                self, "Data Exploration", f"Could not load videos:\n{msg}"
            )

        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_fail)
        worker.start()
        # Keep reference; panel unused in load_fn by design
        _ = panel

    def _after_load_artifacts(self, report: LoadReport) -> None:
        if self._block is None:
            return
        try:
            self._reload_catalog_into_plot()
            if self._info is not None:
                self._info.setText(
                    f"Loaded exploration data for {self._block.display_label}. "
                    + report.message()
                )
        except Exception as exc:
            if self._info is not None:
                self._info.setText(f"Cannot build exploration catalog: {exc}")

    def _update_stale_banner(self, block: BlockHandle) -> None:
        if self._stale_banner is None:
            return
        if eye_csvs_are_stale(block.analysis_path):
            self._stale_banner.setText(
                "Eye-data CSVs are older than final_sync_df.csv — upstream sync may "
                "have changed. Re-export eye data (and Kerr angles) before trusting "
                "pupil_size / k_phi / k_theta traces."
            )
            self._stale_banner.show()
        else:
            self._stale_banner.hide()

    def _update_cache_status(self) -> None:
        if self._cache_status is None:
            return
        if self._video_cache.active:
            self._cache_status.setText("Using local temp copies")
            self._cache_status.setStyleSheet("color: #0f5132;")
        else:
            self._cache_status.setText("Using remote paths")
            self._cache_status.setStyleSheet("color: #555;")

    def _set_time_label(self, ms: float) -> None:
        if self._time_label is not None:
            # Fixed-width field so buttons do not shift as digits change.
            self._time_label.setText(f"t = {ms:12.1f} ms")

    def _on_plot_time_selected(self, ms: float) -> None:
        self._set_time_label(ms)
        video = self._video
        if self._syncing_time or video is None:
            return
        self._syncing_time = True
        try:
            video.seek_ms(ms, emit=False)
        finally:
            self._syncing_time = False

    def _on_video_time_changed(self, ms: float) -> None:
        self._set_time_label(ms)
        if self._syncing_time or self._plot is None:
            return
        self._syncing_time = True
        try:
            self._plot.set_playhead_ms(ms, emit=False)
        finally:
            self._syncing_time = False

    def _on_eye_version_changed(self, _tag: str) -> None:
        try:
            self._reload_catalog_into_plot()
        except Exception as exc:
            if self._info is not None:
                self._info.setText(f"Cannot load eye_data version: {exc}")

    def _ensure_final_sync(self):
        blocksync = self._require_blocksync()
        if getattr(blocksync, "final_sync_df", None) is None:
            sync_path = self._block.analysis_path / "final_sync_df.csv"
            if not sync_path.is_file():
                raise FileNotFoundError(
                    "final_sync_df.csv missing. Complete Sync tab finalize first."
                )
            load_final_sync_df(blocksync, verbose=False)
            self._state.final_sync_df = blocksync.final_sync_df
        final_df = blocksync.final_sync_df
        if final_df is None or final_df.empty:
            raise RuntimeError("final_sync_df is empty.")
        return blocksync, final_df

    def _apply_cache_to_open_panel(self) -> None:
        video = self._video
        if video is None or video._session is None:
            return
        video.remount_videos(
            arena_videos=self._video_cache.map_paths(video._original_arena),
            le_videos=self._video_cache.map_paths(video._original_le),
            re_videos=self._video_cache.map_paths(video._original_re),
        )

    def _discover_videos_for_block(self) -> tuple[list[Path], list[Path], list[Path]]:
        if self._block is None:
            return [], [], []
        video = self._video
        if video is not None and (
            video._original_arena or video._original_le or video._original_re
        ):
            return (
                list(video._original_arena),
                list(video._original_le),
                list(video._original_re),
            )
        return discover_block_videos(self._block.block_path)

    def _cache_videos_locally(self) -> None:
        if self._block is None:
            QtWidgets.QMessageBox.information(
                self, "Data Exploration", "Load a block first."
            )
            return
        arena, le, re = self._discover_videos_for_block()
        if not arena and not le and not re:
            QtWidgets.QMessageBox.warning(
                self, "Data Exploration", "No video files found to cache."
            )
            return

        dlg = ExploreVideoCacheDialog(
            self._video_cache,
            arena=arena,
            le=le,
            re=re,
            parent=self,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._apply_cache_to_open_panel()
            self._update_cache_status()
            if self._info is not None:
                self._info.setText(
                    (self._info.text() + "\nVideos cached to local temp copies.").strip()
                )

    def _reload_videos(self, le_df, re_df) -> None:
        self._pending_le_df = le_df
        self._pending_re_df = re_df
        if self._block is None:
            return
        if self._btn_cache_videos is not None:
            self._btn_cache_videos.setEnabled(True)
        if self._video is None:
            return
        # Window already open — reload in background
        self._start_video_load()

    def _reload_catalog_into_plot(self) -> None:
        if self._block is None or self._plot is None:
            return
        blocksync, final_df = self._ensure_final_sync()

        versions = discover_eye_data_versions(self._block.analysis_path)
        preferred = prefer_eye_data_version(versions)
        selected_tag = self._plot.current_eye_version_tag()
        if selected_tag not in {v.tag for v in versions}:
            selected_tag = preferred.tag if preferred else EYE_VERSION_BASE
            self._plot.set_eye_versions(versions, selected_tag=selected_tag)
        else:
            self._plot.set_eye_versions(versions, selected_tag=selected_tag)

        sample_rate = float(getattr(blocksync, "sample_rate", 30000) or 30000)
        le_df, re_df, le_path, re_path = load_eye_data_version(
            self._block.analysis_path, selected_tag
        )

        oe_rec = getattr(blocksync, "oe_rec", None)
        stale = eye_csvs_are_stale(self._block.analysis_path)
        self._update_stale_banner(self._block)

        catalog = build_explore_catalog(
            final_df,
            sample_rate,
            le_df=le_df,
            re_df=re_df,
            oe_rec=oe_rec,
            stale_sync=stale,
            le_csv_path=le_path,
            re_csv_path=re_path,
            eye_version_tag=selected_tag,
        )
        self._plot.set_catalog(catalog)
        self._reload_videos(le_df, re_df)
        if len(catalog.ms_axis):
            t0 = float(catalog.ms_axis[0])
            self._plot.set_playhead_ms(t0, emit=False)
            self._set_time_label(t0)
            if self._video is not None and self._video._session is not None:
                self._video.seek_ms(t0, emit=False)
