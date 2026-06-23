"""Status bus: per-tab "outputs exist on disk" signal hub.

Each tab declares a ``status_signature(block)`` that returns the list of files
it would produce. The bus checks file existence + mtime and emits a signal
whenever the status of any tab changes. A ``QFileSystemWatcher`` refreshes
status when pipeline files are written externally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6 import QtCore

from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    StageStatus,
)


SignatureFn = Callable[[BlockHandle], list[Path]]


class StatusBus(QtCore.QObject):
    """Aggregator of per-tab disk-output status with live filesystem watching."""

    status_changed = QtCore.pyqtSignal(str, object)  # (tab_id, StageStatus)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signatures: dict[str, SignatureFn] = {}
        self._upstream: dict[str, list[str]] = {}
        self._last_status: dict[str, StageStatus] = {}
        self._current_block: BlockHandle | None = None
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_fs_change)
        self._watcher.fileChanged.connect(self._on_fs_change)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self.refresh_all)

    def register_tab(
        self,
        tab_id: str,
        signature_fn: SignatureFn,
        upstream_tabs: list[str] | None = None,
    ) -> None:
        self._signatures[tab_id] = signature_fn
        self._upstream[tab_id] = list(upstream_tabs or [])

    def set_block(self, block: BlockHandle | None) -> None:
        self._current_block = block
        self._reset_watcher()
        if block is not None:
            self._watch_block_paths(block)
        self.refresh_all()

    def refresh_all(self) -> None:
        if self._current_block is None:
            for tab_id in list(self._signatures):
                self._emit_if_changed(tab_id, StageStatus.NOT_STARTED)
            return
        for tab_id in list(self._signatures):
            status = self._compute_status(tab_id)
            self._emit_if_changed(tab_id, status)

    def status_for(self, tab_id: str) -> StageStatus:
        return self._last_status.get(tab_id, StageStatus.NOT_STARTED)

    def _reset_watcher(self) -> None:
        for path in self._watcher.files():
            self._watcher.removePath(path)
        for path in self._watcher.directories():
            self._watcher.removePath(path)

    def _watch_block_paths(self, block: BlockHandle) -> None:
        dirs: set[str] = set()
        files: set[str] = set()
        analysis = block.analysis_path
        if analysis.is_dir():
            dirs.add(str(analysis))
        else:
            parent = analysis.parent
            if parent.is_dir():
                dirs.add(str(parent))
        for tab_id, sig_fn in self._signatures.items():
            try:
                for p in sig_fn(block):
                    path = Path(p)
                    parent = path.parent
                    if parent.is_dir():
                        dirs.add(str(parent))
                    if path.is_file():
                        files.add(str(path))
            except Exception:
                continue
        for d in sorted(dirs):
            if d not in self._watcher.directories():
                self._watcher.addPath(d)
        for f in sorted(files):
            if f not in self._watcher.files():
                self._watcher.addPath(f)

    def _on_fs_change(self, _path: str) -> None:
        if self._current_block is not None:
            self._watch_block_paths(self._current_block)
        self._refresh_timer.start()

    def _compute_status(self, tab_id: str) -> StageStatus:
        if self._current_block is None:
            return StageStatus.NOT_STARTED
        sig_fn = self._signatures.get(tab_id)
        if sig_fn is None:
            return StageStatus.NOT_STARTED
        try:
            files = [Path(p) for p in sig_fn(self._current_block)]
        except Exception:
            return StageStatus.NOT_STARTED
        if not files:
            return StageStatus.NOT_STARTED
        existing = [p for p in files if p.exists()]
        if not existing:
            return StageStatus.NOT_STARTED
        if len(existing) < len(files):
            return StageStatus.PARTIAL

        own_min_mtime = min(p.stat().st_mtime for p in existing)
        for up_id in self._upstream.get(tab_id, []):
            up_fn = self._signatures.get(up_id)
            if up_fn is None:
                continue
            try:
                up_files = [Path(p) for p in up_fn(self._current_block) if Path(p).exists()]
            except Exception:
                continue
            if up_files and max(p.stat().st_mtime for p in up_files) > own_min_mtime:
                return StageStatus.STALE
        return StageStatus.COMPLETE

    def _emit_if_changed(self, tab_id: str, status: StageStatus) -> None:
        if self._last_status.get(tab_id) != status:
            self._last_status[tab_id] = status
            self.status_changed.emit(tab_id, status)
