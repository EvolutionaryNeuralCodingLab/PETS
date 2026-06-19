"""Status bus: per-tab "outputs exist on disk" signal hub.

Phase 0 implements a polling-only stub. Each tab declares a
``status_signature(block)`` that returns the list of files it would produce.
The bus checks file existence + mtime and emits a single signal whenever the
status of any tab changes.

Later phases (8) plug in a real ``QFileSystemWatcher`` so the icons update
live as the pipeline writes files.
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
    """Lightweight aggregator of per-tab disk-output status."""

    status_changed = QtCore.pyqtSignal(str, object)  # (tab_id, StageStatus)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signatures: dict[str, SignatureFn] = {}
        self._upstream: dict[str, list[str]] = {}
        self._last_status: dict[str, StageStatus] = {}
        self._current_block: BlockHandle | None = None

    def register_tab(
        self,
        tab_id: str,
        signature_fn: SignatureFn,
        upstream_tabs: list[str] | None = None,
    ) -> None:
        """Register a tab and the files whose existence implies it is done.

        ``upstream_tabs`` are tab ids whose completion this tab depends on;
        used to compute STALE (when an upstream output's mtime is newer than
        ours). Phase 0 collects but does not yet act on this information
        beyond the basic computation in :meth:`_compute_status`.
        """
        self._signatures[tab_id] = signature_fn
        self._upstream[tab_id] = list(upstream_tabs or [])

    def set_block(self, block: BlockHandle | None) -> None:
        self._current_block = block
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
