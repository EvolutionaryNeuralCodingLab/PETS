"""Base class for Preprocessing GUI tabs."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    ArtifactLoadState,
    ArtifactScanResult,
    TabArtifactProfile,
    artifact_checklist_text,
    load_tab_artifacts,
    scan_tab_artifacts,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.block_session import (
    BlockSyncSession,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    PreprocConfig,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
)
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync


_LOAD_BTN_STYLE = {
    ArtifactLoadState.NONE: "color: #888888;",
    ArtifactLoadState.PARTIAL: "background-color: #fff3cd; color: #664d03;",
    ArtifactLoadState.READY: "background-color: #d1e7dd; color: #0f5132;",
}


class BaseTab(QtWidgets.QWidget):
    tab_id: str = "base"
    tab_label: str = "Tab"

    def __init__(
        self,
        state: GuiState,
        config: PreprocConfig,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._state = state
        self._config = config
        self._block: BlockHandle | None = None
        self._artifact_panel: QtWidgets.QWidget | None = None
        self._btn_load_prev: QtWidgets.QPushButton | None = None
        self._artifact_checklist: QtWidgets.QPlainTextEdit | None = None
        self._last_scan: ArtifactScanResult | None = None
        self.build_ui()

    @property
    def _session(self) -> BlockSyncSession:
        return self._state.ensure_session()

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._artifact_panel = self._build_artifact_panel()
        layout.addWidget(self._artifact_panel)
        self._placeholder_label = QtWidgets.QLabel(
            f"<i>{self.tab_label}</i> — controls land in a later phase."
        )
        layout.addWidget(self._placeholder_label)
        layout.addStretch(1)

    def _build_artifact_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Previous analysis on disk")
        lay = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        self._btn_load_prev = QtWidgets.QPushButton("Load prev analysis")
        self._btn_load_prev.setEnabled(False)
        self._btn_load_prev.clicked.connect(self._on_load_prev_analysis)
        row.addWidget(self._btn_load_prev)
        row.addStretch(1)
        lay.addLayout(row)
        self._artifact_checklist = QtWidgets.QPlainTextEdit()
        self._artifact_checklist.setReadOnly(True)
        self._artifact_checklist.setMaximumHeight(120)
        self._artifact_checklist.setPlaceholderText("Artifact checklist…")
        lay.addWidget(self._artifact_checklist)
        return box

    def artifact_profile(self) -> TabArtifactProfile | None:
        """Override to return this tab's artifact manifest."""
        return None

    def set_block(self, block: BlockHandle | None) -> None:
        self._block = block
        placeholder = getattr(self, "_placeholder_label", None)
        if placeholder is not None:
            if block is None:
                placeholder.setText(
                    f"<i>{self.tab_label}</i> — no block loaded."
                )
            else:
                placeholder.setText(
                    f"<i>{self.tab_label}</i> — active block: "
                    f"<b>{block.display_label}</b>"
                )
        self._refresh_artifact_ui()

    def status_signature(self, block: BlockHandle) -> list[Path]:
        return []

    def _require_blocksync(self) -> BlockSync:
        if self._block is None:
            raise RuntimeError("No block loaded.")
        return self._session.get(self._block)

    def _refresh_artifact_ui(self) -> None:
        if self._btn_load_prev is None or self._artifact_checklist is None:
            return
        profile = self.artifact_profile()
        if self._block is None or profile is None:
            self._btn_load_prev.setText("Load prev analysis")
            self._btn_load_prev.setEnabled(False)
            self._btn_load_prev.setStyleSheet(_LOAD_BTN_STYLE[ArtifactLoadState.NONE])
            self._artifact_checklist.clear()
            self._last_scan = None
            return
        scan = scan_tab_artifacts(profile, self._block, self._config)
        self._last_scan = scan
        n, total = scan.found_count, scan.total
        if scan.state is ArtifactLoadState.NONE:
            self._btn_load_prev.setText(f"Load prev analysis (0/{total})")
            self._btn_load_prev.setEnabled(False)
        elif scan.state is ArtifactLoadState.PARTIAL:
            self._btn_load_prev.setText(f"Load prev analysis ({n}/{total})")
            self._btn_load_prev.setEnabled(True)
        else:
            self._btn_load_prev.setText(f"Load prev analysis ({n}/{total})")
            self._btn_load_prev.setEnabled(True)
        self._btn_load_prev.setStyleSheet(_LOAD_BTN_STYLE[scan.state])
        self._artifact_checklist.setPlainText(artifact_checklist_text(scan))

    def _on_load_prev_analysis(self) -> None:
        profile = self.artifact_profile()
        if profile is None or self._block is None:
            return
        try:
            report = load_tab_artifacts(
                profile,
                self._session,
                self._block,
                self._state,
                self._config,
            )
            self._after_load_artifacts(report)
            self._refresh_artifact_ui()
            parent = self.window()
            if hasattr(parent, "_status_bus"):
                parent._status_bus.refresh_all()
            QtWidgets.QMessageBox.information(
                self,
                f"{self.tab_label} — load previous analysis",
                report.message(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                f"{self.tab_label} — load previous analysis",
                str(exc),
            )

    def _after_load_artifacts(self, report) -> None:
        """Hook for tab-specific UI refresh after disk load."""

    def on_filesystem_changed(self) -> None:
        """Called when analysis folder changes (status bus watcher)."""
        self._refresh_artifact_ui()
