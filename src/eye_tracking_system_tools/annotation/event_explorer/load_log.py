"""GUI load log panel and auto-written log file."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PyQt6 import QtCore, QtWidgets


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class LoadLog(QtCore.QObject):
    """Timestamped INFO/WARN lines to GUI and optional log file."""

    line_added = QtCore.pyqtSignal(str)

    def __init__(self, log_file: Path | None = None) -> None:
        super().__init__()
        self._log_file = Path(log_file) if log_file else None
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- Event Explorer session { _ts() } ---\n")

    @property
    def log_file(self) -> Path | None:
        return self._log_file

    def set_log_file(self, path: Path | None) -> None:
        self._log_file = Path(path) if path else None
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def info(self, msg: str) -> None:
        self._emit("INFO", msg)

    def warn(self, msg: str) -> None:
        self._emit("WARN", msg)

    def _emit(self, level: str, msg: str) -> None:
        line = f"[{_ts()}] {level}: {msg}"
        self.line_added.emit(line)
        if self._log_file:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def excerpt(self, max_lines: int = 80) -> list[str]:
        if not self._log_file or not self._log_file.exists():
            return []
        lines = self._log_file.read_text(encoding="utf-8").splitlines()
        return lines[-max_lines:]


class LoadLogPanel(QtWidgets.QWidget):
    def __init__(self, load_log: LoadLog, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)
        layout.addWidget(self._text)
        load_log.line_added.connect(self._append)

    def _append(self, line: str) -> None:
        self._text.appendPlainText(line)


def default_log_path(base_dir: Path | None = None) -> Path:
    base = base_dir or Path.home()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"explorer_load_{stamp}.log"
