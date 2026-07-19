"""Dialog to pick explore videos and copy them to a local temp cache."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_cache import (
    ExploreVideoCache,
)


def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _fmt_speed(bps: float) -> str:
    return f"{_fmt_bytes(bps)}/s"


class _CacheWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int, str, int, int, int, int, float)
    finished_ok = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        cache: ExploreVideoCache,
        sources: list[Path],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cache = cache
        self._sources = sources
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self._cache.cache_paths(
                self._sources,
                progress_bytes=self._on_bytes,
                should_cancel=lambda: self._cancel,
            )
            if self._cancel:
                self.failed.emit("cancelled")
                return
            self.finished_ok.emit()
        except RuntimeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _on_bytes(
        self,
        file_i: int,
        n_files: int,
        src: Path,
        file_copied: int,
        file_size: int,
        total_copied: int,
        total_size: int,
        bps: float,
    ) -> None:
        self.progress.emit(
            file_i,
            n_files,
            src.name,
            file_copied,
            file_size,
            total_copied,
            total_size,
            float(bps),
        )


class ExploreVideoCacheDialog(QtWidgets.QDialog):
    """Pick which arena/eye videos to cache locally, with byte-level progress."""

    def __init__(
        self,
        cache: ExploreVideoCache,
        *,
        arena: list[Path],
        le: list[Path],
        re: list[Path],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cache videos locally")
        self.resize(560, 420)
        self._cache = cache
        self._worker: _CacheWorker | None = None
        self._checks: list[tuple[QtWidgets.QCheckBox, Path]] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Select videos to copy into a temporary local folder "
                "(deleted when you leave Explore or switch blocks)."
            )
        )

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        host_lay = QtWidgets.QVBoxLayout(host)

        def add_group(title: str, paths: list[Path], *, default_checked: bool) -> None:
            box = QtWidgets.QGroupBox(title)
            box_lay = QtWidgets.QVBoxLayout(box)
            if not paths:
                lbl = QtWidgets.QLabel("(none found)")
                lbl.setStyleSheet("color: #888;")
                box_lay.addWidget(lbl)
            for i, path in enumerate(paths):
                size = path.stat().st_size if path.is_file() else 0
                chk = QtWidgets.QCheckBox(f"{path.name}  ({_fmt_bytes(size)})")
                chk.setToolTip(str(path))
                # Default: first arena + all eye videos
                if title.startswith("Arena"):
                    chk.setChecked(default_checked and i == 0)
                else:
                    chk.setChecked(default_checked)
                if cache.local_path(path) != path and cache.active:
                    chk.setText(chk.text() + "  [cached]")
                self._checks.append((chk, path))
                box_lay.addWidget(chk)
            host_lay.addWidget(box)

        add_group("Arena", arena, default_checked=True)
        add_group("Left eye", le, default_checked=True)
        add_group("Right eye", re, default_checked=True)
        host_lay.addStretch(1)
        scroll.setWidget(host)
        layout.addWidget(scroll, stretch=1)

        self._overall = QtWidgets.QProgressBar()
        self._overall.setRange(0, 1000)
        self._file_bar = QtWidgets.QProgressBar()
        self._file_bar.setRange(0, 1000)
        self._status = QtWidgets.QLabel("Ready.")
        layout.addWidget(QtWidgets.QLabel("Overall"))
        layout.addWidget(self._overall)
        layout.addWidget(QtWidgets.QLabel("Current file"))
        layout.addWidget(self._file_bar)
        layout.addWidget(self._status)

        btns = QtWidgets.QHBoxLayout()
        self._btn_start = QtWidgets.QPushButton("Start copy")
        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_close = QtWidgets.QPushButton("Close")
        btns.addWidget(self._btn_start)
        btns.addWidget(self._btn_cancel)
        btns.addStretch(1)
        btns.addWidget(self._btn_close)
        layout.addLayout(btns)

        self._btn_cancel.setEnabled(False)
        self._btn_start.clicked.connect(self._start)
        self._btn_cancel.clicked.connect(self._cancel)
        self._btn_close.clicked.connect(self.reject)

    def selected_paths(self) -> list[Path]:
        return [p for chk, p in self._checks if chk.isChecked()]

    def _start(self) -> None:
        sources = self.selected_paths()
        if not sources:
            QtWidgets.QMessageBox.warning(
                self, "Cache videos", "Select at least one video."
            )
            return
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._btn_close.setEnabled(False)
        for chk, _ in self._checks:
            chk.setEnabled(False)
        self._overall.setValue(0)
        self._file_bar.setValue(0)
        self._status.setText("Starting…")

        self._worker = _CacheWorker(self._cache, sources, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self._status.setText("Cancelling…")

    def _on_progress(
        self,
        file_i: int,
        n_files: int,
        name: str,
        file_copied: int,
        file_size: int,
        total_copied: int,
        total_size: int,
        bps: float,
    ) -> None:
        if total_size > 0:
            self._overall.setValue(int(1000 * total_copied / total_size))
        if file_size > 0:
            self._file_bar.setValue(int(1000 * file_copied / file_size))
        self._status.setText(
            f"File {file_i}/{n_files}: {name} — "
            f"{_fmt_bytes(file_copied)} / {_fmt_bytes(file_size)}  |  "
            f"total {_fmt_bytes(total_copied)} / {_fmt_bytes(total_size)}  |  "
            f"{_fmt_speed(bps)}"
        )

    def _on_ok(self) -> None:
        self._overall.setValue(1000)
        self._file_bar.setValue(1000)
        self._status.setText("Done.")
        self._finish_ui()
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._finish_ui()
        if message == "cancelled":
            self._cache.clear()
            self._status.setText("Cancelled — temp copies cleared.")
            return
        QtWidgets.QMessageBox.warning(self, "Cache videos", f"Cache failed: {message}")

    def _finish_ui(self) -> None:
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._btn_close.setEnabled(True)
        for chk, _ in self._checks:
            chk.setEnabled(True)
        self._worker = None
