"""Temporary on-disk copies of explore video files (not decoded frames)."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

ProgressFn = Callable[
    [int, int, Path, int, int, int, int, float],
    None,
]
# (file_i, n_files, src, file_copied, file_size, total_copied, total_size, bytes_per_sec)


class ExploreVideoCache:
    """Session-scoped temp directory for remote video file copies."""

    def __init__(self) -> None:
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._map: dict[Path, Path] = {}  # resolved original -> local copy

    @property
    def active(self) -> bool:
        return bool(self._map)

    @property
    def root(self) -> Path | None:
        if self._tmpdir is None:
            return None
        return Path(self._tmpdir.name)

    def local_path(self, original: Path | str) -> Path:
        """Return cached local path if present, else ``original``."""
        key = Path(original).resolve()
        return self._map.get(key, Path(original))

    def map_paths(self, originals: list[Path | str]) -> list[Path]:
        return [self.local_path(p) for p in originals]

    def cache_paths(
        self,
        sources: list[Path | str],
        *,
        progress: Callable[[int, int, Path], None] | None = None,
        progress_bytes: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> dict[Path, Path]:
        """
        Copy existing source files into a temp directory.

        Returns mapping of resolved original path -> local copy path.
        Sources already under the temp root are skipped (identity mapping).
        """
        existing = [Path(p) for p in sources if Path(p).is_file()]
        if not existing:
            return {}

        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="pets_explore_vid_")

        root = Path(self._tmpdir.name)
        sizes = [p.stat().st_size for p in existing]
        total_size = int(sum(sizes))
        total_copied = 0
        n_files = len(existing)
        t0 = time.monotonic()
        last_report = t0

        for i, src in enumerate(existing, start=1):
            if should_cancel is not None and should_cancel():
                raise RuntimeError("cancelled")
            resolved = src.resolve()
            if progress is not None:
                progress(i, n_files, src)
            try:
                if root in resolved.parents or resolved == root:
                    self._map[resolved] = resolved
                    total_copied += sizes[i - 1]
                    continue
            except Exception:
                pass
            if resolved in self._map and self._map[resolved].is_file():
                total_copied += sizes[i - 1]
                continue

            dest = self._unique_dest(root, src.name)
            file_size = sizes[i - 1]
            file_copied = 0
            with open(src, "rb") as fsrc, open(dest, "wb") as fdst:
                while True:
                    if should_cancel is not None and should_cancel():
                        dest.unlink(missing_ok=True)
                        raise RuntimeError("cancelled")
                    chunk = fsrc.read(chunk_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    file_copied += len(chunk)
                    total_copied += len(chunk)
                    now = time.monotonic()
                    if progress_bytes is not None and (
                        now - last_report >= 0.05 or file_copied >= file_size
                    ):
                        elapsed = max(1e-6, now - t0)
                        progress_bytes(
                            i,
                            n_files,
                            src,
                            file_copied,
                            file_size,
                            total_copied,
                            total_size,
                            total_copied / elapsed,
                        )
                        last_report = now
            # Preserve mtime when possible
            try:
                shutil.copystat(src, dest)
            except OSError:
                pass
            self._map[resolved] = dest
            if progress_bytes is not None:
                elapsed = max(1e-6, time.monotonic() - t0)
                progress_bytes(
                    i,
                    n_files,
                    src,
                    file_size,
                    file_size,
                    total_copied,
                    total_size,
                    total_copied / elapsed,
                )
        return dict(self._map)

    def clear(self) -> None:
        """Delete all temp copies and reset the mapping."""
        self._map.clear()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _unique_dest(self, root: Path, name: str) -> Path:
        dest = root / name
        if not dest.exists():
            return dest
        stem = Path(name).stem
        suffix = Path(name).suffix
        n = 1
        while True:
            candidate = root / f"{stem}__{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1
