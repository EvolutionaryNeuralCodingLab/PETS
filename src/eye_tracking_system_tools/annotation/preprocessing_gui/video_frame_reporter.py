"""Discover block videos and report frame counts via ffprobe for TTL mapping clues."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eye_tracking_system_tools.preprocessing.arena_video_io import (
    list_convertible_arena_videos,
    list_arena_mp4s,
    resolve_arena_path,
)

# Bound ffprobe so Manual TTL never freezes indefinitely on large/corrupt files.
FFPROBE_TIMEOUT_S = 45.0


@dataclass(frozen=True)
class VideoSourceReport:
    source: str
    path: Path
    n_frames: int | None
    closest_ttl_line: int | None
    ttl_n_rising: int | None
    delta: int | None
    error: str | None = None


def require_ffprobe() -> str:
    """Return path to ``ffprobe`` or raise a clear error."""
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError(
            "ffprobe not found on PATH. Install ffmpeg (which includes ffprobe) "
            "to use the video reporter."
        )
    return path


def _parse_ffprobe_int(raw: str, *, label: str) -> int | None:
    text = (raw or "").strip().splitlines()
    if not text:
        return None
    token = text[0].strip().rstrip(",")
    if not token or token.upper() in {"N/A", "NAN", "NULL"}:
        return None
    try:
        value = int(float(token))
    except ValueError:
        return None
    return value if value > 0 else None


def count_frames_ffprobe(
    video_path: Path | str,
    *,
    ffprobe_bin: str | None = None,
    timeout_s: float = FFPROBE_TIMEOUT_S,
) -> int:
    """
    Count frames in a video using ffprobe.

    Prefers fast container metadata (``nb_frames``); falls back to packet counting
    with a hard timeout so the GUI cannot hang.
    """
    bin_path = ffprobe_bin or require_ffprobe()
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffprobe timed out after {timeout_s:.0f}s for {video_path.name}"
            ) from exc

    fast = _run(
        [
            bin_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
    )
    if fast.returncode == 0:
        n = _parse_ffprobe_int(fast.stdout, label="nb_frames")
        if n is not None:
            return n

    slow = _run(
        [
            bin_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
    )
    if slow.returncode != 0:
        err = (slow.stderr or slow.stdout or "").strip() or f"exit {slow.returncode}"
        raise RuntimeError(f"ffprobe failed for {video_path.name}: {err}")

    n = _parse_ffprobe_int(slow.stdout, label="nb_read_packets")
    if n is None:
        raise RuntimeError(f"ffprobe returned no frame count for {video_path.name}")
    return n


def _is_dlc_path(path: Path) -> bool:
    return "DLC" in path.name or "DLC" in str(path)


def _collect_mp4s(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*.mp4") if p.is_file() and not _is_dlc_path(p)
    )


def discover_block_videos(blocksync: Any) -> list[tuple[str, Path]]:
    """
    Return ``(source_label, path)`` for eye and arena videos under a block.

    Prefers BlockSync lists from ``handle_eye_videos`` / ``handle_arena_files``;
    falls back to scanning the block folder.
    """
    block_path = Path(blocksync.block_path)
    sources: list[tuple[str, Path]] = []

    le_videos = getattr(blocksync, "le_videos", None) or []
    re_videos = getattr(blocksync, "re_videos", None) or []
    arena_videos = getattr(blocksync, "arena_videos", None) or []

    if not le_videos:
        le_videos = [str(p) for p in _collect_mp4s(block_path / "eye_videos" / "LE")]
    if not re_videos:
        re_videos = [str(p) for p in _collect_mp4s(block_path / "eye_videos" / "RE")]
    if not arena_videos:
        arena_root = resolve_arena_path(block_path)
        arena_videos = [str(p) for p in list_arena_mp4s(arena_root)]

    if le_videos:
        sources.append(("Left eye", Path(le_videos[0])))
        for i, vid in enumerate(le_videos[1:], start=2):
            sources.append((f"Left eye ({i})", Path(vid)))
    if re_videos:
        sources.append(("Right eye", Path(re_videos[0])))
        for i, vid in enumerate(re_videos[1:], start=2):
            sources.append((f"Right eye ({i})", Path(vid)))

    if len(arena_videos) == 1:
        sources.append(("Arena", Path(arena_videos[0])))
    elif len(arena_videos) > 1:
        for i, vid in enumerate(arena_videos, start=1):
            sources.append((f"Arena ({i})", Path(vid)))
    else:
        # Surface non-mp4 arena clips so the reporter explains why Arena is missing.
        arena_root = resolve_arena_path(block_path)
        for i, vid in enumerate(list_convertible_arena_videos(arena_root), start=1):
            label = "Arena (needs conversion)" if i == 1 else f"Arena (needs conversion {i})"
            sources.append((label, vid))

    return sources


def rising_edge_counts_by_line(events_csv_path: Path | str) -> dict[int, int]:
    """Count rising edges (state==1) per TTL line from ``events.csv``."""
    df = pd.read_csv(events_csv_path)
    if df.empty or "state" not in df.columns or "line" not in df.columns:
        return {}
    df_on = df[df["state"] == 1]
    if df_on.empty:
        return {}
    counts = df_on.groupby("line").size()
    return {int(line): int(n) for line, n in counts.items()}


def closest_ttl_line(
    n_frames: int,
    rising_counts: dict[int, int],
) -> tuple[int | None, int | None, int | None]:
    """
    Return ``(line, ttl_n_rising, abs_delta)`` for the TTL line whose rising-edge
    count is closest to ``n_frames``. Ties prefer the lower line number.
    """
    if not rising_counts:
        return None, None, None
    best_line: int | None = None
    best_count: int | None = None
    best_delta: int | None = None
    for line, count in sorted(rising_counts.items()):
        delta = abs(int(count) - int(n_frames))
        if best_delta is None or delta < best_delta:
            best_line = int(line)
            best_count = int(count)
            best_delta = delta
    return best_line, best_count, best_delta


def build_video_reporter_rows(
    blocksync: Any,
    events_csv_path: Path | str,
    *,
    ffprobe_bin: str | None = None,
    count_fn=None,
    timeout_s: float = FFPROBE_TIMEOUT_S,
) -> list[VideoSourceReport]:
    """
    Build reporter rows: video frame counts vs closest TTL rising-edge counts.

    ``count_fn`` may override ffprobe for tests: ``callable(path) -> int``.
    """
    rising = rising_edge_counts_by_line(events_csv_path)
    rows: list[VideoSourceReport] = []
    for source, path in discover_block_videos(blocksync):
        try:
            if path.suffix.lower() != ".mp4" and "needs conversion" in source.lower():
                rows.append(
                    VideoSourceReport(
                        source=source,
                        path=path,
                        n_frames=None,
                        closest_ttl_line=None,
                        ttl_n_rising=None,
                        delta=None,
                        error=(
                            f"Not .mp4 ({path.suffix}). Run Prepare data and convert "
                            "arena videos to .mp4 first."
                        ),
                    )
                )
                continue
            if count_fn is not None:
                n_frames = int(count_fn(path))
            else:
                n_frames = count_frames_ffprobe(
                    path, ffprobe_bin=ffprobe_bin, timeout_s=timeout_s
                )
            line, ttl_n, delta = closest_ttl_line(n_frames, rising)
            rows.append(
                VideoSourceReport(
                    source=source,
                    path=path,
                    n_frames=n_frames,
                    closest_ttl_line=line,
                    ttl_n_rising=ttl_n,
                    delta=delta,
                )
            )
        except Exception as exc:
            rows.append(
                VideoSourceReport(
                    source=source,
                    path=path,
                    n_frames=None,
                    closest_ttl_line=None,
                    ttl_n_rising=None,
                    delta=None,
                    error=str(exc),
                )
            )
    return rows
