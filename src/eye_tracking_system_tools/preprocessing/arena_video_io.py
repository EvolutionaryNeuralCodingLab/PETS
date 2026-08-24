"""Arena video discovery and ffmpeg conversion helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

CONVERTIBLE_VIDEO_SUFFIXES = frozenset(
    {".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v", ".webm"}
)


class ArenaVideosNeedConversion(RuntimeError):
    """Raised when arena has convertible non-mp4 videos but no .mp4 yet."""

    def __init__(self, convertible: list[Path], arena_path: Path):
        self.convertible = list(convertible)
        self.arena_path = Path(arena_path)
        names = ", ".join(p.name for p in self.convertible[:8])
        more = "" if len(self.convertible) <= 8 else f" (+{len(self.convertible) - 8} more)"
        super().__init__(
            f"No .mp4 arena videos in {self.arena_path}, but found convertible "
            f"file(s): {names}{more}. Convert them to .mp4 to continue."
        )


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to convert arena videos to .mp4."
        )
    return path


def _is_dlc_path(path: Path) -> bool:
    return "DLC" in path.name or "DLC" in str(path)


def _video_files_in(root: Path, suffixes: frozenset[str]) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_file() or _is_dlc_path(p):
            continue
        if p.suffix.lower() in suffixes:
            out.append(p)
    return out


def resolve_arena_path(block_path: Path | str) -> Path:
    """
    Prefer ``arena_videos/videos/`` when it contains video files; otherwise
    ``arena_videos/``. An empty nested ``videos/`` directory no longer hides
    flat-layout clips (e.g. a lone ``.avi`` next to an empty ``videos/``).
    """
    block_path = Path(block_path)
    nested = block_path / "arena_videos" / "videos"
    flat = block_path / "arena_videos"
    video_suffixes = CONVERTIBLE_VIDEO_SUFFIXES | {".mp4"}
    if nested.is_dir() and _video_files_in(nested, video_suffixes):
        return nested
    if flat.is_dir():
        return flat
    return nested if nested.is_dir() else flat


def list_arena_mp4s(arena_path: Path | str) -> list[Path]:
    return _video_files_in(Path(arena_path), frozenset({".mp4"}))


def list_convertible_arena_videos(arena_path: Path | str) -> list[Path]:
    """Non-mp4 video files that can be converted (skips ones that already have .mp4)."""
    arena_path = Path(arena_path)
    existing_mp4_stems = {p.stem.lower() for p in list_arena_mp4s(arena_path)}
    out: list[Path] = []
    for src in _video_files_in(arena_path, CONVERTIBLE_VIDEO_SUFFIXES):
        if src.stem.lower() in existing_mp4_stems:
            continue
        sibling_mp4 = src.with_suffix(".mp4")
        if sibling_mp4.is_file():
            continue
        out.append(src)
    return out


def convert_video_to_mp4(
    src: Path | str,
    dst: Path | str | None = None,
    *,
    ffmpeg_bin: str | None = None,
    timeout_s: float | None = 3600.0,
) -> Path:
    """
    Convert ``src`` to H.264 ``.mp4`` beside it (or ``dst``) via ffmpeg.

    Progress is streamed to the process stderr/stdout (the terminal that launched
    the GUI). Uses a bounded timeout so a stuck encoder cannot hang forever.
    """
    import sys

    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"Video not found: {src}")
    dst = Path(dst) if dst is not None else src.with_suffix(".mp4")
    if dst.resolve() == src.resolve():
        raise ValueError(f"Refusing to overwrite source in place: {src}")

    bin_path = ffmpeg_bin or require_ffmpeg()
    cmd = [
        bin_path,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-stats",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    print(
        f"[arena convert] {src.name} → {dst.name}\n"
        f"[arena convert] ffmpeg: {' '.join(cmd)}",
        flush=True,
    )
    try:
        # Inherit stdout/stderr so ffmpeg's frame/time progress appears in the
        # terminal supporting the GUI (do not capture_output).
        result = subprocess.run(
            cmd,
            check=False,
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        if dst.is_file():
            try:
                dst.unlink()
            except OSError:
                pass
        print(
            f"[arena convert] TIMEOUT converting {src.name} (>{timeout_s}s)",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(
            f"ffmpeg timed out converting {src.name} (>{timeout_s}s). "
            "Check the file or convert manually."
        ) from exc

    if result.returncode != 0 or not dst.is_file():
        print(
            f"[arena convert] FAILED {src.name} (exit {result.returncode})",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError(
            f"ffmpeg failed converting {src.name} (exit {result.returncode}). "
            "See terminal output above for details."
        )
    print(f"[arena convert] done: {dst.name}", flush=True)
    return dst


def convert_arena_videos_to_mp4(
    arena_path: Path | str,
    *,
    ffmpeg_bin: str | None = None,
    timeout_s: float | None = 3600.0,
) -> list[Path]:
    """Convert all convertible arena clips; return paths of created/existing mp4s."""
    arena_path = Path(arena_path)
    to_convert = list_convertible_arena_videos(arena_path)
    created: list[Path] = []
    n = len(to_convert)
    for i, src in enumerate(to_convert, start=1):
        print(f"[arena convert] ({i}/{n}) {src.name}", flush=True)
        created.append(
            convert_video_to_mp4(
                src,
                ffmpeg_bin=ffmpeg_bin,
                timeout_s=timeout_s,
            )
        )
    return created
