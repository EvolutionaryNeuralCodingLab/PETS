"""Tests for arena video discovery and conversion helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from eye_tracking_system_tools.preprocessing.arena_video_io import (
    ArenaVideosNeedConversion,
    convert_video_to_mp4,
    list_convertible_arena_videos,
    list_arena_mp4s,
    resolve_arena_path,
)
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync


def test_resolve_arena_path_prefers_nested_only_when_it_has_videos(tmp_path: Path):
    flat = tmp_path / "arena_videos"
    nested = flat / "videos"
    nested.mkdir(parents=True)
    (flat / "clip.avi").write_bytes(b"x")
    # Empty nested videos/ must not hide flat .avi
    assert resolve_arena_path(tmp_path) == flat

    (nested / "nested.mp4").write_bytes(b"x")
    assert resolve_arena_path(tmp_path) == nested


def test_list_convertible_skips_when_mp4_sibling_exists(tmp_path: Path):
    arena = tmp_path / "arena_videos"
    arena.mkdir()
    (arena / "a.avi").write_bytes(b"x")
    (arena / "b.avi").write_bytes(b"x")
    (arena / "b.mp4").write_bytes(b"x")
    convertible = list_convertible_arena_videos(arena)
    assert [p.name for p in convertible] == ["a.avi"]
    assert list_arena_mp4s(arena)[0].name == "b.mp4"


def test_convert_video_to_mp4_invokes_ffmpeg(tmp_path: Path):
    src = tmp_path / "clip.avi"
    src.write_bytes(b"fake")
    dst = tmp_path / "clip.mp4"

    def _fake_run(cmd, **kwargs):
        # Progress must go to the terminal — never capture_output.
        assert kwargs.get("capture_output") in (False, None)
        assert kwargs.get("stdout") is None
        assert kwargs.get("stderr") is None
        Path(cmd[-1]).write_bytes(b"mp4")
        return type("R", (), {"returncode": 0, "stdout": None, "stderr": None})()

    with patch(
        "eye_tracking_system_tools.preprocessing.arena_video_io.require_ffmpeg",
        return_value="ffmpeg",
    ), patch(
        "eye_tracking_system_tools.preprocessing.arena_video_io.subprocess.run",
        side_effect=_fake_run,
    ):
        out = convert_video_to_mp4(src, dst)
    assert out == dst
    assert dst.is_file()


def test_handle_arena_files_raises_need_conversion(tmp_path: Path, monkeypatch):
    block = tmp_path / "block_001"
    (block / "arena_videos").mkdir(parents=True)
    (block / "eye_videos" / "LE").mkdir(parents=True)
    (block / "eye_videos" / "RE").mkdir(parents=True)
    (block / "oe_files" / "exp").mkdir(parents=True)
    (block / "arena_videos" / "cam.avi").write_bytes(b"avi")

    # Minimal BlockSync without full OE/eye setup: call handle_arena_files on a stub
    class _B:
        block_path = block
        arena_path = block / "arena_videos"

    b = _B()
    # Bind real method
    b.handle_arena_files = BlockSync.handle_arena_files.__get__(b, _B)

    with pytest.raises(ArenaVideosNeedConversion) as ei:
        b.handle_arena_files(convert_non_mp4=False)
    assert ei.value.convertible[0].name == "cam.avi"

    def _fake_convert(arena_path, **kwargs):
        mp4 = Path(arena_path) / "cam.mp4"
        mp4.write_bytes(b"mp4")
        return [mp4]

    monkeypatch.setattr(
        "eye_tracking_system_tools.preprocessing.arena_video_io.convert_arena_videos_to_mp4",
        _fake_convert,
    )
    b.handle_arena_files(convert_non_mp4=True)
    assert len(b.arena_videos) == 1
    assert b.arena_videos[0].suffix == ".mp4"


def test_handle_arena_files_raises_when_no_videos_at_all(tmp_path: Path):
    block = tmp_path / "block_001"
    (block / "arena_videos").mkdir(parents=True)
    (block / "eye_videos" / "LE").mkdir(parents=True)
    (block / "eye_videos" / "RE").mkdir(parents=True)

    class _B:
        block_path = block
        arena_path = block / "arena_videos"

    b = _B()
    b.handle_arena_files = BlockSync.handle_arena_files.__get__(b, _B)
    with pytest.raises(RuntimeError, match="No .mp4 arena videos"):
        b.handle_arena_files(convert_non_mp4=False)
