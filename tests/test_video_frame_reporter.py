"""Tests for ffprobe-based video frame reporter helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.video_frame_reporter import (
    build_video_reporter_rows,
    closest_ttl_line,
    discover_block_videos,
    rising_edge_counts_by_line,
)


def test_closest_ttl_line_picks_nearest_and_ties_lower_line():
    rising = {1: 100, 2: 105, 3: 200}
    line, count, delta = closest_ttl_line(103, rising)
    assert line == 2
    assert count == 105
    assert delta == 2

    # Exact tie on delta → lower line number
    rising_tie = {5: 50, 2: 60}
    line, count, delta = closest_ttl_line(55, rising_tie)
    assert line == 2
    assert count == 60
    assert delta == 5


def test_closest_ttl_line_empty():
    assert closest_ttl_line(10, {}) == (None, None, None)


def test_rising_edge_counts_by_line(tmp_path: Path):
    events = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"line": 1, "state": 1, "sample_number": 10},
            {"line": 1, "state": 0, "sample_number": 11},
            {"line": 1, "state": 1, "sample_number": 20},
            {"line": 2, "state": 1, "sample_number": 30},
        ]
    ).to_csv(events, index=False)
    assert rising_edge_counts_by_line(events) == {1: 2, 2: 1}


def test_discover_block_videos_scans_folders(tmp_path: Path):
    le = tmp_path / "eye_videos" / "LE" / "vid"
    re = tmp_path / "eye_videos" / "RE" / "vid"
    arena = tmp_path / "arena_videos"
    le.mkdir(parents=True)
    re.mkdir(parents=True)
    arena.mkdir(parents=True)
    (le / "left.mp4").write_bytes(b"x")
    (re / "right.mp4").write_bytes(b"x")
    (arena / "arena.mp4").write_bytes(b"x")
    (le / "leftDLC_resnet.mp4").write_bytes(b"x")  # skipped

    blocksync = SimpleNamespace(
        block_path=tmp_path,
        le_videos=None,
        re_videos=None,
        arena_videos=None,
    )
    sources = discover_block_videos(blocksync)
    labels = [s[0] for s in sources]
    assert "Left eye" in labels
    assert "Right eye" in labels
    assert "Arena" in labels
    assert all("DLC" not in s[1].name for s in sources)


def test_discover_block_videos_reports_convertible_arena(tmp_path: Path):
    arena = tmp_path / "arena_videos"
    arena.mkdir()
    avi = arena / "cam.avi"
    avi.write_bytes(b"x")
    blocksync = SimpleNamespace(
        block_path=tmp_path,
        le_videos=[],
        re_videos=[],
        arena_videos=[],
    )
    sources = discover_block_videos(blocksync)
    assert len(sources) == 1
    assert "needs conversion" in sources[0][0].lower()
    assert sources[0][1] == avi


def test_build_video_reporter_rows_marks_convertible_without_ffprobe(tmp_path: Path):
    arena = tmp_path / "arena_videos"
    arena.mkdir()
    (arena / "cam.avi").write_bytes(b"x")
    events = tmp_path / "events.csv"
    pd.DataFrame([{"line": 1, "state": 1, "sample_number": 10}]).to_csv(events, index=False)
    blocksync = SimpleNamespace(
        block_path=tmp_path,
        le_videos=[],
        re_videos=[],
        arena_videos=[],
    )
    rows = build_video_reporter_rows(blocksync, events, count_fn=lambda _p: 1)
    assert len(rows) == 1
    assert rows[0].error is not None
    assert "Not .mp4" in rows[0].error


def test_build_video_reporter_rows_with_mock_count(tmp_path: Path):
    le = tmp_path / "eye_videos" / "LE" / "v"
    le.mkdir(parents=True)
    vid = le / "left.mp4"
    vid.write_bytes(b"x")

    events = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"line": 5, "state": 1, "sample_number": i * 100}
            for i in range(50)
        ]
        + [
            {"line": 8, "state": 1, "sample_number": i * 100}
            for i in range(60)
        ]
    ).to_csv(events, index=False)

    blocksync = SimpleNamespace(
        block_path=tmp_path,
        le_videos=[str(vid)],
        re_videos=[],
        arena_videos=[],
    )
    rows = build_video_reporter_rows(
        blocksync,
        events,
        count_fn=lambda _p: 50,
    )
    assert len(rows) == 1
    assert rows[0].source == "Left eye"
    assert rows[0].n_frames == 50
    assert rows[0].closest_ttl_line == 5
    assert rows[0].ttl_n_rising == 50
    assert rows[0].delta == 0
