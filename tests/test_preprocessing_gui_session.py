"""Tests for shared BlockSync session and GuiState block management."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.block_session import (
    BlockSyncSession,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
)


def _handle(block_path: Path, num: str = "011") -> BlockHandle:
    return BlockHandle(
        animal_call="PV_228",
        experiment_date="2026_06_15",
        block_num=num,
        block_path=block_path,
        path_to_animal_folder=block_path.parent.parent.parent,
    )


def test_blocksync_session_caches_by_path(tmp_path, monkeypatch):
    block_path = tmp_path / "block_011"
    block_path.mkdir(parents=True)
    handle = _handle(block_path)

    created = []

    def fake_blocksync(*args, **kwargs):
        bs = MagicMock()
        bs.le_videos = None
        bs.re_videos = None
        created.append(bs)
        return bs

    monkeypatch.setattr(
        "eye_tracking_system_tools.annotation.preprocessing_gui.block_session.BlockSync",
        fake_blocksync,
    )

    session = BlockSyncSession()
    a = session.get(handle)
    b = session.get(handle)
    assert a is b
    assert len(created) == 1


def test_blocksync_session_invalidate_forces_new(tmp_path, monkeypatch):
    block_path = tmp_path / "block_011"
    block_path.mkdir(parents=True)
    handle = _handle(block_path)
    counter = {"n": 0}

    def fake_blocksync(*args, **kwargs):
        counter["n"] += 1
        bs = MagicMock()
        bs.le_videos = None
        bs.re_videos = None
        return bs

    monkeypatch.setattr(
        "eye_tracking_system_tools.annotation.preprocessing_gui.block_session.BlockSync",
        fake_blocksync,
    )

    session = BlockSyncSession()
    session.get(handle)
    session.invalidate(handle)
    session.get(handle)
    assert counter["n"] == 2


def test_ensure_eye_videos_calls_once():
    bs = SimpleNamespace(
        le_videos=None,
        re_videos=None,
        _gui_eye_videos_prepared=False,
    )
    calls = {"n": 0}

    def handle_eye_videos():
        calls["n"] += 1
        bs.le_videos = ["le.mp4"]
        bs.re_videos = ["re.mp4"]

    bs.handle_eye_videos = handle_eye_videos
    BlockSyncSession.ensure_eye_videos(bs)
    BlockSyncSession.ensure_eye_videos(bs)
    assert calls["n"] == 1


def test_gui_state_add_blocks_dedup(tmp_path):
    p1 = tmp_path / "block_011"
    p2 = tmp_path / "block_012"
    p1.mkdir()
    p2.mkdir()
    state = GuiState()
    h1 = _handle(p1, "011")
    h2 = _handle(p2, "012")
    assert state.add_blocks([h1, h2]) == 2
    assert state.add_blocks([h1]) == 0
    assert len(state.blocks) == 2


def test_gui_state_remove_block_at_clamps_index(tmp_path):
    paths = [tmp_path / f"block_{n:03d}" for n in (11, 12, 13)]
    for p in paths:
        p.mkdir()
    state = GuiState(
        blocks=[_handle(p, p.name.split("_")[1]) for p in paths],
        current_index=2,
    )
    removed = state.remove_block_at(2)
    assert removed is not None
    assert state.current_index == 1
    assert len(state.blocks) == 2

    state.current_index = 0
    state.remove_block_at(0)
    assert state.current_index == 0
    assert len(state.blocks) == 1

    state.remove_block_at(0)
    assert state.blocks == []
    assert state.current_index == 0


def test_session_release_drops_cache(tmp_path, monkeypatch):
    block_path = tmp_path / "block_011"
    block_path.mkdir(parents=True)
    handle = _handle(block_path)
    instances: list[object] = []

    def fake_blocksync(*args, **kwargs):
        bs = SimpleNamespace(
            le_videos=None,
            re_videos=None,
            _gui_eye_videos_prepared=False,
        )
        instances.append(bs)
        return bs

    monkeypatch.setattr(
        "eye_tracking_system_tools.annotation.preprocessing_gui.block_session.BlockSync",
        fake_blocksync,
    )
    session = BlockSyncSession()
    first = session.get(handle)
    session.release(handle)
    second = session.get(handle)
    assert first is not second
    assert len(instances) == 2
