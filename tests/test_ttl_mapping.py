"""Tests for TTL channel mapping reuse helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.block_session import (
    BlockSyncSession,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.manual_ttl_dialog import (
    ManualTtlDialog,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.ttl_mapping import (
    DEFAULT_CHANNELDICT,
    channeldict_from_line_map,
    is_default_channeldict,
    line_map_from_channeldict,
    list_mapping_sources,
    load_ttl_sidecar,
)
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync


def _write_synthetic_events_csv(path: Path) -> dict:
    arena_samples = [100_000 + i * 10_000 for i in range(20)]
    rows = []
    for s in arena_samples:
        rows.append({"line": 3, "state": 1, "sample_number": s})
    for i in range(50):
        rows.append({"line": 1, "state": 1, "sample_number": 200_000 + i * 5_000})
    for i in range(60):
        rows.append({"line": 2, "state": 1, "sample_number": 300_000 + i * 5_000})
    for i in range(40):
        rows.append({"line": 4, "state": 1, "sample_number": 400_000 + i * 8_000})
    pd.DataFrame(rows).to_csv(path, index=False)
    return {
        "manual_line_map": {
            "Arena_TTL": 3,
            "L_eye_TTL": 1,
            "R_eye_TTL": 2,
            "LED_driver": 4,
        },
        "arena_window": {
            "arena_start_timestamp": int(arena_samples[0]),
            "arena_end_timestamp": int(arena_samples[-1]),
            "arena_start_index": 0,
        },
    }


@pytest.fixture
def ttl_fixture(tmp_path: Path):
    events_csv = tmp_path / "events.csv"
    golden = _write_synthetic_events_csv(events_csv)
    blocksync = SimpleNamespace(
        block_num="015",
        block_path=tmp_path,
        sample_rate=30_000,
        le_frame_count=50,
        re_frame_count=60,
        le_videos=[],
        re_videos=[],
    )
    blocksync._summarize_ttl_lines_from_events_csv = (
        lambda path: BlockSync._summarize_ttl_lines_from_events_csv(blocksync, path)
    )
    blocksync._plot_ttl_raster = lambda *args, **kwargs: None
    return blocksync, events_csv, golden


def _block_handle(tmp_path: Path, *, block_num: str = "001") -> BlockHandle:
    block_path = tmp_path / f"block_{block_num}"
    block_path.mkdir(parents=True)
    return BlockHandle(
        animal_call="mouse1",
        experiment_date="2024-01-01",
        block_num=block_num,
        block_path=block_path,
        path_to_animal_folder=tmp_path,
    )


def test_channeldict_line_map_round_trip():
    line_map = {
        "Arena_TTL": 1,
        "L_eye_TTL": 5,
        "R_eye_TTL": 8,
        "LED_driver": 4,
        "Logical ON/OFF": 7,
    }
    channeldict = channeldict_from_line_map(line_map)
    assert line_map_from_channeldict(channeldict) == line_map


def test_is_default_channeldict():
    assert is_default_channeldict(DEFAULT_CHANNELDICT) is True
    assert is_default_channeldict({1: "Arena_TTL", 2: "L_eye_TTL"}) is False


def test_load_ttl_sidecar(tmp_path: Path):
    block = _block_handle(tmp_path)
    oe_dir = block.block_path / "oe_files" / "exp1"
    oe_dir.mkdir(parents=True)
    payload = {
        "manual_line_map": {"Arena_TTL": 3, "L_eye_TTL": 1},
        "arena_window": {
            "arena_start_timestamp": 100,
            "arena_end_timestamp": 200,
            "arena_start_index": 0,
        },
    }
    (oe_dir / "ttl_manual_mapping.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_ttl_sidecar(block.block_path, "exp1")
    assert loaded is not None
    assert loaded["manual_line_map"]["Arena_TTL"] == 3
    assert loaded["arena_window"]["arena_start_index"] == 0


def test_block_sync_session_resolve_channeldict():
    session = BlockSyncSession()
    block = BlockHandle(
        animal_call="m",
        experiment_date=None,
        block_num="1",
        block_path=Path("/tmp/block_1"),
        path_to_animal_folder=Path("/tmp"),
    )
    custom = {1: "Arena_TTL", 2: "L_eye_TTL"}
    block.channeldict = custom
    assert session.resolve_channeldict(block) == custom

    block.channeldict = None
    session.remember_channeldict({3: "Arena_TTL", 4: "R_eye_TTL"})
    assert session.resolve_channeldict(block) == {3: "Arena_TTL", 4: "R_eye_TTL"}

    session.last_channeldict = None
    assert session.resolve_channeldict(block) is None


def test_list_mapping_sources_from_sidecar(tmp_path: Path):
    source = _block_handle(tmp_path, block_num="001")
    target = _block_handle(tmp_path, block_num="002")
    oe_dir = source.block_path / "oe_files" / "exp1"
    oe_dir.mkdir(parents=True)
    sidecar = {
        "manual_line_map": {
            "Arena_TTL": 3,
            "L_eye_TTL": 1,
            "R_eye_TTL": 2,
            "LED_driver": 4,
        },
        "arena_window": {
            "arena_start_timestamp": 1,
            "arena_end_timestamp": 2,
            "arena_start_index": 0,
        },
    }
    (oe_dir / "ttl_manual_mapping.json").write_text(json.dumps(sidecar), encoding="utf-8")

    session = BlockSyncSession()
    sources = list_mapping_sources(session, [source, target], exclude=target)
    assert len(sources) == 1
    assert sources[0][0] == source.display_label
    assert sources[0][1]["Arena_TTL"] == 3


def test_manual_ttl_dialog_apply_line_map(qapp_session, ttl_fixture):
    blocksync, events_csv, golden = ttl_fixture
    dlg = ManualTtlDialog(blocksync, events_csv)
    dlg.apply_line_map(golden["manual_line_map"])
    assert dlg._arena_line.value() == golden["manual_line_map"]["Arena_TTL"]
    assert dlg._l_eye_line.value() == golden["manual_line_map"]["L_eye_TTL"]
    assert dlg._r_eye_line.value() == golden["manual_line_map"]["R_eye_TTL"]
    assert dlg._led_driver_line.value() == golden["manual_line_map"]["LED_driver"]
