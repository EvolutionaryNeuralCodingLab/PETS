"""Tests for manual LED_driver replacement helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.manual_led_dialog import (
    BLINK_INTERVAL_S,
    BLINK_OFF_DURATION_S,
    apply_led_manual_replacement_to_blocksync,
    build_led_replacement_payload,
    frames_to_oe_samples,
    generate_blink_frame_series,
    inject_led_into_oe_events,
    synthesize_led_fall_samples,
    try_apply_saved_led_replacement,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.manual_ttl_dialog import (
    build_manual_ttl_payload,
    save_ttl_manual_sidecar,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.ttl_mapping import (
    led_driver_is_ready,
    load_led_manual_replacement,
    load_ttl_sidecar,
    oe_events_have_led_driver,
)


def test_generate_blink_frame_series_sixty_second_spacing():
    # 5 minutes of video at 60 fps → frames 0..17999 (exactly 300 s).
    # Candidates every 3600 frames; the t=300 s candidate (frame 18000) is past end.
    n_frames = 60 * 60 * 5
    frames = generate_blink_frame_series(0, n_frames, fps=60.0, interval_s=60.0)
    assert list(frames) == [0, 3600, 7200, 10800, 14400]
    assert np.all(np.diff(frames) == 3600)

    offset = generate_blink_frame_series(100, n_frames, fps=60.0, interval_s=60.0)
    assert list(offset) == [100, 3700, 7300, 10900, 14500]


def test_synthesize_led_fall_samples_uses_34ms():
    fs = 30_000.0
    on = np.array([1_000_000, 2_000_000], dtype=np.int64)
    fall = synthesize_led_fall_samples(on, fs)
    expected_delta = int(round(BLINK_OFF_DURATION_S * fs))
    assert list(fall) == [1_000_000 - expected_delta, 2_000_000 - expected_delta]


def test_frames_to_oe_samples_stops_past_ttl():
    eye_ttl = np.arange(10, dtype=np.int64) * 1000 + 500
    samples = frames_to_oe_samples(np.array([0, 5, 9, 10, 11]), eye_ttl)
    assert list(samples) == [500, 5500, 9500]


def test_build_led_replacement_payload_and_inject(tmp_path: Path):
    fs = 30_000.0
    eye_ttl = np.arange(0, 60 * 60 * 3, dtype=np.int64) * int(fs / 60) + 10_000
    n_frames = len(eye_ttl)
    payload = build_led_replacement_payload(
        first_frame=120,
        n_frames=n_frames,
        eye_ttl=eye_ttl,
        sample_rate=fs,
        fps=60.0,
        interval_s=BLINK_INTERVAL_S,
    )
    assert payload["first_frame"] == 120
    assert len(payload["on_samples"]) >= 2
    assert len(payload["fall_samples"]) == len(payload["on_samples"])
    assert payload["on_samples"][0] == int(eye_ttl[120])

    oe = pd.DataFrame(
        {
            "L_eye_TTL": eye_ttl[:100],
            "R_eye_TTL": eye_ttl[:100] + 1,
        }
    )
    out = inject_led_into_oe_events(oe, payload["on_samples"], payload["fall_samples"])
    assert oe_events_have_led_driver(out)
    assert int(out["LED_driver"].dropna().iloc[0]) == payload["on_samples"][0]


def test_apply_and_reload_led_sidecar(tmp_path: Path):
    oe_dirname = "exp1"
    oe_dir = tmp_path / "oe_files" / oe_dirname
    oe_dir.mkdir(parents=True)
    eye_ttl = np.arange(5000, dtype=np.int64) * 500 + 1000
    oe = pd.DataFrame({"L_eye_TTL": eye_ttl, "R_eye_TTL": eye_ttl + 3})
    blocksync = SimpleNamespace(
        block_path=tmp_path,
        oe_dirname=oe_dirname,
        sample_rate=30_000.0,
        oe_events=oe,
        block_num="001",
    )
    payload = build_led_replacement_payload(
        10, len(eye_ttl), eye_ttl, 30_000.0, fps=60.0
    )
    apply_led_manual_replacement_to_blocksync(blocksync, payload)
    assert led_driver_is_ready(blocksync)
    loaded = load_led_manual_replacement(tmp_path, oe_dirname)
    assert loaded is not None
    assert loaded["first_frame"] == 10
    assert loaded["on_samples"] == payload["on_samples"]

    # Clear in-memory LED and re-apply from sidecar
    blocksync.oe_events = oe.copy()
    assert not led_driver_is_ready(blocksync)
    assert try_apply_saved_led_replacement(blocksync) is True
    assert led_driver_is_ready(blocksync)


def test_led_driver_is_ready_false_without_columns():
    blocksync = SimpleNamespace(oe_events=pd.DataFrame({"L_eye_TTL": [1, 2, 3]}))
    assert led_driver_is_ready(blocksync) is False
    assert oe_events_have_led_driver(None) is False


def test_build_manual_ttl_payload_led_missing(ttl_events_fixture):
    blocksync, events_csv = ttl_events_fixture
    manual_line_map, arena_window = build_manual_ttl_payload(
        blocksync,
        events_csv,
        arena_line=3,
        l_eye_line=1,
        r_eye_line=2,
        led_driver_missing=True,
        window_mode="i",
        start_index=0,
        end_index=-1,
    )
    assert "LED_driver" not in manual_line_map
    assert manual_line_map["Arena_TTL"] == 3
    assert arena_window["arena_start_index"] == 0

    path = save_ttl_manual_sidecar(
        events_csv,
        blocksync,
        manual_line_map,
        arena_window,
        led_driver_missing=True,
    )
    loaded = load_ttl_sidecar(blocksync.block_path, events_csv.parent.name)
    # events_csv is directly under tmp; sidecar_path needs oe_files layout
    assert path.is_file()
    assert loaded is None  # wrong oe layout for load_ttl_sidecar helper

    # Proper OE layout
    oe_dir = blocksync.block_path / "oe_files" / "exp1"
    oe_dir.mkdir(parents=True)
    events2 = oe_dir / "events.csv"
    events2.write_text(events_csv.read_text(encoding="utf-8"), encoding="utf-8")
    save_ttl_manual_sidecar(
        events2,
        blocksync,
        manual_line_map,
        arena_window,
        led_driver_missing=True,
    )
    loaded = load_ttl_sidecar(blocksync.block_path, "exp1")
    assert loaded is not None
    assert loaded["led_driver_missing"] is True
    assert "LED_driver" not in loaded["manual_line_map"]


@pytest.fixture
def ttl_events_fixture(tmp_path: Path):
    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync

    events_csv = tmp_path / "events.csv"
    arena_samples = [100_000 + i * 10_000 for i in range(20)]
    rows = []
    for s in arena_samples:
        rows.append({"line": 3, "state": 1, "sample_number": s})
    for i in range(50):
        rows.append({"line": 1, "state": 1, "sample_number": 200_000 + i * 5_000})
    for i in range(60):
        rows.append({"line": 2, "state": 1, "sample_number": 300_000 + i * 5_000})
    pd.DataFrame(rows).to_csv(events_csv, index=False)
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
    return blocksync, events_csv
