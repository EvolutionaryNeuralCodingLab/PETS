"""Phase 3 tests — manual TTL dialog and Qt ROI picker."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.manual_ttl_dialog import (
    ManualTtlDialog,
    build_manual_ttl_payload,
    compute_arena_window,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.qt_roi_picker import (
    QtRoiPickerDialog,
    jitter_report_needs_computation,
    normalize_correlation_roi,
)
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync


def _write_synthetic_events_csv(path: Path) -> dict:
    """Write a minimal events.csv and return the golden manual payload."""
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


def test_manual_ttl_dialog_payload_shape(qapp_session, ttl_fixture):
    blocksync, events_csv, golden = ttl_fixture
    dlg = ManualTtlDialog(blocksync, events_csv)
    dlg.set_mapping_for_test(
        arena_line=3,
        l_eye_line=1,
        r_eye_line=2,
        led_driver_line=4,
        window_mode="i",
        start_index="0",
        end_index="-1",
    )
    manual_line_map, arena_window = dlg.build_payload()
    assert manual_line_map == golden["manual_line_map"]
    assert arena_window == golden["arena_window"]


def test_build_manual_ttl_payload_helper(ttl_fixture):
    blocksync, events_csv, golden = ttl_fixture
    manual_line_map, arena_window = build_manual_ttl_payload(
        blocksync,
        events_csv,
        arena_line=3,
        l_eye_line=1,
        r_eye_line=2,
        led_driver_line=4,
        window_mode="i",
        start_index=0,
        end_index=-1,
    )
    assert manual_line_map == golden["manual_line_map"]
    assert arena_window == golden["arena_window"]


def test_build_manual_ttl_payload_with_extra_roles(ttl_fixture):
    blocksync, events_csv, golden = ttl_fixture
    manual_line_map, arena_window = build_manual_ttl_payload(
        blocksync,
        events_csv,
        arena_line=3,
        l_eye_line=1,
        r_eye_line=2,
        led_driver_line=4,
        extra_roles={"Stim_TTL": 5},
        window_mode="i",
        start_index=0,
        end_index=-1,
    )
    assert manual_line_map["Stim_TTL"] == 5
    assert arena_window == golden["arena_window"]


def test_build_manual_ttl_payload_rejects_duplicate_lines(ttl_fixture):
    blocksync, events_csv, _golden = ttl_fixture
    with pytest.raises(ValueError, match="distinct line"):
        build_manual_ttl_payload(
            blocksync,
            events_csv,
            arena_line=3,
            l_eye_line=3,
            r_eye_line=2,
            led_driver_line=4,
        )


def test_compute_arena_window_sample_mode(ttl_fixture):
    blocksync, events_csv, _golden = ttl_fixture
    arena_window = compute_arena_window(
        blocksync,
        events_csv,
        arena_line=3,
        mode="s",
        start_sample=150_000,
        end_sample=250_000,
    )
    assert arena_window["arena_start_timestamp"] == 150_000
    assert arena_window["arena_end_timestamp"] == 250_000
    assert arena_window["arena_start_index"] == 5


def test_qt_roi_picker_returns_tuple(qapp_session):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[20:60, 10:40] = 255
    dlg = QtRoiPickerDialog(frame, title="Test ROI")
    dlg.set_selection_rect(10, 20, 30, 40)
    roi = dlg.selected_roi()
    assert roi == (10, 20, 30, 40)


def test_normalize_correlation_roi_forces_odd_dimensions():
    assert normalize_correlation_roi((10, 20, 30, 40)) == [10, 20, 31, 41]
    assert normalize_correlation_roi((0, 0, 5, 7)) == [0, 0, 5, 7]


def test_jitter_report_needs_computation(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    blocksync = SimpleNamespace(analysis_path=analysis)
    assert jitter_report_needs_computation(blocksync) is True
    (analysis / "jitter_report_dict.pkl").write_bytes(b"stub")
    assert jitter_report_needs_computation(blocksync) is False
    assert jitter_report_needs_computation(blocksync, overwrite=True) is True
