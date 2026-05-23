"""Smoke tests for Block Annotator (no GUI)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from eye_tracking_system_tools.annotation.block_annotator.config_io import (
    ensure_config_template,
    load_config,
)
from eye_tracking_system_tools.annotation.block_annotator.models import (
    AnnotatorConfig,
    compute_ms_axis,
)
from eye_tracking_system_tools.annotation.block_annotator.persistence import (
    annotation_filename,
    save_annotations,
)
from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
    infer_metadata,
)
from eye_tracking_system_tools.annotation.block_annotator.oe_streams import (
    OEStream,
    _as_sequence,
    list_oe_streams,
)


def test_video_reader_sequential_advance_without_seek(monkeypatch):
    from eye_tracking_system_tools.annotation.block_annotator import video_widget as vw

    calls = {"set": 0, "read": 0}

    class FakeCap:
        def __init__(self):
            self._pos = 0

        def set(self, prop, value):
            calls["set"] += 1
            self._pos = int(value)

        def read(self):
            calls["read"] += 1
            idx = self._pos
            self._pos += 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8) + idx

        def get(self, prop):
            return 100

        def release(self):
            pass

    fake = FakeCap()
    reader = vw.VideoReader(None)
    reader.path = Path("dummy.mp4")
    reader._cap = fake
    reader._nframes = 100

    reader.read_frame(10)
    assert calls["set"] == 1
    calls["read"] = 0
    calls["set"] = 0
    reader.read_frame(11)
    assert calls["set"] == 0
    assert calls["read"] == 1
    calls["read"] = 0
    reader.read_frame(9)
    assert calls["set"] == 1


def test_apply_display_transforms_flip_then_annotate_order():
    from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
        apply_display_transforms,
    )

    arr = np.zeros((100, 80, 3), dtype=np.uint8)
    arr[10:20, 30:40] = (255, 0, 0)
    out = apply_display_transforms(
        arr,
        show_annotations=False,
        flip_vertical=True,
    )
    assert out.shape == (100, 80, 3)
    # red blob should move to bottom after vertical flip
    assert out[89, 30, 0] == 255


def test_safe_frame_rejects_int64_sentinel():
    from eye_tracking_system_tools.annotation.block_annotator.models import _safe_frame

    assert _safe_frame(-9.223372e18) is None
    assert _safe_frame(float("nan")) is None
    assert _safe_frame(120) == 120


def test_compute_ms_axis_from_arena_ttl():
    df = pd.DataFrame({"Arena_TTL": [0.0, 30000.0, 60000.0]})
    ms = compute_ms_axis(df, 30000.0)
    np.testing.assert_allclose(ms, [0.0, 1000.0, 2000.0])


def test_compute_ms_axis_uses_existing_column():
    df = pd.DataFrame(
        {"Arena_TTL": [0.0, 30000.0], "ms_axis": [1.0, 2.0]}
    )
    ms = compute_ms_axis(df, 30000.0)
    np.testing.assert_allclose(ms, [1.0, 2.0])


def test_add_event_type_dedup():
    cfg = AnnotatorConfig(event_types=["saccade"])
    assert cfg.add_event_type("blink") is True
    assert cfg.event_types == ["saccade", "blink"]
    assert cfg.add_event_type("Blink") is False
    assert cfg.add_event_type("  ") is False


def test_config_template_created(tmp_path):
    path = ensure_config_template(tmp_path)
    assert path.exists()
    cfg = load_config(None, tmp_path)
    assert "saccade" in cfg.event_types
    assert cfg.default_range_half_width_ms == 100.0


def test_list_oe_streams_with_numpy_channel_numbers():
    class FakeRec:
        channelNumbers = np.array([1, 2, 3])
        analogChannelNumbers = np.array([4])
        accel_files = ["AUX1.continuous", "AUX2.continuous"]

    streams = list_oe_streams(FakeRec())
    assert len(streams) == 6
    assert streams[0] == OEStream("hs", 1, "HS ch1")
    assert streams[3] == OEStream("adc", 4, "ADC ch4")
    assert streams[4].kind == "aux"


def test_as_sequence_none_and_array():
    assert _as_sequence(None) == []
    np.testing.assert_array_equal(_as_sequence(np.array([5])), [5])


def test_infer_metadata_with_date(tmp_path):
    block = tmp_path / "PV_99" / "2024_01_15" / "block_007"
    block.mkdir(parents=True)
    animal, date, num, root = infer_metadata(block)
    assert animal == "PV_99"
    assert date == "2024_01_15"
    assert num == "007"
    assert root == tmp_path


def test_save_annotations_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "Arena_TTL": [0.0, 30000.0],
            "Arena_frame": [0, 1],
            "L_eye_frame": [0, 1],
            "R_eye_frame": [0, 1],
            "L_values": [0, 0],
            "R_values": [0, 0],
        }
    )
    from eye_tracking_system_tools.annotation.block_annotator.models import (
        BlockSession,
        AnnotationEvent,
    )

    cfg = AnnotatorConfig()
    session = BlockSession(
        animal_call="PV_test",
        experiment_date="2024_01_01",
        block_num="001",
        block_path=tmp_path / "block",
        output_folder=tmp_path / "out",
        config=cfg,
        final_sync_df=df,
        ms_axis=compute_ms_axis(df, 30000.0),
        sample_rate_hz=30000.0,
        arena_videos=[],
        le_videos=[],
        re_videos=[],
    )
    events = [
        AnnotationEvent(
            event_type="saccade",
            timepoint_ms=500.0,
            start_ms=400.0,
            end_ms=600.0,
        )
    ]
    out = save_annotations(tmp_path / "out", session, events)
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == 1
    assert len(data["events"]) == 1
    assert annotation_filename(session).endswith("_annotations.json")
