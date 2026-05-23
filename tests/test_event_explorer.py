"""Tests for Event Explorer (no GUI)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eye_tracking_system_tools.annotation.event_explorer.catalog import (
    build_catalog,
    discover_annotation_files,
    load_events_from_file,
)
from eye_tracking_system_tools.annotation.event_explorer.eye_csv_resolver import (
    detect_pupil_column,
    glob_eye_candidates,
    pick_newest_csv,
    pupil_values,
)
from eye_tracking_system_tools.annotation.event_explorer.snippet_extractor import (
    extract_eye_snippet,
)
from eye_tracking_system_tools.annotation.event_explorer.models import (
    BlockDataCache,
    ColumnMap,
    EventRecord,
)
from eye_tracking_system_tools.annotation.event_explorer.load_log import LoadLog


def test_catalog_load_sample_json(tmp_path):
    ann = tmp_path / "PV_106_2025_09_04_block_015_annotations.json"
    ann.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "animal_call": "PV_106",
                "experiment_date": "2025_09_04",
                "block_num": "015",
                "block_path": str(tmp_path / "block_015"),
                "events": [
                    {
                        "id": "a",
                        "event_type": "saccade",
                        "timepoint_ms": 100.0,
                        "start_ms": 0,
                        "end_ms": 200,
                        "l_eye_frame": 10,
                    },
                    {
                        "id": "b",
                        "event_type": "blink",
                        "timepoint_ms": 200.0,
                        "start_ms": 100,
                        "end_ms": 300,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    records = load_events_from_file(ann)
    assert len(records) == 2
    assert records[0].event_type == "saccade"


def test_discover_scan_dir(tmp_path):
    sub = tmp_path / "out"
    sub.mkdir()
    f = sub / "x_annotations.json"
    f.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "animal_call": "A",
                "block_num": "001",
                "block_path": "/tmp",
                "events": [],
            }
        )
    )
    found = discover_annotation_files(scan_dirs=[tmp_path])
    assert f.resolve() in [p.resolve() for p in found]


def test_pick_newest_eye_csv(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    old = analysis / "left_eye_data_old.csv"
    new = analysis / "left_eye_data_new.csv"
    old.write_text("eye_frame,ms_axis,width,height\n0,0,1,1\n", encoding="utf-8")
    time.sleep(0.05)
    new.write_text("eye_frame,ms_axis,width,height\n0,0,2,2\n", encoding="utf-8")
    cands = glob_eye_candidates(analysis, "left")
    chosen = pick_newest_csv(cands)
    assert chosen == new


def test_snippet_frame_slice_synthetic():
    df = pd.DataFrame(
        {
            "eye_frame": np.arange(100),
            "ms_axis": np.arange(100) * 10.0,
            "width": np.ones(100) * 2.0,
            "height": np.ones(100) * 2.0,
            "k_theta": np.linspace(-1, 1, 100),
        }
    )
    cache = BlockDataCache(
        block_path=Path("/fake"),
        final_sync_df=df,
        ms_axis=df["ms_axis"].to_numpy(),
        sample_rate_hz=30000.0,
        le_df=df,
        column_map=ColumnMap(
            pupil=detect_pupil_column(df),
            l_degrees="k_theta",
        ),
    )
    rec = EventRecord(
        event_id="e1",
        event_type="saccade",
        animal_call="A",
        experiment_date=None,
        block_num="001",
        timepoint_ms=500.0,
        start_ms=400,
        end_ms=600,
        row_index=50,
        arena_frame=None,
        l_eye_frame=50,
        r_eye_frame=None,
        note="",
        annotation_path=Path("a.json"),
        block_path=Path("/fake"),
    )
    log = LoadLog()
    from eye_tracking_system_tools.annotation.event_explorer.snippet_extractor import (
        STREAM_L_PUPIL,
    )

    snip = extract_eye_snippet(rec, cache, STREAM_L_PUPIL, 100.0, log)
    assert snip is not None
    assert snip.source == "frame"
    assert len(snip.time_rel_ms) > 0
    col = cache.column_map.pupil
    assert col is not None
    vals = pupil_values(df, col)
    assert vals is not None
