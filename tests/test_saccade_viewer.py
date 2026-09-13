"""Unit tests for saccade_viewer models, artifacts, and clip logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.saccade_viewer.artifacts import (
    apply_saved_tags,
    merge_tag_rows,
    merge_verification_tags,
    read_verification_tags,
    save_verification_tags,
    verification_filter_good_only,
)
from eye_tracking_system_tools.analysis.saccade_viewer.event_clip import (
    bounds_for_event,
    clip_indices,
    clip_ms_bounds,
    next_unset_index,
)
from eye_tracking_system_tools.analysis.saccade_viewer.models import (
    VerificationEvent,
    compute_event_id,
    normalize_event_batch,
)


def _sample_df(tmp_path: Path) -> pd.DataFrame:
    bp = tmp_path / "block_012"
    bp.mkdir(parents=True)
    return pd.DataFrame(
        {
            "animal": ["M_002", "M_002"],
            "block": ["012", "012"],
            "eye": ["L", "R"],
            "saccade_on_ms": [1000.0, 2000.0],
            "saccade_off_ms": [1030.0, 2030.0],
            "block_path": [str(bp), str(bp)],
        }
    )


def test_compute_event_id_stable():
    a = compute_event_id(animal="M_002", block="012", eye="L", onset_ms=1000.0, off_ms=1030.0)
    b = compute_event_id(animal="M_002", block="012", eye="L", onset_ms=1000.0, off_ms=1030.0)
    assert a == b
    assert len(a) == 16


def test_normalize_event_batch(tmp_path: Path):
    df = _sample_df(tmp_path)
    spec = BlockSpec(animal="M_002", block_path=tmp_path / "block_012", block_num="012")
    batch = normalize_event_batch(df, [spec])
    assert len(batch.events) == 2
    assert batch.events[0].block_key == "M_002_block_012"
    assert batch.events[0].onset_ms == 1000.0


def test_clip_bounds():
    t0, t1 = clip_ms_bounds(5000.0, pre_ms=250.0, post_ms=250.0)
    assert t0 == 4750.0
    assert t1 == 5250.0
    axis = np.linspace(0, 10000, 101)
    b = clip_indices(axis, t0, t1)
    assert b.i_start <= b.i_end


def test_bounds_for_event():
    ev = VerificationEvent(
        event_id="abc",
        animal="M_002",
        block="012",
        block_key="M_002_block_012",
        block_path=Path("/tmp/block_012"),
        eye="L",
        onset_ms=5000.0,
        off_ms=5030.0,
    )
    axis = np.arange(0, 10000, 100, dtype=float)
    b = bounds_for_event(axis, ev, pre_ms=200.0, post_ms=200.0)
    assert b.t_start_ms == 4800.0
    assert b.t_end_ms == 5200.0


def test_next_unset_index():
    events = [
        VerificationEvent("a", "M", "01", "M_block_01", Path("/b"), "L", 1, 2, "good"),
        VerificationEvent("b", "M", "01", "M_block_01", Path("/b"), "R", 3, 4, "unset"),
        VerificationEvent("c", "M", "01", "M_block_01", Path("/b"), "L", 5, 6, "unset"),
    ]
    assert next_unset_index(events, 0, forward=True) == 1
    assert next_unset_index(events, 2, forward=True) == 1  # wrap


def test_tag_artifact_round_trip(tmp_path: Path):
    df = _sample_df(tmp_path)
    spec = BlockSpec(animal="M_002", block_path=tmp_path / "block_012", block_num="012")
    batch = normalize_event_batch(df, [spec])
    ev = batch.events[0]
    tagged = VerificationEvent(
        event_id=ev.event_id,
        animal=ev.animal,
        block=ev.block,
        block_key=ev.block_key,
        block_path=ev.block_path,
        eye=ev.eye,
        onset_ms=ev.onset_ms,
        off_ms=ev.off_ms,
        verification_status="bad",
        pairing_tag="monocular",
        notes="blink artifact",
        row_index=ev.row_index,
        extras=dict(ev.extras),
    )
    path = save_verification_tags(ev.block_path, [tagged])
    assert path.is_file()
    loaded = read_verification_tags(ev.block_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["verification_status"] == "bad"
    assert loaded.iloc[0]["pairing_tag"] == "monocular"
    assert loaded.iloc[0]["notes"] == "blink artifact"


def test_merge_tag_rows_preserves_other_events():
    existing = pd.DataFrame(
        [
            {
                "event_id": "keep",
                "animal": "M",
                "block": "01",
                "eye": "L",
                "onset_ms": 1.0,
                "off_ms": 2.0,
                "verification_status": "good",
                "verified_at": "t0",
                "notes": "",
            }
        ]
    )
    merged = merge_tag_rows(
        existing,
        [
            {
                "event_id": "new",
                "animal": "M",
                "block": "01",
                "eye": "R",
                "onset_ms": 3.0,
                "off_ms": 4.0,
                "verification_status": "bad",
                "verified_at": "",
                "notes": "",
            }
        ],
    )
    assert len(merged) == 2
    assert set(merged["event_id"]) == {"keep", "new"}


def test_apply_saved_tags_loads_pairing_and_notes(tmp_path: Path):
    df = _sample_df(tmp_path)
    batch = normalize_event_batch(df, [])
    ev = batch.events[0]
    tagged = VerificationEvent(
        event_id=ev.event_id,
        animal=ev.animal,
        block=ev.block,
        block_key=ev.block_key,
        block_path=ev.block_path,
        eye=ev.eye,
        onset_ms=ev.onset_ms,
        off_ms=ev.off_ms,
        verification_status="good",
        pairing_tag="concurrent",
        notes="nice example",
    )
    save_verification_tags(ev.block_path, [tagged])
    reloaded = apply_saved_tags([ev], ev.block_path)
    assert reloaded[0].pairing_tag == "concurrent"
    assert reloaded[0].notes == "nice example"


def test_merge_verification_tags(tmp_path: Path):
    df = _sample_df(tmp_path)
    block_path = tmp_path / "block_012"
    ev = normalize_event_batch(df, []).events[0]
    tagged = VerificationEvent(
        event_id=ev.event_id,
        animal=ev.animal,
        block=ev.block,
        block_key=ev.block_key,
        block_path=ev.block_path,
        eye=ev.eye,
        onset_ms=ev.onset_ms,
        off_ms=ev.off_ms,
        verification_status="good",
        row_index=0,
    )
    save_verification_tags(block_path, [tagged])
    merged = merge_verification_tags(df, {"M_002_block_012": block_path})
    assert "verification_status" in merged.columns
    assert "pairing_tag" in merged.columns
    assert "notes" in merged.columns
    assert merged.iloc[0]["verification_status"] == "good"
    assert merged.iloc[1]["verification_status"] == "unset"


def test_launch_uses_local_event_loop_in_ipython(monkeypatch):
    from eye_tracking_system_tools.analysis.saccade_viewer import launch as launch_mod
    from PyQt6 import QtCore

    loops: list = []

    class FakeLoop:
        def exec(self):
            loops.append(True)

        def quit(self) -> None:
            pass

    class FakeWin(QtCore.QObject):
        closed = QtCore.pyqtSignal()

    monkeypatch.setattr(launch_mod, "_running_in_ipython", lambda: True)
    monkeypatch.setattr(launch_mod.QtCore, "QEventLoop", FakeLoop)

    app = launch_mod._ensure_qapplication()
    assert app.quitOnLastWindowClosed() is False

    win = FakeWin()
    launch_mod._block_until_window_closed(win, app)  # type: ignore[arg-type]
    assert len(loops) == 1


def test_verification_filter_good_only():
    df = pd.DataFrame({"verification_status": ["good", "bad", "unset"]})
    kept = verification_filter_good_only(df)
    assert len(kept) == 2
