"""Qt tests for saccade verification viewer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PyQt6")


@pytest.fixture
def qapp_session():
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def _make_batch(tmp_path: Path):
    from eye_tracking_system_tools.analysis.block_registry import BlockSpec
    from eye_tracking_system_tools.analysis.saccade_viewer.models import normalize_event_batch

    bp1 = tmp_path / "block_012"
    bp2 = tmp_path / "block_013"
    bp1.mkdir(parents=True)
    bp2.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "animal": ["M_002", "M_002", "M_002"],
            "block": ["012", "012", "013"],
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [1000.0, 2000.0, 3000.0],
            "saccade_off_ms": [1030.0, 2030.0, 3030.0],
            "block_path": [str(bp1), str(bp1), str(bp2)],
        }
    )
    specs = [
        BlockSpec(animal="M_002", block_path=bp1, block_num="012"),
        BlockSpec(animal="M_002", block_path=bp2, block_num="013"),
    ]
    return normalize_event_batch(df, specs)


def test_viewer_block_filter(qapp_session, tmp_path: Path, monkeypatch):
    from eye_tracking_system_tools.analysis.saccade_viewer.app import SaccadeViewerWindow

    monkeypatch.setattr(SaccadeViewerWindow, "_start_video_load", lambda self, spec: None)
    monkeypatch.setattr(SaccadeViewerWindow, "_load_traces", lambda self, spec: None)

    batch = _make_batch(tmp_path)
    win = SaccadeViewerWindow(batch)
    assert win._block_key == "M_002_block_012"
    assert len(win._block_events) == 2
    win._block_combo.setCurrentIndex(1)
    assert win._block_key == "M_002_block_013"
    assert len(win._block_events) == 1
    win.close()


def test_viewer_tag_updates_status(qapp_session, tmp_path: Path, monkeypatch):
    from eye_tracking_system_tools.analysis.saccade_viewer.app import SaccadeViewerWindow

    monkeypatch.setattr(SaccadeViewerWindow, "_start_video_load", lambda self, spec: None)
    monkeypatch.setattr(SaccadeViewerWindow, "_load_traces", lambda self, spec: None)

    saved: list = []

    def fake_save(block_path, events, **kwargs):
        saved.append((block_path, list(events)))
        from eye_tracking_system_tools.analysis.saccade_viewer.artifacts import tags_path

        path = tags_path(block_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(
        "eye_tracking_system_tools.analysis.saccade_viewer.app.save_verification_tags",
        fake_save,
    )

    batch = _make_batch(tmp_path)
    win = SaccadeViewerWindow(batch)
    win._set_status("good")
    assert win._block_events[0].verification_status == "good"
    assert len(saved) == 1
    win.close()


def test_viewer_pairing_and_comment(qapp_session, tmp_path: Path, monkeypatch):
    from eye_tracking_system_tools.analysis.saccade_viewer.app import SaccadeViewerWindow

    monkeypatch.setattr(SaccadeViewerWindow, "_start_video_load", lambda self, spec: None)
    monkeypatch.setattr(SaccadeViewerWindow, "_load_traces", lambda self, spec: None)

    saved: list = []

    def fake_save(block_path, events, **kwargs):
        saved.append(list(events))
        from eye_tracking_system_tools.analysis.saccade_viewer.artifacts import tags_path

        path = tags_path(block_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(
        "eye_tracking_system_tools.analysis.saccade_viewer.app.save_verification_tags",
        fake_save,
    )

    batch = _make_batch(tmp_path)
    win = SaccadeViewerWindow(batch)
    win._set_pairing("monocular")
    assert win._block_events[0].pairing_tag == "monocular"
    win._comment.setText("test note")
    win._save_comment()
    assert win._block_events[0].notes == "test note"
    assert len(saved) == 2
    win.close()


def test_viewer_emits_closed_on_close(qapp_session, tmp_path: Path, monkeypatch):
    from eye_tracking_system_tools.analysis.saccade_viewer.app import SaccadeViewerWindow

    monkeypatch.setattr(SaccadeViewerWindow, "_start_video_load", lambda self, spec: None)
    monkeypatch.setattr(SaccadeViewerWindow, "_load_traces", lambda self, spec: None)

    batch = _make_batch(tmp_path)
    win = SaccadeViewerWindow(batch)
    seen: list[bool] = []
    win.closed.connect(lambda: seen.append(True))
    win.close()
    assert seen == [True]


def test_viewer_step_event(qapp_session, tmp_path: Path, monkeypatch):
    from eye_tracking_system_tools.analysis.saccade_viewer.app import SaccadeViewerWindow

    monkeypatch.setattr(SaccadeViewerWindow, "_start_video_load", lambda self, spec: None)
    monkeypatch.setattr(SaccadeViewerWindow, "_load_traces", lambda self, spec: None)

    batch = _make_batch(tmp_path)
    win = SaccadeViewerWindow(batch)
    first_id = win._current_event().event_id
    win._step_event(1)
    second_id = win._current_event().event_id
    assert first_id != second_id
    win.close()
