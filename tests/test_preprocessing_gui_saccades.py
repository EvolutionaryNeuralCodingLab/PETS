"""Tests for Saccades tab playhead, load-existing, and video saccade markers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_tab_order_explore_before_saccades():
    from eye_tracking_system_tools.annotation.preprocessing_gui.app import _TAB_CLASSES
    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs import (
        ExploreTab,
        SaccadesTab,
    )

    ids = [cls.tab_id for cls in _TAB_CLASSES]
    assert ids.index("verify") < ids.index("refine") < ids.index("conicoid") < ids.index("kerr")
    assert ids.index("explore") < ids.index("saccades")
    assert ExploreTab.tab_label == "Data Exploration"
    assert SaccadesTab.tab_label == "Saccades"


def test_saccade_intervals_by_eye_and_ms_in_intervals():
    from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
        ms_in_intervals,
        saccade_intervals_by_eye,
    )

    events = pd.DataFrame(
        {
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [10.0, 50.0, 200.0],
            "saccade_off_ms": [20.0, 60.0, 210.0],
        }
    )
    left, right = saccade_intervals_by_eye(events)
    assert left.shape == (2, 2)
    assert right.shape == (1, 2)
    assert ms_in_intervals(15.0, left) is True
    assert ms_in_intervals(15.0, right) is False
    assert ms_in_intervals(55.0, right) is True
    assert ms_in_intervals(100.0, left) is False


def test_draw_saccade_circle_marks_corner():
    from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
        draw_saccade_circle,
    )

    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    out = draw_saccade_circle(frame, radius=10, margin=20, thickness=2)
    assert out is not frame
    # Green BGR pixel near the margin corner
    assert int(out[20, 20, 1]) > 200
    assert int(out[20, 20, 0]) < 50
    assert int(frame[20, 20, 1]) == 0


def test_video_panel_saccade_active_in_prepare_frame(qapp_session):
    from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
        VideoPanel,
        apply_display_transforms,
    )

    arr = np.zeros((40, 50, 3), dtype=np.uint8)
    off = apply_display_transforms(arr, saccade_active=False)
    on = apply_display_transforms(arr, saccade_active=True)
    assert not np.array_equal(off, on)

    panel = VideoPanel("test")

    class StubReader:
        def read_frame(self, frame_idx):
            return arr.copy()

    panel._reader = StubReader()
    panel.set_saccade_active(True)
    qimg = panel.prepare_frame(0)
    assert qimg is not None
    assert not qimg.isNull()
    assert panel._saccade_active is True


def test_explore_video_panel_sets_eye_saccade_flags(qapp_session):
    from eye_tracking_system_tools.annotation.block_annotator.models import (
        AnnotatorConfig,
        BlockSession,
        compute_ms_axis,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
        ExploreVideoPanel,
    )

    n = 10
    final = pd.DataFrame(
        {
            "Arena_TTL": np.arange(n, dtype=float) * 500.0,
            "Arena_frame": np.arange(n),
            "L_eye_frame": np.arange(n),
            "R_eye_frame": np.arange(n),
        }
    )
    session = BlockSession(
        animal_call="PV_test",
        experiment_date="2024_01_01",
        block_num="001",
        block_path=Path("block"),
        output_folder=Path("out"),
        config=AnnotatorConfig(),
        final_sync_df=final,
        ms_axis=compute_ms_axis(final, 30000.0),
        sample_rate_hz=30000.0,
        arena_videos=[],
        le_videos=[],
        re_videos=[],
    )
    panel = ExploreVideoPanel()
    panel._bind_session(session)

    # At ms≈0 initially — no events yet
    assert panel._left_panel._saccade_active is False
    assert panel._right_panel._saccade_active is False

    ms0 = float(session.ms_at(0))
    events = pd.DataFrame(
        {
            "eye": ["L", "R"],
            "saccade_on_ms": [ms0 - 1.0, 1e9],
            "saccade_off_ms": [ms0 + 50.0, 1e9 + 10.0],
        }
    )
    panel.set_saccade_events(events)
    assert panel._left_panel._saccade_active is True
    assert panel._right_panel._saccade_active is False


def test_saccade_velocity_panel_playhead_absolute_ms(qapp_session):
    from eye_tracking_system_tools.annotation.preprocessing_gui.saccade_plot_panel import (
        SaccadeVelocityPanel,
    )

    panel = SaccadeVelocityPanel()
    selected: list[float] = []
    previewed: list[float] = []
    panel.time_selected.connect(selected.append)
    panel.time_preview.connect(previewed.append)

    t0 = 100_000.0
    left = pd.DataFrame(
        {
            "ms_axis": np.linspace(t0, t0 + 5_000.0, 50),
            "angular_speed_r": np.linspace(0.1, 2.0, 50),
            "speed_r": np.linspace(0.1, 2.0, 50),
        }
    )
    panel.set_traces(left, None, window_t0_ms=t0)
    panel.set_playhead_ms(t0 + 2500.0, emit=False)
    assert panel.playhead_ms() == pytest.approx(t0 + 2500.0)
    assert panel._playhead_line.value() == pytest.approx(2500.0)

    panel._playhead_line.setPos(1000.0)
    assert previewed
    assert previewed[-1] == pytest.approx(t0 + 1000.0)
    panel._playhead_line.sigPositionChangeFinished.emit(panel._playhead_line)
    assert selected
    assert selected[-1] == pytest.approx(t0 + 1000.0)


def test_saccades_tab_load_existing_button_styles(qapp_session, tmp_path: Path):
    from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
        PreprocConfig,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
        BlockHandle,
        GuiState,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.saccades_tab import (
        SaccadesTab,
        _LOAD_EXISTING_STYLE_DISABLED,
        _LOAD_EXISTING_STYLE_READY,
    )
    from eye_tracking_system_tools.analysis.saccade_export import (
        EVENTS_CSV,
        PARAMS_YAML,
        saccades_dir,
    )

    state = GuiState(output_folder=tmp_path)
    tab = SaccadesTab(state, PreprocConfig())
    assert not tab._btn_load_existing.isEnabled()
    assert _LOAD_EXISTING_STYLE_DISABLED in tab._btn_load_existing.styleSheet()

    block_path = tmp_path / "block"
    analysis = block_path / "analysis"
    sac = saccades_dir(block_path)
    sac.mkdir(parents=True)
    (sac / EVENTS_CSV).write_text("eye,saccade_on_ms,saccade_off_ms\nL,0,10\n")
    (sac / PARAMS_YAML).write_text("saccade: {}\nbinocular: {}\n")
    (analysis / "final_sync_df.csv").write_text("Arena_TTL\n0\n")

    block = BlockHandle(
        animal_call="PV_1",
        experiment_date="2024_01_01",
        block_num="001",
        block_path=block_path,
        path_to_animal_folder=tmp_path,
    )
    tab.set_block(block)
    assert tab._btn_load_existing.isEnabled()
    assert _LOAD_EXISTING_STYLE_READY in tab._btn_load_existing.styleSheet()
    assert tab._btn_open_video is not None


def test_tag_bad_detections_span_and_query_filter():
    from eye_tracking_system_tools.analysis.saccade_export import (
        BAD_DETECTIONS_COL,
        apply_bad_detection_spans,
        ensure_bad_detections_column,
        merge_time_spans,
        subtract_time_span,
        tag_bad_detections_span,
    )

    events = pd.DataFrame(
        {
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [10.0, 50.0, 200.0],
            "saccade_off_ms": [20.0, 60.0, 210.0],
        }
    )
    tagged = tag_bad_detections_span(events, 15.0, 55.0)
    assert list(tagged[BAD_DETECTIONS_COL]) == [True, True, False]
    kept = tagged.query("bad_detections==False")
    assert len(kept) == 1
    assert float(kept.iloc[0]["saccade_on_ms"]) == 200.0

    spans = merge_time_spans([], new_span=(15.0, 55.0))
    spans = merge_time_spans(spans, new_span=(40.0, 80.0))
    assert spans == [(15.0, 80.0)]
    spans = subtract_time_span(spans, 30.0, 50.0)
    assert spans == [(15.0, 30.0), (50.0, 80.0)]

    reset = apply_bad_detection_spans(
        ensure_bad_detections_column(events), [(190.0, 220.0)], reset=True
    )
    assert list(reset[BAD_DETECTIONS_COL]) == [False, False, True]


def test_saccade_panel_mark_bad_region_emits_absolute_ms(qapp_session):
    from eye_tracking_system_tools.annotation.preprocessing_gui.saccade_plot_panel import (
        SaccadeVelocityPanel,
    )

    panel = SaccadeVelocityPanel()
    tagged: list[tuple[float, float]] = []
    panel.bad_span_tag_requested.connect(lambda a, b: tagged.append((a, b)))

    t0 = 50_000.0
    left = pd.DataFrame(
        {
            "ms_axis": np.linspace(t0, t0 + 10_000.0, 40),
            "angular_speed_r": np.ones(40),
            "speed_r": np.ones(40),
        }
    )
    panel.set_traces(left, None, window_t0_ms=t0)
    panel._btn_mark_bad.setChecked(True)
    assert panel._region_mode == "bad"
    assert panel._select_region is not None
    panel._select_region.setRegion((1000.0, 2500.0))
    panel._btn_tag_bad.click()
    assert tagged
    assert tagged[-1][0] == pytest.approx(t0 + 1000.0)
    assert tagged[-1][1] == pytest.approx(t0 + 2500.0)


def test_saccades_tab_tags_bad_detections_column(qapp_session):
    from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
        PreprocConfig,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.models import GuiState
    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.saccades_tab import (
        SaccadesTab,
    )
    from eye_tracking_system_tools.analysis.saccade_export import BAD_DETECTIONS_COL

    tab = SaccadesTab(GuiState(), PreprocConfig())
    tab._events = pd.DataFrame(
        {
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [10.0, 50.0, 200.0],
            "saccade_off_ms": [20.0, 60.0, 210.0],
            "concurrency": ["monocular", "monocular", "monocular"],
        }
    )
    tab._on_tag_bad_span(0.0, 30.0)
    assert BAD_DETECTIONS_COL in tab._events.columns
    assert list(tab._events[BAD_DETECTIONS_COL]) == [True, False, False]
    assert tab._bad_spans == [(0.0, 30.0)]
    tab._on_clear_bad_spans()
    assert not tab._events[BAD_DETECTIONS_COL].any()
    assert tab._bad_spans == []
