"""Phase 0 tests — Data Exploration series catalog and plot panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    ensure_config_template,
    load_config,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_plot_panel import (
    ExplorePlotPanel,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.explore_series import (
    EYE_K_PHI,
    EYE_K_THETA,
    EYE_PUPIL_SIZE,
    EYE_VERSION_BASE,
    available_eye_metric_ids,
    build_explore_catalog,
    discover_eye_data_versions,
    eye_csvs_are_stale,
    load_eye_data_version,
    prefer_eye_data_version,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.explore_tab import (
    ExploreTab,
)


def _synthetic_final_sync(n: int = 50, fs: float = 30000.0) -> pd.DataFrame:
    ttl = np.arange(n, dtype=np.float64) * (fs / 60.0)
    return pd.DataFrame(
        {
            "Arena_TTL": ttl,
            "Arena_frame": np.arange(n),
            "L_eye_frame": np.arange(n),
            "R_eye_frame": np.arange(n),
            "L_values": np.sin(np.linspace(0, 6, n)),
            "R_values": np.cos(np.linspace(0, 6, n)),
        }
    )


def _synthetic_eye_df(
    n: int = 50,
    fs: float = 30000.0,
    *,
    with_kerr: bool = True,
    bare_angles: bool = False,
) -> pd.DataFrame:
    ttl = np.arange(n, dtype=np.float64) * (fs / 60.0)
    data = {
        "OE_timestamp": ttl,
        "eye_frame": np.arange(n),
        "ms_axis": ttl / (fs / 1000.0),
        "width": np.full(n, 8.0),
        "height": np.full(n, 6.0),
        "center_x": np.linspace(10, 20, n),
        "center_y": np.linspace(15, 25, n),
        "phi": np.zeros(n),  # ellipse phi — ignored unless theta also present
    }
    if with_kerr and not bare_angles:
        data["k_theta"] = np.linspace(-5, 5, n)
        data["k_phi"] = np.linspace(0, 2, n)
    if bare_angles:
        data["theta"] = np.linspace(-5, 5, n)
        data["phi"] = np.linspace(0, 2, n)
    return pd.DataFrame(data)


def test_build_explore_catalog_paired_eye_metrics():
    final = _synthetic_final_sync()
    le = _synthetic_eye_df(with_kerr=True)
    re = _synthetic_eye_df(with_kerr=True)
    catalog = build_explore_catalog(final, 30000.0, le_df=le, re_df=re)
    ids = available_eye_metric_ids(catalog)
    assert ids == [EYE_PUPIL_SIZE, EYE_K_PHI, EYE_K_THETA]
    for mid in ids:
        pair = catalog.eye_metrics[mid]
        assert pair.available
        assert pair.left is not None and pair.left.available
        assert pair.right is not None and pair.right.available


def test_build_explore_catalog_bare_phi_theta_aliases():
    final = _synthetic_final_sync()
    le = _synthetic_eye_df(with_kerr=False, bare_angles=True)
    re = _synthetic_eye_df(with_kerr=False, bare_angles=True)
    catalog = build_explore_catalog(final, 30000.0, le_df=le, re_df=re)
    ids = available_eye_metric_ids(catalog)
    assert EYE_K_PHI in ids
    assert EYE_K_THETA in ids


def test_ellipse_phi_alone_does_not_become_k_phi():
    final = _synthetic_final_sync()
    le = _synthetic_eye_df(with_kerr=False, bare_angles=False)
    re = _synthetic_eye_df(with_kerr=False, bare_angles=False)
    catalog = build_explore_catalog(final, 30000.0, le_df=le, re_df=re)
    ids = available_eye_metric_ids(catalog)
    assert ids == [EYE_PUPIL_SIZE]
    assert EYE_K_PHI not in ids


def test_discover_and_load_eye_data_versions(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    base = _synthetic_eye_df(with_kerr=False)
    tagged = _synthetic_eye_df(with_kerr=True)
    base.to_csv(analysis / "left_eye_data.csv", index=False)
    base.to_csv(analysis / "right_eye_data.csv", index=False)
    tagged.to_csv(analysis / "left_eye_data_raw_verified.csv", index=False)
    tagged.to_csv(analysis / "right_eye_data_raw_verified.csv", index=False)

    versions = discover_eye_data_versions(analysis)
    tags = [v.tag for v in versions]
    assert EYE_VERSION_BASE in tags
    assert "raw_verified" in tags

    preferred = prefer_eye_data_version(versions)
    assert preferred is not None
    assert preferred.tag == "raw_verified"

    le, re, le_path, re_path = load_eye_data_version(analysis, "raw_verified")
    assert le is not None and re is not None
    assert "k_phi" in le.columns
    assert le_path is not None and le_path.name.endswith("raw_verified.csv")


def test_eye_csvs_are_stale(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    sync = analysis / "final_sync_df.csv"
    left = analysis / "left_eye_data.csv"
    right = analysis / "right_eye_data.csv"
    _synthetic_final_sync().to_csv(sync, index=False)
    _synthetic_eye_df().to_csv(left, index=False)
    _synthetic_eye_df().to_csv(right, index=False)
    sync.touch()
    assert eye_csvs_are_stale(analysis) is True


def test_explore_plot_panel_version_combo(qapp_session, tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    tagged = _synthetic_eye_df(30, with_kerr=True)
    tagged.to_csv(analysis / "left_eye_data_raw_verified.csv", index=False)
    tagged.to_csv(analysis / "right_eye_data_raw_verified.csv", index=False)
    versions = discover_eye_data_versions(analysis)

    final = _synthetic_final_sync(30)
    catalog = build_explore_catalog(
        final, 30000.0, le_df=tagged, re_df=tagged, eye_version_tag="raw_verified"
    )
    panel = ExplorePlotPanel()
    panel.set_eye_versions(versions, selected_tag="raw_verified")
    panel.set_catalog(catalog)
    assert panel.current_eye_version_tag() == "raw_verified"
    assert EYE_PUPIL_SIZE in panel._eye_checks
    assert EYE_K_PHI in panel._eye_checks
    assert EYE_K_THETA in panel._eye_checks
    assert panel._box_zoom_btn.isChecked()
    for plot in panel._plot_rows.values():
        assert plot.getViewBox().state["mouseMode"] == pg.ViewBox.RectMode
    panel._box_zoom_btn.setChecked(False)
    panel._apply_mouse_mode()
    for plot in panel._plot_rows.values():
        assert plot.getViewBox().state["mouseMode"] == pg.ViewBox.PanMode


def test_explore_video_panel_instantiates(qapp_session):
    from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
        ExploreVideoPanel,
    )

    panel = ExploreVideoPanel()
    assert panel._left_panel is not None
    assert panel._arena_panel is not None
    assert panel._right_panel is not None
    assert hasattr(panel, "_le_flip_v") and hasattr(panel, "_re_flip_v")
    assert hasattr(panel, "_le_ann") and hasattr(panel, "_re_ann")
    panel.clear()
    assert panel.current_ms() is None


def test_explore_tab_instantiates(qapp_session, tmp_path: Path):
    ensure_config_template(tmp_path)
    config = load_config(None, tmp_path)
    state = GuiState(output_folder=tmp_path)
    state.ensure_session()
    tab = ExploreTab(state, config)
    assert tab.tab_id == "explore"
    assert tab._plot is not None
    assert tab._video is None
    assert tab._video_window is None
    assert tab._btn_open_video is not None
    assert tab._btn_cache_videos is not None
    assert tab._time_label is not None
    assert tab._plot.chrome_widget is not None
    assert tab._plot.chrome_widget.parent() is not tab._plot
    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=tmp_path / "block_015",
        path_to_animal_folder=tmp_path,
    )
    (tmp_path / "block_015" / "analysis").mkdir(parents=True)
    tab.set_block(block)
    assert "015" in tab._info.text() or "block" in tab._info.text().lower()


def test_explore_playhead_emits_on_release_not_drag(qapp_session):
    final = _synthetic_final_sync(30)
    eye = _synthetic_eye_df(30, with_kerr=True)
    catalog = build_explore_catalog(final, 30000.0, le_df=eye, re_df=eye)
    panel = ExplorePlotPanel()
    panel.set_catalog(catalog)
    assert panel._playheads

    selected: list[float] = []
    previewed: list[float] = []
    panel.time_selected.connect(selected.append)
    panel.time_preview.connect(previewed.append)

    line = next(iter(panel._playheads.values()))
    line.setPos(float(catalog.ms_axis[10]))
    assert selected == []
    assert previewed
    assert previewed[-1] == pytest.approx(float(catalog.ms_axis[10]))

    line.sigPositionChangeFinished.emit(line)
    assert selected
    assert selected[-1] == pytest.approx(float(catalog.ms_axis[10]))


def test_video_panel_prepare_frame_returns_qimage(qapp_session):
    from PyQt6 import QtGui

    from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
        VideoPanel,
        numpy_array_to_qimage,
    )

    panel = VideoPanel("test")
    panel.show()

    arr = np.zeros((24, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = 200
    qimg = numpy_array_to_qimage(arr)
    assert isinstance(qimg, QtGui.QImage)
    assert not qimg.isNull()

    calls: list[int | None] = []

    class StubReader:
        def read_frame(self, frame_idx):
            calls.append(frame_idx)
            return arr.copy()

    panel._reader = StubReader()
    out = panel.prepare_frame(3, fast_scale=True)
    assert isinstance(out, QtGui.QImage)
    assert not out.isNull()
    assert calls == [3]


def test_video_panel_display_image_sets_pixmap(qapp_session):
    from PyQt6 import QtGui

    from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
        MISSING_TEXT,
        VideoPanel,
        numpy_array_to_qimage,
    )

    panel = VideoPanel("test")
    panel.show()
    qimg = numpy_array_to_qimage(np.full((10, 12, 3), 128, dtype=np.uint8))
    panel.display_image(qimg, "frame 1")
    assert panel._info.text() == "frame 1"
    assert panel._label.pixmap() is not None
    assert not panel._label.pixmap().isNull()
    assert panel._label.text() == ""

    panel.display_image(None, "")
    assert panel._label.text() == MISSING_TEXT


def test_video_panel_fast_scale_toggle(qapp_session):
    from PyQt6 import QtCore

    from eye_tracking_system_tools.annotation.block_annotator.video_widget import (
        VideoPanel,
    )

    panel = VideoPanel("test")
    panel.show()

    arr = np.zeros((24, 32, 3), dtype=np.uint8)

    class StubReader:
        def read_frame(self, frame_idx):
            return arr.copy()

    panel._reader = StubReader()
    modes: list[QtCore.Qt.TransformationMode] = []
    orig_prepare = panel.prepare_frame

    def _spy(frame_idx, target_size=None, *, fast_scale=False):
        modes.append(fast_scale)
        return orig_prepare(frame_idx, target_size, fast_scale=fast_scale)

    panel.prepare_frame = _spy

    panel.set_fast_scale(True)
    panel.show_frame(0, "")
    panel.set_fast_scale(False)
    panel.show_frame(0, "")
    assert modes == [True, False]


def _make_video_session(n: int = 5):
    from eye_tracking_system_tools.annotation.block_annotator.models import (
        AnnotatorConfig,
        BlockSession,
        compute_ms_axis,
    )

    final = _synthetic_final_sync(n)
    return BlockSession(
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


def test_explore_video_panel_skips_unchanged_frame_ids(qapp_session):
    from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
        ExploreVideoPanel,
    )

    panel = ExploreVideoPanel()
    panel._bind_session(_make_video_session(5))
    panel._playback.play(forward=True)

    calls: list[int | None] = []
    panel._arena_panel.show_frame = lambda idx, info="": calls.append(idx)
    panel._left_panel.show_frame = lambda idx, info="": None
    panel._right_panel.show_frame = lambda idx, info="": None

    panel._playback.set_index(1)
    panel._playback.set_index(1)
    assert len(calls) == 1


def test_explore_video_panel_pause_repaints_smoothly(qapp_session):
    from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
        ExploreVideoPanel,
    )

    panel = ExploreVideoPanel()
    panel._bind_session(_make_video_session(5))
    panel._playback.play(forward=True)
    assert panel._arena_panel._fast_scale is True

    panel._playback.pause()
    assert panel._arena_panel._fast_scale is False
    assert panel._left_panel._fast_scale is False
    assert panel._right_panel._fast_scale is False


def test_explore_video_cache_copy_and_clear(tmp_path: Path):
    from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_cache import (
        ExploreVideoCache,
    )

    src_a = tmp_path / "arena.mp4"
    src_b = tmp_path / "le.mp4"
    src_a.write_bytes(b"aaa" * 1000)
    src_b.write_bytes(b"bbb" * 1000)

    cache = ExploreVideoCache()
    reports: list[tuple] = []

    def on_bytes(*args):
        reports.append(args)

    mapping = cache.cache_paths([src_a, src_b], progress_bytes=on_bytes)
    assert cache.active
    assert src_a.resolve() in mapping
    assert reports
    local_a = cache.local_path(src_a)
    assert local_a.is_file()
    assert local_a.read_bytes() == src_a.read_bytes()
    assert local_a != src_a.resolve()

    root = cache.root
    assert root is not None and root.is_dir()
    cache.clear()
    assert not cache.active
    assert cache.local_path(src_a) == src_a
    assert root is None or not root.exists()


def test_explore_plot_x_window_and_y_controls(qapp_session):
    final = _synthetic_final_sync(30)
    eye = _synthetic_eye_df(30, with_kerr=True)
    catalog = build_explore_catalog(final, 30000.0, le_df=eye, re_df=eye)
    panel = ExplorePlotPanel()
    panel.set_catalog(catalog)
    assert panel._btn_apply_x is not None
    assert panel._center_ms is not None
    assert panel._window_ms is not None
    assert panel._y_controls
    row_id = next(iter(panel._plot_rows))
    panel._on_y_limits(row_id, 1.0, 2.0)
    assert panel._y_limits[row_id] == (1.0, 2.0)
    ymin, ymax = panel._plot_rows[row_id].viewRange()[1]
    assert ymin == pytest.approx(1.0)
    assert ymax == pytest.approx(2.0)
