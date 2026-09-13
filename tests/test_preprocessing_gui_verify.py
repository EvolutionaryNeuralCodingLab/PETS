"""Phase 4 tests — EllipseVerifierWidget and Verify tab."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    ensure_config_template,
    load_config,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.ellipse_verifier import (
    EllipseVerifierWidget,
    display_roi_from_drag,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.verify_tab import (
    VerifyTab,
    eye_data_is_stale,
)


def _write_tiny_video(path: Path, *, width: int = 160, height: int = 120, nframes: int = 3) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    for i in range(nframes):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (20 + i * 10, 40, 60)
        writer.write(frame)
    writer.release()


def _sample_eye_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eye_frame": np.arange(n),
            "center_x": np.linspace(20, 40, n),
            "center_y": np.linspace(30, 50, n),
            "width": np.full(n, 8),
            "height": np.full(n, 6),
            "phi": np.zeros(n),
        }
    )


def test_ellipse_verifier_yflip_click_math(qapp_session, tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    _write_tiny_video(video, height=200)
    widget = EllipseVerifierWidget(_sample_eye_df(), video, "left")
    ref = widget.pick_ref_from_display_xy(100, 50)
    assert ref == (100, 200 - 1 - 50)
    assert widget.ref_xy() == ref


def test_ellipse_verifier_exposes_df_and_ref(qapp_session, tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    _write_tiny_video(video)
    widget = EllipseVerifierWidget(_sample_eye_df(), video, "left")
    widget.pick_ref_from_display_xy(12, 8)
    df_out = widget.df()
    assert "center_x" in df_out.columns
    assert widget.ref_xy() == (12, widget.frame_height - 1 - 8)


def test_display_roi_from_drag_rounding_and_clamp():
    assert display_roi_from_drag(10, 20, 50, 80, 160, 120) == (10, 20, 40, 60)
    assert display_roi_from_drag(50, 80, 10, 20, 160, 120) == (10, 20, 40, 60)
    assert display_roi_from_drag(-5, -5, 200, 200, 160, 120) == (0, 0, 160, 120)
    assert display_roi_from_drag(10.4, 20.6, 50.4, 80.4, 160, 120) == (10, 21, 40, 59)
    assert display_roi_from_drag(10, 20, 10, 20, 160, 120) is None


def test_ellipse_verifier_measure_roi_labels(qapp_session, tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    _write_tiny_video(video, width=160, height=120)
    widget = EllipseVerifierWidget(_sample_eye_df(), video, "left")
    assert widget._rb_measure.text() == "Measure ROI"
    widget._rb_measure.setChecked(True)
    assert widget._mode == "measure"
    widget._on_drag_finished(10.0, 20.0, 50.0, 80.0)
    assert widget.measure_roi() == (10, 20, 40, 60)
    texts = widget._frame_view.measure_overlay_texts()
    assert "40 px" in texts
    assert "60 px" in texts
    widget._show_frame(0)
    assert "40 px" in widget._frame_view.measure_overlay_texts()
    widget._on_clear_measure()
    assert widget.measure_roi() is None
    assert widget._frame_view.measure_overlay_texts() == []


def test_ellipse_verifier_has_compute_angles_span_button(qapp_session, tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    _write_tiny_video(video)
    widget = EllipseVerifierWidget(_sample_eye_df(), video, "left")
    assert hasattr(widget, "_btn_angles_span")
    assert widget._btn_angles_span.text() == "Compute angles span"
    assert hasattr(widget, "_kerr_span_status")
    assert "none" in widget._kerr_span_status.text().lower()


def test_kerr_angles_span_dialog_builds(qapp_session):
    from eye_tracking_system_tools.annotation.preprocessing_gui.kerr_angles_span_dialog import (
        KerrAnglesSpanDialog,
        KerrRefSpanMetrics,
        grid_points_in_roi,
        index_closest_span_match,
        span_metrics_from_preview,
    )
    from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
        KerrAnglePreview,
    )

    preview = KerrAnglePreview(
        phi=np.linspace(-5.0, 5.0, 50),
        theta=np.linspace(-3.0, 4.0, 50),
        f_z=120.0,
        ref_x=100.0,
        ref_y=80.0,
        n_input=50,
        n_finite=50,
    )
    other_preview = KerrAnglePreview(
        phi=np.linspace(-4.5, 4.8, 40),
        theta=np.linspace(-2.8, 3.7, 40),
        f_z=118.0,
        ref_x=210.0,
        ref_y=90.0,
        n_input=40,
        n_finite=40,
    )
    eye_df = pd.DataFrame(
        {
            "eye_frame": np.arange(50),
            "OE_timestamp": np.arange(50, dtype=np.int64),
            "ms_axis": np.arange(50, dtype=float),
            "center_x": 100.0 + np.linspace(-3, 3, 50),
            "center_y": 80.0 + np.linspace(-2, 2, 50),
            "width": np.full(50, 12.0),
            "height": np.full(50, 10.0),
            "phi": np.zeros(50),
        }
    )
    frame_rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    dialog = KerrAnglesSpanDialog(
        preview,
        eye="left",
        eye_df=eye_df,
        frame_rgb=frame_rgb,
        other_preview=other_preview,
        other_eye="right",
    )
    assert "Left" in dialog.windowTitle()
    assert dialog._heat_plot is not None
    assert dialog._btn_recompute.isEnabled()
    assert dialog._btn_roi_grid.isEnabled()
    assert dialog._chk_show_other.isEnabled()
    assert "Right" in dialog._other_summary.text()
    dialog._chk_show_other.setChecked(True)
    assert dialog._showing_other is True
    dialog._spin_ref_x.setValue(101.0)
    dialog._spin_ref_y.setValue(81.0)
    dialog._on_recompute()
    assert dialog.ref_xy() == (101, 81)

    metrics = span_metrics_from_preview(preview)
    assert metrics.phi_full is not None and metrics.phi_full > 0
    assert metrics.theta_p5_95 is not None and metrics.theta_p5_95 > 0
    pts = grid_points_in_roi((10, 20, 21, 11), 3, 3)
    assert len(pts) == 9
    assert pts[0] == (10.0, 20.0)
    assert pts[-1] == (30.0, 30.0)

    target = span_metrics_from_preview(other_preview)
    # Candidate identical to target should win over a far-away one.
    near = target
    far = KerrRefSpanMetrics(
        ref_x=1.0,
        ref_y=1.0,
        f_z=1.0,
        n_finite=1,
        n_input=1,
        phi_full=100.0,
        phi_p5_95=80.0,
        theta_full=90.0,
        theta_p5_95=70.0,
        phi_mad=40.0,
        theta_mad=35.0,
    )
    hit = index_closest_span_match([far, near], target)
    assert hit is not None
    assert hit[0] == 1
    assert hit[1] == 0.0
    assert metrics.phi_mad is not None and metrics.phi_mad >= 0
    assert metrics.theta_mad is not None and metrics.theta_mad >= 0
    assert dialog._chk_include_mad.isChecked()
    dialog.close()


def test_eye_data_is_stale_detects_newer_final_sync(sample_block_path: Path, tmp_path: Path):
    analysis = sample_block_path / "analysis"
    if not (analysis / "left_eye_data.csv").is_file():
        pytest.skip("Sample eye data not present.")
    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=sample_block_path,
        path_to_animal_folder=sample_block_path.parents[2],
    )
    final_sync = analysis / "final_sync_df.csv"
    if not final_sync.is_file():
        pytest.skip("final_sync_df.csv missing on sample block.")
    assert isinstance(eye_data_is_stale(block), bool)


def test_verify_tab_writes_self_kerr_refs(sample_block_path, qapp_session, tmp_path):
    left_path = sample_block_path / "analysis" / "left_eye_data.csv"
    right_path = sample_block_path / "analysis" / "right_eye_data.csv"
    if not left_path.is_file() or not right_path.is_file():
        pytest.skip("Sample eye-data CSVs not present.")

    ensure_config_template(tmp_path)
    config = load_config(None, tmp_path)
    state = GuiState(output_folder=tmp_path)
    block = BlockHandle(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        block_path=sample_block_path,
        path_to_animal_folder=sample_block_path.parents[2],
    )

    tab = VerifyTab(state, config)
    tab.set_block(block)
    tab._on_load_prev_analysis()
    assert tab._left_verifier is not None and tab._right_verifier is not None

    tab._left_verifier.pick_ref_from_display_xy(101, 50)
    tab._right_verifier.pick_ref_from_display_xy(303, 60)
    tab._save_all()

    refs_path = sample_block_path / "analysis" / "self_kerr_refs.csv"
    assert refs_path.is_file()
    refs = pd.read_csv(refs_path)
    for col in ("kerr_ref_r_x", "kerr_ref_r_y", "kerr_ref_l_x", "kerr_ref_l_y"):
        assert col in refs.columns
        assert pd.notna(refs.iloc[0][col])
    assert int(refs.iloc[0]["kerr_ref_l_x"]) == 101
    assert int(refs.iloc[0]["kerr_ref_l_y"]) == tab._left_verifier.frame_height - 1 - 50
    assert int(refs.iloc[0]["kerr_ref_r_x"]) == 303
