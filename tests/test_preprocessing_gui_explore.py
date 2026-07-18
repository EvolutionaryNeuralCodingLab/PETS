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
    assert tab._video is not None
    assert tab._time_label is not None
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
