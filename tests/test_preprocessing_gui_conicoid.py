"""Conicoid tab: Swirski default, Dierkes note, v2 guard, Kerr comparison summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    ensure_config_template,
    load_config,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
)
from eye_tracking_system_tools.preprocessing.compare_conicoid_kerr import (
    comparison_paths,
)
from eye_tracking_system_tools.preprocessing.ellipse_geometry_meta import (
    write_ellipse_geometry_meta,
)


def _tab(tmp_path: Path):
    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.conicoid_tab import (
        ConicoidTab,
    )

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    ensure_config_template(cfg_dir)
    return ConicoidTab(GuiState(output_folder=cfg_dir), load_config(None, cfg_dir))


def _handle(tmp_path: Path, *, v2: bool) -> BlockHandle:
    block_path = tmp_path / "PV_000" / "2026_01_01" / "block_001"
    (block_path / "analysis").mkdir(parents=True)
    if v2:
        write_ellipse_geometry_meta(block_path / "analysis")
    return BlockHandle(
        animal_call="PV_000",
        experiment_date="2026_01_01",
        block_num="001",
        block_path=block_path,
        path_to_animal_folder=tmp_path / "PV_000",
    )


def test_swirski_is_default_and_dierkes_is_flagged(qapp_session, tmp_path):
    tab = _tab(tmp_path)
    assert tab._sphere_method.currentText() == "Swirski 2D"
    assert tab._sphere_method_value() == "swirski_2d"
    assert tab._dierkes_note.isHidden()
    tab._sphere_method.setCurrentIndex(1)
    assert tab._sphere_method_value() == "dierkes_3d"
    assert not tab._dierkes_note.isHidden()
    assert "provisional" in tab._sphere_method.currentText()
    assert tab._ransac.isChecked()


def test_geometry_guard_controls_calculate(qapp_session, tmp_path):
    tab = _tab(tmp_path)
    tab.set_block(_handle(tmp_path / "legacy", v2=False))
    assert not tab._btn_calculate.isEnabled()
    assert "Not geometry v2" in tab._geometry_status.text()
    tab.set_block(_handle(tmp_path / "current", v2=True))
    assert tab._btn_calculate.isEnabled()
    assert tab._geometry_status.text().startswith("v2")


def test_compare_summary_shown_from_disk(qapp_session, tmp_path):
    tab = _tab(tmp_path)
    handle = _handle(tmp_path, v2=True)
    tab._name_tag.setText("t1")
    tab.set_block(handle)
    assert not tab._btn_compare.isEnabled()
    assert "Conicoid and Kerr" in tab._compare_summary.text()

    ap = handle.analysis_path
    frame = pd.DataFrame({"OE_timestamp": [0, 1], "eye_frame": [0, 1]})
    for side in ("left", "right"):
        frame.assign(c_nx=0.0, c_ny=0.0, c_nz=-1.0).to_csv(ap / f"{side}_conicoid_angle_t1.csv")
        frame.assign(phi=0.0, theta=0.0).to_csv(ap / f"{side}_kerr_angle_t1.csv")
    tab._refresh_compare_ui()
    assert tab._btn_compare.isEnabled()

    pd.DataFrame(
        [
            {
                "eye": side,
                "n_frames": 2,
                "frac_conicoid_finite": 1.0,
                "frac_kerr_finite": 1.0,
                "frac_both_finite": 1.0,
                "median_angle_deg": 1.25,
                "p90_angle_deg": 2.5,
            }
            for side in ("left", "right")
        ]
    ).to_csv(comparison_paths(ap, "t1").summary, index=False)
    tab._refresh_compare_ui()
    text = tab._compare_summary.text()
    assert "median 1.25 deg" in text
    assert text.startswith("left:") and "\nright:" in text
