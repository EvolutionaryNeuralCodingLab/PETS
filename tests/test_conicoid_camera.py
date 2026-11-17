"""Pinhole-intrinsics resolution order for the conicoid pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eye_tracking_system_tools.preprocessing.conicoid.camera import (
    DEFAULT_FOCAL_PX,
    default_intrinsics,
    kerr_ref_principal_point,
    resolve_intrinsics,
)


def test_default_intrinsics_uses_documented_fallback():
    cam = default_intrinsics("left")
    assert cam.width == 640
    assert cam.height == 480
    assert cam.cx == 320.0
    assert cam.cy == 240.0
    assert cam.focal_px == DEFAULT_FOCAL_PX
    assert "qilens" in cam.source.lower() or "ov5647" in cam.source.lower()


def test_resolve_prefers_focal_length_csv(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame([{"L_focal_length": 700.0, "R_focal_length": 710.0}]).to_csv(
        analysis / "LR_focal_length.csv", index=False
    )
    pd.DataFrame([{"L_pix_size": 0.01, "R_pix_size": 0.02}]).to_csv(
        analysis / "LR_pix_size.csv", index=False
    )
    left = resolve_intrinsics(tmp_path, "left")
    right = resolve_intrinsics(tmp_path, "right")
    assert left.focal_px == 700.0
    assert right.focal_px == 710.0
    assert "LR_focal_length.csv" in left.source


def test_resolve_ignores_eye_plane_pixel_size(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame([{"L_pix_size": 0.01, "R_pix_size": 0.02}]).to_csv(
        analysis / "LR_pix_size.csv", index=False
    )
    left = resolve_intrinsics(tmp_path, "left")
    assert left.focal_px == DEFAULT_FOCAL_PX
    assert "LR_pix_size.csv" not in left.source


def test_principal_point_defaults_to_image_center(tmp_path: Path):
    (tmp_path / "analysis").mkdir()
    cam = resolve_intrinsics(tmp_path, "left")
    assert (cam.cx, cam.cy) == (320.0, 240.0)
    assert cam.principal_point_source == "image_center"


def test_principal_point_from_kerr_refs(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame(
        [{"kerr_ref_r_x": 301.5, "kerr_ref_r_y": 255.0, "kerr_ref_l_x": 330.0, "kerr_ref_l_y": 231.25}]
    ).to_csv(analysis / "self_kerr_refs.csv", index=False)
    pd.DataFrame([{"L_focal_length": 700.0, "R_focal_length": 710.0}]).to_csv(
        analysis / "LR_focal_length.csv", index=False
    )
    left_ref = kerr_ref_principal_point(analysis, "left")
    right_ref = kerr_ref_principal_point(tmp_path, "right")
    assert left_ref == (330.0, 231.25)
    assert right_ref == (301.5, 255.0)
    cam = resolve_intrinsics(
        tmp_path, "left", principal_point=left_ref, principal_point_source="self_kerr_refs.csv"
    )
    assert (cam.cx, cam.cy) == left_ref
    assert cam.focal_px == 700.0
    assert cam.principal_point_source == "self_kerr_refs.csv"
    fallback = resolve_intrinsics(None, "left", principal_point=left_ref)
    assert (fallback.cx, fallback.cy) == left_ref
    assert fallback.focal_px == DEFAULT_FOCAL_PX


def test_kerr_ref_missing_returns_none(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    assert kerr_ref_principal_point(tmp_path, "left") is None
    pd.DataFrame([{"kerr_ref_r_x": 300.0, "kerr_ref_r_y": 250.0}]).to_csv(
        analysis / "self_kerr_refs.csv", index=False
    )
    assert kerr_ref_principal_point(tmp_path, "left") is None


def test_resolve_logs_fallback_when_missing(tmp_path: Path, caplog):
    (tmp_path / "analysis").mkdir()
    with caplog.at_level("WARNING"):
        cam = resolve_intrinsics(tmp_path, "left")
    assert cam.focal_px == DEFAULT_FOCAL_PX
    assert any("fallback" in rec.message.lower() for rec in caplog.records)
