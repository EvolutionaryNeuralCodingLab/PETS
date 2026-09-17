"""Pinhole-intrinsics resolution order for the conicoid pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eye_tracking_system_tools.preprocessing.conicoid.camera import (
    DEFAULT_FOCAL_PX,
    default_intrinsics,
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


def test_resolve_falls_back_to_pixel_size(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame([{"L_pix_size": 0.01, "R_pix_size": 0.02}]).to_csv(
        analysis / "LR_pix_size.csv", index=False
    )
    left = resolve_intrinsics(tmp_path, "left")
    assert left.focal_px == 3.7 / 0.01
    assert "LR_pix_size.csv" in left.source


def test_resolve_logs_fallback_when_missing(tmp_path: Path, caplog):
    (tmp_path / "analysis").mkdir()
    with caplog.at_level("WARNING"):
        cam = resolve_intrinsics(tmp_path, "left")
    assert cam.focal_px == DEFAULT_FOCAL_PX
    assert any("fallback" in rec.message.lower() for rec in caplog.records)
