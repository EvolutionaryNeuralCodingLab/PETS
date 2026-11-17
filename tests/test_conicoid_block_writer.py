"""Block-level conicoid writer (v2 guard, Kerr principal point) and Kerr comparison."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.calculate_conicoid_angles import (
    EllipseGeometryError,
    calculate_conicoid_angles_for_block,
    check_block_geometry,
    read_conicoid_fit_meta,
)
from eye_tracking_system_tools.preprocessing.compare_conicoid_kerr import (
    compare_conicoid_kerr_for_block,
    comparison_paths,
    kerr_outputs_available,
)
from eye_tracking_system_tools.preprocessing.conicoid.angles import (
    angle_between_normals_deg,
    gazes_to_kerr_angles,
    yaw_pitch_to_normals,
)
from eye_tracking_system_tools.preprocessing.conicoid.geometry import Circle3D
from eye_tracking_system_tools.preprocessing.conicoid.unproject import project_circle
from eye_tracking_system_tools.preprocessing.ellipse_geometry_meta import (
    write_ellipse_geometry_meta,
)

FOCAL = 650.0
REF = {"left": (300.0, 250.0), "right": (335.0, 228.0)}
TAG = "test"


def _eye_table(ref: tuple[float, float]) -> tuple[pd.DataFrame, np.ndarray]:
    eye_c = np.array([0.4, -0.3, 13.0])
    radius = 4.0
    yaw, pitch = np.meshgrid(np.linspace(-25, 25, 6), np.linspace(-15, 15, 5))
    normals = yaw_pitch_to_normals(yaw.ravel(), pitch.ravel())
    rows = []
    for k, n in enumerate(normals):
        el = project_circle(Circle3D(centre=eye_c + radius * n, normal=n, radius=1.0), FOCAL)
        rows.append(
            {
                "OE_timestamp": 1000 + 10 * k,
                "eye_frame": k,
                "ms_axis": float(k),
                "center_x": el.cx + ref[0],
                "center_y": el.cy + ref[1],
                "width": el.major_radius,
                "height": el.minor_radius,
                "major_ax": el.major_radius,
                "minor_ax": el.minor_radius,
                "phi": float(np.mod(el.angle, np.pi)),
            }
        )
    df = pd.DataFrame(rows)
    # One blink row stays NaN end to end.
    blink = {c: np.nan for c in df.columns}
    blink.update(OE_timestamp=5, eye_frame=len(df), ms_axis=np.nan)
    df = pd.concat([df, pd.DataFrame([blink])], ignore_index=True)
    return df, normals


def _make_block(tmp_path: Path, *, v2_meta: bool = True) -> SimpleNamespace:
    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "kerr_ref_l_x": REF["left"][0],
                "kerr_ref_l_y": REF["left"][1],
                "kerr_ref_r_x": REF["right"][0],
                "kerr_ref_r_y": REF["right"][1],
            }
        ]
    ).to_csv(analysis / "self_kerr_refs.csv", index=False)
    pd.DataFrame([{"L_focal_length": FOCAL, "R_focal_length": FOCAL}]).to_csv(
        analysis / "LR_focal_length.csv", index=False
    )
    if v2_meta:
        write_ellipse_geometry_meta(analysis)
    left, left_n = _eye_table(REF["left"])
    right, right_n = _eye_table(REF["right"])
    return SimpleNamespace(
        block_num="001",
        block_path=tmp_path,
        analysis_path=analysis,
        left_eye_data=left,
        right_eye_data=right,
        truth={"left": left_n, "right": right_n},
    )


def _run(block, **kwargs):
    return calculate_conicoid_angles_for_block(
        block, name_tag=TAG, load_eye_data_flag=False, export_flag=False, use_ransac=False, **kwargs
    )


def test_writer_uses_kerr_ref_and_recovers_gaze(tmp_path: Path):
    block = _make_block(tmp_path)
    meta = _run(block)
    assert meta["sphere_method"] == "swirski_2d"
    for side in ("left", "right"):
        eye = meta["eyes"][side]
        assert eye["principal_point"] == list(REF[side])
        assert eye["principal_point_source"] == "self_kerr_refs.csv"
        assert eye["focal_px"] == FOCAL
        assert "LR_focal_length.csv" in eye["focal_source"]
        out = pd.read_csv(block.analysis_path / f"{side}_conicoid_angle_{TAG}.csv", index_col=0)
        n = out[["c_nx", "c_ny", "c_nz"]].to_numpy(dtype=float)
        assert np.isnan(n[-1]).all()
        err = angle_between_normals_deg(n[:-1], block.truth[side])
        assert np.nanmax(err) < 1e-4
    assert read_conicoid_fit_meta(block.analysis_path, TAG)["eyes"]["left"]["n_finite_normal"] == 30


def test_writer_keeps_near_circular_rows(tmp_path: Path):
    block = _make_block(tmp_path)
    block.left_eye_data.loc[0, "phi"] = np.nan
    _run(block)
    out = pd.read_csv(block.analysis_path / f"left_conicoid_angle_{TAG}.csv", index_col=0)
    assert np.isfinite(out.loc[0, ["c_nx", "c_ny", "c_nz"]].to_numpy(dtype=float)).all()


def test_writer_refuses_non_v2_geometry(tmp_path: Path):
    block = _make_block(tmp_path, v2_meta=False)
    with pytest.raises(EllipseGeometryError, match="ellipse_recovery"):
        _run(block)
    assert not (block.analysis_path / f"left_conicoid_angle_{TAG}.csv").exists()


def test_writer_uses_repaired_ellipse_v2_tables(tmp_path: Path):
    block = _make_block(tmp_path, v2_meta=False)
    repaired = block.analysis_path / "ellipse_v2"
    repaired.mkdir()
    write_ellipse_geometry_meta(repaired)
    block.left_eye_data.to_csv(repaired / "left_eye_data.csv")
    block.right_eye_data.to_csv(repaired / "right_eye_data.csv")
    sources = check_block_geometry(block.analysis_path)
    assert sources["left"] == "ellipse_v2/left_eye_data.csv"
    meta = _run(block)
    assert meta["eyes"]["right"]["ellipse_table"] == "ellipse_v2/right_eye_data.csv"


def test_repaired_dir_without_eye_tables_is_refused(tmp_path: Path):
    block = _make_block(tmp_path, v2_meta=False)
    repaired = block.analysis_path / "ellipse_v2"
    repaired.mkdir()
    write_ellipse_geometry_meta(repaired)
    with pytest.raises(EllipseGeometryError, match="manual review"):
        check_block_geometry(block.analysis_path)


def _write_kerr(block, yaw_offset_deg: float) -> dict[str, float]:
    """Kerr CSVs from the conicoid normals rotated about the vertical axis.

    Returns the expected median 3D angle per eye (frames 2.. are compared).
    """
    expected = {}
    for side in ("left", "right"):
        df = block.left_eye_data if side == "left" else block.right_eye_data
        conicoid = pd.read_csv(block.analysis_path / f"{side}_conicoid_angle_{TAG}.csv", index_col=0)
        n = conicoid[["c_nx", "c_ny", "c_nz"]].to_numpy(dtype=float)
        rot = np.deg2rad(yaw_offset_deg)
        r = np.array([[np.cos(rot), 0, np.sin(rot)], [0, 1, 0], [-np.sin(rot), 0, np.cos(rot)]])
        rotated = n @ r.T
        expected[side] = float(np.nanmedian(angle_between_normals_deg(n[2:], rotated[2:])))
        phi, theta = gazes_to_kerr_angles(rotated)
        kerr = pd.DataFrame(
            {
                "OE_timestamp": df["OE_timestamp"],
                "eye_frame": df["eye_frame"],
                "ms_axis": df["ms_axis"],
                "r": 1.0,
                "theta": theta,
                "phi": phi,
            }
        )
        kerr.iloc[:2, kerr.columns.get_loc("phi")] = np.nan
        kerr.to_csv(block.analysis_path / f"{side}_kerr_angle_{TAG}.csv")
    return expected


def test_compare_reports_3d_angle_and_finite_fractions(tmp_path: Path):
    block = _make_block(tmp_path)
    _run(block)
    assert not kerr_outputs_available(block, TAG)
    expected = _write_kerr(block, yaw_offset_deg=2.0)
    assert kerr_outputs_available(block, TAG)
    summary = compare_conicoid_kerr_for_block(block, name_tag=TAG)
    paths = comparison_paths(block.analysis_path, TAG)
    assert paths.frames.is_file() and paths.summary.is_file() and paths.plot.is_file()
    for row in summary.itertuples():
        assert 1.5 < expected[row.eye] <= 2.0
        assert row.median_angle_deg == pytest.approx(expected[row.eye], abs=1e-6)
        assert row.n_frames == 31
        assert row.frac_conicoid_finite == pytest.approx(30 / 31)
        assert row.frac_kerr_finite == pytest.approx(28 / 31)
        assert row.frac_both_finite == pytest.approx(28 / 31)
        assert row.principal_point_source == "self_kerr_refs.csv"
        assert row.sphere_method == "swirski_2d"
    frames = pd.read_csv(paths.frames)
    assert set(frames["eye"]) == {"left", "right"}


def test_compare_prefers_stored_normal(tmp_path: Path):
    block = _make_block(tmp_path)
    _run(block)
    _write_kerr(block, yaw_offset_deg=0.0)
    for side in ("left", "right"):
        path = block.analysis_path / f"{side}_conicoid_angle_{TAG}.csv"
        df = pd.read_csv(path, index_col=0)
        df["c_phi"] = 45.0
        df.to_csv(path)
    summary = compare_conicoid_kerr_for_block(block, name_tag=TAG, make_plot=False)
    assert summary["median_angle_deg"].max() < 1e-5


def test_compare_requires_both_outputs(tmp_path: Path):
    block = _make_block(tmp_path)
    _run(block)
    with pytest.raises(FileNotFoundError, match="kerr_angle"):
        compare_conicoid_kerr_for_block(block, name_tag=TAG)


def test_fit_meta_is_json(tmp_path: Path):
    block = _make_block(tmp_path)
    _run(block)
    data = json.loads((block.analysis_path / f"conicoid_fit_meta_{TAG}.json").read_text())
    assert data["use_ransac"] is False
    assert set(data["eyes"]) == {"left", "right"}
