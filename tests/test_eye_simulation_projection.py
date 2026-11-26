"""Analytic projection and Swirski recovery for the physical eye simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from physical_eye_simulation.analytic_projection import (
    load_eye_geometry,
    project_pupil_ellipses,
    rotation_matrix_yaw_pitch,
)
from physical_eye_simulation.camera_model import PaperCamera

GEOM = Path(__file__).resolve().parents[1] / "physical_eye_simulation" / "geometry"
HAS_GEOM = (GEOM / "eye_geometry_right.json").is_file()

pytestmark = pytest.mark.skipif(not HAS_GEOM, reason="eye_geometry_right.json missing")


def test_rotation_maps_rest_gaze_to_command_normal():
    yaw = np.array([0.0, 30.0, -20.0, 15.0])
    pitch = np.array([0.0, 0.0, 10.0, -25.0])
    r = rotation_matrix_yaw_pitch(yaw, pitch)
    rest_gaze = np.array([0.0, 0.0, -1.0])
    got = np.einsum("nij,j->ni", r, rest_gaze)
    x = np.deg2rad(yaw)
    y = np.deg2rad(pitch)
    want = np.stack([np.sin(x) * np.cos(y), np.sin(y), -np.cos(x) * np.cos(y)], axis=-1)
    assert np.allclose(got, want, atol=1e-10)


def test_rest_pupil_projects_to_image_centre():
    eye = load_eye_geometry("right", path=GEOM / "eye_geometry_right.json")
    frame = project_pupil_ellipses(
        eye,
        pivot_name="eye_centre",
        camera_distance_mm=13.0,
        yaw_deg=np.array([0.0]),
        pitch_deg=np.array([0.0]),
        camera=PaperCamera(),
    )
    row = frame.iloc[0]
    assert abs(row.center_x - 320.0) < 0.05
    assert abs(row.center_y - 240.0) < 0.05
    assert row.width == pytest.approx(row.height, rel=0.02)


def test_swirski_recovers_gaze_on_analytic_grid():
    from eye_tracking_system_tools.preprocessing.conicoid import (
        CameraIntrinsics,
        angle_between_normals_deg,
        fit_eye_and_gaze,
    )

    eye = load_eye_geometry("right", path=GEOM / "eye_geometry_right.json")
    cam = PaperCamera()
    yaw = np.arange(-40.0, 41.0, 5.0)
    pitch = np.arange(-40.0, 41.0, 5.0)
    yy, pp = np.meshgrid(yaw, pitch, indexing="ij")
    table = project_pupil_ellipses(
        eye,
        pivot_name="eye_centre",
        camera_distance_mm=13.0,
        yaw_deg=yy.ravel(),
        pitch_deg=pp.ravel(),
        camera=cam,
    )
    swirski_table = pd.DataFrame(
        {
            "center_x": table["center_x"],
            "center_y": table["center_y"],
            "width": table["width"],
            "height": table["height"],
            "major_ax": table["width"],
            "minor_ax": table["height"],
            "phi": table["phi"],
        }
    )
    camera = CameraIntrinsics(
        width=cam.width,
        height=cam.height,
        cx=cam.cx,
        cy=cam.cy,
        focal_px=cam.focal_px,
        source="paper",
    )
    pivot_z = float(table.attrs["pivot_cam"][2])
    out, _fit = fit_eye_and_gaze(
        swirski_table,
        camera,
        eye_z=pivot_z,
        pupil_radius=eye.pupil_radius_mm,
        use_ransac=False,
        sphere_method="swirski_2d",
    )
    reported = out[["c_nx", "c_ny", "c_nz"]].to_numpy(dtype=float)
    truth = table[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=float)
    err = angle_between_normals_deg(reported, truth)
    assert np.nanmedian(err) < 0.3
