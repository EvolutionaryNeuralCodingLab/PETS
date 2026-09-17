"""Refine residuals, Dierkes 3D intersection, refraction apply, refined CSV I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.conicoid.camera import CameraIntrinsics
from eye_tracking_system_tools.preprocessing.conicoid.eye_model import (
    Sphere,
    circle_from_params,
    dierkes_eyeball_centre,
    fit_sphere_from_ellipses,
    params_from_circle,
)
from eye_tracking_system_tools.preprocessing.conicoid.geometry import Circle3D
from eye_tracking_system_tools.preprocessing.conicoid.refine import (
    PixelEllipse,
    RefineSettings,
    edge_distance_cost,
    refine_circle,
    refine_eye_table,
    region_contrast_cost,
    signed_ellipse_distance,
)
from eye_tracking_system_tools.preprocessing.conicoid.refined_io import (
    overlay_refined_geometry,
    promote_refined_to_eye_data,
    refined_eye_csv_path,
    write_refined_eye_table,
)
from eye_tracking_system_tools.preprocessing.conicoid.refraction import (
    apply_refraction_to_fit,
    identity_refraction_maps,
    save_refraction_maps,
)
from eye_tracking_system_tools.preprocessing.conicoid.unproject import (
    angular_error_deg,
    project_circle,
)


def _grid_pupils(eye_c: np.ndarray, radius: float, focal: float):
    ellipses = []
    true_n = []
    circles = []
    for ph in np.linspace(-0.4, 0.4, 6):
        for th in np.linspace(-0.3, 0.3, 5):
            nx = np.sin(ph)
            ny = np.sin(th) * np.cos(ph)
            nz = -np.sqrt(max(1e-9, 1.0 - nx * nx - ny * ny))
            n = np.array([nx, ny, nz])
            n = n / np.linalg.norm(n)
            circ = Circle3D(centre=eye_c + radius * n, normal=n, radius=1.0)
            ellipses.append(project_circle(circ, focal))
            true_n.append(n)
            circles.append(circ)
    return ellipses, true_n, circles


def test_dierkes_line_intersection_recovers_sphere():
    f = 250.0
    eye_c = np.array([0.0, 1.0, 13.0])
    radius = 4.0
    ellipses, true_n, circles = _grid_pupils(eye_c, radius, f)
    recovered = dierkes_eyeball_centre(circles, radius)
    assert np.linalg.norm(recovered - eye_c) == pytest.approx(0.0, abs=1e-8)

    fit_d = fit_sphere_from_ellipses(
        ellipses,
        focal_length=f,
        pupil_radius=1.0,
        eye_z=13.0,
        use_ransac=False,
        sphere_method="dierkes_3d",
    )
    fit_s = fit_sphere_from_ellipses(
        ellipses,
        focal_length=f,
        pupil_radius=1.0,
        eye_z=13.0,
        use_ransac=False,
        sphere_method="swirski_2d",
    )
    assert np.linalg.norm(fit_d.sphere.centre - eye_c) == pytest.approx(0.0, abs=1e-7)
    assert np.linalg.norm(fit_s.sphere.centre - eye_c) == pytest.approx(0.0, abs=1e-7)
    assert np.linalg.norm(fit_d.sphere.centre - fit_s.sphere.centre) < 1e-6
    errs = [
        angular_error_deg(obs.circle.normal, n)
        for obs, n in zip(fit_d.pupils, true_n)
        if obs.circle is not None
    ]
    assert max(errs) == pytest.approx(0.0, abs=1e-5)


def test_identity_refraction_leaves_fit_unchanged():
    f = 250.0
    eye_c = np.array([0.4, -0.3, 13.0])
    ellipses, true_n, _ = _grid_pupils(eye_c, 4.0, f)
    fit = fit_sphere_from_ellipses(
        ellipses, focal_length=f, eye_z=13.0, use_ransac=False
    )
    before = fit.sphere.centre.copy()
    apply_refraction_to_fit(fit, identity_refraction_maps())
    assert np.linalg.norm(fit.sphere.centre - before) == pytest.approx(0.0, abs=1e-10)
    assert angular_error_deg(fit.pupils[0].circle.normal, true_n[0]) < 1e-5


def test_reye_translation_map_shifts_centre(tmp_path: Path):
    f = 250.0
    eye_c = np.array([0.0, 1.0, 13.0])
    ellipses, _, _ = _grid_pupils(eye_c, 4.0, f)
    fit = fit_sphere_from_ellipses(
        ellipses, focal_length=f, eye_z=13.0, use_ransac=False
    )
    maps = identity_refraction_maps()
    coef = np.array(maps.reye_coef, copy=True)
    coef[0, 0] = 0.25  # intercept → +0.25 on Ex
    from eye_tracking_system_tools.preprocessing.conicoid.refraction import RefractionMaps

    maps = RefractionMaps(reye_coef=coef, n_ref=maps.n_ref)
    path = tmp_path / "conicoid_refraction.npz"
    save_refraction_maps(path, maps)
    apply_refraction_to_fit(fit, maps)
    assert fit.sphere.centre[0] == pytest.approx(eye_c[0] + 0.25, abs=1e-6)


def _facing_disk_scene(camera: CameraIntrinsics):
    sphere = Sphere(centre=np.array([0.0, 0.0, 13.0]), radius=4.0)
    circ = Circle3D(
        centre=sphere.centre + sphere.radius * np.array([0.0, 0.0, -1.0]),
        normal=np.array([0.0, 0.0, -1.0]),
        radius=1.0,
    )
    el = project_circle(circ, camera.focal_px)
    pel = PixelEllipse(
        cx=el.cx + camera.cx,
        cy=el.cy + camera.cy,
        major_radius=el.major_radius,
        minor_radius=el.minor_radius,
        angle=el.angle,
    )
    h, w = int(camera.height), int(camera.width)
    img = np.full((h, w), 180, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    d = signed_ellipse_distance(xx, yy, pel)
    img[d > 0] = 20
    return sphere, circ, pel, img


def test_region_contrast_prefers_true_ellipse_over_offset():
    camera = CameraIntrinsics(width=128, height=128, cx=64.0, cy=64.0, focal_px=250.0)
    _sphere, _circ, pel, img = _facing_disk_scene(camera)
    true_cost = region_contrast_cost(img, pel, band_width=5.0, epsilon=0.5)
    bad = PixelEllipse(
        cx=pel.cx + 4.0,
        cy=pel.cy - 3.0,
        major_radius=pel.major_radius * 0.85,
        minor_radius=pel.minor_radius * 0.85,
        angle=pel.angle,
    )
    bad_cost = region_contrast_cost(img, bad, band_width=5.0, epsilon=0.5)
    assert true_cost < bad_cost


def test_refine_contrast_shrinks_residual_on_dark_disk():
    camera = CameraIntrinsics(width=128, height=128, cx=64.0, cy=64.0, focal_px=250.0)
    sphere, circ, pel, img = _facing_disk_scene(camera)
    theta, psi, r = params_from_circle(sphere, circ)
    start = circle_from_params(sphere, theta + 0.08, psi - 0.05, r * 0.85)
    before = region_contrast_cost(
        img, PixelEllipse(
            cx=project_circle(start, camera.focal_px).cx + camera.cx,
            cy=project_circle(start, camera.focal_px).cy + camera.cy,
            major_radius=project_circle(start, camera.focal_px).major_radius,
            minor_radius=project_circle(start, camera.focal_px).minor_radius,
            angle=project_circle(start, camera.focal_px).angle,
        ),
        band_width=5.0,
        epsilon=0.5,
    )
    refined = refine_circle(
        img,
        start,
        sphere,
        camera,
        settings=RefineSettings(metric="contrast", lock_sphere=True),
    )
    after_pel = PixelEllipse(
        cx=project_circle(refined, camera.focal_px).cx + camera.cx,
        cy=project_circle(refined, camera.focal_px).cy + camera.cy,
        major_radius=project_circle(refined, camera.focal_px).major_radius,
        minor_radius=project_circle(refined, camera.focal_px).minor_radius,
        angle=project_circle(refined, camera.focal_px).angle,
    )
    after = region_contrast_cost(img, after_pel, band_width=5.0, epsilon=0.5)
    assert after < before
    assert abs(after_pel.cx - pel.cx) < abs(
        project_circle(start, camera.focal_px).cx + camera.cx - pel.cx
    )


def test_edge_distance_refine_snaps_to_inliers():
    camera = CameraIntrinsics(width=128, height=128, cx=64.0, cy=64.0, focal_px=250.0)
    sphere, circ, pel, img = _facing_disk_scene(camera)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    c, s = np.cos(pel.angle), np.sin(pel.angle)
    xr = pel.major_radius * np.cos(angles)
    yr = pel.minor_radius * np.sin(angles)
    xs = pel.cx + c * xr - s * yr
    ys = pel.cy + s * xr + c * yr
    pts = np.column_stack([xs, ys])
    theta, psi, r = params_from_circle(sphere, circ)
    start = circle_from_params(sphere, theta + 0.1, psi, r * 0.7)
    start_el = project_circle(start, camera.focal_px)
    start_pel = PixelEllipse(
        cx=start_el.cx + camera.cx,
        cy=start_el.cy + camera.cy,
        major_radius=start_el.major_radius,
        minor_radius=start_el.minor_radius,
        angle=start_el.angle,
    )
    before = edge_distance_cost(pts, start_pel)
    refined = refine_circle(
        img,
        start,
        sphere,
        camera,
        settings=RefineSettings(metric="edge"),
        inliers=pts,
    )
    after_el = project_circle(refined, camera.focal_px)
    after_pel = PixelEllipse(
        cx=after_el.cx + camera.cx,
        cy=after_el.cy + camera.cy,
        major_radius=after_el.major_radius,
        minor_radius=after_el.minor_radius,
        angle=after_el.angle,
    )
    after = edge_distance_cost(pts, after_pel)
    assert after < before


def test_refine_eye_table_writes_refined_columns():
    camera = CameraIntrinsics(width=128, height=128, cx=64.0, cy=64.0, focal_px=250.0)
    f = camera.focal_px
    eye_c = np.array([0.0, 0.0, 13.0])
    ellipses, _, circles = _grid_pupils(eye_c, 4.0, f)
    rows = []
    for k, (el, circ) in enumerate(zip(ellipses, circles)):
        rows.append(
            {
                "OE_timestamp": k,
                "eye_frame": k,
                "center_x": el.cx + camera.cx,
                "center_y": el.cy + camera.cy,
                "width": el.major_radius,
                "height": el.minor_radius,
                "major_ax": el.major_radius,
                "minor_ax": el.minor_radius,
                "phi": el.angle,
            }
        )
    df = pd.DataFrame(rows)
    sphere, circ0, pel, img = _facing_disk_scene(camera)
    frames = {int(r["eye_frame"]): img for r in rows}

    def grab(idx: int):
        return frames.get(idx)

    out, fit = refine_eye_table(
        df,
        camera,
        frame_image=grab,
        settings=RefineSettings(
            metric="contrast", use_ransac=False, sphere_method="dierkes_3d"
        ),
        frame_indices=[0, 1],
    )
    assert "center_x_refined" in out.columns
    assert np.isfinite(out.loc[0, "center_x_refined"])
    assert fit.sphere.radius > 0


def test_refine_eye_table_leaves_nan_without_image():
    camera = CameraIntrinsics(width=128, height=128, cx=64.0, cy=64.0, focal_px=250.0)
    f = camera.focal_px
    eye_c = np.array([0.0, 0.0, 13.0])
    ellipses, _, _ = _grid_pupils(eye_c, 4.0, f)
    rows = []
    for k, el in enumerate(ellipses[:6]):
        rows.append(
            {
                "OE_timestamp": k,
                "eye_frame": k,
                "center_x": el.cx + camera.cx,
                "center_y": el.cy + camera.cy,
                "width": el.major_radius,
                "height": el.minor_radius,
                "major_ax": el.major_radius,
                "minor_ax": el.minor_radius,
                "phi": el.angle,
            }
        )
    df = pd.DataFrame(rows)
    out, _fit = refine_eye_table(
        df,
        camera,
        frame_image=None,
        settings=RefineSettings(
            metric="contrast", use_ransac=False, sphere_method="dierkes_3d"
        ),
        frame_indices=[0, 1],
    )
    assert out.loc[0:1, "center_x_refined"].isna().all()
    assert out.loc[0:1, "phi_refined"].isna().all()


def test_promote_refined_backs_up_and_copies_geometry(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    eye = pd.DataFrame(
        {
            "OE_timestamp": [0, 1],
            "eye_frame": [0, 1],
            "center_x": [10.0, 11.0],
            "center_y": [20.0, 21.0],
            "width": [5.0, 5.0],
            "height": [4.0, 4.0],
            "phi": [0.1, 0.2],
            "major_ax": [5.0, 5.0],
            "minor_ax": [4.0, 4.0],
        }
    )
    eye_path = analysis / "left_eye_data.csv"
    eye.to_csv(eye_path, index=False)
    refined = eye.copy()
    refined["center_x_refined"] = [30.0, 31.0]
    refined["center_y_refined"] = [40.0, 41.0]
    refined["width_refined"] = [6.0, 6.5]
    refined["height_refined"] = [3.0, 3.5]
    refined["phi_refined"] = [0.5, 0.6]
    ref_path = refined_eye_csv_path(tmp_path, "left", "gui")
    write_refined_eye_table(refined, ref_path)
    backup = promote_refined_to_eye_data(
        tmp_path, "left", ref_path, timestamp="20260101T000000Z"
    )
    assert backup.name == "left_eye_data_pre_promote_20260101T000000Z.csv"
    backed = pd.read_csv(backup)
    pd.testing.assert_frame_equal(backed, eye)
    promoted = pd.read_csv(eye_path)
    assert promoted.loc[0, "center_x"] == pytest.approx(30.0)
    assert promoted.loc[0, "width"] == pytest.approx(6.0)
    assert promoted.loc[0, "major_ax"] == pytest.approx(6.0)
    assert promoted.loc[0, "minor_ax"] == pytest.approx(3.0)
    assert "center_x_refined" not in promoted.columns


def test_overlay_refined_geometry_merges_on_eye_frame():
    eye = pd.DataFrame(
        {
            "eye_frame": [0, 1, 2],
            "center_x": [1.0, 2.0, 3.0],
            "width": [4.0, 4.0, 4.0],
            "height": [3.0, 3.0, 3.0],
            "phi": [0.0, 0.0, 0.0],
            "center_y": [0.0, 0.0, 0.0],
        }
    )
    refined = pd.DataFrame(
        {
            "eye_frame": [1, 0],
            "center_x_refined": [20.0, 10.0],
            "center_y_refined": [0.0, 0.0],
            "width_refined": [5.0, 6.0],
            "height_refined": [2.0, 2.5],
            "phi_refined": [0.2, 0.1],
        }
    )
    out = overlay_refined_geometry(eye, refined)
    assert out.loc[0, "center_x"] == pytest.approx(10.0)
    assert out.loc[1, "center_x"] == pytest.approx(20.0)
    assert out.loc[2, "center_x"] == pytest.approx(3.0)
