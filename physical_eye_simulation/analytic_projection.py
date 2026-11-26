"""Exact pinhole projection of a circular pupil under yaw/pitch eye rotations.

Camera frame: origin at the pinhole, +Z toward the scene, +X right, +Y down.
At rest the optical axis is +Z and the gaze toward the camera is −Z. Rotations
use the same nested yaw-then-pitch chart as ``yaw_pitch_to_normals``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from physical_eye_simulation.camera_model import (
    CAMERA_DISTANCES_MM,
    PIVOTS,
    PaperCamera,
    project_points,
)

ANNOTATIONS = Path(__file__).resolve().parent / "geometry"
CIRCLE_SAMPLES = 128


@dataclass(frozen=True)
class EyeGeometry:
    side: str
    eye_centre_mm: np.ndarray
    sclera_sphere_centre_mm: np.ndarray
    lens_centre_mm: np.ndarray
    fovea_pit_mm: np.ndarray
    optical_axis: np.ndarray  # unit, fovea → lens
    pupil_centre_mm: np.ndarray
    pupil_normal: np.ndarray  # unit, roughly along optical_axis
    pupil_radius_mm: float
    raw: dict

    def pivot(self, name: str) -> np.ndarray:
        if name == "eye_centre":
            return self.eye_centre_mm
        if name == "sclera_sphere_centre":
            return self.sclera_sphere_centre_mm
        if name == "lens_centre":
            return self.lens_centre_mm
        raise KeyError(name)


def load_eye_geometry(side: str, path: Path | None = None) -> EyeGeometry:
    path = path or (ANNOTATIONS / f"eye_geometry_{side}.json")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    axis = np.asarray(raw["optical_axis_from_fovea_to_lens"], dtype=float)
    axis = axis / np.linalg.norm(axis)
    pupil = raw["pupil"]
    normal = np.asarray(pupil["normal_from_fovea_to_lens"], dtype=float)
    normal = normal / np.linalg.norm(normal)
    if normal.dot(axis) < 0:
        normal = -normal
    return EyeGeometry(
        side=side,
        eye_centre_mm=np.asarray(raw["eye_centre_mm"], dtype=float),
        sclera_sphere_centre_mm=np.asarray(raw["sclera_sphere"]["centre_mm"], dtype=float),
        lens_centre_mm=np.asarray(raw["lens_centre_mm"], dtype=float),
        fovea_pit_mm=np.asarray(raw["fovea_pit_mm"], dtype=float),
        optical_axis=axis,
        pupil_centre_mm=np.asarray(pupil["centre_mm"], dtype=float),
        pupil_normal=normal,
        pupil_radius_mm=float(pupil["radius_mm"]),
        raw=raw,
    )


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (axis, u, v) with u,v spanning the plane perpendicular to axis."""
    a = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, u, v


def rotation_matrix_yaw_pitch(yaw_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    """Rotation that maps rest gaze (−Z) to yaw_pitch_to_normals(yaw, pitch).

    Returns (N, 3, 3). Columns are the images of the camera-frame basis vectors.
    """
    x = np.deg2rad(np.asarray(yaw_deg, dtype=float))
    y = np.deg2rad(np.asarray(pitch_deg, dtype=float))
    # Gaze n = (sin x cos y, sin y, −cos x cos y); anterior = −n.
    # Build R that maps e_z → −n and keeps a right-handed frame with
    # e_x → (cos x, 0, sin x) style horizontal.
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    # New Z (optical / anterior): −n = (−sin x cos y, −sin y, cos x cos y)
    z_new = np.stack([-sx * cy, -sy, cx * cy], axis=-1)
    # New X: horizontal, perpendicular to world-Y then orthonormalised to Z.
    # At y=0: x_new = (cos x, 0, sin x).
    x_tmp = np.stack([cx, np.zeros_like(cx), sx], axis=-1)
    y_new = np.cross(z_new, x_tmp)
    y_new /= np.linalg.norm(y_new, axis=-1, keepdims=True)
    x_new = np.cross(y_new, z_new)
    # Rows of R map camera coords; we want p' = R @ p for column vectors,
    # with R's columns = images of basis → stack x_new, y_new, z_new as columns.
    return np.stack([x_new, y_new, z_new], axis=-1)  # (N, 3, 3)


def mri_to_camera_rest(
    eye: EyeGeometry,
    pivot_name: str,
    camera_distance_mm: float,
) -> dict:
    """Rigid transform placing the eye at rest in the camera frame.

    Camera at the origin looks along +Z. The pupil (anterior) sits at
    ``Z = camera_distance_mm``, closer to the camera than the pivot. The MRI
    optical axis (fovea → lens, anterior) maps to **−Z** (toward the camera),
    which is also the rest gaze normal.
    """
    axis, u, v = _orthonormal_basis(eye.optical_axis)
    # MRI anterior (+axis) → camera −Z; MRI u → +X; v → +Y (down).
    # camera = R @ (mri - pupil) + (0,0,D) with R rows = camera axes in MRI.
    r_align = np.stack([u, v, -axis], axis=0)

    def xform(points: np.ndarray) -> np.ndarray:
        local = np.asarray(points, dtype=float) - eye.pupil_centre_mm
        return (r_align @ local.T).T + np.array([0.0, 0.0, float(camera_distance_mm)])

    pivot_mri = eye.pivot(pivot_name)
    pupil_cam = np.array([0.0, 0.0, float(camera_distance_mm)])
    pivot_raw = xform(pivot_mri)
    # Centred-pupil baseline: keep only the axial component of the pivot so the
    # pupil sits on the gaze ray through the pivot (Swirski's assumption). The
    # discarded lateral offset is anatomical and is recorded for the report.
    lateral = float(np.linalg.norm(pivot_raw[:2]))
    pivot_cam = np.array([0.0, 0.0, float(pivot_raw[2])])
    return {
        "R_align": r_align,
        "pupil_centre_cam": pupil_cam,
        "pupil_normal_cam": np.array([0.0, 0.0, -1.0]),  # toward camera
        "pupil_radius_mm": eye.pupil_radius_mm,
        "pivot_cam": pivot_cam,
        "pivot_raw_cam": pivot_raw.tolist(),
        "pivot_lateral_offset_mm": lateral,
        "xform": xform,
    }


def rotate_about_pivot(
    points: np.ndarray,
    pivot: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    """Apply (N,3,3) rotations about ``pivot`` to points (M,3) → (N,M,3)."""
    rel = points - pivot
    # (N,3,3) @ (M,3) → (N,M,3)
    return np.einsum("nij,mj->nmi", rotations, rel) + pivot


def fit_ellipse_from_points(uv: np.ndarray) -> tuple[float, float, float, float, float]:
    """Algebraic ellipse fit; returns centre_x, centre_y, major, minor, phi_rad."""
    u = uv[:, 0]
    v = uv[:, 1]
    design = np.column_stack([u * u, u * v, v * v, u, v, np.ones_like(u)])
    _, _, vt = np.linalg.svd(design)
    a, b, c, d, e, f = vt[-1]
    mat = np.array([[a, b / 2.0], [b / 2.0, c]])
    centre = np.linalg.solve(2.0 * mat, -np.array([d, e]))
    k = (
        a * centre[0] ** 2
        + b * centre[0] * centre[1]
        + c * centre[1] ** 2
        + d * centre[0]
        + e * centre[1]
        + f
    )
    w, vec = np.linalg.eigh(mat)
    if np.any(w == 0) or np.any((-k / w) <= 0):
        return (
            float(centre[0]),
            float(centre[1]),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    semi = np.sqrt(-k / w)
    major_idx = int(np.argmax(semi))
    major_dir = vec[:, major_idx]
    phi = float(np.mod(np.arctan2(major_dir[1], major_dir[0]), np.pi))
    return float(centre[0]), float(centre[1]), float(semi.max()), float(semi.min()), phi


def project_pupil_ellipses(
    eye: EyeGeometry,
    *,
    pivot_name: str,
    camera_distance_mm: float,
    yaw_deg: np.ndarray,
    pitch_deg: np.ndarray,
    camera: PaperCamera = PaperCamera(),
    n_circle: int = CIRCLE_SAMPLES,
) -> pd.DataFrame:
    """Exact projected ellipses for every (yaw, pitch) sample."""
    rest = mri_to_camera_rest(eye, pivot_name, camera_distance_mm)
    # Pupil circle in the rest camera frame.
    t = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)
    # Basis in the pupil plane (XY at rest).
    circle = np.stack(
        [
            rest["pupil_radius_mm"] * np.cos(t),
            rest["pupil_radius_mm"] * np.sin(t),
            np.zeros_like(t),
        ],
        axis=1,
    )
    circle += rest["pupil_centre_cam"]

    rotations = rotation_matrix_yaw_pitch(yaw_deg, pitch_deg)
    # (N, n_circle, 3)
    world = rotate_about_pivot(circle, rest["pivot_cam"], rotations)
    # Gaze normals (toward camera).
    x = np.deg2rad(np.asarray(yaw_deg, dtype=float))
    y = np.deg2rad(np.asarray(pitch_deg, dtype=float))
    gaze = np.stack(
        [np.sin(x) * np.cos(y), np.sin(y), -np.cos(x) * np.cos(y)],
        axis=-1,
    )
    pupil_centres = rotate_about_pivot(
        rest["pupil_centre_cam"][None, :],
        rest["pivot_cam"],
        rotations,
    )[:, 0, :]

    rows = []
    for i in range(len(yaw_deg)):
        pts = world[i]
        if np.any(pts[:, 2] <= 1e-6):
            rows.append(
                {
                    "frame": i,
                    "x_angle": float(yaw_deg[i]),
                    "y_angle": float(pitch_deg[i]),
                    "gaze_x": float(gaze[i, 0]),
                    "gaze_y": float(gaze[i, 1]),
                    "gaze_z": float(gaze[i, 2]),
                    "pupil_centre_x_mm": float(pupil_centres[i, 0]),
                    "pupil_centre_y_mm": float(pupil_centres[i, 1]),
                    "pupil_centre_z_mm": float(pupil_centres[i, 2]),
                    "center_x": np.nan,
                    "center_y": np.nan,
                    "width": np.nan,
                    "height": np.nan,
                    "phi": np.nan,
                }
            )
            continue
        uv = project_points(pts, camera)
        cx, cy, major, minor, phi = fit_ellipse_from_points(uv)
        rows.append(
            {
                "frame": i,
                "x_angle": float(yaw_deg[i]),
                "y_angle": float(pitch_deg[i]),
                "gaze_x": float(gaze[i, 0]),
                "gaze_y": float(gaze[i, 1]),
                "gaze_z": float(gaze[i, 2]),
                "pupil_centre_x_mm": float(pupil_centres[i, 0]),
                "pupil_centre_y_mm": float(pupil_centres[i, 1]),
                "pupil_centre_z_mm": float(pupil_centres[i, 2]),
                "center_x": cx,
                "center_y": cy,
                "width": major,
                "height": minor,
                "phi": phi,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["pivot"] = pivot_name
    frame.attrs["camera_distance_mm"] = camera_distance_mm
    frame.attrs["pivot_cam"] = rest["pivot_cam"].tolist()
    frame.attrs["pivot_lateral_offset_mm"] = rest["pivot_lateral_offset_mm"]
    return frame


def angle_grid(step_deg: float = 1.0, span_deg: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(-span_deg, span_deg + 0.5 * step_deg, step_deg)
    yy, xx = np.meshgrid(angles, angles, indexing="xy")
    # xx varies over columns (yaw), yy over rows (pitch) — flatten C-order so
    # consecutive frames scan yaw fastest? Use yaw outer for readability:
    yaw, pitch = np.meshgrid(angles, angles, indexing="ij")
    return yaw.ravel(), pitch.ravel()


def run_scenario(
    side: str,
    pivot_name: str,
    camera_distance_mm: float,
    *,
    step_deg: float = 1.0,
    span_deg: float = 60.0,
    out_dir: Path | None = None,
    geometry_path: Path | None = None,
    camera: PaperCamera | None = None,
) -> pd.DataFrame:
    cam = camera or PaperCamera()
    eye = load_eye_geometry(side, path=geometry_path)
    yaw, pitch = angle_grid(step_deg=step_deg, span_deg=span_deg)
    frame = project_pupil_ellipses(
        eye,
        pivot_name=pivot_name,
        camera_distance_mm=camera_distance_mm,
        yaw_deg=yaw,
        pitch_deg=pitch,
        camera=cam,
    )
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "ground_truth_analytic.csv"
        frame.to_csv(csv_path, index=False)
        meta = {
            "side": side,
            "pivot": pivot_name,
            "camera_distance_mm": camera_distance_mm,
            "step_deg": step_deg,
            "span_deg": span_deg,
            "n_frames": int(len(frame)),
            "camera": cam.to_dict(),
            "eye_geometry": eye.raw,
            "pivot_cam_mm": frame.attrs["pivot_cam"],
            "pivot_lateral_offset_mm": frame.attrs["pivot_lateral_offset_mm"],
        }
        (out_dir / "scenario_analytic.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return frame


def run_all_analytic(
    *,
    step_deg: float = 1.0,
    span_deg: float = 60.0,
    sides: tuple[str, ...] = ("right", "left"),
    pivots: tuple[str, ...] = PIVOTS,
    distances_mm: tuple[float, ...] = CAMERA_DISTANCES_MM,
    out_root: Path | None = None,
    geometry_dir: Path | None = None,
    camera: PaperCamera | None = None,
) -> list[Path]:
    out_root = Path(
        out_root
        or Path(__file__).resolve().parents[1]
        / "outputs"
        / "physical_eye_simulation_runs"
    )
    geometry_dir = Path(geometry_dir or ANNOTATIONS)
    written: list[Path] = []
    for side in sides:
        geom = geometry_dir / f"eye_geometry_{side}.json"
        for pivot in pivots:
            for distance in distances_mm:
                out = out_root / side / pivot / f"d{distance:g}mm"
                print(f"analytic {side} {pivot} d={distance:g} mm …")
                run_scenario(
                    side,
                    pivot,
                    distance,
                    step_deg=step_deg,
                    span_deg=span_deg,
                    out_dir=out,
                    geometry_path=geom if geom.is_file() else None,
                    camera=camera,
                )
                written.append(out / "ground_truth_analytic.csv")
    return written


if __name__ == "__main__":
    run_all_analytic(step_deg=1.0)
