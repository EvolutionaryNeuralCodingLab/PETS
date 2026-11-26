"""Build a Blender scenario scene for one eye / pivot / camera distance.

Eye geometry is transformed into the camera frame once, then parented to a
pivot empty at the rotation centre. Keyframes rotate that empty through the
yaw/pitch grid. Run with Blender headless::

    Blender -b --python development/eye_simulation/blender_build_scenario.py -- \\
      --side right --pivot eye_centre --distance 13 --step 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "outputs" / "physical_eye_simulation_runs" / "cache"
ANNOT = REPO / "physical_eye_simulation" / "geometry"
OUT_ROOT = REPO / "outputs" / "physical_eye_simulation_runs"

LENS_MM = 3.7
SENSOR_WIDTH_MM = 2592 * 0.0014
SENSOR_HEIGHT_MM = SENSOR_WIDTH_MM * 480 / 640
WIDTH, HEIGHT = 640, 480
# Blender camera optics are in millimetres and treat scene distances as
# metres when unit scale is 1. Express all geometry in metres.
MM = 0.001


def _parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=("right", "left"), default="right")
    p.add_argument(
        "--pivot",
        choices=("eye_centre", "sclera_sphere_centre", "lens_centre"),
        default="eye_centre",
    )
    p.add_argument("--distance", type=float, default=13.0)
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--span", type=float, default=60.0)
    p.add_argument(
        "--out-root",
        type=str,
        default=str(OUT_ROOT),
        help="Root directory for scenario outputs",
    )
    return p.parse_args(argv)


def _clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _emit(name: str, color, strength: float = 1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*color, 1.0)
    emit.inputs["Strength"].default_value = strength
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _basis(axis: np.ndarray):
    a = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, u, v


def _align_np(optical_axis: np.ndarray) -> np.ndarray:
    axis, u, v = _basis(np.asarray(optical_axis, dtype=float))
    return np.stack([u, v, -axis], axis=0)  # rows = camera axes in MRI


def _yaw_pitch_quat(yaw_deg: float, pitch_deg: float) -> Quaternion:
    x = math.radians(yaw_deg)
    y = math.radians(pitch_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    z_new = Vector((-sx * cy, -sy, cx * cy))
    x_tmp = Vector((cx, 0.0, sx))
    y_new = z_new.cross(x_tmp).normalized()
    x_new = y_new.cross(z_new).normalized()
    m = Matrix((
        (x_new.x, y_new.x, z_new.x),
        (x_new.y, y_new.y, z_new.y),
        (x_new.z, y_new.z, z_new.z),
    ))
    return m.to_quaternion()


def _load_obj_verts_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts = []
    faces = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:4]]
            faces.append(idx)
    return np.asarray(verts, float), np.asarray(faces, int)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv)
    _clear()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    if hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 8

    geom = json.loads((ANNOT / f"eye_geometry_{args.side}.json").read_text(encoding="utf-8"))
    pupil = geom["pupil"]
    optical_axis = np.asarray(geom["optical_axis_from_fovea_to_lens"], float)
    r_align = _align_np(optical_axis)
    pupil_mri = np.asarray(pupil["centre_mm"], float)
    pivot_mri = np.asarray(geom["pivots"][args.pivot], float)

    def to_cam_m(points_mm: np.ndarray) -> np.ndarray:
        local = np.asarray(points_mm, float) - pupil_mri
        cam_mm = (r_align @ local.T).T + np.array([0.0, 0.0, args.distance])
        return cam_mm * MM

    pupil_cam_m = np.array([0.0, 0.0, args.distance * MM])
    pivot_raw_m = to_cam_m(pivot_mri)
    pivot_cam_m = np.array([0.0, 0.0, float(pivot_raw_m[2])])
    # Keep mm copies for metadata / analytic comparison.
    pupil_cam = pupil_cam_m / MM
    pivot_cam = pivot_cam_m / MM
    pivot_raw = pivot_raw_m / MM

    # Camera at origin looking +Z. Blender cameras look along local −Z with
    # +Y up; a 180° rotation about X maps that to world +Z with +Y down
    # (OpenCV), without mirroring X.
    cam_data = bpy.data.cameras.new("PaperCamera")
    cam_data.type = "PERSP"
    cam_data.lens = LENS_MM
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = SENSOR_WIDTH_MM
    cam_data.sensor_height = SENSOR_HEIGHT_MM
    cam_data.clip_start = 0.001
    cam_data.clip_end = 1.0
    cam = bpy.data.objects.new("PaperCamera", cam_data)
    cam.location = (0, 0, 0)
    cam.rotation_euler = (math.pi, 0, 0)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 1.0

    # Pivot empty at the rotation centre (metres).
    pivot = bpy.data.objects.new("Pivot", None)
    pivot.empty_display_type = "SPHERE"
    pivot.empty_display_size = 0.001
    pivot.location = Vector(pivot_cam_m)
    bpy.context.collection.objects.link(pivot)

    # Mesh: rewrite vertices into pivot-local camera coordinates (metres).
    obj_path = CACHE / f"{args.side}_eye_decimated.obj"
    verts_mri, faces = _load_obj_verts_faces(obj_path)
    verts_cam_m = to_cam_m(verts_mri)
    verts_local = verts_cam_m - pivot_cam_m
    mesh_data = bpy.data.meshes.new(f"{args.side}_mesh")
    mesh_data.from_pydata(verts_local.tolist(), [], faces.tolist())
    mesh_data.update()
    mesh = bpy.data.objects.new(f"{args.side}_mesh", mesh_data)
    # Mid-light grey so the black pupil stays separable under a dark threshold.
    mesh.data.materials.append(_emit("mesh_mat", (0.72, 0.72, 0.74), 1.0))
    bpy.context.collection.objects.link(mesh)
    mesh.parent = pivot
    mesh.matrix_parent_inverse.identity()
    mesh.location = (0, 0, 0)
    mesh.rotation_euler = (0, 0, 0)
    mesh.scale = (1, 1, 1)

    # Pupil disc in pivot-local coords: centre at pupil_cam - pivot_cam.
    radius_mm = float(pupil["radius_mm"])
    radius_m = radius_mm * MM
    mesh_disc = bpy.data.meshes.new("pupil_disc")
    n_pts = 64
    angles = [2.0 * math.pi * i / n_pts for i in range(n_pts)]
    verts = [(radius_m * math.cos(a), radius_m * math.sin(a), 0.0) for a in angles]
    faces = [list(range(n_pts))]
    mesh_disc.from_pydata(verts, [], faces)
    mesh_disc.update()
    disc = bpy.data.objects.new("pupil_disc", mesh_disc)
    disc.data.materials.append(_emit("pupil_mat", (0.0, 0.0, 0.0), 1.0))
    bpy.context.collection.objects.link(disc)
    disc.parent = pivot
    disc.matrix_parent_inverse.identity()
    # Sit slightly toward the camera of the nominal pupil plane so the disc
    # wins depth tests against the MRI mesh at the lens opening.
    disc.location = Vector(pupil_cam_m - pivot_cam_m) + Vector((0.0, 0.0, -0.12 * MM))
    disc.rotation_euler = (0.0, 0.0, 0.0)
    disc.scale = (1.0, 1.0, 1.0)
    radius = radius_mm

    # Keyframes
    angles = np.arange(-args.span, args.span + 0.5 * args.step, args.step)
    yaw_grid, pitch_grid = np.meshgrid(angles, angles, indexing="ij")
    yaw = yaw_grid.ravel()
    pitch = pitch_grid.ravel()
    scene.frame_start = 0
    scene.frame_end = int(len(yaw) - 1)
    pivot.rotation_mode = "QUATERNION"
    for i, (y, p) in enumerate(zip(yaw, pitch)):
        pivot.rotation_quaternion = _yaw_pitch_quat(float(y), float(p))
        pivot.keyframe_insert(data_path="rotation_quaternion", frame=i)

    out_root = Path(args.out_root)
    out_dir = out_root / args.side / args.pivot / f"d{args.distance:g}mm"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear old frames
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir / "frame_")
    blend_path = out_dir / "scenario.blend"
    meta = {
        "side": args.side,
        "pivot": args.pivot,
        "camera_distance_mm": args.distance,
        "step_deg": args.step,
        "span_deg": args.span,
        "n_frames": int(len(yaw)),
        "pivot_cam_mm": pivot_cam.tolist(),
        "pivot_raw_cam_mm": pivot_raw.tolist(),
        "pupil_cam_mm": pupil_cam.tolist(),
        "pupil_radius_mm": radius,
        "camera": {
            "lens_mm": LENS_MM,
            "sensor_width_mm": SENSOR_WIDTH_MM,
            "resolution": [WIDTH, HEIGHT],
        },
        "yaw": yaw.tolist(),
        "pitch": pitch.tolist(),
    }
    (out_dir / "scenario_render.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"Wrote {blend_path} ({len(yaw)} frames)")
    print(f"pivot_cam={pivot_cam.tolist()} pupil_local={(pupil_cam - pivot_cam).tolist()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
