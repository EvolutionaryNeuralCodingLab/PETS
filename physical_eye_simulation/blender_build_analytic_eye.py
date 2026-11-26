"""Build a lightweight analytical Blender eye (sphere + pupil), no MRI mesh.

Run with Blender headless::

    Blender -b --python physical_eye_simulation/blender_build_analytic_eye.py -- \\
      --side right --pivot sclera_sphere_centre --distance 16 --step 2 \\
      --geometry physical_eye_simulation/geometry/eye_geometry_right.json \\
      --out outputs/physical_eye_simulation_runs/demo
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

LENS_MM = 3.7
SENSOR_WIDTH_MM = 2592 * 0.0014
SENSOR_HEIGHT_MM = SENSOR_WIDTH_MM * 480 / 640
WIDTH, HEIGHT = 640, 480
MM = 0.001


def _parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    p = argparse.ArgumentParser()
    p.add_argument("--side", choices=("right", "left"), default="right")
    p.add_argument(
        "--pivot",
        choices=("eye_centre", "sclera_sphere_centre", "lens_centre"),
        default="sclera_sphere_centre",
    )
    p.add_argument("--distance", type=float, default=16.0)
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--span", type=float, default=60.0)
    p.add_argument("--geometry", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--lens-mm", type=float, default=LENS_MM)
    p.add_argument("--sensor-width-mm", type=float, default=SENSOR_WIDTH_MM)
    p.add_argument("--width", type=int, default=WIDTH)
    p.add_argument("--height", type=int, default=HEIGHT)
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
    return np.stack([u, v, -axis], axis=0)


def _yaw_pitch_quat(yaw_deg: float, pitch_deg: float) -> Quaternion:
    x = math.radians(yaw_deg)
    y = math.radians(pitch_deg)
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    z_new = Vector((-sx * cy, -sy, cx * cy))
    x_tmp = Vector((cx, 0.0, sx))
    y_new = z_new.cross(x_tmp).normalized()
    x_new = y_new.cross(z_new).normalized()
    m = Matrix(
        (
            (x_new.x, y_new.x, z_new.x),
            (x_new.y, y_new.y, z_new.y),
            (x_new.z, y_new.z, z_new.z),
        )
    )
    return m.to_quaternion()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv)
    _clear()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    if hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 8

    geom = json.loads(Path(args.geometry).read_text(encoding="utf-8"))
    pupil = geom["pupil"]
    optical_axis = np.asarray(geom["optical_axis_from_fovea_to_lens"], float)
    r_align = _align_np(optical_axis)
    pupil_mri = np.asarray(pupil["centre_mm"], float)
    pivot_mri = np.asarray(geom["pivots"][args.pivot], float)
    sclera_r = float(geom["sclera_sphere"]["radius_mm"]) * MM
    sclera_mri = np.asarray(geom["sclera_sphere"]["centre_mm"], float)

    def to_cam_m(points_mm: np.ndarray) -> np.ndarray:
        local = np.asarray(points_mm, float) - pupil_mri
        return ((r_align @ local.T).T + np.array([0.0, 0.0, args.distance])) * MM

    pupil_cam_m = np.array([0.0, 0.0, args.distance * MM])
    pivot_raw_m = to_cam_m(pivot_mri)
    pivot_cam_m = np.array([0.0, 0.0, float(pivot_raw_m[2])])
    sclera_cam_m = to_cam_m(sclera_mri)

    cam_data = bpy.data.cameras.new("PaperCamera")
    cam_data.lens = args.lens_mm
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = args.sensor_width_mm
    cam_data.sensor_height = args.sensor_width_mm * args.height / args.width
    cam_data.clip_start = 0.001
    cam_data.clip_end = 1.0
    cam = bpy.data.objects.new("PaperCamera", cam_data)
    cam.rotation_euler = (math.pi, 0, 0)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)

    pivot = bpy.data.objects.new("Pivot", None)
    pivot.empty_display_type = "SPHERE"
    pivot.empty_display_size = 0.001
    pivot.location = Vector(pivot_cam_m)
    bpy.context.collection.objects.link(pivot)

    # Lightweight sclera sphere (no MRI mesh).
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sclera_r, location=(0, 0, 0), segments=48, ring_count=24)
    sphere = bpy.context.active_object
    sphere.name = f"{args.side}_sclera"
    sphere.data.materials.append(_emit("sclera_mat", (0.72, 0.72, 0.74), 1.0))
    sphere.parent = pivot
    sphere.matrix_parent_inverse.identity()
    sphere.location = Vector(sclera_cam_m - pivot_cam_m)

    radius_m = float(pupil["radius_mm"]) * MM
    n_pts = 64
    angles = [2.0 * math.pi * i / n_pts for i in range(n_pts)]
    verts = [(radius_m * math.cos(a), radius_m * math.sin(a), 0.0) for a in angles]
    mesh_disc = bpy.data.meshes.new("pupil_disc")
    mesh_disc.from_pydata(verts, [], [list(range(n_pts))])
    mesh_disc.update()
    disc = bpy.data.objects.new("pupil_disc", mesh_disc)
    disc.data.materials.append(_emit("pupil_mat", (0.0, 0.0, 0.0), 1.0))
    bpy.context.collection.objects.link(disc)
    disc.parent = pivot
    disc.matrix_parent_inverse.identity()
    disc.location = Vector(pupil_cam_m - pivot_cam_m) + Vector((0.0, 0.0, -0.12 * MM))

    grid = np.arange(-args.span, args.span + 0.5 * args.step, args.step)
    yaw_grid, pitch_grid = np.meshgrid(grid, grid, indexing="ij")
    yaw = yaw_grid.ravel()
    pitch = pitch_grid.ravel()
    scene.frame_start = 0
    scene.frame_end = int(len(yaw) - 1)
    pivot.rotation_mode = "QUATERNION"
    for i, (y, p) in enumerate(zip(yaw, pitch)):
        pivot.rotation_quaternion = _yaw_pitch_quat(float(y), float(p))
        pivot.keyframe_insert(data_path="rotation_quaternion", frame=i)

    out_dir = Path(args.out) / args.side / args.pivot / f"d{args.distance:g}mm"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames_mesh"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(frames_dir / "frame_")
    meta = {
        "side": args.side,
        "pivot": args.pivot,
        "camera_distance_mm": args.distance,
        "step_deg": args.step,
        "span_deg": args.span,
        "n_frames": int(len(yaw)),
        "pivot_cam_mm": (pivot_cam_m / MM).tolist(),
        "pupil_cam_mm": (pupil_cam_m / MM).tolist(),
        "pupil_radius_mm": float(pupil["radius_mm"]),
        "analytic_eye": True,
        "camera": {
            "lens_mm": args.lens_mm,
            "sensor_width_mm": args.sensor_width_mm,
            "resolution": [args.width, args.height],
            "focal_px": args.lens_mm / args.sensor_width_mm * args.width,
            "width": args.width,
            "height": args.height,
            "cx": args.width / 2.0,
            "cy": args.height / 2.0,
        },
        "yaw": yaw.tolist(),
        "pitch": pitch.tolist(),
    }
    (out_dir / "scenario_render.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scenario_analytic_eye.blend"))
    print(f"Wrote {out_dir / 'scenario_analytic_eye.blend'} ({len(yaw)} frames)")


if __name__ == "__main__":
    main()
