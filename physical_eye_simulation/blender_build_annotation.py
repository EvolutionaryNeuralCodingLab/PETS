"""Build the interactive annotation .blend for Pogona eye landmarks.

Run with Blender headless::

    /Applications/Blender.app/Contents/MacOS/Blender -b \\
      --python development/eye_simulation/blender_build_annotation.py

Opens a scene with left and right eyes (decimated), movable empties for
lens_centre, fovea_pit, eye_centre, pupil_disc, and a read-only sclera sphere.
Units are millimetres. Save the file after adjusting landmarks and return it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "development" / "eye_simulation" / "cache"
OUT = REPO / "development" / "eye_simulation" / "annotations" / "pogona_eye_annotations.blend"
README = REPO / "development" / "eye_simulation" / "annotations" / "README.md"


def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Factory empty scenes still have a default cube in some versions.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _set_units_mm() -> None:
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 0.001  # 1 Blender unit = 1 mm
    scene.unit_settings.length_unit = "MILLIMETERS"


def _import_obj(path: Path, name: str) -> bpy.types.Object:
    # Blender 4+/5: wm.obj_import
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    obj = bpy.context.selected_objects[0]
    obj.name = name
    obj.data.name = name
    # Keep mesh coordinates as millimetres (Blender unit = 1 mm via scale_length).
    return obj


def _empty(name: str, location, *, size: float = 0.8, display: str = "PLAIN_AXES") -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = display
    empty.empty_display_size = size
    empty.location = Vector(location)
    bpy.context.collection.objects.link(empty)
    return empty


def _sphere(name: str, centre, radius: float, *, wire: bool = True) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    # Build an icosphere manually via ops then rename.
    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, location=centre, subdivisions=3)
    created = bpy.context.active_object
    created.name = name
    created.data.name = name
    if wire:
        created.display_type = "WIRE"
    # Remove the empty mesh object we created first.
    if obj.name != created.name:
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    return created


def _disc(name: str, centre, normal, radius: float) -> bpy.types.Object:
    # Unit circle in the mesh; radius is carried by the object scale so the
    # annotation reader can recover it from matrix_world.
    bpy.ops.mesh.primitive_circle_add(vertices=64, radius=1.0, fill_type="NGON", location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name
    n = Vector(normal).normalized()
    rot = Vector((0, 0, 1)).rotation_difference(n).to_matrix().to_4x4()
    scale = Matrix.Diagonal((radius, radius, radius, 1.0))
    obj.matrix_world = Matrix.Translation(Vector(centre)) @ rot @ scale
    return obj


def _curve_line(name: str, a, b) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.05
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (*a, 1.0)
    spline.points[1].co = (*b, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    return obj


def _collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def _move_to_collection(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    for other in list(obj.users_collection):
        other.objects.unlink(obj)
    col.objects.link(obj)


def _emit_material(name: str, color, strength: float = 1.0) -> bpy.types.Material:
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


def _build_eye(side: str, prep: dict) -> None:
    col = _collection(f"{side}_eye")
    g = prep["guesses"]
    obj_path = Path(prep["decimated"]["obj"])
    mesh_obj = _import_obj(obj_path, f"{side}_mesh")
    mesh_obj.data.materials.append(_emit_material(f"{side}_mesh_mat", (0.55, 0.55, 0.6), 0.4))
    _move_to_collection(mesh_obj, col)

    eye_c = _empty(f"{side}_eye_centre", g["eye_centre_mm"], size=1.0, display="SPHERE")
    lens = _empty(f"{side}_lens_centre", g["lens"]["centre_mm"], size=0.7, display="SPHERE")
    fovea = _empty(f"{side}_fovea_pit", g["fovea"]["point_mm"], size=0.5, display="PLAIN_AXES")
    for empty, color in (
        (eye_c, (0.2, 0.7, 0.3)),
        (lens, (0.9, 0.7, 0.1)),
        (fovea, (0.9, 0.2, 0.2)),
    ):
        empty.color = (*color, 1.0)
        _move_to_collection(empty, col)

    sclera = g["sclera_sphere"]
    sphere = _sphere(
        f"{side}_sclera_sphere",
        sclera["centre_mm"],
        sclera["radius_mm"],
        wire=True,
    )
    sphere.hide_select = True  # reference only
    _move_to_collection(sphere, col)
    sclera_c = _empty(
        f"{side}_sclera_sphere_centre",
        sclera["centre_mm"],
        size=0.6,
        display="CUBE",
    )
    sclera_c.hide_select = True
    _move_to_collection(sclera_c, col)

    axis = g["optical_axis_from_fovea_to_lens"]
    pupil = _disc(
        f"{side}_pupil_disc",
        g["pupil"]["centre_mm"],
        axis,
        g["pupil"]["radius_mm"],
    )
    pupil.data.materials.append(_emit_material(f"{side}_pupil_mat", (0.05, 0.05, 0.05), 1.0))
    _move_to_collection(pupil, col)

    axis_line = _curve_line(
        f"{side}_optical_axis",
        g["fovea"]["point_mm"],
        g["lens"]["centre_mm"],
    )
    _move_to_collection(axis_line, col)

    # Store guess JSON on the collection for the reader.
    col["pets_eye_side"] = side
    col["pets_prep_json"] = json.dumps(
        {
            "side": side,
            "source_stl": prep["source_stl"],
            "axial_depth_fovea_to_lens_mm": g["axial_depth_fovea_to_lens_mm"],
            "sclera_radius_mm": sclera["radius_mm"],
        }
    )


def _write_readme() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    README.write_text(
        """# Pogona eye landmark annotation

Open `pogona_eye_annotations.blend` in Blender. Scene units are millimetres.

## What to adjust (per eye)

Movable objects (select and grab / scale):

| Object | Meaning |
|--------|---------|
| `{side}_lens_centre` | Centre of the lens (pivot c; also defines the optical axis) |
| `{side}_fovea_pit` | Foveal pit on the retina (optical axis = fovea → lens) |
| `{side}_eye_centre` | Pivot (a): centre of the eye |
| `{side}_pupil_disc` | Flat pupil: move to set the plane/centre, scale in XY for radius |

Read-only (locked):

| Object | Meaning |
|--------|---------|
| `{side}_sclera_sphere` | Wire sphere fitted to the outer sclera (pivot b centre is its origin) |
| `{side}_sclera_sphere_centre` | Pivot (b) |
| `{side}_optical_axis` | Line from fovea to lens (visual aid; recomputed by the reader) |
| `{side}_mesh` | Decimated MRI surface |

## How to return the annotation

1. Move the empties / disc until they sit where you want them.
2. File → Save (overwrite this .blend, or Save As next to it).
3. Tell the agent the path. It will run `read_annotations.py` to write
   `eye_geometry_left.json` and `eye_geometry_right.json`.

The optical axis used by the simulation is the line from `fovea_pit` to
`lens_centre`. The pupil disc should stay roughly perpendicular to that axis.
""",
        encoding="utf-8",
    )


def main() -> None:
    _clear_scene()
    _set_units_mm()
    for side in ("right", "left"):
        prep_path = CACHE / f"{side}_eye_prep.json"
        if not prep_path.is_file():
            raise SystemExit(f"Missing {prep_path}; run mesh_tools.py first")
        prep = json.loads(prep_path.read_text(encoding="utf-8"))
        _build_eye(side, prep)

    # Camera for a sensible default view of the right eye.
    cam_data = bpy.data.cameras.new("AnnotationCamera")
    cam = bpy.data.objects.new("AnnotationCamera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (10, -40, 25)
    cam.rotation_euler = (math.radians(60), 0, math.radians(15))
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("Key", type="SUN")
    light = bpy.data.objects.new("Key", light_data)
    bpy.context.collection.objects.link(light)
    light.rotation_euler = (math.radians(40), math.radians(20), 0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT))
    _write_readme()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    # When run inside Blender, ``bpy`` is already available.
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
