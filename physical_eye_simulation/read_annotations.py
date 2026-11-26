"""Read landmark objects from the annotated Blender file into eye_geometry JSON.

Run with Blender headless::

    /Applications/Blender.app/Contents/MacOS/Blender -b \\
      development/eye_simulation/annotations/pogona_eye_annotations.blend \\
      --python development/eye_simulation/read_annotations.py

Or, after the user returns a edited copy::

    ... -b /path/to/returned.blend --python development/eye_simulation/read_annotations.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
OUT_DIR = PACKAGE / "geometry"
CACHE = REPO / "outputs" / "physical_eye_simulation_runs" / "cache"


def _loc(name: str) -> list[float]:
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise KeyError(f"Missing object {name!r} in the annotation file")
    return [float(v) for v in obj.matrix_world.translation]


def _pupil(side: str, axis: Vector) -> dict:
    obj = bpy.data.objects.get(f"{side}_pupil_disc")
    if obj is None:
        raise KeyError(f"Missing {side}_pupil_disc")
    centre = Vector(obj.matrix_world.translation)
    # Disc is authored as a unit circle with radius in the object scale.
    sx = obj.matrix_world.to_scale()
    radius = float((abs(sx.x) + abs(sx.y)) / 2.0)
    if radius < 1e-6 and obj.type == "MESH" and obj.data.vertices:
        # Fallback: world-space XY extent of the mesh.
        pts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        radius = float(max((p - centre).length for p in pts))
    # Prefer the disc's own normal (+Z after orientation) if it is roughly aligned.
    normal = (obj.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    if normal.dot(axis) < 0:
        normal = -normal
    if abs(normal.dot(axis)) < 0.5:
        # Disc was left badly oriented; fall back to the optical axis.
        normal = axis.normalized()
    return {
        "centre_mm": [float(v) for v in centre],
        "normal_from_fovea_to_lens": [float(v) for v in normal],
        "radius_mm": radius,
    }


def _geometry(side: str) -> dict:
    lens = Vector(_loc(f"{side}_lens_centre"))
    fovea = Vector(_loc(f"{side}_fovea_pit"))
    eye_centre = Vector(_loc(f"{side}_eye_centre"))
    sclera_centre = Vector(_loc(f"{side}_sclera_sphere_centre"))
    axis = (lens - fovea).normalized()
    pupil = _pupil(side, axis)

    sclera_obj = bpy.data.objects.get(f"{side}_sclera_sphere")
    if sclera_obj is not None:
        sclera_radius = float(sum(sclera_obj.dimensions) / 6.0)  # diameter/2 average
    else:
        prep = json.loads((CACHE / f"{side}_eye_prep.json").read_text(encoding="utf-8"))
        sclera_radius = float(prep["guesses"]["sclera_sphere"]["radius_mm"])

    prep_path = CACHE / f"{side}_eye_prep.json"
    source_stl = None
    if prep_path.is_file():
        source_stl = json.loads(prep_path.read_text(encoding="utf-8")).get("source_stl")

    axial_depth = float((lens - fovea).length)
    return {
        "side": side,
        "units": "mm",
        "source_stl": source_stl,
        "annotation_blend": bpy.data.filepath,
        "eye_centre_mm": [float(v) for v in eye_centre],
        "sclera_sphere": {
            "centre_mm": [float(v) for v in sclera_centre],
            "radius_mm": sclera_radius,
        },
        "lens_centre_mm": [float(v) for v in lens],
        "fovea_pit_mm": [float(v) for v in fovea],
        "optical_axis_from_fovea_to_lens": [float(v) for v in axis],
        "axial_depth_fovea_to_lens_mm": axial_depth,
        "pupil": pupil,
        "pivots": {
            "eye_centre": [float(v) for v in eye_centre],
            "sclera_sphere_centre": [float(v) for v in sclera_centre],
            "lens_centre": [float(v) for v in lens],
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for side in ("right", "left"):
        geom = _geometry(side)
        path = OUT_DIR / f"eye_geometry_{side}.json"
        path.write_text(json.dumps(geom, indent=2), encoding="utf-8")
        report[side] = {
            "path": str(path),
            "axial_depth_fovea_to_lens_mm": geom["axial_depth_fovea_to_lens_mm"],
            "pupil_radius_mm": geom["pupil"]["radius_mm"],
            "sclera_radius_mm": geom["sclera_sphere"]["radius_mm"],
        }
        print(
            f"{side}: axial depth {geom['axial_depth_fovea_to_lens_mm']:.3f} mm, "
            f"pupil r={geom['pupil']['radius_mm']:.3f} mm, "
            f"sclera r={geom['sclera_sphere']['radius_mm']:.3f} mm → {path}"
        )
    summary = OUT_DIR / "annotation_summary.json"
    summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"summary → {summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
