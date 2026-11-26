"""Render all frames of a scenario.blend (mesh + pupil, or pupil-only).

Usage::

    Blender -b outputs/.../scenario.blend --python \\
      development/eye_simulation/blender_render_frames.py -- \\
      --out-subdir frames_mesh --show-mesh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def _parse_args(argv: list[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    p = argparse.ArgumentParser()
    p.add_argument("--out-subdir", default="frames_mesh")
    p.add_argument(
        "--show-mesh",
        action="store_true",
        help="Include the eye mesh in the render (default: hide mesh).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv)
    scene = bpy.context.scene
    blend = Path(bpy.data.filepath).resolve()
    out_dir = blend.parent / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for obj in bpy.data.objects:
        name = obj.name.lower()
        if "pupil" in name:
            obj.hide_render = False
        elif "mesh" in name:
            obj.hide_render = not args.show_mesh

    scene.render.filepath = str(out_dir / "frame_")
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = "PNG"
    print(
        f"Rendering frames {scene.frame_start}..{scene.frame_end} → {out_dir} "
        f"(mesh={'on' if args.show_mesh else 'off'})"
    )
    bpy.ops.render.render(animation=True)
    print("Done.")


if __name__ == "__main__":
    main()
