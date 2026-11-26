"""Single entry point for the physical-eye Kerr vs Swirski simulation.

Examples::

    PYTHONPATH=.:src python -m physical_eye_simulation.run
    PYTHONPATH=.:src python -m physical_eye_simulation.run --mode analytic \\
      --distance-mm 16 --pivot sclera_sphere_centre --sides right
    PYTHONPATH=.:src python -m physical_eye_simulation.run --mode blender --step-deg 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from physical_eye_simulation.analytic_projection import run_scenario
from physical_eye_simulation.camera_model import PIVOTS, PaperCamera
from physical_eye_simulation.evaluate import evaluate_scenario_dir

PACKAGE = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        # Minimal fallback for the default config shape (no nested lists of maps).
        return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict:
    """Tiny subset parser so the default config works without PyYAML."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(0, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        cur = stack[-1][1]
        if line.startswith("- "):
            # Treat as list item under the last key — not used in default.
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            child: dict = {}
            cur[key] = child
            stack.append((indent + 2, child))
        else:
            if val.startswith("[") and val.endswith("]"):
                items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
                parsed = []
                for item in items:
                    try:
                        parsed.append(float(item) if "." in item else int(item))
                    except ValueError:
                        parsed.append(item)
                cur[key] = parsed
            else:
                val = val.split("#", 1)[0].strip().strip("'\"")
                try:
                    if "." in val:
                        cur[key] = float(val)
                    else:
                        cur[key] = int(val)
                except ValueError:
                    cur[key] = val
    return root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=PACKAGE / "config_default.yaml")
    p.add_argument("--mode", choices=("analytic", "blender"), default=None)
    p.add_argument("--lens-mm", type=float, default=None)
    p.add_argument("--sensor-width-mm", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--distance-mm", type=float, nargs="+", default=None)
    p.add_argument("--pivot", choices=PIVOTS, nargs="+", default=None)
    p.add_argument("--sides", choices=("right", "left"), nargs="+", default=None)
    p.add_argument("--step-deg", type=float, default=None)
    p.add_argument("--span-deg", type=float, default=None)
    p.add_argument("--geometry-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--blender-binary", type=str, default=None)
    return p.parse_args(argv)


def _merge(args: argparse.Namespace) -> dict:
    cfg = _load_yaml(args.config) if args.config.is_file() else {}
    cam = dict(cfg.get("camera") or {})
    if args.lens_mm is not None:
        cam["lens_mm"] = args.lens_mm
    if args.sensor_width_mm is not None:
        cam["sensor_width_mm"] = args.sensor_width_mm
    if args.width is not None:
        cam["width"] = args.width
    if args.height is not None:
        cam["height"] = args.height
    blender = dict(cfg.get("blender") or {})
    if args.blender_binary is not None:
        blender["binary"] = args.blender_binary
    return {
        "mode": args.mode or cfg.get("mode", "analytic"),
        "camera": cam,
        "geometry_dir": Path(
            args.geometry_dir
            or cfg.get("geometry_dir")
            or (PACKAGE / "geometry")
        ),
        "sides": tuple(args.sides or cfg.get("sides") or ("right", "left")),
        "pivots": tuple(args.pivot or cfg.get("pivots") or ("sclera_sphere_centre",)),
        "distances_mm": tuple(
            args.distance_mm or cfg.get("distances_mm") or (16.0,)
        ),
        "step_deg": float(args.step_deg if args.step_deg is not None else cfg.get("step_deg", 1)),
        "span_deg": float(args.span_deg if args.span_deg is not None else cfg.get("span_deg", 60)),
        "out": Path(args.out or cfg.get("out") or "outputs/physical_eye_simulation_runs/default"),
        "blender": blender,
    }


def _camera_from_cfg(cam: dict) -> PaperCamera:
    return PaperCamera.from_optics(
        lens_mm=float(cam.get("lens_mm", 3.7)),
        sensor_width_mm=float(cam.get("sensor_width_mm", 3.6288)),
        width=int(cam.get("width", 640)),
        height=int(cam.get("height", 480)),
        cx=cam.get("cx"),
        cy=cam.get("cy"),
    )


def _run_analytic(cfg: dict, camera: PaperCamera) -> list[Path]:
    outs: list[Path] = []
    for side in cfg["sides"]:
        geom = cfg["geometry_dir"] / f"eye_geometry_{side}.json"
        if not geom.is_file():
            raise FileNotFoundError(geom)
        for pivot in cfg["pivots"]:
            for distance in cfg["distances_mm"]:
                out = cfg["out"] / side / pivot / f"d{float(distance):g}mm"
                print(f"[analytic] {side} {pivot} d={distance:g}")
                run_scenario(
                    side,
                    pivot,
                    float(distance),
                    step_deg=cfg["step_deg"],
                    span_deg=cfg["span_deg"],
                    out_dir=out,
                    geometry_path=geom,
                    camera=camera,
                )
                evaluate_scenario_dir(out, source="analytic")
                outs.append(out)
    return outs


def _run_blender(cfg: dict, camera: PaperCamera) -> list[Path]:
    binary = cfg["blender"].get("binary") or "blender"
    outs: list[Path] = []
    for side in cfg["sides"]:
        geom = cfg["geometry_dir"] / f"eye_geometry_{side}.json"
        for pivot in cfg["pivots"]:
            for distance in cfg["distances_mm"]:
                out_root = cfg["out"]
                scen = out_root / side / pivot / f"d{float(distance):g}mm"
                print(f"[blender] build {side} {pivot} d={distance:g}")
                cmd_build = [
                    binary,
                    "-b",
                    "--python",
                    str(PACKAGE / "blender_build_analytic_eye.py"),
                    "--",
                    "--side",
                    side,
                    "--pivot",
                    pivot,
                    "--distance",
                    str(distance),
                    "--step",
                    str(cfg["step_deg"]),
                    "--span",
                    str(cfg["span_deg"]),
                    "--geometry",
                    str(geom),
                    "--out",
                    str(out_root),
                    "--lens-mm",
                    str(camera.lens_mm),
                    "--sensor-width-mm",
                    str(camera.sensor_width_mm),
                    "--width",
                    str(camera.width),
                    "--height",
                    str(camera.height),
                ]
                subprocess.run(cmd_build, check=True)
                blend = scen / "scenario_analytic_eye.blend"
                print(f"[blender] render {blend}")
                cmd_render = [
                    binary,
                    "-b",
                    str(blend),
                    "--python",
                    str(PACKAGE / "blender_render_frames.py"),
                    "--",
                    "--out-subdir",
                    "frames_mesh",
                ]
                subprocess.run(cmd_render, check=True)
                # Analytic GT for the same grid (for merge + scoring truth).
                run_scenario(
                    side,
                    pivot,
                    float(distance),
                    step_deg=cfg["step_deg"],
                    span_deg=cfg["span_deg"],
                    out_dir=scen,
                    geometry_path=geom,
                    camera=camera,
                )
                from physical_eye_simulation.render_and_fit import fit_scenario

                fit_scenario(
                    scen,
                    frames_subdir="frames_mesh",
                    out_csv_name="ground_truth.csv",
                    eval_tag="rendered_analytic_eye",
                    dark_threshold=int(cfg["blender"].get("dark_threshold", 40)),
                )
                # Ensure camera meta is present for evaluate.
                meta_path = scen / "scenario.json"
                if meta_path.is_file():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["camera"] = camera.to_dict()
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                evaluate_scenario_dir(scen, source="analytic")
                evaluate_scenario_dir(scen, source="rendered")
                outs.append(scen)
    return outs


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _merge(args)
    cfg["out"] = Path(cfg["out"])
    if not cfg["out"].is_absolute():
        cfg["out"] = REPO / cfg["out"]
    geom_dir = Path(cfg["geometry_dir"])
    if not geom_dir.is_absolute():
        geom_dir = REPO / geom_dir
    cfg["geometry_dir"] = geom_dir
    cfg["out"].mkdir(parents=True, exist_ok=True)
    camera = _camera_from_cfg(cfg["camera"])
    print(
        f"mode={cfg['mode']}  f={camera.focal_px:.2f}px  "
        f"sides={cfg['sides']} pivots={cfg['pivots']} "
        f"distances={cfg['distances_mm']} step={cfg['step_deg']}"
    )
    if cfg["mode"] == "analytic":
        outs = _run_analytic(cfg, camera)
    else:
        outs = _run_blender(cfg, camera)
    print(f"Done. Scenarios written under {cfg['out']} ({len(outs)} dirs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
