"""Render a scenario (Blender EEVEE) and fit ellipses to the black pupil.

Two stages:

1. Build + render (Blender)::

       Blender -b --python development/eye_simulation/blender_build_scenario.py -- \\
         --side right --pivot eye_centre --distance 13 --step 2
       Blender -b outputs/.../scenario.blend -a

2. Fit (this script, conda env)::

       python -m development.eye_simulation.render_and_fit \\
         --scenario outputs/pogona_eye_simulation/right/eye_centre/d13mm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from eye_tracking_system_tools.preprocessing.ellipse_fit import LsqEllipse  # noqa: E402


def _fit_pupil(image_bgr: np.ndarray, *, dark_threshold: int = 40) -> dict:
    """Threshold the black pupil on a light / grey background and fit an ellipse.

    ``dark_threshold`` should sit between the black pupil and the mid-grey mesh
    (mesh≈180–200, pupil≈0–10 with the emissive setup).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Pupil is near-black; mesh is mid-grey; background is white. Keep the
    # largest dark connected component so mesh speckles do not dominate.
    mask = (gray < int(dark_threshold)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_label, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_label <= 1:
        return {
            "r_center_x": np.nan,
            "r_center_y": np.nan,
            "r_width": np.nan,
            "r_height": np.nan,
            "r_phi": np.nan,
            "r_visible_frac": 0.0,
        }
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    component = (labels == best).astype(np.uint8) * 255
    visible_frac = float((component > 0).mean())
    # LsqEllipse is a boundary fit; filled-disc pixels bias the semi-axes down.
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {
            "r_center_x": np.nan,
            "r_center_y": np.nan,
            "r_width": np.nan,
            "r_height": np.nan,
            "r_phi": np.nan,
            "r_visible_frac": visible_frac,
        }
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 20:
        return {
            "r_center_x": np.nan,
            "r_center_y": np.nan,
            "r_width": np.nan,
            "r_height": np.nan,
            "r_phi": np.nan,
            "r_visible_frac": visible_frac,
        }
    pts = contour.reshape(-1, 2).astype(float)
    if pts.shape[0] > 2000:
        idx = np.linspace(0, pts.shape[0] - 1, 2000).astype(int)
        pts = pts[idx]
    try:
        reg = LsqEllipse().fit(pts)
        center, width, height, phi = reg.as_parameters()
        center_x, center_y = center
    except Exception:
        return {
            "r_center_x": np.nan,
            "r_center_y": np.nan,
            "r_width": np.nan,
            "r_height": np.nan,
            "r_phi": np.nan,
            "r_visible_frac": visible_frac,
        }
    if not np.isfinite(phi):
        phi = 0.0
    phi = float(np.mod(phi, np.pi))
    return {
        "r_center_x": float(center_x),
        "r_center_y": float(center_y),
        "r_width": float(width),
        "r_height": float(height),
        "r_phi": phi,
        "r_visible_frac": visible_frac,
    }


def fit_scenario(
    scenario_dir: Path,
    *,
    frames_subdir: str = "frames",
    out_csv_name: str = "ground_truth.csv",
    eval_tag: str = "rendered",
    dark_threshold: int = 40,
) -> Path:
    scenario_dir = Path(scenario_dir)
    meta = json.loads((scenario_dir / "scenario_render.json").read_text(encoding="utf-8"))
    frames_dir = scenario_dir / frames_subdir
    n = int(meta["n_frames"])
    yaw = np.asarray(meta["yaw"], dtype=float)
    pitch = np.asarray(meta["pitch"], dtype=float)

    analytic_path = scenario_dir / "ground_truth_analytic.csv"
    analytic = pd.read_csv(analytic_path) if analytic_path.is_file() else None

    rows = []
    missing = 0
    for i in range(n):
        candidates = [
            frames_dir / f"frame_{i:04d}.png",
            frames_dir / f"frame_{i:05d}.png",
            frames_dir / f"{i:04d}.png",
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            missing += 1
            fit = {
                "r_center_x": np.nan,
                "r_center_y": np.nan,
                "r_width": np.nan,
                "r_height": np.nan,
                "r_phi": np.nan,
                "r_visible_frac": 0.0,
            }
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            fit = _fit_pupil(image, dark_threshold=dark_threshold)
        row = {
            "frame": i,
            "x_angle": float(yaw[i]),
            "y_angle": float(pitch[i]),
            **fit,
        }
        rows.append(row)
        if (i + 1) % 500 == 0:
            print(f"  fitted {i + 1}/{n}")

    rendered = pd.DataFrame(rows)
    if analytic is not None:
        merged = rendered.copy()
        for col in (
            "gaze_x",
            "gaze_y",
            "gaze_z",
            "pupil_centre_x_mm",
            "pupil_centre_y_mm",
            "pupil_centre_z_mm",
            "center_x",
            "center_y",
            "width",
            "height",
            "phi",
        ):
            if col not in analytic.columns:
                continue
            key = analytic.set_index(["x_angle", "y_angle"])[col]
            vals = []
            for y, p in zip(rendered["x_angle"], rendered["y_angle"]):
                try:
                    vals.append(float(key.loc[(y, p)]))
                except KeyError:
                    d = (analytic["x_angle"] - y) ** 2 + (analytic["y_angle"] - p) ** 2
                    vals.append(float(analytic.loc[d.idxmin(), col]))
            merged[col] = vals
    else:
        merged = rendered

    out_csv = scenario_dir / out_csv_name
    merged.to_csv(out_csv, index=False)
    scenario = {
        **meta,
        "n_missing_frames": missing,
        "frames_subdir": frames_subdir,
        "eval_tag": eval_tag,
        "fit_script": "render_and_fit.py",
        "dark_threshold": dark_threshold,
        "pivot_cam_mm": meta.get("pivot_cam_mm"),
        "eye_geometry": {
            "pupil": {"radius_mm": meta.get("pupil_radius_mm", 1.5)},
        },
    }
    (scenario_dir / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    print(f"wrote {out_csv} (missing frames: {missing})")
    return out_csv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", type=Path, required=True)
    p.add_argument("--frames-subdir", default="frames")
    p.add_argument("--out-csv", default="ground_truth.csv")
    p.add_argument("--eval-tag", default="rendered")
    p.add_argument(
        "--dark-threshold",
        type=int,
        default=40,
        help="Grey < threshold counts as pupil (lower = stricter vs mesh).",
    )
    args = p.parse_args(argv)
    fit_scenario(
        args.scenario,
        frames_subdir=args.frames_subdir,
        out_csv_name=args.out_csv,
        eval_tag=args.eval_tag,
        dark_threshold=args.dark_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
