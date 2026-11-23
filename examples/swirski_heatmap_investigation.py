#!/usr/bin/env python3
"""In-depth look at the Swirski ground-truth heatmaps and the best focal length.

Reads the Blender scene that rendered the synthetic eye (when Blender and the
.blend are available), fits a pinhole model to the ground-truth ellipses,
builds noise-free replicas of the dataset, and sweeps Swirski's focal length on
the real table, the replicas, and hybrids that swap one measured quantity into
a replica. Heatmaps use the turbo colormap with a per-map colour range, with
and without error arrows.

    python examples/swirski_heatmap_investigation.py
    python examples/swirski_heatmap_investigation.py --blend /path/to/file.blend
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-pets-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_kerr_conicoid_ground_truth as gt  # noqa: E402

REPO = HERE.parent
DEFAULT_OUT = REPO / "outputs" / "in-depth swirski heatmap investigation"
DEFAULT_BLEND = Path("/Volumes/Data-1/Nimrod/Eyeball_map_generator_FINAL.blend")
BLENDER_EXE = Path("/Applications/Blender.app/Contents/MacOS/Blender")

SWEEP_FOCALS = np.arange(300.0, 411.0, 5.0)
PROFILE_FOCALS = np.arange(300.0, 421.0, 10.0)
FIT_STRIDE = 7
CIRCLE_SAMPLES = 120
BLENDER_DUMP = r'''
import bpy, json, math
import numpy as np
sc = bpy.context.scene
r = sc.render
cam = sc.camera
c = cam.data
eye = bpy.data.objects["Eye"]
me = eye.data
co = np.array([v.co[:] for v in me.vertices])
mat_of = [eye.material_slots[p.material_index].material.name for p in me.polygons]
vert_mats = {}
for p, m in zip(me.polygons, mat_of):
    for v in p.vertices:
        vert_mats.setdefault(v, set()).add(m)
rings = {}
for v, ms in vert_mats.items():
    if len(ms) > 1:
        rings.setdefault(" | ".join(sorted(ms)), []).append(v)
scale = float(eye.scale[0])
ring_info = {}
for key, idx in rings.items():
    p = co[idx]
    centre = p.mean(0)
    ring_info[key] = {
        "n_vertices": len(idx),
        "radius_mm": float(np.linalg.norm(p - centre, axis=1).mean() * scale * 1000),
        "plane_distance_from_eye_centre_mm": float(np.linalg.norm(centre) * scale * 1000),
    }
fcurves = []
act = eye.animation_data.action if eye.animation_data else None
if act is not None:
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                for fc in bag.fcurves:
                    vals = [k.co[1] for k in fc.keyframe_points]
                    fcurves.append({"path": fc.data_path, "index": fc.array_index,
                                    "n_keys": len(vals),
                                    "min_deg": math.degrees(min(vals)), "max_deg": math.degrees(max(vals))})
outer = bpy.data.objects.get("outer eye")
out = {
    "render": {"resolution": [r.resolution_x, r.resolution_y], "percentage": r.resolution_percentage,
               "pixel_aspect": [r.pixel_aspect_x, r.pixel_aspect_y], "engine": r.engine,
               "frames": [sc.frame_start, sc.frame_end], "film_transparent": r.film_transparent},
    "camera": {"type": c.type, "lens_mm": c.lens, "sensor_fit": c.sensor_fit,
               "sensor_width_mm": c.sensor_width, "sensor_height_mm": c.sensor_height,
               "shift": [c.shift_x, c.shift_y], "location_mm": [v * 1000 for v in cam.location],
               "rotation_deg": [math.degrees(a) for a in cam.rotation_euler]},
    "eye": {"location_mm": [v * 1000 for v in eye.location], "rotation_mode": eye.rotation_mode,
            "sphere_radius_mm": float(np.linalg.norm(co, axis=1).max() * scale * 1000),
            "material_rings": ring_info, "rotation_fcurves": fcurves},
    "outer_eye": None if outer is None else {
        "radius_mm": float(max(outer.dimensions) / 2 * 1000),
        "location_mm": [v * 1000 for v in outer.location],
        "materials": [s.material.name for s in outer.material_slots if s.material]},
}
print("JSONSTART" + json.dumps(out) + "JSONEND")
'''


def blender_scene(blend: Path, out_dir: Path) -> dict | None:
    """Dump the camera and eye rig from the .blend, or reuse a previous dump."""
    cached = out_dir / "blender_scene.json"
    if BLENDER_EXE.is_file() and blend.is_file():
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(BLENDER_DUMP)
            script = handle.name
        run = subprocess.run(
            [str(BLENDER_EXE), "-b", str(blend), "--python", script],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"JSONSTART(.*)JSONEND", run.stdout, re.S)
        if match:
            scene = json.loads(match.group(1))
            scene["blend_file"] = str(blend)
            cached.write_text(json.dumps(scene, indent=2), encoding="utf-8")
            return scene
        print(run.stdout[-2000:], run.stderr[-2000:])
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    return None


def blender_focal_px(scene: dict) -> float:
    cam = scene["camera"]
    width, height = scene["render"]["resolution"]
    fit = cam["sensor_fit"]
    if fit == "VERTICAL" or (fit == "AUTO" and height > width):
        return cam["lens_mm"] / cam["sensor_width_mm"] * height
    return cam["lens_mm"] / cam["sensor_width_mm"] * width


def project_pupils(x_deg, y_deg, f, d_ratio, r_ratio, cx, cy) -> pd.DataFrame:
    """Exact pinhole image of the pupil circle for every command angle.

    Lengths are in units of the camera-to-rotation-centre distance. The eye
    turns about Blender X (pitch, ``y_deg``) then world Z (yaw, ``x_deg``);
    the camera looks down +y. Each projected circle is fitted with a conic.
    """
    x = np.deg2rad(np.asarray(x_deg, dtype=float))
    y = np.deg2rad(np.asarray(y_deg, dtype=float))
    g = np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.cos(y), -np.sin(y)], axis=1)
    a1 = np.stack([np.cos(x), np.sin(x), np.zeros_like(x)], axis=1)
    a2 = np.cross(g, a1)
    t = np.linspace(0.0, 2.0 * np.pi, CIRCLE_SAMPLES, endpoint=False)
    pts = (
        d_ratio * g[:, None, :]
        + r_ratio * (np.cos(t)[None, :, None] * a1[:, None, :] + np.sin(t)[None, :, None] * a2[:, None, :])
    )
    depth = pts[..., 1] + 1.0
    u = cx + f * pts[..., 0] / depth
    v = cy - f * pts[..., 2] / depth
    design = np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], axis=-1)
    coef = np.linalg.svd(design)[2][:, -1, :]
    a, b, c, d, e, k0 = coef.T
    m = np.stack([np.stack([a, b / 2], -1), np.stack([b / 2, c], -1)], -2)
    centre = np.linalg.solve(2.0 * m, np.stack([-d, -e], -1)[..., None])[..., 0]
    k = a * centre[:, 0] ** 2 + b * centre[:, 0] * centre[:, 1] + c * centre[:, 1] ** 2 + d * centre[:, 0] + e * centre[:, 1] + k0
    w, vec = np.linalg.eigh(m)
    semi = np.sqrt(-k[:, None] / w)
    major_idx = np.argmax(semi, axis=1)
    major_dir = vec[np.arange(len(x)), :, major_idx]
    phi = np.mod(np.arctan2(major_dir[:, 1], major_dir[:, 0]), np.pi)
    major = semi.max(axis=1)
    minor = semi.min(axis=1)
    return pd.DataFrame(
        {
            "center_x": centre[:, 0],
            "center_y": centre[:, 1],
            "major_ax": major,
            "minor_ax": minor,
            "width": major,
            "height": minor,
            "phi": phi,
        }
    )


def _observed(work: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [work["center_x"], work["center_y"], work["minor_ax"] / 2.0, work["major_ax"] / 2.0]
    )


def fit_pinhole(work: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Least-squares pinhole fit of the measured ellipses; free f and an f profile."""
    sub = work.iloc[::FIT_STRIDE]
    obs = _observed(sub)

    def residual(f, d_ratio, r_ratio, cx, cy):
        pred = project_pupils(sub["x_angle"], sub["y_angle"], f, d_ratio, r_ratio, cx, cy)
        return (pred[["center_x", "center_y", "minor_ax", "major_ax"]].to_numpy() - obs).ravel()

    free = least_squares(lambda p: residual(*p), [372.9, 0.17, 0.12, 320.0, 240.0])
    rows = []
    start = [0.17, 0.12, 320.0, 240.0]
    for f in PROFILE_FOCALS:
        sol = least_squares(lambda q: residual(float(f), *q), start)
        res = residual(float(f), *sol.x).reshape(-1, 4)
        rms = np.sqrt((res**2).mean(axis=0))
        rows.append(
            {
                "focal_px": float(f),
                "pupil_distance_ratio": sol.x[0],
                "pupil_radius_ratio": sol.x[1],
                "cx": sol.x[2],
                "cy": sol.x[3],
                "rms_center_x_px": rms[0],
                "rms_center_y_px": rms[1],
                "rms_minor_semi_px": rms[2],
                "rms_major_semi_px": rms[3],
                "rms_total_px": float(np.sqrt((res**2).mean())),
            }
        )
    res = residual(*free.x).reshape(-1, 4)
    best = {
        "focal_px": float(free.x[0]),
        "pupil_distance_ratio": float(free.x[1]),
        "pupil_radius_ratio": float(free.x[2]),
        "cx": float(free.x[3]),
        "cy": float(free.x[4]),
        "rms_total_px": float(np.sqrt((res**2).mean())),
        "rms_per_column_px": dict(zip(["center_x", "center_y", "minor_semi", "major_semi"], np.sqrt((res**2).mean(0)).tolist())),
    }
    return best, pd.DataFrame(rows)


def swirski_errors(work: pd.DataFrame, table: pd.DataFrame, focal: float, cx=gt.REF_X, cy=gt.REF_Y) -> pd.DataFrame:
    err, _note = gt._evaluate_table(
        work,
        table.set_index(work.index),
        focal_px=float(focal),
        eye_z=gt.EYE_Z,
        pupil_radius=gt.PUPIL_RADIUS,
        cx=float(cx),
        cy=float(cy),
        sphere_method="swirski_2d",
    )
    return err


def focal_sweep(work: pd.DataFrame, tables: dict[str, tuple[pd.DataFrame, float, float]]) -> pd.DataFrame:
    rows = []
    for name, (table, cx, cy) in tables.items():
        for f in SWEEP_FOCALS:
            err = swirski_errors(work, table, f, cx, cy)
            total = err["total"].to_numpy(dtype=float)
            rows.append(
                {
                    "variant": name,
                    "focal_px": float(f),
                    "median_deg": float(np.nanmedian(total)),
                    "mean_deg": float(np.nanmean(total)),
                    "p90_deg": float(np.nanpercentile(total, 90)),
                    "frac_finite": float(np.isfinite(total).mean()),
                }
            )
        best = min((r for r in rows if r["variant"] == name), key=lambda r: r["median_deg"])
        print(f"  {name}: best {best['focal_px']:.0f} px, median {best['median_deg']:.3f}°")
    return pd.DataFrame(rows)


def _turbo():
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="0.85")
    return cmap


def draw_map(ax, field: dict, title: str, *, arrows: bool, vmax: float | None = None, norm=None, step: int = gt.QUIVER_STEP_DEG):
    mag = field["mag"]
    if norm is None:
        top = vmax if vmax is not None else float(np.nanpercentile(mag, 99))
        kwargs = {"vmin": 0.0, "vmax": max(top, 1e-3)}
    else:
        kwargs = {"norm": norm}
    span = gt.SPAN_DEG + 0.5
    mesh = ax.imshow(
        np.ma.masked_invalid(mag),
        origin="lower",
        extent=(-span, span, -span, span),
        cmap=_turbo(),
        interpolation="nearest",
        aspect="equal",
        **kwargs,
    )
    if arrows:
        angles = field["angles"]
        idx = np.arange(0, angles.size, step)
        xx, yy = np.meshgrid(angles[idx], angles[idx])
        uu = field["phi"][np.ix_(idx, idx)] - xx
        vv = field["theta"][np.ix_(idx, idx)] - yy
        ok = np.isfinite(uu) & np.isfinite(vv)
        typical = float(np.nanmedian(np.hypot(uu[ok], vv[ok]))) if np.any(ok) else 0.0
        gain = 1.0 if typical >= 1.0 else float(10 ** np.ceil(np.log10(2.0 / max(typical, 1e-3))))
        ax.quiver(
            xx[ok], yy[ok], gain * uu[ok], gain * vv[ok],
            angles="xy", scale_units="xy", scale=1, width=0.0035,
            color="white", edgecolor="black", linewidth=0.3, pivot="tail", zorder=3,
        )
        if gain != 1.0:
            title = f"{title}\narrows ×{gain:g}"
    ax.set_xlim(-span - 2, span + 2)
    ax.set_ylim(-span - 2, span + 2)
    ax.set_xlabel("Command yaw (deg)")
    ax.set_ylabel("Command pitch (deg)")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return mesh


def save_map(field: dict, title: str, path: Path, *, arrows: bool) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=160)
    mesh = draw_map(ax, field, title, arrows=arrows)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("3D gaze error (deg)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_grid(fields: dict[str, dict], path: Path, *, arrows: bool) -> None:
    names = list(fields)
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.9 * rows), dpi=150,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.25})
    peaks = [float(np.nanmax(fields[n]["mag"])) for n in names]
    norm = LogNorm(vmin=0.02, vmax=max(peaks))
    mesh = None
    for ax, name in zip(axes.ravel(), names):
        median = float(np.nanmedian(fields[name]["mag"]))
        label = name.replace(" (", "\n(") + f", median bin {median:.2f}°"
        mesh = draw_map(ax, fields[name], label, arrows=arrows, norm=norm, step=10)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("3D gaze error (deg, log scale)")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_profile(profile: pd.DataFrame, best: dict, blender_f: float | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    for col, label in (
        ("rms_center_x_px", "centre x"),
        ("rms_center_y_px", "centre y"),
        ("rms_minor_semi_px", "minor semi-axis"),
        ("rms_major_semi_px", "major semi-axis"),
        ("rms_total_px", "all"),
    ):
        ax.plot(profile["focal_px"], profile[col], marker="o", ms=3, lw=1.5 if col == "rms_total_px" else 1.0, label=label)
    ax.axvline(best["focal_px"], color="k", ls="--", lw=0.9, label=f"free fit {best['focal_px']:.1f} px")
    if blender_f is not None:
        ax.axvline(blender_f, color="tab:red", ls=":", lw=1.2, label=f"Blender lens {blender_f:.1f} px")
    ax.set_xlabel("Pinhole focal length (px)")
    ax.set_ylabel("RMS residual to CSV (px)")
    ax.set_title("Which pinhole camera reproduces the ground-truth ellipses?", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_sweep(sweep: pd.DataFrame, marks: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    styles = {
        "CSV": ("black", "-", 2.0),
        "CSV, fitted principal point": ("0.4", "--", 1.4),
        "replica f=334": ("tab:blue", "-", 1.4),
        "replica f=373": ("tab:red", "-", 1.4),
        "replica f=335": ("tab:blue", "-", 1.4),
        "replica 334 + CSV centres": ("tab:purple", "-.", 1.4),
        "replica 335 + CSV centres": ("tab:purple", "-.", 1.4),
        "replica 334 + CSV axes": ("tab:green", "-.", 1.4),
        "replica 335 + CSV axes": ("tab:green", "-.", 1.4),
    }
    for name, group in sweep.groupby("variant", sort=False):
        color, ls, lw = styles.get(name, ("0.5", "-", 1.0))
        best = group.loc[group["median_deg"].idxmin()]
        ax.plot(group["focal_px"], group["median_deg"], color=color, ls=ls, lw=lw,
                label=f"{name} (best {best['focal_px']:.0f} px, {best['median_deg']:.2f}°)")
    for label, f in marks.items():
        ax.axvline(f, color="0.6", lw=0.8, ls=":")
        ax.text(f, ax.get_ylim()[1] * 0.97, label, rotation=90, va="top", ha="right", fontsize=7, color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel("Swirski focal length (px)")
    ax.set_ylabel("Median 3D gaze error (deg)")
    ax.set_title("Focal sweep: real table, noise-free replicas, and one-quantity swaps", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_centre_residual(work: pd.DataFrame, replica: pd.DataFrame, path: Path) -> pd.DataFrame:
    dx = work["center_x"].to_numpy() - replica["center_x"].to_numpy()
    dy = work["center_y"].to_numpy() - replica["center_y"].to_numpy()
    frame = pd.DataFrame({"x_angle": work["x_angle"], "y_angle": work["y_angle"], "total": np.hypot(dx, dy),
                          "phi": work["x_angle"] + 20 * dx, "theta": work["y_angle"] + 20 * dy})
    field = gt._bin_field(frame)
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=160)
    mesh = draw_map(ax, field, "", arrows=False)
    angles = field["angles"]
    idx = np.arange(0, angles.size, gt.QUIVER_STEP_DEG)
    xx, yy = np.meshgrid(angles[idx], angles[idx])
    uu = field["phi"][np.ix_(idx, idx)] - xx
    vv = field["theta"][np.ix_(idx, idx)] - yy
    ok = np.isfinite(uu)
    ax.quiver(xx[ok], yy[ok], uu[ok], vv[ok], angles="xy", scale_units="xy", scale=1, width=0.0035,
              color="white", edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_title("CSV ellipse centre − exact pinhole centre (f=335)\narrows: image offset ×20", fontsize=9)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Centre offset (px)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    ecc = np.hypot(work["x_angle"], work["y_angle"])
    radial_dir = np.stack([replica["center_x"] - gt.REF_X, replica["center_y"] - gt.REF_Y], 1)
    radial_dir /= np.maximum(np.linalg.norm(radial_dir, axis=1, keepdims=True), 1e-9)
    radial = dx * radial_dir[:, 0] + dy * radial_dir[:, 1]
    rows = []
    for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90)):
        m = (ecc >= lo) & (ecc < hi)
        rows.append({"ecc_from": lo, "ecc_to": hi, "n": int(m.sum()),
                     "median_offset_px": float(np.median(np.hypot(dx, dy)[m])),
                     "median_radial_offset_px": float(np.median(radial[m]))})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=gt.DEFAULT_CSV)
    parser.add_argument("--blend", type=Path, default=DEFAULT_BLEND)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out = args.out
    (out / "heatmaps_with_arrows").mkdir(parents=True, exist_ok=True)
    (out / "heatmaps_without_arrows").mkdir(parents=True, exist_ok=True)

    scene = blender_scene(args.blend, out)
    blender_f = blender_focal_px(scene) if scene else None
    work = gt._prepare(pd.read_csv(args.csv))
    lever_px, _ = gt._fit_focal_px(work)
    print(f"Blender lens focal: {blender_f}")

    print("pinhole fit")
    best, profile = fit_pinhole(work)
    profile.to_csv(out / "pinhole_fit_profile.csv", index=False)
    plot_profile(profile, best, blender_f, out / "pinhole_fit_profile")
    print(f"  free fit f={best['focal_px']:.1f} px, rms {best['rms_total_px']:.3f} px")
    at_blender = profile.iloc[(profile["focal_px"] - (blender_f or 372.9)).abs().argmin()]
    replica_fit = project_pupils(work["x_angle"], work["y_angle"], best["focal_px"], best["pupil_distance_ratio"],
                                 best["pupil_radius_ratio"], best["cx"], best["cy"])
    blender_profile = least_squares(
        lambda q: (project_pupils(work["x_angle"].iloc[::FIT_STRIDE], work["y_angle"].iloc[::FIT_STRIDE],
                                  float(blender_f or 372.9), *q)[["center_x", "center_y", "minor_ax", "major_ax"]].to_numpy()
                   - _observed(work.iloc[::FIT_STRIDE])).ravel(),
        [at_blender["pupil_distance_ratio"], at_blender["pupil_radius_ratio"], at_blender["cx"], at_blender["cy"]],
    )
    replica_blender = project_pupils(work["x_angle"], work["y_angle"], float(blender_f or 372.9), *blender_profile.x)

    csv_table = gt._swirski_table(work, "major_plus_90", 0.5).reset_index(drop=True)
    centres = replica_fit.copy()
    centres[["center_x", "center_y"]] = csv_table[["center_x", "center_y"]]
    axes = replica_fit.copy()
    axes[["major_ax", "minor_ax", "width", "height"]] = csv_table[["major_ax", "minor_ax", "width", "height"]]
    pp = (best["cx"], best["cy"])
    f_fit = round(best["focal_px"])
    f_blend = round(blender_f or 372.9)
    tables = {
        "CSV": (csv_table, gt.REF_X, gt.REF_Y),
        "CSV, fitted principal point": (csv_table, *pp),
        f"replica f={f_fit}": (replica_fit, *pp),
        f"replica f={f_blend}": (replica_blender, blender_profile.x[2], blender_profile.x[3]),
        f"replica {f_fit} + CSV centres": (centres, *pp),
        f"replica {f_fit} + CSV axes": (axes, *pp),
    }
    print("focal sweeps")
    sweep = focal_sweep(work, tables)
    sweep.to_csv(out / "swirski_focal_sweep_decomposition.csv", index=False)
    best_csv = float(sweep.loc[sweep["variant"] == "CSV"].sort_values("median_deg").iloc[0]["focal_px"])
    marks = {f"pinhole fit {best['focal_px']:.0f}": best["focal_px"]}
    if blender_f:
        marks[f"Blender lens {blender_f:.0f}"] = blender_f
    plot_sweep(sweep, marks, out / "swirski_focal_sweep_decomposition")
    centre_bins = plot_centre_residual(work, replica_fit, out / "csv_centre_vs_pinhole")
    centre_bins.to_csv(out / "csv_centre_vs_pinhole_by_eccentricity.csv", index=False)

    print("heatmaps")
    kerr_phi, kerr_theta, kerr_fz = gt._run_kerr(work)
    errors = {"Kerr": gt._error_frame(kerr_phi, kerr_theta, work)}
    settings = [
        (f"Swirski f={lever_px:.1f} px (old baseline: centre lever arm)", lever_px),
        (f"Swirski f={gt.LAB_FOCAL_PX:.1f} px (lab OV5647 default)", gt.LAB_FOCAL_PX),
        (f"Swirski f={blender_f:.1f} px (Blender lens)" if blender_f else "Swirski f=372.9 px", blender_f or 372.9),
        (f"Swirski f={best['focal_px']:.1f} px (pinhole fit)", best["focal_px"]),
        (f"Swirski f={best_csv:.0f} px (best on CSV)", best_csv),
    ]
    rows = []
    csv_indexed = gt._swirski_table(work, "major_plus_90", 0.5)
    for title, f in settings:
        errors[title] = swirski_errors(work, csv_indexed, f)
    fields = {}
    for title, err in errors.items():
        total = err["total"].to_numpy(dtype=float)
        rows.append({"map": title, "median_deg": float(np.nanmedian(total)), "mean_deg": float(np.nanmean(total)),
                     "p90_deg": float(np.nanpercentile(total, 90)), "frac_finite": float(np.isfinite(total).mean()),
                     "slope_yaw": gt._slope(err["x_angle"].to_numpy(float), err["phi"].to_numpy(float)),
                     "slope_pitch": gt._slope(err["y_angle"].to_numpy(float), err["theta"].to_numpy(float))})
        fields[title] = gt._bin_field(err)
        stem = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
        label = f"{title}\nmedian {rows[-1]['median_deg']:.2f}°"
        save_map(fields[title], label, out / "heatmaps_with_arrows" / stem, arrows=True)
        save_map(fields[title], label, out / "heatmaps_without_arrows" / stem, arrows=False)
        print(f"  {title}: median {rows[-1]['median_deg']:.3f}°")
    pd.DataFrame(rows).to_csv(out / "heatmap_summary.csv", index=False)
    save_grid(fields, out / "heatmap_grid_log_with_arrows", arrows=True)
    save_grid(fields, out / "heatmap_grid_log_without_arrows", arrows=False)

    params = {
        "csv": str(args.csv),
        "n_frames": int(len(work)),
        "blender_scene": scene,
        "blender_focal_px": blender_f,
        "blender_pixel_pitch_um": (scene["camera"]["sensor_width_mm"] / scene["render"]["resolution"][0] * 1000) if scene else None,
        "blender_horizontal_fov_deg": float(np.degrees(2 * np.arctan(320 / blender_f))) if blender_f else None,
        "lab_focal_px": gt.LAB_FOCAL_PX,
        "lab_horizontal_fov_deg": float(np.degrees(2 * np.arctan(320 / gt.LAB_FOCAL_PX))),
        "old_baseline_focal_px": lever_px,
        "kerr_f_z_px": kerr_fz,
        "pinhole_fit_free": best,
        "pinhole_fit_at_blender_focal": {"pupil_distance_ratio": blender_profile.x[0], "pupil_radius_ratio": blender_profile.x[1],
                                         "cx": blender_profile.x[2], "cy": blender_profile.x[3]},
        "best_focal_on_csv_px": best_csv,
        "swirski_settings": {"eye_z": gt.EYE_Z, "pupil_radius": gt.PUPIL_RADIUS, "principal_point": [gt.REF_X, gt.REF_Y],
                             "sphere_method": "swirski_2d", "ransac": False, "ellipse_convention": "major_plus_90, semi-axes = stored/2"},
        "commit": gt._repo_commit(),
    }
    (out / "parameters.json").write_text(json.dumps(params, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
