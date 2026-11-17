#!/usr/bin/env python3
"""Compare Kerr and Swirski gaze angles to the Blender ground-truth grid.

The input is the synthetic eye used for the Kerr accuracy check: 15,367 frames,
gaze from -60° to 60° in 1° steps, with a fitted ellipse on each frame. Kerr
uses the pupil center and the axis ratio. Swirski
(:func:`eye_tracking_system_tools.preprocessing.conicoid.fit_eye_and_gaze`)
uses the full ellipse.

The main figures are a pair of ground-truth error maps on the ±60° command
grid. The color is the angle between two gaze directions, not a subtraction of
Kerr's nested ``(φ, θ)``. Kerr's pair is inverted to the direction it implies,
and Swirski's stored pair is inverted the same way, which recovers its
toward-camera normal. Arrows use the yaw/pitch chart of that direction, which
returns the command angles exactly when the direction is right.

A second pass scores ellipse-angle conventions, then sweeps the Swirski knobs
that can move the angles (focal length, eye distance, pupil radius, principal
point, sphere method).

Usage::

    python examples/compare_kerr_conicoid_ground_truth.py
"""

from __future__ import annotations

import argparse
import json
import os
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
from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eye_tracking_system_tools.preprocessing.conicoid import (  # noqa: E402
    CameraIntrinsics,
    angle_between_normals_deg,
    chart_yaw_pitch,
    fit_eye_and_gaze,
    kerr_angles_to_normals,
    yaw_pitch_to_normals,
)

DEFAULT_CSV = (
    REPO
    / "development"
    / "kerr_relative_error"
    / "ellipse_angle_mapping_correct_diameter_08mm_distance_13mm.csv"
)
DEFAULT_OUT = REPO / "outputs" / "kerr_vs_conicoid_ground_truth"

REF_X = 320.0
REF_Y = 240.0
EYE_Z = 13.0
PUPIL_RADIUS = 1.0
PAPER_SPAN_DEG = 35.0
SPAN_DEG = 60
QUIVER_STEP_DEG = 5
LAB_FOCAL_PX = 3.7 / 0.0014 * (640 / 2592)
MIN_FINITE_FRAC = 0.95
FOCAL_SWEEP_START_PX = 120.0
FOCAL_SWEEP_STOP_PX = 1000.0
FOCAL_SWEEP_STEP_PX = 5.0


def focal_sweep_values() -> list[float]:
    """Focal lengths from 120 to 1000 px, step 5, plus the 3.7 mm lab default."""
    grid = np.arange(
        FOCAL_SWEEP_START_PX,
        FOCAL_SWEEP_STOP_PX + 0.1 * FOCAL_SWEEP_STEP_PX,
        FOCAL_SWEEP_STEP_PX,
    )
    values = [float(v) for v in grid]
    values.append(float(LAB_FOCAL_PX))
    values.sort()
    unique: list[float] = []
    for value in values:
        if not unique or abs(value - unique[-1]) > 1e-6:
            unique.append(value)
    return unique

ANGLE_MODES = (
    "major_plus_90",
    "phi_as_major",
    "major_plus_90_reflected",
    "phi_as_major_reflected",
)
AXIS_SCALES = (0.5, 1.0)
EYE_Z_GRID = (4.0, 8.0, 13.0, 20.0, 40.0, 80.0)
PUPIL_RADII = (0.5, 1.0, 2.0, 4.0)

def _repo_commit() -> str:
    """Short commit of the checkout that provides the conicoid package."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fit_focal_px(df: pd.DataFrame) -> tuple[float, float]:
    """Pinhole focal length from on-axis horizontal center motion."""
    band = df.loc[df["y_angle"].abs() < 0.5].copy()
    band = band.loc[band["x_angle"].abs() < 15.0]
    pred = np.tan(np.deg2rad(band["x_angle"].to_numpy(dtype=float)))
    centers = band["center_x"].to_numpy(dtype=float)
    design = np.column_stack([pred, np.ones(pred.size)])
    slope, intercept = np.linalg.lstsq(design, centers, rcond=None)[0]
    return float(slope), float(intercept)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["x_angle"] = pd.to_numeric(work["x_angle"], errors="coerce")
    work["y_angle"] = pd.to_numeric(work["y_angle"], errors="coerce")
    work["axs_1"] = pd.to_numeric(work["axs_1"], errors="coerce")
    work["axs_2"] = pd.to_numeric(work["axs_2"], errors="coerce")
    work["ellipse_phi"] = pd.to_numeric(work["ellipse_phi"], errors="coerce")
    work["center_x"] = pd.to_numeric(work["center_x"], errors="coerce")
    work["center_y"] = pd.to_numeric(work["center_y"], errors="coerce")
    work["major_ax"] = np.maximum(work["axs_1"], work["axs_2"])
    work["minor_ax"] = np.minimum(work["axs_1"], work["axs_2"])
    work["ratio2"] = work["minor_ax"] / work["major_ax"]
    work["ecc_gt"] = np.hypot(work["x_angle"], work["y_angle"])
    work["OE_timestamp"] = 0
    work["eye_frame"] = np.arange(len(work))
    work["ms_axis"] = 0.0
    return work


def _major_axis_angle(ellipse_phi_deg: np.ndarray, mode: str) -> np.ndarray:
    """Major-axis angle in radians for one ellipse-column convention.

    On this table ``ellipse_phi`` is the angle of ``axs_1``, and ``axs_1`` is
    the shorter axis, so the major axis is 90° on from that angle.
    """
    minor = np.deg2rad(np.asarray(ellipse_phi_deg, dtype=float))
    if mode == "major_plus_90":
        angle = minor + 0.5 * np.pi
    elif mode == "phi_as_major":
        angle = minor
    elif mode == "major_plus_90_reflected":
        angle = np.pi - (minor + 0.5 * np.pi)
    elif mode == "phi_as_major_reflected":
        angle = np.pi - minor
    else:
        raise ValueError(f"unknown angle mode {mode!r}")
    return np.mod(angle, np.pi)


def _swirski_table(work: pd.DataFrame, angle_mode: str, axis_scale: float) -> pd.DataFrame:
    major = work["major_ax"].to_numpy(dtype=float) * float(axis_scale)
    minor = work["minor_ax"].to_numpy(dtype=float) * float(axis_scale)
    phi = _major_axis_angle(work["ellipse_phi"].to_numpy(dtype=float), angle_mode)
    return pd.DataFrame(
        {
            "center_x": work["center_x"].to_numpy(dtype=float),
            "center_y": work["center_y"].to_numpy(dtype=float),
            "major_ax": major,
            "minor_ax": minor,
            "width": major,
            "height": minor,
            "phi": phi,
        },
        index=work.index,
    )


def _nearest_row(work: pd.DataFrame, x_deg: float, y_deg: float) -> dict:
    dist = (work["x_angle"] - x_deg) ** 2 + (work["y_angle"] - y_deg) ** 2
    row = work.loc[dist.idxmin()]
    return {
        "target_x": x_deg,
        "target_y": y_deg,
        "x_angle": float(row["x_angle"]),
        "y_angle": float(row["y_angle"]),
        "ellipse_phi_deg": float(row["ellipse_phi"]),
        "axs_1": float(row["axs_1"]),
        "axs_2": float(row["axs_2"]),
        "center_x": float(row["center_x"]),
        "center_y": float(row["center_y"]),
    }


def _run_kerr(work: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync

    f_z, out = BlockSync.kerr(work, aEC=REF_X, bEC=REF_Y)
    return out["phi"], out["theta"], float(f_z)


def _run_swirski(
    table: pd.DataFrame,
    *,
    focal_px: float,
    eye_z: float = EYE_Z,
    pupil_radius: float = PUPIL_RADIUS,
    cx: float = REF_X,
    cy: float = REF_Y,
    sphere_method: str = "swirski_2d",
):
    camera = CameraIntrinsics(
        width=640,
        height=480,
        cx=float(cx),
        cy=float(cy),
        focal_px=float(focal_px),
        source="sweep",
    )
    out, fit = fit_eye_and_gaze(
        table,
        camera,
        eye_z=float(eye_z),
        pupil_radius=float(pupil_radius),
        use_ransac=False,
        sphere_method=sphere_method,
    )
    sphere = {
        "centre_xyz": [float(v) for v in np.asarray(fit.sphere.centre).reshape(3)],
        "radius": float(fit.sphere.radius),
        "n_inliers": int(fit.n_inliers),
        "n_observations": int(fit.n_observations),
    }
    return out["c_phi"], out["c_theta"], _repo_commit(), sphere


def _command_normals(work: pd.DataFrame) -> np.ndarray:
    """Unit gaze for a horizontal command ``x`` followed by a vertical ``y``.

    The pupil centers in this table track ``(sin x cos y, sin y)``, the first
    two components of this normal.
    """
    return yaw_pitch_to_normals(work["x_angle"], work["y_angle"])


def _as_float(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=float)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)


def _error_frame(phi, theta, work: pd.DataFrame) -> pd.DataFrame:
    """Score both methods by the angle between gaze directions.

    ``phi`` and ``theta`` are Kerr's nested pair, which is what both
    algorithms emit. They are inverted to a direction and compared with the
    command-angle normal. ``total`` is that angle. The ``phi`` and ``theta``
    columns of the result are the yaw/pitch chart of the reported direction,
    so a perfect direction lines up with the command angles.
    """
    reported = kerr_angles_to_normals(phi, theta)
    truth = _command_normals(work)
    yaw, pitch = chart_yaw_pitch(reported)
    gt_x = work["x_angle"].to_numpy(dtype=float)
    gt_y = work["y_angle"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "x_angle": gt_x,
            "y_angle": gt_y,
            "phi": yaw,
            "theta": pitch,
            "ecc_gt": work["ecc_gt"].to_numpy(dtype=float),
            "dphi": yaw - gt_x,
            "dtheta": pitch - gt_y,
            "total": angle_between_normals_deg(reported, truth),
            "ratio2": work["ratio2"].to_numpy(dtype=float),
        },
        index=work.index,
    )


def _verify_direction_metric(work: pd.DataFrame) -> None:
    """Fail if a perfect direction does not score as zero error."""
    truth = _command_normals(work)
    yaw, pitch = chart_yaw_pitch(truth)
    gt_x = work["x_angle"].to_numpy(dtype=float)
    gt_y = work["y_angle"].to_numpy(dtype=float)
    chart_err = np.hypot(yaw - gt_x, pitch - gt_y)
    self_err = angle_between_normals_deg(truth, truth)
    kerr_phi, kerr_theta = _euler_kerr_angles(work)
    recovered = kerr_angles_to_normals(kerr_phi, kerr_theta)
    roundtrip = angle_between_normals_deg(recovered, truth)
    old = np.hypot(kerr_phi - gt_x, kerr_theta - gt_y)
    print(
        "direction metric: "
        f"chart vs command max {np.nanmax(chart_err):.3e}°, "
        f"self-error max {np.nanmax(self_err):.3e}°, "
        f"Kerr-pair roundtrip max {np.nanmax(roundtrip):.3e}°, "
        f"old pair-subtraction median {np.nanmedian(old):.3f}°"
    )
    if (
        float(np.nanmax(chart_err)) > 1e-8
        or float(np.nanmax(self_err)) > 1e-5
        or float(np.nanmax(roundtrip)) > 1e-5
    ):
        raise RuntimeError("direction metric failed its roundtrip check")


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 10:
        return None
    design = np.column_stack([x[mask], np.ones(int(mask.sum()))])
    coef = np.linalg.lstsq(design, y[mask], rcond=None)[0]
    return float(coef[0])


def _score_record(
    err: pd.DataFrame,
    *,
    family: str,
    name: str,
    focal_px: float | None = None,
    eye_z: float | None = None,
    pupil_radius: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    sphere_method: str | None = None,
    angle_mode: str | None = None,
    axis_scale: float | None = None,
    note: str = "",
) -> dict:
    total = err["total"].to_numpy(dtype=float)
    finite = np.isfinite(total)
    n_finite = int(finite.sum())
    frac = float(n_finite / len(err)) if len(err) else 0.0
    inside = finite & (err["ecc_gt"].to_numpy(dtype=float) <= PAPER_SPAN_DEG)

    def _med(mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        return float(np.median(total[mask]))

    def _mean(mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        return float(np.mean(total[mask]))

    return {
        "family": family,
        "name": name,
        "focal_px": focal_px,
        "eye_z": eye_z,
        "pupil_radius": pupil_radius,
        "cx": cx,
        "cy": cy,
        "sphere_method": sphere_method,
        "angle_mode": angle_mode,
        "axis_scale": axis_scale,
        "n_finite": n_finite,
        "frac_finite": frac,
        "median_total_deg": _med(finite),
        "mean_total_deg": _mean(finite),
        "median_within_35_deg": _med(inside),
        "mean_within_35_deg": _mean(inside),
        "slope_phi": _slope(err["x_angle"].to_numpy(dtype=float), err["phi"].to_numpy(dtype=float)),
        "slope_theta": _slope(err["y_angle"].to_numpy(dtype=float), err["theta"].to_numpy(dtype=float)),
        "note": note,
    }


def _summarize(name: str, err: pd.DataFrame) -> dict:
    scored = _score_record(err, family="baseline", name=name)
    ok = err.dropna(subset=["total"])
    outside = ok.loc[ok["ecc_gt"] > PAPER_SPAN_DEG, "total"]
    return {
        "method": name,
        "n_finite": scored["n_finite"],
        "n_input": int(len(err)),
        "mean_total_deg": scored["mean_total_deg"],
        "median_total_deg": scored["median_total_deg"],
        "p95_total_deg": (
            float(np.nanpercentile(ok["total"], 95)) if len(ok) else None
        ),
        "mean_abs_dphi_deg": float(np.nanmean(np.abs(ok["dphi"]))) if len(ok) else None,
        "mean_abs_dtheta_deg": float(np.nanmean(np.abs(ok["dtheta"]))) if len(ok) else None,
        "median_total_within_35_deg": scored["median_within_35_deg"],
        "mean_total_within_35_deg": scored["mean_within_35_deg"],
        "n_within_35_deg": int((ok["ecc_gt"] <= PAPER_SPAN_DEG).sum()) if len(ok) else 0,
        "median_total_beyond_35_deg": float(np.median(outside)) if len(outside) else None,
        "mean_total_beyond_35_deg": float(np.mean(outside)) if len(outside) else None,
        "n_beyond_35_deg": int(len(outside)),
        "slope_phi": scored["slope_phi"],
        "slope_theta": scored["slope_theta"],
    }


def _bin_field(err: pd.DataFrame) -> dict:
    """1° mean-error grid over ground-truth (x_angle, y_angle)."""
    angles = np.arange(-SPAN_DEG, SPAN_DEG + 1, dtype=int)
    n = int(angles.size)
    mag = np.full((n, n), np.nan)
    phi_grid = np.full((n, n), np.nan)
    theta_grid = np.full((n, n), np.nan)
    frame = err.copy()
    frame["x_bin"] = np.round(frame["x_angle"]).astype(int)
    frame["y_bin"] = np.round(frame["y_angle"]).astype(int)
    binned = frame.groupby(["x_bin", "y_bin"], as_index=False).agg(
        err_mag=("total", "mean"),
        phi=("phi", "mean"),
        theta=("theta", "mean"),
    )
    index = {int(v): i for i, v in enumerate(angles)}
    for row in binned.itertuples(index=False):
        if row.x_bin not in index or row.y_bin not in index:
            continue
        i = index[row.x_bin]
        j = index[row.y_bin]
        mag[j, i] = row.err_mag
        phi_grid[j, i] = row.phi
        theta_grid[j, i] = row.theta
    return {"angles": angles, "mag": mag, "phi": phi_grid, "theta": theta_grid}


def _gaze_limits(fields: list[dict], pad: float = 8.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis window covering the ground-truth grid and every reported arrow tip."""
    xs = [-float(SPAN_DEG) - 0.5, float(SPAN_DEG) + 0.5]
    ys = [-float(SPAN_DEG) - 0.5, float(SPAN_DEG) + 0.5]
    for field in fields:
        phi = np.asarray(field["phi"], dtype=float)
        theta = np.asarray(field["theta"], dtype=float)
        finite = np.isfinite(phi) & np.isfinite(theta)
        if not np.any(finite):
            continue
        xs.extend([float(np.min(phi[finite])), float(np.max(phi[finite]))])
        ys.extend([float(np.min(theta[finite])), float(np.max(theta[finite]))])
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def _draw_heatmap(ax, field: dict, title: str, vmin: float, vmax: float, limits=None):
    mag = field["mag"]
    angles = field["angles"]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="white")
    mesh = ax.imshow(
        np.ma.masked_invalid(mag),
        origin="lower",
        extent=(-SPAN_DEG - 0.5, SPAN_DEG + 0.5, -SPAN_DEG - 0.5, SPAN_DEG + 0.5),
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    step = QUIVER_STEP_DEG
    q_idx = np.arange(0, angles.size, step)
    x_q = angles[q_idx]
    y_q = angles[q_idx]
    xx, yy = np.meshgrid(x_q, y_q)
    uu = field["phi"][np.ix_(q_idx, q_idx)] - xx
    vv = field["theta"][np.ix_(q_idx, q_idx)] - yy
    finite = np.isfinite(uu) & np.isfinite(vv)
    ax.quiver(
        xx[finite],
        yy[finite],
        uu[finite],
        vv[finite],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        color="black",
        pivot="tail",
        zorder=3,
    )
    if limits is None:
        limits = _gaze_limits([field])
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xlabel("Command yaw (deg)")
    ax.set_ylabel("Command pitch (deg)")
    ax.set_title(title)
    ax.tick_params(labelsize=8)
    return mesh


def _save_heatmap(field: dict, title: str, path: Path, vmin: float, vmax: float, limits=None) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=150)
    mesh = _draw_heatmap(ax, field, title, vmin, vmax, limits=limits)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("3D gaze error (deg)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if path.suffix == ".pdf":
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _save_heatmap_pair(fields: dict[str, dict], path: Path, vmin: float, vmax: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), dpi=150)
    limits = _gaze_limits([fields[name] for name in ("Kerr", "Swirski") if name in fields])
    mesh = None
    for ax, name in zip(axes, ("Kerr", "Swirski")):
        mesh = _draw_heatmap(ax, fields[name], name, vmin, vmax, limits=limits)
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("3D gaze error (deg)", fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _shared_vmax(fields: dict[str, dict]) -> float:
    peaks = [float(np.nanmax(field["mag"])) for field in fields.values()]
    peak = max(peaks)
    if not np.isfinite(peak) or peak <= 0:
        return 1.0
    return peak


def _dipole_note(err: pd.DataFrame) -> dict:
    """Signed phi error on the left and right halves of the ground-truth field."""
    dphi = err["dphi"].to_numpy(dtype=float)
    x = err["x_angle"].to_numpy(dtype=float)
    left = np.isfinite(dphi) & (x < -10)
    right = np.isfinite(dphi) & (x > 10)
    left_mean = float(np.mean(dphi[left])) if np.any(left) else None
    right_mean = float(np.mean(dphi[right])) if np.any(right) else None
    gap = None
    if left_mean is not None and right_mean is not None:
        gap = float(right_mean - left_mean)
    return {
        "mean_dphi_x_lt_m10": left_mean,
        "mean_dphi_x_gt_10": right_mean,
        "right_minus_left_deg": gap,
    }


def _eccentricity_bins(named_errors: dict[str, pd.DataFrame], n_bins: int) -> pd.DataFrame:
    ecc = np.concatenate([e["ecc_gt"].to_numpy(dtype=float) for e in named_errors.values()])
    ecc = ecc[np.isfinite(ecc)]
    edges = np.linspace(0.0, float(np.max(ecc)), int(n_bins) + 1)
    rows = []
    for name, err in named_errors.items():
        ok = err.dropna(subset=["total"]).copy()
        cats = pd.cut(ok["ecc_gt"], edges, include_lowest=True)
        grouped = ok.groupby(cats, observed=False)["total"]
        for interval, series in grouped:
            values = series.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "method": name,
                    "ecc_left": float(interval.left),
                    "ecc_right": float(interval.right),
                    "ecc_center": float(interval.mid),
                    "n": int(values.size),
                    "mean_total_deg": float(np.mean(values)) if values.size else np.nan,
                    "median_total_deg": float(np.median(values)) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_eccentricity(bins: pd.DataFrame, path_pdf: Path, path_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
    colors = {"Kerr": "#1f4e79", "Swirski": "#b85c38"}
    for name, color in colors.items():
        curve = bins.loc[(bins["method"] == name) & (bins["n"] >= 20)]
        ax.plot(
            curve["ecc_center"],
            curve["mean_total_deg"],
            color=color,
            linewidth=1.8,
            label=name,
        )
    ax.axvline(PAPER_SPAN_DEG, color="0.35", linestyle="--", linewidth=1.0, label="35° span")
    ax.set_xlabel("Ground-truth eccentricity (deg)")
    ax.set_ylabel("Mean total error (deg)")
    ax.set_xlim(0, float(bins["ecc_right"].max()))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_pdf)
    fig.savefig(path_png)
    plt.close(fig)


def _try_swirski(table: pd.DataFrame, **kwargs):
    try:
        phi, theta, _commit, sphere = _run_swirski(table, **kwargs)
        return phi, theta, sphere, ""
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


def _evaluate_table(work: pd.DataFrame, table: pd.DataFrame, **kwargs) -> tuple[dict, pd.DataFrame | None]:
    phi, theta, sphere, note = _try_swirski(table, **kwargs)
    if phi is None:
        empty = _error_frame(np.full(len(work), np.nan), np.full(len(work), np.nan), work)
        err = empty
    else:
        err = _error_frame(phi, theta, work)
        if sphere is not None:
            note = note or f"sphere_radius={sphere['radius']:.3f}"
    return err, note


def _pick_best(rows: list[dict], frac_floor: float = MIN_FINITE_FRAC) -> dict:
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame["frac_finite"] >= frac_floor]
    if eligible.empty:
        eligible = frame.loc[frame["n_finite"] > 0]
    if eligible.empty:
        raise RuntimeError("no finite Swirski settings to choose from")
    return eligible.sort_values(["median_total_deg", "frac_finite"], ascending=[True, False]).iloc[0].to_dict()


def _plot_sweep_lines(frame: pd.DataFrame, x_col: str, path: Path, title: str) -> None:
    ok = frame.dropna(subset=[x_col, "median_total_deg"]).sort_values(x_col)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.6), dpi=150, sharex=True)
    axes[0].plot(ok[x_col], ok["median_total_deg"], color="#b85c38", marker="o", linewidth=1.4)
    axes[0].set_ylabel("Median error (deg)")
    axes[0].set_title(title)
    axes[1].plot(ok[x_col], ok["slope_phi"], color="#1f4e79", marker="o", linewidth=1.4)
    axes[1].axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel(r"Slope $\phi$ vs $x$")
    axes[1].set_xlabel(x_col)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _plot_joint(grid: pd.DataFrame, path: Path) -> None:
    focals = np.sort(grid["focal_px"].unique())
    depths = np.sort(grid["eye_z"].unique())
    mag = np.full((len(depths), len(focals)), np.nan)
    f_index = {float(v): i for i, v in enumerate(focals)}
    z_index = {float(v): i for i, v in enumerate(depths)}
    for row in grid.itertuples(index=False):
        mag[z_index[float(row.eye_z)], f_index[float(row.focal_px)]] = row.median_total_deg
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    mesh = ax.imshow(mag, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(focals)))
    ax.set_xticklabels([f"{v:.0f}" for v in focals])
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([f"{v:g}" for v in depths])
    ax.set_xlabel("Focal length (px)")
    ax.set_ylabel("eye_z")
    ax.set_title("Median Swirski error")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Median error (deg)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _print_table(summary: pd.DataFrame) -> None:
    show = summary.copy()
    skip = {"method", "n_finite", "n_input", "n_within_35_deg", "n_beyond_35_deg"}
    for col in show.columns:
        if col in skip:
            continue
        show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")
    print(show.to_string(index=False))


def _base_kwargs(focal_px: float, angle_mode: str, axis_scale: float) -> dict:
    return {
        "focal_px": float(focal_px),
        "eye_z": EYE_Z,
        "pupil_radius": PUPIL_RADIUS,
        "cx": REF_X,
        "cy": REF_Y,
        "sphere_method": "swirski_2d",
        "angle_mode": angle_mode,
        "axis_scale": float(axis_scale),
    }


def _best_reported_focal(out_dir: Path) -> float:
    """Focal length the latest 3D-error sweep selected, or 310 px if it is absent."""
    path = out_dir / "sweep_focal_120_1000.csv"
    if not path.is_file():
        return 310.0
    frame = pd.read_csv(path)
    return float(_pick_best(frame.to_dict(orient="records"))["focal_px"])


def _euler_kerr_angles(work: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Kerr readout of a yaw-then-pitch normal built from the command angles.

    ``n = (sin x cos y, sin y, -cos x cos y)`` is the gaze toward the camera
    after a horizontal rotation ``x`` and a vertical rotation ``y``. Kerr's
    nested arcsin does not invert that normal back to ``(x, y)``.
    """
    x = np.deg2rad(work["x_angle"].to_numpy(dtype=float))
    y = np.deg2rad(work["y_angle"].to_numpy(dtype=float))
    nx = np.sin(x) * np.cos(y)
    ny = np.sin(y)
    phi = np.degrees(np.arcsin(np.clip(nx, -1.0, 1.0)))
    cos_phi = np.cos(np.deg2rad(phi))
    theta = np.degrees(np.arcsin(np.clip(ny / cos_phi, -1.0, 1.0)))
    return phi, theta


def _run_knob_sweep(csv_path: Path, out_dir: Path) -> int:
    """Re-score eye distance, pupil radius, and principal point at 310 and 653 px."""
    raw = pd.read_csv(csv_path)
    work = _prepare(raw)
    _verify_direction_metric(work)
    table = _swirski_table(work, "major_plus_90", 0.5)
    focals = (_best_reported_focal(out_dir), float(LAB_FOCAL_PX))
    records = []

    def score(family: str, name: str, **kwargs) -> dict:
        err, note = _evaluate_table(work, table, **kwargs)
        record = _score_record(
            err,
            family=family,
            name=name,
            angle_mode="major_plus_90",
            axis_scale=0.5,
            note=note,
            **kwargs,
        )
        records.append(record)
        print(
            f"  {family} {name}: median {record['median_total_deg']} "
            f"frac {record['frac_finite']:.3f} "
            f"slope {record['slope_phi']:.3f}/{record['slope_theta']:.3f}"
        )
        return record

    for focal in focals:
        label = "lab" if abs(focal - LAB_FOCAL_PX) < 1e-6 else f"{focal:.0f}"
        print(f"focal {focal:.3f} px")
        phi, theta, _commit, sphere = _run_swirski(
            table,
            focal_px=float(focal),
            eye_z=EYE_Z,
            pupil_radius=PUPIL_RADIUS,
            cx=REF_X,
            cy=REF_Y,
            sphere_method="swirski_2d",
        )
        err = _error_frame(phi, theta, work)
        chart = np.hypot(err["dphi"].to_numpy(dtype=float), err["dtheta"].to_numpy(dtype=float))
        print(
            f"  3D median {np.nanmedian(err['total']):.3f}°"
            f"  chart residual median {np.nanmedian(chart):.3f}°"
            f"  sphere r={sphere['radius']:.3f} centre={sphere['centre_xyz']}"
        )
        for eye_z in EYE_Z_GRID:
            score(
                "eye_z",
                f"f{label}_z{eye_z:g}",
                focal_px=float(focal),
                eye_z=float(eye_z),
                pupil_radius=PUPIL_RADIUS,
                cx=REF_X,
                cy=REF_Y,
                sphere_method="swirski_2d",
            )
        for radius in PUPIL_RADII:
            score(
                "pupil_radius",
                f"f{label}_r{radius:g}",
                focal_px=float(focal),
                eye_z=EYE_Z,
                pupil_radius=float(radius),
                cx=REF_X,
                cy=REF_Y,
                sphere_method="swirski_2d",
            )
        for dx in (-4.0, 0.0, 4.0):
            for dy in (-4.0, 0.0, 4.0):
                score(
                    "principal_point",
                    f"f{label}_dx{dx:g}_dy{dy:g}",
                    focal_px=float(focal),
                    eye_z=EYE_Z,
                    pupil_radius=PUPIL_RADIUS,
                    cx=REF_X + dx,
                    cy=REF_Y + dy,
                    sphere_method="swirski_2d",
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sweep_knobs_focal.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"wrote {path}")
    return 0


def _run_focal_sweep_only(csv_path: Path, out_dir: Path) -> int:
    """Score Swirski from 120 to 1000 px, plus the exact 3.7 mm lab focal length."""
    raw = pd.read_csv(csv_path)
    work = _prepare(raw)
    _verify_direction_metric(work)
    table = _swirski_table(work, "major_plus_90", 0.5)
    focals = focal_sweep_values()
    print(f"focal sweep: {len(focals)} values, {focals[0]:.1f}–{focals[-1]:.1f} px")
    print(f"lab focal length inserted at {LAB_FOCAL_PX:.3f} px")
    records = []
    for focal in focals:
        err, note = _evaluate_table(
            work,
            table,
            focal_px=float(focal),
            eye_z=EYE_Z,
            pupil_radius=PUPIL_RADIUS,
            cx=REF_X,
            cy=REF_Y,
            sphere_method="swirski_2d",
        )
        record = _score_record(
            err,
            family="focal",
            name="lab_3.7mm" if abs(focal - LAB_FOCAL_PX) < 1e-6 else f"f{focal:.1f}",
            focal_px=float(focal),
            eye_z=EYE_Z,
            pupil_radius=PUPIL_RADIUS,
            cx=REF_X,
            cy=REF_Y,
            sphere_method="swirski_2d",
            angle_mode="major_plus_90",
            axis_scale=0.5,
            note=note,
        )
        records.append(record)
        print(
            f"  f={focal:.3f}: median {record['median_total_deg']} "
            f"within35 {record['median_within_35_deg']} "
            f"frac {record['frac_finite']:.3f} slope_phi {record['slope_phi']}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    path = out_dir / "sweep_focal_120_1000.csv"
    frame.to_csv(path, index=False)
    best = _pick_best(records)
    print(
        f"best high-coverage focal: {float(best['focal_px']):.3f} px, "
        f"median {float(best['median_total_deg']):.3f}°"
    )
    print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument(
        "--only-focal",
        action="store_true",
        help="Run only the 120–1000 px focal sweep, including the 3.7 mm lab value.",
    )
    parser.add_argument(
        "--only-knobs",
        action="store_true",
        help="Sweep eye distance, pupil radius, and principal point at 310 px and at 3.7 mm.",
    )
    args = parser.parse_args(argv)

    if not args.csv.is_file():
        raise SystemExit(f"Ground-truth table not found: {args.csv}")
    if args.only_focal and args.only_knobs:
        raise SystemExit("Choose one of --only-focal or --only-knobs")
    if args.only_focal:
        return _run_focal_sweep_only(args.csv, args.out)
    if args.only_knobs:
        return _run_knob_sweep(args.csv, args.out)

    raw = pd.read_csv(args.csv)
    work = _prepare(raw)
    focal_px, focal_intercept = _fit_focal_px(work)
    phi_check = [
        _nearest_row(work, 0.0, 0.0),
        _nearest_row(work, 30.0, 0.0),
        _nearest_row(work, 0.0, 30.0),
    ]

    kerr_phi, kerr_theta, kerr_fz = _run_kerr(work)
    baseline_table = _swirski_table(work, "major_plus_90", 0.5)
    sw_phi, sw_theta, sw_commit, sphere = _run_swirski(baseline_table, focal_px=focal_px)

    errors = {
        "Kerr": _error_frame(kerr_phi, kerr_theta, work),
        "Swirski": _error_frame(sw_phi, sw_theta, work),
    }
    fields = {name: _bin_field(err) for name, err in errors.items()}
    vmax = _shared_vmax(fields)
    summary = pd.DataFrame([_summarize(name, err) for name, err in errors.items()])
    bins = _eccentricity_bins(errors, args.bins)
    dipole = _dipole_note(errors["Swirski"])

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "comparison_table.csv"
    bins_path = args.out / "error_vs_eccentricity.csv"
    ecc_pdf = args.out / "error_vs_eccentricity.pdf"
    ecc_png = args.out / "error_vs_eccentricity.png"
    summary.to_csv(summary_path, index=False)
    bins.to_csv(bins_path, index=False)
    _plot_eccentricity(bins, ecc_pdf, ecc_png)
    _save_heatmap(fields["Kerr"], "Kerr", args.out / "kerr_error_heatmap.pdf", 0.0, vmax)
    _save_heatmap(fields["Swirski"], "Swirski", args.out / "swirski_error_heatmap.pdf", 0.0, vmax)
    _save_heatmap_pair(fields, args.out / "error_heatmaps.pdf", 0.0, vmax)

    print(f"rows: {len(work)}")
    print(f"Kerr f_z: {kerr_fz:.3f} px")
    print(f"Swirski focal length: {focal_px:.3f} px (commit {sw_commit})")
    print(
        "ellipse_phi check: "
        + ", ".join(
            f"({row['x_angle']:.0f},{row['y_angle']:.0f}) -> {row['ellipse_phi_deg']:.1f}°"
            for row in phi_check
        )
    )
    print()
    _print_table(summary)
    print()

    records: list[dict] = []
    tables: dict[tuple[str, float], pd.DataFrame] = {("major_plus_90", 0.5): baseline_table}

    def table_for(mode: str, scale: float) -> pd.DataFrame:
        key = (mode, float(scale))
        if key not in tables:
            tables[key] = _swirski_table(work, mode, scale)
        return tables[key]

    print("scoring ellipse conventions")
    for mode in ANGLE_MODES:
        for scale in AXIS_SCALES:
            if mode == "major_plus_90" and scale == 0.5:
                err = errors["Swirski"]
                note = "baseline"
            else:
                err, note = _evaluate_table(
                    work,
                    table_for(mode, scale),
                    focal_px=focal_px,
                    eye_z=EYE_Z,
                    pupil_radius=PUPIL_RADIUS,
                    cx=REF_X,
                    cy=REF_Y,
                    sphere_method="swirski_2d",
                )
            record = _score_record(
                err,
                family="convention",
                name=f"{mode}_scale{scale:g}",
                **_base_kwargs(focal_px, mode, scale),
                note=note,
            )
            records.append(record)
            print(
                f"  {record['name']}: median {record['median_total_deg']} "
                f"frac {record['frac_finite']:.3f} slope_phi {record['slope_phi']}"
            )

    convention_rows = [row for row in records if row["family"] == "convention"]
    best_convention = _pick_best(convention_rows)
    locked_mode = str(best_convention["angle_mode"])
    locked_scale = float(best_convention["axis_scale"])
    locked_table = table_for(locked_mode, locked_scale)
    print(f"locked convention: {locked_mode}, axis scale {locked_scale:g}")

    def score_setting(family: str, name: str, **kwargs) -> dict:
        err, note = _evaluate_table(work, locked_table, **kwargs)
        fit_kwargs = {
            "focal_px": kwargs.get("focal_px", focal_px),
            "eye_z": kwargs.get("eye_z", EYE_Z),
            "pupil_radius": kwargs.get("pupil_radius", PUPIL_RADIUS),
            "cx": kwargs.get("cx", REF_X),
            "cy": kwargs.get("cy", REF_Y),
            "sphere_method": kwargs.get("sphere_method", "swirski_2d"),
        }
        record = _score_record(
            err,
            family=family,
            name=name,
            angle_mode=locked_mode,
            axis_scale=locked_scale,
            note=note,
            **fit_kwargs,
        )
        records.append(record)
        return record

    print("focal-length sweep")
    focal_values = focal_sweep_values()
    for focal in focal_values:
        record = score_setting(
            "focal",
            f"f{focal:.1f}",
            focal_px=float(focal),
            eye_z=EYE_Z,
            pupil_radius=PUPIL_RADIUS,
            cx=REF_X,
            cy=REF_Y,
            sphere_method="swirski_2d",
        )
        print(f"  f={focal:.1f}: median {record['median_total_deg']} slope {record['slope_phi']}")

    print("eye_z sweep")
    for eye_z in EYE_Z_GRID:
        record = score_setting(
            "eye_z",
            f"z{eye_z:g}",
            focal_px=focal_px,
            eye_z=float(eye_z),
            pupil_radius=PUPIL_RADIUS,
            cx=REF_X,
            cy=REF_Y,
            sphere_method="swirski_2d",
        )
        print(f"  eye_z={eye_z:g}: median {record['median_total_deg']} slope {record['slope_phi']}")

    print("focal x eye_z grid")
    joint_focals = list(np.arange(40.0, 121.0, 10.0))
    for focal in joint_focals:
        for eye_z in EYE_Z_GRID:
            record = score_setting(
                "focal_eyez",
                f"f{focal:.0f}_z{eye_z:g}",
                focal_px=float(focal),
                eye_z=float(eye_z),
                pupil_radius=PUPIL_RADIUS,
                cx=REF_X,
                cy=REF_Y,
                sphere_method="swirski_2d",
            )
            print(
                f"  f={focal:.0f} z={eye_z:g}: median {record['median_total_deg']} "
                f"slope {record['slope_phi']}"
            )

    print("pupil-radius sweep")
    for radius in PUPIL_RADII:
        record = score_setting(
            "pupil_radius",
            f"r{radius:g}",
            focal_px=focal_px,
            eye_z=EYE_Z,
            pupil_radius=float(radius),
            cx=REF_X,
            cy=REF_Y,
            sphere_method="swirski_2d",
        )
        print(f"  r={radius:g}: median {record['median_total_deg']}")

    print("principal-point grid")
    for dx in (-4.0, 0.0, 4.0):
        for dy in (-4.0, 0.0, 4.0):
            record = score_setting(
                "principal_point",
                f"dx{dx:g}_dy{dy:g}",
                focal_px=focal_px,
                eye_z=EYE_Z,
                pupil_radius=PUPIL_RADIUS,
                cx=REF_X + dx,
                cy=REF_Y + dy,
                sphere_method="swirski_2d",
            )
            print(f"  ({REF_X + dx:.0f}, {REF_Y + dy:.0f}): median {record['median_total_deg']}")

    print("sphere methods")
    for method in ("swirski_2d", "dierkes_3d"):
        record = score_setting(
            "sphere_method",
            method,
            focal_px=focal_px,
            eye_z=EYE_Z,
            pupil_radius=PUPIL_RADIUS,
            cx=REF_X,
            cy=REF_Y,
            sphere_method=method,
        )
        print(f"  {method}: median {record['median_total_deg']} slope {record['slope_phi']}")

    sweep = pd.DataFrame(records)
    joint = sweep.loc[sweep["family"] == "focal_eyez"].copy()
    focal_family = sweep.loc[sweep["family"] == "focal"]
    best_joint = _pick_best(focal_family.to_dict(orient="records"))
    best_phi, best_theta, best_sphere, best_note = _try_swirski(
        locked_table,
        focal_px=float(best_joint["focal_px"]),
        eye_z=float(best_joint["eye_z"]),
        pupil_radius=PUPIL_RADIUS,
        cx=REF_X,
        cy=REF_Y,
        sphere_method="swirski_2d",
    )
    if best_phi is None:
        raise RuntimeError(f"best Swirski setting failed: {best_note}")
    best_err = _error_frame(best_phi, best_theta, work)
    best_field = _bin_field(best_err)
    best_title = (
        f"Swirski best ({locked_mode}, scale {locked_scale:g}, "
        f"f={float(best_joint['focal_px']):.0f}, z={float(best_joint['eye_z']):g})"
    )
    _save_heatmap(
        best_field,
        best_title,
        args.out / "swirski_error_heatmap_best.pdf",
        0.0,
        vmax,
    )

    sweep_path = args.out / "sweep_summary.csv"
    joint_path = args.out / "sweep_focal_eyez.csv"
    sweep.to_csv(sweep_path, index=False)
    joint.to_csv(joint_path, index=False)
    focal_rows = sweep.loc[sweep["family"] == "focal"]
    _plot_sweep_lines(
        focal_rows,
        "focal_px",
        args.out / "sweep_focal.pdf",
        f"Focal length at eye_z={EYE_Z:g}, {locked_mode}",
    )
    _plot_sweep_lines(
        sweep.loc[sweep["family"] == "eye_z"],
        "eye_z",
        args.out / "sweep_eye_z.pdf",
        f"eye_z at f={focal_px:.1f} px, {locked_mode}",
    )
    _plot_joint(joint, args.out / "sweep_focal_eyez.pdf")

    metadata = {
        "csv": str(args.csv),
        "n_rows": int(len(work)),
        "reference_xy": [REF_X, REF_Y],
        "kerr_f_z_px": kerr_fz,
        "swirski_focal_px": focal_px,
        "swirski_focal_fit_intercept_x": focal_intercept,
        "lab_focal_px": LAB_FOCAL_PX,
        "swirski_eye_z": EYE_Z,
        "swirski_sphere_method": "swirski_2d",
        "swirski_branch": "conicoid_approach",
        "swirski_commit": sw_commit,
        "swirski_sphere": sphere,
        "heatmap_vmax_deg": vmax,
        "phi_geometric_check": phi_check,
        "swirski_dipole": dipole,
        "locked_convention": {
            "angle_mode": locked_mode,
            "axis_scale": locked_scale,
            "median_total_deg": best_convention["median_total_deg"],
            "slope_phi": best_convention["slope_phi"],
            "frac_finite": best_convention["frac_finite"],
        },
        "best_joint": {
            "focal_px": best_joint["focal_px"],
            "eye_z": best_joint["eye_z"],
            "median_total_deg": best_joint["median_total_deg"],
            "mean_total_deg": best_joint["mean_total_deg"],
            "slope_phi": best_joint["slope_phi"],
            "slope_theta": best_joint["slope_theta"],
            "frac_finite": best_joint["frac_finite"],
            "sphere": best_sphere,
        },
        "error_definition": "hypot(phi - x_angle, theta - y_angle)",
        "baseline_axis_convention": "semi-axes = stored/2; major angle = ellipse_phi+90deg",
        "outputs": {
            "table": str(summary_path),
            "kerr_heatmap": str(args.out / "kerr_error_heatmap.pdf"),
            "swirski_heatmap": str(args.out / "swirski_error_heatmap.pdf"),
            "heatmap_pair": str(args.out / "error_heatmaps.pdf"),
            "best_heatmap": str(args.out / "swirski_error_heatmap_best.pdf"),
            "sweep": str(sweep_path),
            "joint": str(joint_path),
        },
    }
    meta_path = args.out / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(
        f"best joint: f={float(best_joint['focal_px']):.0f} px, "
        f"eye_z={float(best_joint['eye_z']):g}, "
        f"median {float(best_joint['median_total_deg']):.3f}°"
    )
    print(f"maps:  {args.out / 'error_heatmaps.pdf'}")
    print(f"sweep: {sweep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
