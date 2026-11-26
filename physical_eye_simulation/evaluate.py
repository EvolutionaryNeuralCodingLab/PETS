"""Score Kerr and Swirski on analytic (and optional rendered) ground-truth CSVs."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-pets-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.conicoid import (
    CameraIntrinsics,
    angle_between_normals_deg,
    chart_yaw_pitch,
    fit_eye_and_gaze,
    kerr_angles_to_normals,
)
from physical_eye_simulation.camera_model import (
    CAMERA_DISTANCES_MM,
    FOCAL_PX,
    PIVOTS,
    PaperCamera,
)

OUT_ROOT = REPO / "outputs" / "physical_eye_simulation_runs"
SPAN_DEG = 60
QUIVER_STEP = 5


def _swirski_table(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    def col(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    return pd.DataFrame(
        {
            "center_x": frame[col("center_x")],
            "center_y": frame[col("center_y")],
            "width": frame[col("width")],
            "height": frame[col("height")],
            "major_ax": frame[col("width")],
            "minor_ax": frame[col("height")],
            "phi": frame[col("phi")],
            "OE_timestamp": 0,
            "eye_frame": np.arange(len(frame)),
        }
    )


def _truth_normals(frame: pd.DataFrame) -> np.ndarray:
    return frame[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=float)


def score_swirski(
    frame: pd.DataFrame,
    *,
    eye_z: float,
    pupil_radius: float,
    prefix: str = "",
    use_ransac: bool = False,
    camera: PaperCamera | None = None,
) -> tuple[pd.DataFrame, dict]:
    table = _swirski_table(frame, prefix=prefix)
    cam = camera or PaperCamera()
    camera = CameraIntrinsics(
        width=int(cam.width),
        height=int(cam.height),
        cx=float(cam.cx),
        cy=float(cam.cy),
        focal_px=float(cam.focal_px),
        source="paper",
    )
    out, fit = fit_eye_and_gaze(
        table,
        camera,
        eye_z=float(eye_z),
        pupil_radius=float(pupil_radius),
        use_ransac=use_ransac,
        sphere_method="swirski_2d",
    )
    reported = np.column_stack(
        [
            out["c_nx"].to_numpy(dtype=float),
            out["c_ny"].to_numpy(dtype=float),
            out["c_nz"].to_numpy(dtype=float),
        ]
    )
    truth = _truth_normals(frame)
    yaw, pitch = chart_yaw_pitch(reported)
    err = pd.DataFrame(
        {
            "x_angle": frame["x_angle"].to_numpy(dtype=float),
            "y_angle": frame["y_angle"].to_numpy(dtype=float),
            "phi": yaw,
            "theta": pitch,
            "total": angle_between_normals_deg(reported, truth),
            "ecc_gt": np.hypot(frame["x_angle"], frame["y_angle"]),
        }
    )
    meta = {
        "sphere_centre": [float(v) for v in np.asarray(fit.sphere.centre).reshape(3)],
        "sphere_radius": float(fit.sphere.radius),
        "n_inliers": int(fit.n_inliers),
        "use_ransac": use_ransac,
    }
    return err, meta


def score_kerr(frame: pd.DataFrame, *, prefix: str = "") -> tuple[pd.DataFrame, dict]:
    table = _swirski_table(frame, prefix=prefix)
    # Kerr needs axs_1/axs_2 and ellipse_phi in degrees, plus ratio columns.
    work = table.copy()
    work["axs_1"] = work["minor_ax"] * 2.0
    work["axs_2"] = work["major_ax"] * 2.0
    work["ellipse_phi"] = np.degrees(work["phi"].to_numpy(dtype=float))
    work["ratio2"] = work["axs_1"] / work["axs_2"]
    work["ms_axis"] = 0.0
    # Reference = pupil at rest (0,0)
    rest = frame.loc[(frame["x_angle"].abs() < 0.5) & (frame["y_angle"].abs() < 0.5)]
    if len(rest):
        aec = float(rest.iloc[0][f"{prefix}center_x" if prefix else "center_x"])
        bec = float(rest.iloc[0][f"{prefix}center_y" if prefix else "center_y"])
    else:
        aec, bec = 320.0, 240.0
    f_z, out = BlockSync.kerr(work, aEC=aec, bEC=bec)
    reported = kerr_angles_to_normals(out["phi"], out["theta"])
    truth = _truth_normals(frame)
    yaw, pitch = chart_yaw_pitch(reported)
    err = pd.DataFrame(
        {
            "x_angle": frame["x_angle"].to_numpy(dtype=float),
            "y_angle": frame["y_angle"].to_numpy(dtype=float),
            "phi": yaw,
            "theta": pitch,
            "total": angle_between_normals_deg(reported, truth),
            "ecc_gt": np.hypot(frame["x_angle"], frame["y_angle"]),
        }
    )
    return err, {"f_z_px": float(f_z), "aEC": aec, "bEC": bec}


def _bin_field(err: pd.DataFrame) -> dict:
    angles = np.arange(-SPAN_DEG, SPAN_DEG + 1, dtype=int)
    n = int(angles.size)
    mag = np.full((n, n), np.nan)
    phi = np.full((n, n), np.nan)
    theta = np.full((n, n), np.nan)
    frame = err.copy()
    frame["x_bin"] = np.round(frame["x_angle"]).astype(int)
    frame["y_bin"] = np.round(frame["y_angle"]).astype(int)
    binned = frame.groupby(["x_bin", "y_bin"], as_index=False).agg(
        err_mag=("total", "mean"), phi=("phi", "mean"), theta=("theta", "mean")
    )
    index = {int(v): i for i, v in enumerate(angles)}
    for row in binned.itertuples(index=False):
        if row.x_bin not in index or row.y_bin not in index:
            continue
        mag[index[row.y_bin], index[row.x_bin]] = row.err_mag
        phi[index[row.y_bin], index[row.x_bin]] = row.phi
        theta[index[row.y_bin], index[row.x_bin]] = row.theta
    return {"angles": angles, "mag": mag, "phi": phi, "theta": theta}


def _turbo():
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="0.85")
    return cmap


def save_heatmap(field: dict, title: str, path: Path, *, arrows: bool) -> None:
    mag = field["mag"]
    vmax = float(np.nanpercentile(mag, 99)) if np.isfinite(mag).any() else 1.0
    vmax = max(vmax, 1e-3)
    span = SPAN_DEG + 0.5
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=150)
    mesh = ax.imshow(
        np.ma.masked_invalid(mag),
        origin="lower",
        extent=(-span, span, -span, span),
        cmap=_turbo(),
        interpolation="nearest",
        aspect="equal",
        vmin=0.0,
        vmax=vmax,
    )
    if arrows:
        angles = field["angles"]
        idx = np.arange(0, angles.size, QUIVER_STEP)
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
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("3D gaze error (deg)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize(err: pd.DataFrame) -> dict:
    total = err["total"].to_numpy(dtype=float)
    y = err["y_angle"].to_numpy(dtype=float)
    x = err["x_angle"].to_numpy(dtype=float)
    finite = np.isfinite(total)

    def med(mask: np.ndarray) -> float | None:
        m = finite & mask
        return float(np.median(total[m])) if np.any(m) else None

    return {
        "n_frames": int(len(err)),
        "frac_finite": float(finite.mean()) if len(err) else 0.0,
        "median_deg": float(np.nanmedian(total)),
        "mean_deg": float(np.nanmean(total)),
        "p90_deg": float(np.nanpercentile(total[finite], 90)) if np.any(finite) else None,
        "median_pitch_gt_30": med(y > 30),
        "median_pitch_lt_m30": med(y < -30),
        "median_yaw_gt_30": med(x > 30),
        "median_yaw_lt_m30": med(x < -30),
        "median_within_30": med(np.hypot(x, y) <= 30),
    }


def evaluate_scenario_dir(scenario_dir: Path, *, source: str = "analytic") -> dict:
    if source == "analytic":
        csv_path = scenario_dir / "ground_truth_analytic.csv"
        prefix = ""
        meta_path = scenario_dir / "scenario_analytic.json"
    else:
        csv_path = scenario_dir / "ground_truth.csv"
        prefix = "r_"
        meta_path = scenario_dir / "scenario.json"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    pupil_radius = float(meta.get("eye_geometry", {}).get("pupil", {}).get("radius_mm", 1.5))
    pivot_cam = meta.get("pivot_cam_mm", [0.0, 0.0, 13.0])
    eye_z = float(pivot_cam[2]) if len(pivot_cam) == 3 else 13.0

    # Drop rows without the requested ellipse columns.
    cols = [f"{prefix}center_x", f"{prefix}center_y", f"{prefix}width", f"{prefix}height", f"{prefix}phi"]
    if prefix:
        # fall back to analytic columns if rendered missing
        if cols[0] not in frame.columns:
            prefix = ""
            cols = ["center_x", "center_y", "width", "height", "phi"]
    ok = frame.dropna(subset=cols if prefix == "" else cols)
    if prefix:
        # rename for shared scorers by creating a view with prefix stripped via _swirski_table
        pass

    cam_meta = meta.get("camera") or {}
    if cam_meta:
        paper_cam = PaperCamera(
            width=int(cam_meta.get("width", 640)),
            height=int(cam_meta.get("height", 480)),
            focal_px=float(cam_meta.get("focal_px", FOCAL_PX)),
            cx=float(cam_meta.get("cx", 320.0)),
            cy=float(cam_meta.get("cy", 240.0)),
            lens_mm=float(cam_meta.get("lens_mm", 3.7)),
            sensor_width_mm=float(cam_meta.get("sensor_width_mm", 3.6288)),
        )
    else:
        paper_cam = PaperCamera()
    sw_err, sw_meta = score_swirski(
        ok, eye_z=eye_z, pupil_radius=pupil_radius, prefix=prefix, camera=paper_cam
    )
    kerr_err, kerr_meta = score_kerr(ok, prefix=prefix)

    out = scenario_dir / f"eval_{source}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, err, extra in (
        ("Swirski", sw_err, sw_meta),
        ("Kerr", kerr_err, kerr_meta),
    ):
        summary = summarize(err)
        summary.update({"method": name, "source": source, **extra})
        rows.append(summary)
        field = _bin_field(err)
        save_heatmap(field, f"{name} ({source})\nmedian {summary['median_deg']:.2f}°",
                     out / f"{name.lower()}_heatmap_arrows", arrows=True)
        save_heatmap(field, f"{name} ({source})\nmedian {summary['median_deg']:.2f}°",
                     out / f"{name.lower()}_heatmap", arrows=False)
        err.to_csv(out / f"{name.lower()}_errors.csv", index=False)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "summary.csv", index=False)
    (out / "eval_meta.json").write_text(
        json.dumps({"scenario": meta, "swirski": sw_meta, "kerr": kerr_meta}, indent=2),
        encoding="utf-8",
    )
    return {"dir": str(out), "summary": rows}


def cross_scenario_figure(
    out_root: Path = OUT_ROOT,
    *,
    source: str = "analytic",
) -> Path:
    rows = []
    eval_dir = f"eval_{source}"
    for side in ("right", "left"):
        for pivot in PIVOTS:
            for distance in CAMERA_DISTANCES_MM:
                path = out_root / side / pivot / f"d{distance:g}mm" / eval_dir / "summary.csv"
                if not path.is_file():
                    continue
                frame = pd.read_csv(path)
                for row in frame.itertuples(index=False):
                    rows.append(
                        {
                            "side": side,
                            "pivot": pivot,
                            "distance_mm": distance,
                            "method": row.method,
                            "median_deg": row.median_deg,
                            "p90_deg": row.p90_deg,
                        }
                    )
    if not rows:
        raise FileNotFoundError(f"No {eval_dir} summaries found")
    table = pd.DataFrame(rows)
    suffix = "" if source == "analytic" else f"_{source}"
    table.to_csv(out_root / f"cross_scenario_summary{suffix}.csv", index=False)

    methods = sorted(table["method"].unique())
    pivots = list(PIVOTS)
    distances = list(CAMERA_DISTANCES_MM)
    fig, axes = plt.subplots(len(methods), 2, figsize=(9.5, 3.6 * len(methods)), dpi=150, squeeze=False)
    for r, method in enumerate(methods):
        for c, side in enumerate(("right", "left")):
            ax = axes[r][c]
            sub = table.loc[(table["method"] == method) & (table["side"] == side)]
            mag = np.full((len(pivots), len(distances)), np.nan)
            for i, pivot in enumerate(pivots):
                for j, distance in enumerate(distances):
                    hit = sub.loc[(sub["pivot"] == pivot) & (sub["distance_mm"] == distance)]
                    if len(hit):
                        mag[i, j] = float(hit.iloc[0]["median_deg"])
            mesh = ax.imshow(mag, origin="upper", aspect="auto", cmap=_turbo())
            ax.set_xticks(range(len(distances)))
            ax.set_xticklabels([f"{d:g}" for d in distances])
            ax.set_yticks(range(len(pivots)))
            ax.set_yticklabels(pivots)
            ax.set_xlabel("Camera distance (mm)")
            ax.set_title(f"{method} · {side} · {source} · median (deg)", fontsize=9)
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_root / f"cross_scenario_median_error{suffix}"
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="Evaluate a single scenario directory (else all analytic).",
    )
    p.add_argument(
        "--source",
        choices=("analytic", "rendered", "both"),
        default="analytic",
    )
    p.add_argument("--cross-figure", action="store_true", help="Rebuild cross-scenario figure.")
    args = p.parse_args(argv)

    results = []
    sources = ("analytic", "rendered") if args.source == "both" else (args.source,)

    if args.scenario is not None:
        for source in sources:
            print(f"eval {args.scenario} source={source}")
            results.append(evaluate_scenario_dir(args.scenario, source=source))
    else:
        for side in ("right", "left"):
            for pivot in PIVOTS:
                for distance in CAMERA_DISTANCES_MM:
                    scenario = OUT_ROOT / side / pivot / f"d{distance:g}mm"
                    if not (scenario / "ground_truth_analytic.csv").is_file():
                        print(f"skip missing {scenario}")
                        continue
                    for source in sources:
                        if source == "rendered" and not (scenario / "ground_truth.csv").is_file():
                            continue
                        print(f"eval {side} {pivot} d={distance:g} source={source}")
                        results.append(evaluate_scenario_dir(scenario, source=source))
        if args.source in ("analytic", "both") or args.cross_figure:
            fig = cross_scenario_figure()
            print(f"cross-scenario figure → {fig}")

    if args.cross_figure and args.scenario is not None:
        fig = cross_scenario_figure()
        print(f"cross-scenario figure → {fig}")

    print(f"evaluated {len(results)} scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
