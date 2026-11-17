"""Compare conicoid (Swirski) gaze with Kerr gaze as a 3D angle, per frame.

Sibling of ``calculate_kerr_angles.py`` / ``calculate_conicoid_angles.py``.
Kerr stays the sanity check; this module only reads both outputs.

Both methods are turned into toward-camera unit normals before scoring:

- conicoid: the stored ``c_nx`` / ``c_ny`` / ``c_nz`` (the Kerr-chart pair
  ``c_phi`` / ``c_theta`` is inverted only when the normal is absent);
- Kerr: ``phi`` / ``theta`` inverted with
  :func:`conicoid.angles.kerr_angles_to_normals`.

The disagreement is :func:`conicoid.angles.angle_between_normals_deg`, the
same score used by ``examples/compare_kerr_conicoid_ground_truth.py``. Pair
subtraction is never used.

Outputs under ``analysis/`` for ``name_tag``:

- ``conicoid_vs_kerr_{tag}.csv`` -- one row per eye frame
- ``conicoid_vs_kerr_summary_{tag}.csv`` -- one row per eye
- ``conicoid_vs_kerr_{tag}.png`` -- angle histogram and yaw/pitch map per eye
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eye_tracking_system_tools.preprocessing.calculate_conicoid_angles import (
    read_conicoid_fit_meta,
)
from eye_tracking_system_tools.preprocessing.conicoid.angles import (
    angle_between_normals_deg,
    chart_yaw_pitch,
    kerr_angles_to_normals,
)

SIDES = ("left", "right")
_JOIN_KEYS = ("eye_frame", "OE_timestamp")


@dataclass(frozen=True)
class ComparisonPaths:
    frames: Path
    summary: Path
    plot: Path


def comparison_paths(analysis_path, name_tag: str) -> ComparisonPaths:
    ap = Path(analysis_path)
    return ComparisonPaths(
        frames=ap / f"conicoid_vs_kerr_{name_tag}.csv",
        summary=ap / f"conicoid_vs_kerr_summary_{name_tag}.csv",
        plot=ap / f"conicoid_vs_kerr_{name_tag}.png",
    )


def kerr_angle_csv(analysis_path, side: str, name_tag: str) -> Path:
    return Path(analysis_path) / f"{side}_kerr_angle_{name_tag}.csv"


def conicoid_angle_csv(analysis_path, side: str, name_tag: str) -> Path:
    return Path(analysis_path) / f"{side}_conicoid_angle_{name_tag}.csv"


def _analysis_path(block_or_path) -> Path:
    return Path(getattr(block_or_path, "analysis_path", block_or_path))


def kerr_outputs_available(block_or_path, name_tag: str) -> bool:
    ap = _analysis_path(block_or_path)
    return all(kerr_angle_csv(ap, s, name_tag).is_file() for s in SIDES)


def conicoid_outputs_available(block_or_path, name_tag: str) -> bool:
    ap = _analysis_path(block_or_path)
    return all(conicoid_angle_csv(ap, s, name_tag).is_file() for s in SIDES)


def conicoid_normals(df: pd.DataFrame) -> np.ndarray:
    """Stored conicoid normal, or the inverse of ``c_phi`` / ``c_theta``."""
    if {"c_nx", "c_ny", "c_nz"}.issubset(df.columns):
        return df[["c_nx", "c_ny", "c_nz"]].to_numpy(dtype=float)
    if {"c_phi", "c_theta"}.issubset(df.columns):
        return kerr_angles_to_normals(df["c_phi"], df["c_theta"])
    raise ValueError("conicoid table has neither c_nx/c_ny/c_nz nor c_phi/c_theta")


def kerr_normals(df: pd.DataFrame) -> np.ndarray:
    """Normal implied by Kerr's nested pair (``phi``/``theta`` or ``k_phi``/``k_theta``)."""
    for phi_col, theta_col in (("phi", "theta"), ("k_phi", "k_theta")):
        if {phi_col, theta_col}.issubset(df.columns):
            return kerr_angles_to_normals(df[phi_col], df[theta_col])
    raise ValueError("Kerr table has neither phi/theta nor k_phi/k_theta")


def _join_key(conicoid_df: pd.DataFrame, kerr_df: pd.DataFrame) -> str:
    for key in _JOIN_KEYS:
        if key in conicoid_df.columns and key in kerr_df.columns:
            if conicoid_df[key].notna().all() and conicoid_df[key].is_unique:
                return key
    raise ValueError(
        f"cannot align conicoid and Kerr rows: need a unique shared column from {_JOIN_KEYS}"
    )


def compare_eye(conicoid_df: pd.DataFrame, kerr_df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Per-frame comparison for one eye (outer join on ``eye_frame``)."""
    key = _join_key(conicoid_df, kerr_df)
    c = pd.DataFrame({key: conicoid_df[key].to_numpy()})
    c[["c_nx", "c_ny", "c_nz"]] = conicoid_normals(conicoid_df)
    k = pd.DataFrame({key: kerr_df[key].to_numpy()})
    k[["k_nx", "k_ny", "k_nz"]] = kerr_normals(kerr_df)
    extra = [col for col in _JOIN_KEYS if col != key and col in conicoid_df.columns]
    for col in extra:
        c[col] = conicoid_df[col].to_numpy()
    merged = pd.merge(c, k, on=key, how="outer", sort=True)

    cn = merged[["c_nx", "c_ny", "c_nz"]].to_numpy(dtype=float)
    kn = merged[["k_nx", "k_ny", "k_nz"]].to_numpy(dtype=float)
    c_yaw, c_pitch = chart_yaw_pitch(cn)
    k_yaw, k_pitch = chart_yaw_pitch(kn)
    out = pd.DataFrame(
        {
            "eye": side,
            key: merged[key].to_numpy(),
            **{col: merged[col].to_numpy() for col in extra},
            "conicoid_finite": np.isfinite(cn).all(axis=1),
            "kerr_finite": np.isfinite(kn).all(axis=1),
            "angle_deg": angle_between_normals_deg(cn, kn),
            "c_yaw_deg": c_yaw,
            "c_pitch_deg": c_pitch,
            "k_yaw_deg": k_yaw,
            "k_pitch_deg": k_pitch,
        }
    )
    return out


def summarize_eye(frames: pd.DataFrame, side: str, fit_meta: dict | None = None) -> dict:
    """One summary row: 3D-angle stats, finite fractions, and fit inputs."""
    n = int(len(frames))
    angle = frames["angle_deg"].to_numpy(dtype=float)
    finite = angle[np.isfinite(angle)]

    def frac(col: str) -> float:
        return float(frames[col].mean()) if n else float("nan")

    row = {
        "eye": side,
        "n_frames": n,
        "frac_conicoid_finite": frac("conicoid_finite"),
        "frac_kerr_finite": frac("kerr_finite"),
        "frac_both_finite": float(finite.size / n) if n else float("nan"),
        "median_angle_deg": float(np.median(finite)) if finite.size else float("nan"),
        "p90_angle_deg": float(np.percentile(finite, 90)) if finite.size else float("nan"),
        "mean_angle_deg": float(np.mean(finite)) if finite.size else float("nan"),
    }
    eye_meta = ((fit_meta or {}).get("eyes") or {}).get(side) or {}
    pp = eye_meta.get("principal_point") or [np.nan, np.nan]
    row.update(
        {
            "focal_px": eye_meta.get("focal_px", np.nan),
            "focal_source": eye_meta.get("focal_source", ""),
            "principal_point_x": pp[0],
            "principal_point_y": pp[1],
            "principal_point_source": eye_meta.get("principal_point_source", ""),
            "ellipse_table": eye_meta.get("ellipse_table", ""),
            "sphere_method": (fit_meta or {}).get("sphere_method", ""),
            "use_ransac": (fit_meta or {}).get("use_ransac", ""),
        }
    )
    return row


def plot_comparison(frames: pd.DataFrame, summary: pd.DataFrame, path: Path) -> Path:
    """Histogram of the 3D angle and conicoid yaw/pitch map coloured by it, per eye."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sides = [s for s in SIDES if (frames["eye"] == s).any()]
    fig = Figure(figsize=(10, 4.2 * max(1, len(sides))), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(max(1, len(sides)), 2, squeeze=False)
    finite_all = frames["angle_deg"].to_numpy(dtype=float)
    finite_all = finite_all[np.isfinite(finite_all)]
    vmax = float(np.percentile(finite_all, 99)) if finite_all.size else 1.0
    vmax = max(vmax, 1e-3)
    for i, side in enumerate(sides):
        sub = frames.loc[frames["eye"] == side]
        srow = summary.loc[summary["eye"] == side].iloc[0]
        angle = sub["angle_deg"].to_numpy(dtype=float)
        ok = np.isfinite(angle)
        ax_h, ax_s = axes[i]
        ax_h.hist(angle[ok], bins=60, range=(0.0, vmax), color="0.35")
        ax_h.axvline(srow["median_angle_deg"], color="tab:red", lw=1.2)
        ax_h.set_xlabel("Angle between conicoid and Kerr gaze (deg)")
        ax_h.set_ylabel("Frames")
        ax_h.set_title(
            f"{side} eye: median {srow['median_angle_deg']:.2f} deg, "
            f"p90 {srow['p90_angle_deg']:.2f} deg, "
            f"both finite {100 * srow['frac_both_finite']:.1f}%",
            fontsize=9,
        )
        sc = ax_s.scatter(
            sub["c_yaw_deg"].to_numpy(dtype=float)[ok],
            sub["c_pitch_deg"].to_numpy(dtype=float)[ok],
            c=angle[ok],
            s=2,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            rasterized=True,
        )
        ax_s.set_xlabel("Conicoid yaw (deg)")
        ax_s.set_ylabel("Conicoid pitch (deg)")
        ax_s.set_aspect("equal", adjustable="datalim")
        ax_s.set_title(f"{side} eye: colour = 3D angle to Kerr", fontsize=9)
        fig.colorbar(sc, ax=ax_s, label="deg")
    path = Path(path)
    fig.savefig(path, dpi=130)
    return path


def compare_conicoid_kerr_for_block(
    block,
    name_tag: str = "default",
    *,
    make_plot: bool = True,
) -> pd.DataFrame:
    """Write the per-frame CSV, per-eye summary CSV, and plot; return the summary.

    Needs ``{side}_conicoid_angle_{tag}.csv`` and ``{side}_kerr_angle_{tag}.csv``
    for both eyes under ``block.analysis_path``.
    """
    ap = _analysis_path(block)
    missing = [
        p.name
        for s in SIDES
        for p in (conicoid_angle_csv(ap, s, name_tag), kerr_angle_csv(ap, s, name_tag))
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing outputs for tag {name_tag!r} in {ap}: {', '.join(missing)}. "
            "Run Conicoid and Kerr with the same name_tag first."
        )
    fit_meta = read_conicoid_fit_meta(ap, name_tag)
    frames_list = []
    rows = []
    for side in SIDES:
        c_df = pd.read_csv(conicoid_angle_csv(ap, side, name_tag), index_col=0)
        k_df = pd.read_csv(kerr_angle_csv(ap, side, name_tag), index_col=0)
        frames = compare_eye(c_df, k_df, side)
        frames_list.append(frames)
        rows.append(summarize_eye(frames, side, fit_meta))
    frames_all = pd.concat(frames_list, ignore_index=True)
    summary = pd.DataFrame(rows)

    paths = comparison_paths(ap, name_tag)
    frames_all.to_csv(paths.frames, index=False)
    summary.to_csv(paths.summary, index=False)
    if make_plot:
        plot_comparison(frames_all, summary, paths.plot)
    for row in rows:
        print(
            f"  {row['eye']}: median 3D angle {row['median_angle_deg']:.2f} deg, "
            f"both finite {100 * row['frac_both_finite']:.1f}% of {row['n_frames']} frames"
        )
    return summary


def format_summary(summary: pd.DataFrame) -> str:
    """Short text for the GUI: per-eye median / p90 angle and finite fractions."""
    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"{row['eye']}: median {row['median_angle_deg']:.2f} deg, "
            f"p90 {row['p90_angle_deg']:.2f} deg | finite: conicoid "
            f"{100 * row['frac_conicoid_finite']:.1f}%, Kerr {100 * row['frac_kerr_finite']:.1f}%, "
            f"both {100 * row['frac_both_finite']:.1f}% of {int(row['n_frames'])}"
        )
    return "\n".join(lines)
