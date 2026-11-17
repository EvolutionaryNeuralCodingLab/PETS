#!/usr/bin/env python3
"""Compare release-style vs corrected ellipse fits on a block.

Default comparison block::

    /Volumes/Data-1/Nimrod/experiments/PV_106/2025_09_04/block_015

Runs the current (corrected) ``LsqEllipse`` / ``eye_tracking_analysis`` on the
block DLC annotations, compares against existing ``le_df``/``re_df`` (release
fits), and writes a JSON report under ``analysis/ellipse_v2/``.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.ellipse_fit import (
    canonicalize_ellipse_axes_phi,
)
from eye_tracking_system_tools.preprocessing.ellipse_geometry_meta import (
    write_ellipse_geometry_meta,
)
from eye_tracking_system_tools.preprocessing.sync_free_eye_io import (
    align_ellipse_df_to_video_frames,
    build_data_for_eye_tracking,
    find_dlc_csv,
    video_frame_count,
)


DEFAULT_BLOCK = Path(
    "/Volumes/Data-1/Nimrod/experiments/PV_106/2025_09_04/block_015"
)


def _ang_diff_mod_pi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return np.abs(d)


def _compare_fit_tables(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    frame_col: str | None = None,
) -> dict[str, Any]:
    o = old.reset_index(drop=True).copy()
    n = new.reset_index(drop=True).copy()
    if frame_col and frame_col in o.columns and frame_col in n.columns:
        merged = o.merge(n, on=frame_col, suffixes=("_old", "_new"), how="inner")
    else:
        m = min(len(o), len(n))
        merged = pd.concat(
            [
                o.iloc[:m].add_suffix("_old").reset_index(drop=True),
                n.iloc[:m].add_suffix("_new").reset_index(drop=True),
            ],
            axis=1,
        )

    cx_o = pd.to_numeric(merged.get("center_x_old"), errors="coerce")
    cy_o = pd.to_numeric(merged.get("center_y_old"), errors="coerce")
    cx_n = pd.to_numeric(merged.get("center_x_new"), errors="coerce")
    cy_n = pd.to_numeric(merged.get("center_y_new"), errors="coerce")
    both = cx_o.notna() & cy_o.notna() & cx_n.notna() & cy_n.notna()
    dist = np.sqrt((cx_o - cx_n) ** 2 + (cy_o - cy_n) ** 2)

    w_o = pd.to_numeric(merged.get("width_old"), errors="coerce").to_numpy()
    h_o = pd.to_numeric(merged.get("height_old"), errors="coerce").to_numpy()
    p_o = pd.to_numeric(merged.get("phi_old"), errors="coerce").to_numpy()
    w_c, h_c, p_c = canonicalize_ellipse_axes_phi(w_o, h_o, p_o)

    w_n = pd.to_numeric(merged.get("width_new"), errors="coerce").to_numpy()
    h_n = pd.to_numeric(merged.get("height_new"), errors="coerce").to_numpy()
    p_n = pd.to_numeric(merged.get("phi_new"), errors="coerce").to_numpy()

    axis_ok = np.isfinite(w_c) & np.isfinite(w_n)
    phi_ok = np.isfinite(p_c) & np.isfinite(p_n) & axis_ok
    # Near-circle filter on new fits
    ratio = np.where(w_n > 0, h_n / w_n, np.nan)
    elongated = phi_ok & (ratio < 0.98)

    null_old = ~(
        pd.to_numeric(merged.get("center_x_old"), errors="coerce").notna()
        & pd.to_numeric(merged.get("center_y_old"), errors="coerce").notna()
    )
    null_new = ~(
        pd.to_numeric(merged.get("center_x_new"), errors="coerce").notna()
        & pd.to_numeric(merged.get("center_y_new"), errors="coerce").notna()
    )

    return {
        "n_rows_compared": int(len(merged)),
        "n_both_finite_centers": int(both.sum()),
        "median_center_delta_px": float(np.nanmedian(dist[both])) if both.any() else None,
        "max_center_delta_px": float(np.nanmax(dist[both])) if both.any() else None,
        "median_abs_major_delta_after_canonicalize": (
            float(np.nanmedian(np.abs(w_c[axis_ok] - w_n[axis_ok])))
            if axis_ok.any()
            else None
        ),
        "median_phi_abs_err_deg_elongated": (
            float(np.degrees(np.nanmedian(_ang_diff_mod_pi(p_c[elongated], p_n[elongated]))))
            if elongated.any()
            else None
        ),
        "n_elongated_comparable_phi": int(elongated.sum()),
        "null_mask_agreement": float((null_old == null_new).mean()) if len(merged) else None,
        "acceptance": {
            "centers_median_lt_0_5px": (
                bool(np.nanmedian(dist[both]) < 0.5) if both.any() else False
            ),
            "phi_median_err_lt_2deg": (
                bool(
                    np.degrees(np.nanmedian(_ang_diff_mod_pi(p_c[elongated], p_n[elongated])))
                    < 2.0
                )
                if elongated.any()
                else None
            ),
            "null_masks_identical": (
                bool((null_old == null_new).all()) if len(merged) else False
            ),
        },
    }


def _fit_eye(block_path: Path, eye: str, uncertainty_thr: float) -> pd.DataFrame:
    folder = block_path / "eye_videos" / ("LE" if eye == "left" else "RE")
    try:
        dlc_path = find_dlc_csv(folder)
    except FileNotFoundError:
        # Some blocks nest DLC under a trial subfolder.
        nested = sorted(folder.rglob("*DLC*.csv"))
        if not nested:
            raise
        dlc_path = nested[0]
    dlc = pd.read_csv(dlc_path, header=1)
    data = build_data_for_eye_tracking(dlc)
    ellipse = BlockSync.eye_tracking_analysis(dlc, uncertainty_thr)
    videos = sorted(p for p in folder.rglob("*.mp4") if "DLC" not in p.name)
    if not videos:
        raise FileNotFoundError(f"No eye video under {folder}")
    n_vid = video_frame_count(videos[0])
    return align_ellipse_df_to_video_frames(
        ellipse, n_video_frames=n_vid, n_data_rows=len(data)
    )


def _kerr_parity(block_path: Path) -> dict[str, Any]:
    """Orientation-only: existing Kerr columns should be unchanged by canonicalize."""
    analysis = block_path / "analysis"
    out: dict[str, Any] = {}
    for side in ("left", "right"):
        eye = analysis / f"{side}_eye_data.csv"
        kerr = analysis / f"{side}_kerr_angle_raw_verified.csv"
        if not eye.is_file() or not kerr.is_file():
            out[side] = {"status": "missing_inputs"}
            continue
        ed = pd.read_csv(eye)
        kd = pd.read_csv(kerr)
        # Canonicalize should not change major/minor used by Kerr if we take max/min
        w = pd.to_numeric(ed["width"], errors="coerce").to_numpy()
        h = pd.to_numeric(ed["height"], errors="coerce").to_numpy()
        stacked = np.column_stack([w, h])
        with np.errstate(all="ignore"):
            major = np.nanmax(stacked, axis=1)
            minor = np.nanmin(stacked, axis=1)
        phi = pd.to_numeric(ed["phi"], errors="coerce").to_numpy(dtype=float)
        w2, h2, _ = canonicalize_ellipse_axes_phi(w, h, phi)
        finite = np.isfinite(major) & np.isfinite(w2)
        out[side] = {
            "status": "ok",
            "max_abs_major_delta": (
                float(np.nanmax(np.abs(major[finite] - w2[finite])))
                if finite.any()
                else None
            ),
            "max_abs_minor_delta": (
                float(np.nanmax(np.abs(minor[finite] - h2[finite])))
                if finite.any()
                else None
            ),
            "kerr_rows": int(len(kd)),
            "kerr_columns": [
                c for c in kd.columns if c.startswith("k_") or c in ("phi", "theta")
            ],
            "note": (
                "major/minor from max/min match canonicalize; "
                "Kerr should be unchanged for orientation-only repair"
            ),
        }
    return out


def run_comparison(
    block_path: Path,
    *,
    uncertainty_thr: float = 0.95,
) -> dict[str, Any]:
    block_path = Path(block_path)
    analysis = block_path / "analysis"
    report: dict[str, Any] = {
        "block_path": str(block_path),
        "uncertainty_thr": uncertainty_thr,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "eyes": {},
    }

    for eye, legacy_name, frame_col in (
        ("left", "le_df.csv", "L_eye_frame"),
        ("right", "re_df.csv", "R_eye_frame"),
    ):
        legacy_path = analysis / legacy_name
        eye_report: dict[str, Any] = {"legacy_table": str(legacy_path)}
        if not legacy_path.is_file():
            eye_report["status"] = "missing_legacy"
            report["eyes"][eye] = eye_report
            continue
        legacy = pd.read_csv(legacy_path)
        new = _fit_eye(block_path, eye, uncertainty_thr)
        # Align legacy synced table on eye frame when present
        if frame_col in legacy.columns:
            legacy_f = legacy.rename(columns={frame_col: "eye_frame"})
            cmp_ = _compare_fit_tables(legacy_f, new, frame_col="eye_frame")
        else:
            cmp_ = _compare_fit_tables(legacy, new)
        eye_report["status"] = "compared"
        eye_report["comparison"] = cmp_
        eye_report["n_new_fitted"] = int(new["center_x"].notna().sum())
        report["eyes"][eye] = eye_report

    report["kerr_parity"] = _kerr_parity(block_path)

    # Sync-free frame map check on new fits
    report["sync_free_frame_map"] = {
        "expected": "ellipse.iloc[k] -> eye_frame k; last two frames NaN",
        "note": "validated by unit test + new align implementation used above",
    }

    # Historical le_df/re_df on this block were fit with the old x-flip convention
    # (frame_width - x). When centers are mirrors and phi_new ≈ (π - phi_old),
    # orientation parity after reflection is the fair geometric check.
    for eye, eye_report in report["eyes"].items():
        cmp_ = eye_report.get("comparison") or {}
        eye_report["notes"] = (
            "If median center delta ≈ |W - 2x| and phi differs by π reflection, "
            "legacy tables likely used the historical DLC x-flip convention; "
            "axis lengths after canonicalize should still match (~1e-7). "
            "Canonical geometry/unit tests cover eigh orientation independently."
        )
        report["eyes"][eye] = eye_report

    out_dir = analysis / "ellipse_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_ellipse_geometry_meta(
        out_dir,
        extra={"source": "compare_ellipse_pipelines", "comparison_report": str(out_path)},
    )
    report["report_path"] = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--block", type=Path, default=DEFAULT_BLOCK)
    p.add_argument("--uncertainty-thr", type=float, default=0.95)
    args = p.parse_args(argv)
    report = run_comparison(args.block, uncertainty_thr=args.uncertainty_thr)
    print(json.dumps({k: report[k] for k in ("block_path", "eyes", "kerr_parity", "report_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
