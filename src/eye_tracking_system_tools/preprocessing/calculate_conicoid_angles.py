"""Calculate conicoid (Safaee-Rad / Swirski) gaze angles from ellipse tables.

Parallel to ``calculate_kerr_angles.py``: consumes ``left_eye_data.csv`` /
``right_eye_data.csv`` produced after ``read_dlc_data`` + ``create_eye_data``,
fits a 3D pupil-on-sphere model, and writes the gaze normal ``c_nx`` /
``c_ny`` / ``c_nz`` (plus the Kerr-chart pair ``c_phi`` / ``c_theta`` for old
consumers) without replacing Kerr's ``k_phi`` / ``k_theta``.

Inputs per eye:

- Ellipses must follow geometry version 2 (see ``ellipse_fit``). The main
  ``left/right_eye_data.csv`` are used when ``analysis/ellipse_geometry_meta.json``
  says v2; otherwise the repaired ``analysis/ellipse_v2/`` tables written by
  ``ellipse_recovery``. Blocks with neither raise :class:`EllipseGeometryError`.
- Principal point: the Kerr reference from ``analysis/self_kerr_refs.csv``
  (chosen in Verify), else the image centre with a warning.
- Focal length: ``LR_focal_length.csv``, else the documented 652.6 px
  fallback (``conicoid.camera``).

Each run also writes ``analysis/conicoid_fit_meta_{name_tag}.json`` with the
camera, sources, and sphere fit per eye.

Usage
-----
    from eye_tracking_system_tools.preprocessing.calculate_conicoid_angles import (
        calculate_conicoid_angles_for_block,
    )
    calculate_conicoid_angles_for_block(block, name_tag="raw_verified")
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import load_eye_data
from eye_tracking_system_tools.preprocessing.conicoid import fit_eye_and_gaze
from eye_tracking_system_tools.preprocessing.conicoid.camera import (
    kerr_ref_principal_point,
    resolve_intrinsics,
)
from eye_tracking_system_tools.preprocessing.conicoid.refined_io import (
    latest_refined_eye_csv,
    overlay_refined_geometry,
    read_refined_eye_table,
    refined_eye_csv_path,
    refined_tag_from_name,
)

from eye_tracking_system_tools.preprocessing.ellipse_fit import ELLIPSE_GEOMETRY_VERSION
from eye_tracking_system_tools.preprocessing.ellipse_geometry_meta import (
    read_ellipse_geometry_version,
)

CONICOID_ANGLE_COLS = ("c_phi", "c_theta")
DEFAULT_SPHERE_METHOD = "swirski_2d"
VALIDATED_SPHERE_METHODS = ("swirski_2d",)
REPAIRED_GEOMETRY_DIR = "ellipse_v2"


class EllipseGeometryError(RuntimeError):
    """The block's ellipse tables are not geometry version 2."""


def conicoid_fit_meta_path(analysis_path, name_tag: str) -> Path:
    return Path(analysis_path) / f"conicoid_fit_meta_{name_tag}.json"


def read_conicoid_fit_meta(analysis_path, name_tag: str) -> dict | None:
    path = conicoid_fit_meta_path(analysis_path, name_tag)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def v2_eye_table_path(analysis_path, side: str, *, block_label: str = "?") -> tuple[Path | None, str]:
    """Locate the geometry-v2 eye table for ``side`` without loading it.

    Returns ``(None, label)`` when the main ``{side}_eye_data.csv`` is v2
    (``analysis/ellipse_geometry_meta.json``), or ``(path, label)`` for the
    repaired ``analysis/ellipse_v2/{side}_eye_data.csv``. Raises
    :class:`EllipseGeometryError` otherwise.
    """
    analysis = Path(analysis_path)
    if read_ellipse_geometry_version(analysis, default=1) >= ELLIPSE_GEOMETRY_VERSION:
        return None, f"{side}_eye_data.csv (v2 meta)"
    repaired_dir = analysis / REPAIRED_GEOMETRY_DIR
    repaired = repaired_dir / f"{side}_eye_data.csv"
    if (
        repaired.is_file()
        and read_ellipse_geometry_version(repaired_dir, default=1) >= ELLIPSE_GEOMETRY_VERSION
    ):
        return repaired, f"{REPAIRED_GEOMETRY_DIR}/{repaired.name}"
    if repaired_dir.is_dir():
        hint = (
            f"{REPAIRED_GEOMETRY_DIR}/ exists but has no v2 {side}_eye_data.csv "
            "(verified tables flagged for manual review or refit)."
        )
    else:
        hint = (
            "Run `python -m eye_tracking_system_tools.preprocessing.ellipse_recovery` "
            "or re-run Sync/Verify with the current pipeline."
        )
    raise EllipseGeometryError(
        f"Block {block_label} {side} eye: ellipse tables are not geometry version "
        f"{ELLIPSE_GEOMETRY_VERSION}; conicoid will not run on them. {hint}"
    )


def resolve_v2_eye_table(block, side: str) -> tuple[pd.DataFrame, str]:
    """Return ``(eye_table, source_label)`` in ellipse geometry version 2.

    The in-memory main table when ``analysis/`` carries v2 metadata, else the
    repaired ``analysis/ellipse_v2/{side}_eye_data.csv`` from ``ellipse_recovery``.
    """
    path, label = v2_eye_table_path(
        block.analysis_path, side, block_label=str(getattr(block, "block_num", "?"))
    )
    if path is None:
        return (block.left_eye_data if side == "left" else block.right_eye_data), label
    return pd.read_csv(path, index_col=0, engine="python"), label


def check_block_geometry(analysis_path, *, block_label: str = "?") -> dict[str, str]:
    """Raise :class:`EllipseGeometryError` unless both eyes have v2 tables.

    Returns ``{side: source_label}``.
    """
    return {
        side: v2_eye_table_path(analysis_path, side, block_label=block_label)[1]
        for side in ("left", "right")
    }
_CONICOID_COL_RE = re.compile(
    r"^c_(?:phi|theta|nx|ny|nz|ex|ey|ez|eradius)(?:_[xy])?$"
)


def _conicoid_columns(columns) -> list[str]:
    return [c for c in columns if _CONICOID_COL_RE.match(str(c))]


def append_conicoid_angle_data(eye_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge conicoid angle / vector columns onto ``eye_df`` by ``OE_timestamp``."""
    if eye_df is None:
        raise ValueError("eye_df is None")
    if new_df is None or new_df.empty:
        raise ValueError("Conicoid angle dataframe is empty")
    if "OE_timestamp" not in new_df.columns:
        raise ValueError("Angle dataframe missing OE_timestamp")
    if "OE_timestamp" not in eye_df.columns:
        raise ValueError("eye_df has no OE_timestamp")

    base = eye_df.copy()
    drop = _conicoid_columns(base.columns)
    if drop:
        base = base.drop(columns=drop)

    extra = [
        c
        for c in (
            "c_phi",
            "c_theta",
            "c_nx",
            "c_ny",
            "c_nz",
            "c_ex",
            "c_ey",
            "c_ez",
            "c_eradius",
        )
        if c in new_df.columns
    ]
    return pd.merge(
        base,
        new_df[["OE_timestamp", *extra]],
        on="OE_timestamp",
        how="left",
    )


def export_eye_data_w_conicoid(block, name_tag: str = "default") -> None:
    """Write ``left/right_eye_data_degrees_{tag}_conicoid.csv``."""
    left_path = block.analysis_path / f"left_eye_data_degrees_{name_tag}_conicoid.csv"
    right_path = block.analysis_path / f"right_eye_data_degrees_{name_tag}_conicoid.csv"
    block.left_eye_data.to_csv(left_path)
    block.right_eye_data.to_csv(right_path)
    print(f"Exported conicoid eye data (tag: {name_tag}) for block {block.block_num}")


def _fit_one_eye(eye_df: pd.DataFrame, camera, **kwargs):
    angles, fit = fit_eye_and_gaze(eye_df, camera, **kwargs)
    print(
        f"  sphere centre={fit.sphere.centre}, radius={fit.sphere.radius:.4f}, "
        f"inliers={fit.n_inliers}/{fit.n_observations}, camera={camera.source}, "
        f"principal point=({camera.cx:.1f}, {camera.cy:.1f}) [{camera.principal_point_source}]"
    )
    return angles, fit


def _camera_for_eye(block, side: str):
    block_path = getattr(block, "block_path", None) or getattr(block, "analysis_path", None)
    ref = kerr_ref_principal_point(block.analysis_path, side)
    if ref is None:
        warnings.warn(
            f"Block {getattr(block, 'block_num', '?')} {side} eye: no Kerr reference in "
            "self_kerr_refs.csv; conicoid principal point falls back to the image centre.",
            stacklevel=3,
        )
        return resolve_intrinsics(block_path, side)
    return resolve_intrinsics(
        block_path,
        side,
        principal_point=ref,
        principal_point_source="self_kerr_refs.csv",
    )


def _eye_fit_meta(camera, fit, angles: pd.DataFrame, table_source: str) -> dict:
    normals = angles[["c_nx", "c_ny", "c_nz"]].to_numpy(dtype=float)
    return {
        "ellipse_table": table_source,
        "focal_px": float(camera.focal_px),
        "focal_source": str(camera.source),
        "principal_point": [float(camera.cx), float(camera.cy)],
        "principal_point_source": str(camera.principal_point_source),
        "n_rows": int(len(angles)),
        "n_finite_normal": int(np.isfinite(normals).all(axis=1).sum()),
        "n_inliers": int(fit.n_inliers),
        "n_observations": int(fit.n_observations),
        "sphere_centre": [float(v) for v in np.asarray(fit.sphere.centre).reshape(3)],
        "sphere_radius": float(fit.sphere.radius),
    }


def _ellipse_table_for_source(
    block, side: str, ellipse_source: str, *, base: pd.DataFrame | None = None
) -> pd.DataFrame:
    """``base`` (default: the in-memory eye table) or a refined sidecar overlaid on it."""
    df = base if base is not None else (
        block.left_eye_data if side == "left" else block.right_eye_data
    )
    src = (ellipse_source or "original").strip()
    if src in ("original", "eye_data", ""):
        return df
    if src in ("refined", "latest_refined"):
        path = latest_refined_eye_csv(block.analysis_path, side)
        if path is None:
            raise FileNotFoundError(f"No refined ellipse CSV for {side} eye")
    elif src.startswith("refined:"):
        path = refined_eye_csv_path(block.analysis_path, side, src.split(":", 1)[1])
        if not path.is_file():
            raise FileNotFoundError(f"No refined ellipse CSV for {side} eye tag {src}")
    else:
        path = Path(src)
        if not path.is_file():
            path = Path(block.analysis_path) / src
        tag = refined_tag_from_name(path.name)
        if tag:
            path = refined_eye_csv_path(block.analysis_path, side, tag)
        if not path.is_file():
            raise FileNotFoundError(f"Ellipse source not found: {ellipse_source}")
    refined = read_refined_eye_table(path)
    print(f"  ellipse source ({side}): {path.name}")
    return overlay_refined_geometry(df, refined)


def calculate_conicoid_angles_for_block(
    block,
    name_tag: str = "default",
    *,
    load_eye_data_flag: bool = True,
    export_flag: bool = True,
    eye_z: float = 13.0,
    pupil_radius: float = 1.0,
    use_ransac: bool = True,
    sphere_method: str = DEFAULT_SPHERE_METHOD,
    ellipse_source: str = "original",
    refraction_maps=None,
) -> dict:
    """Fit the conicoid model on both eyes of a ``BlockSync`` instance.

    ``ellipse_source`` is ``original`` (the v2 eye table, see
    :func:`resolve_v2_eye_table`), ``refined`` (latest
    ``*_eye_data_refined_*.csv`` overlaid on it), or a CSV path. Returns the
    fit metadata also written to ``conicoid_fit_meta_{name_tag}.json``.
    Raises :class:`EllipseGeometryError` on non-v2 geometry.
    """
    print(f"\n{'=' * 60}")
    print(f"Conicoid angles for block {block.block_num}")
    print(f"{'=' * 60}")

    if load_eye_data_flag:
        try:
            load_eye_data(block)
        except FileNotFoundError:
            print(f"Skipping block {block.block_num} - eye data not found")
            return {}

    check_block_geometry(block.analysis_path, block_label=str(block.block_num))
    if sphere_method not in VALIDATED_SPHERE_METHODS:
        warnings.warn(
            f"sphere_method={sphere_method!r} is provisional and not validated "
            f"against ground truth; {DEFAULT_SPHERE_METHOD!r} is the tested route.",
            stacklevel=2,
        )
    kwargs = dict(
        pupil_radius=pupil_radius,
        eye_z=eye_z,
        use_ransac=use_ransac,
        sphere_method=sphere_method,
        refraction_maps=refraction_maps,
    )

    meta: dict = {
        "name_tag": name_tag,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "sphere_method": sphere_method,
        "use_ransac": bool(use_ransac),
        "eye_z": float(eye_z),
        "pupil_radius": float(pupil_radius),
        "ellipse_source": ellipse_source,
        "refraction": refraction_maps is not None,
        "eyes": {},
    }
    for side in ("left", "right"):
        print(f"{side.capitalize()} eye")
        camera = _camera_for_eye(block, side)
        base, table_source = resolve_v2_eye_table(block, side)
        table = _ellipse_table_for_source(block, side, ellipse_source, base=base)
        angles, fit = _fit_one_eye(table, camera, **kwargs)
        angles.to_csv(block.analysis_path / f"{side}_conicoid_angle_{name_tag}.csv")
        attr = f"{side}_eye_data"
        setattr(block, attr, append_conicoid_angle_data(getattr(block, attr), angles))
        meta["eyes"][side] = _eye_fit_meta(camera, fit, angles, table_source)

    with open(conicoid_fit_meta_path(block.analysis_path, name_tag), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if export_flag:
        export_eye_data_w_conicoid(block, name_tag=name_tag)

    print(f"Completed conicoid processing for block {block.block_num}\n")
    return meta


def calculate_conicoid_angles_for_collection(
    block_collection: List,
    name_tag: str = "default",
    *,
    load_eye_data_flag: bool = True,
    export_flag: bool = True,
    continue_on_error: bool = True,
    eye_z: float = 13.0,
    pupil_radius: float = 1.0,
    use_ransac: bool = True,
    sphere_method: str = DEFAULT_SPHERE_METHOD,
    ellipse_source: str = "original",
    refraction_maps=None,
    compare_with_kerr: bool = False,
) -> None:
    """Run :func:`calculate_conicoid_angles_for_block` on many blocks.

    With ``compare_with_kerr`` each block that already has Kerr outputs for
    ``name_tag`` also gets the Swirski-vs-Kerr comparison
    (:func:`compare_conicoid_kerr.compare_conicoid_kerr_for_block`).
    """
    print(f"\n{'=' * 60}")
    print(f"Conicoid processing {len(block_collection)} blocks")
    print(f"{'=' * 60}\n")

    successful = 0
    failed = 0
    for block in block_collection:
        try:
            calculate_conicoid_angles_for_block(
                block,
                name_tag=name_tag,
                load_eye_data_flag=load_eye_data_flag,
                export_flag=export_flag,
                eye_z=eye_z,
                pupil_radius=pupil_radius,
                use_ransac=use_ransac,
                sphere_method=sphere_method,
                ellipse_source=ellipse_source,
                refraction_maps=refraction_maps,
            )
            if compare_with_kerr:
                from eye_tracking_system_tools.preprocessing.compare_conicoid_kerr import (
                    compare_conicoid_kerr_for_block,
                    kerr_outputs_available,
                )

                if kerr_outputs_available(block, name_tag):
                    compare_conicoid_kerr_for_block(block, name_tag=name_tag)
                else:
                    print(f"  block {block.block_num}: no Kerr outputs for {name_tag}; comparison skipped")
            successful += 1
        except Exception as exc:
            failed += 1
            if continue_on_error:
                print(f"Error processing block {block.block_num}: {exc}")
                print("Continuing with next block...\n")
            else:
                raise
    print(f"Conicoid complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    print("This script is designed to be imported from notebooks or other scripts.")
    print("See the module docstring for usage examples.")
