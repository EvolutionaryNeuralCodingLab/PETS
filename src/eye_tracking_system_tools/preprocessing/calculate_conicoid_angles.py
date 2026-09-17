"""Calculate conicoid (Safaee-Rad / Swirski) gaze angles from ellipse tables.

Parallel to ``calculate_kerr_angles.py``: consumes ``left_eye_data.csv`` /
``right_eye_data.csv`` produced after ``read_dlc_data`` + ``create_eye_data``,
fits a 3D pupil-on-sphere model, and writes ``c_phi`` / ``c_theta`` without
replacing Kerr's ``k_phi`` / ``k_theta``.

Usage
-----
    from eye_tracking_system_tools.preprocessing.calculate_conicoid_angles import (
        calculate_conicoid_angles_for_block,
    )
    calculate_conicoid_angles_for_block(block, name_tag="raw_verified")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd

from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import load_eye_data
from eye_tracking_system_tools.preprocessing.conicoid import fit_eye_and_gaze
from eye_tracking_system_tools.preprocessing.conicoid.camera import resolve_intrinsics
from eye_tracking_system_tools.preprocessing.conicoid.refined_io import (
    latest_refined_eye_csv,
    overlay_refined_geometry,
    read_refined_eye_table,
    refined_eye_csv_path,
    refined_tag_from_name,
)

CONICOID_ANGLE_COLS = ("c_phi", "c_theta")
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


def _fit_one_eye(eye_df: pd.DataFrame, camera, **kwargs) -> pd.DataFrame:
    angles, fit = fit_eye_and_gaze(eye_df, camera, **kwargs)
    print(
        f"  sphere centre={fit.sphere.centre}, radius={fit.sphere.radius:.4f}, "
        f"inliers={fit.n_inliers}/{fit.n_observations}, camera={camera.source}"
    )
    return angles


def _ellipse_table_for_source(block, side: str, ellipse_source: str) -> pd.DataFrame:
    """Original ``*_eye_data.csv`` or a refined sidecar, overlaid onto that table."""
    df = block.left_eye_data if side == "left" else block.right_eye_data
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
    sphere_method: str = "dierkes_3d",
    ellipse_source: str = "original",
    refraction_maps=None,
) -> None:
    """Fit the conicoid model on both eyes of a ``BlockSync`` instance.

    ``ellipse_source`` is ``original`` (``left/right_eye_data.csv``),
    ``refined`` (latest ``*_eye_data_refined_*.csv``), or a CSV path.
    """
    print(f"\n{'=' * 60}")
    print(f"Conicoid angles for block {block.block_num}")
    print(f"{'=' * 60}")

    if load_eye_data_flag:
        try:
            load_eye_data(block)
        except FileNotFoundError:
            print(f"Skipping block {block.block_num} - eye data not found")
            return

    block_path = getattr(block, "block_path", None) or getattr(block, "analysis_path", None)
    kwargs = dict(
        pupil_radius=pupil_radius,
        eye_z=eye_z,
        use_ransac=use_ransac,
        sphere_method=sphere_method,
        refraction_maps=refraction_maps,
    )

    print("Left eye")
    cam_l = resolve_intrinsics(block_path, "left")
    left_table = _ellipse_table_for_source(block, "left", ellipse_source)
    left_angles = _fit_one_eye(left_table, cam_l, **kwargs)
    left_out = block.analysis_path / f"left_conicoid_angle_{name_tag}.csv"
    left_angles.to_csv(left_out)
    block.left_eye_data = append_conicoid_angle_data(block.left_eye_data, left_angles)

    print("Right eye")
    cam_r = resolve_intrinsics(block_path, "right")
    right_table = _ellipse_table_for_source(block, "right", ellipse_source)
    right_angles = _fit_one_eye(right_table, cam_r, **kwargs)
    right_out = block.analysis_path / f"right_conicoid_angle_{name_tag}.csv"
    right_angles.to_csv(right_out)
    block.right_eye_data = append_conicoid_angle_data(block.right_eye_data, right_angles)

    if export_flag:
        export_eye_data_w_conicoid(block, name_tag=name_tag)

    print(f"Completed conicoid processing for block {block.block_num}\n")


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
    sphere_method: str = "dierkes_3d",
    ellipse_source: str = "original",
    refraction_maps=None,
) -> None:
    """Run :func:`calculate_conicoid_angles_for_block` on many blocks."""
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
