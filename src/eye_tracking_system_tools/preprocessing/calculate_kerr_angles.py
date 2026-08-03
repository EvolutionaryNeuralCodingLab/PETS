"""
Calculate Kerr angles for eye-tracking data.

This script performs the Kerr angle calculation pipeline:
1. Loads eye data from CSV files (created by synchronization pipeline)
2. Loads Kerr reference coordinates from self_kerr_refs.csv
3. Calculates Kerr angles (phi and theta) using the BlockSync method
4. Appends angle data (k_phi, k_theta) to eye dataframes
5. Exports the updated dataframes with a specified tag

Expected workflow:
    After running data_verification.ipynb (where Kerr references are chosen and saved),
    call this script to calculate and append Kerr angles.

Usage:
    # For a single block:
    from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import calculate_kerr_angles_for_block
    
    calculate_kerr_angles_for_block(block, name_tag='raw_verified')
    
    # For a collection of blocks:
    from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import calculate_kerr_angles_for_collection
    
    calculate_kerr_angles_for_collection(block_collection, name_tag='raw_verified')
    
    # If eye data is already loaded (e.g., from data_verification.ipynb):
    calculate_kerr_angles_for_block(block, name_tag='raw_verified', load_eye_data_flag=False)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Optional
import re
import pandas as pd
import numpy as np

# Canonical Kerr angle columns written by append_angle_data.
KERR_ANGLE_COLS = ("k_phi", "k_theta")

# Pandas merge suffixes / leftover names that must not survive a re-append.
_KERR_ANGLE_COL_RE = re.compile(r"^k_(?:phi|theta)(?:_[xy])?$")


def load_eye_data(block) -> None:
    """
    Load the eye dataframes from CSV files created by the synchronization pipeline.
    
    Parameters
    ----------
    block : BlockSync
        The BlockSync instance to load data for.
        
    Raises
    ------
    FileNotFoundError
        If the eye data CSV files are not found.
    """
    try:
        block.left_eye_data = pd.read_csv(
            block.analysis_path / 'left_eye_data.csv', 
            index_col=0, 
            engine='python'
        )
        block.right_eye_data = pd.read_csv(
            block.analysis_path / 'right_eye_data.csv', 
            index_col=0, 
            engine='python'
        )
        print(f'Loaded eye data for block {block.block_num}')
    except FileNotFoundError:
        print(f'Warning: Eye data files not found for block {block.block_num}. '
              f'Run the synchronization pipeline first!')
        raise


def load_self_kerr_refs(block, filename: str = "self_kerr_refs.csv") -> bool:
    """
    Load Kerr reference coordinates from the analysis folder CSV and set them on `block`.

    Reads a single-row CSV with columns:
        kerr_ref_r_x, kerr_ref_r_y, kerr_ref_l_x, kerr_ref_l_y

    Parameters
    ----------
    block : BlockSync
        The BlockSync instance to load references for.
    filename : str, optional
        Name of the CSV file containing Kerr references. Default is "self_kerr_refs.csv".

    Returns
    -------
    bool
        True if refs were loaded and applied, False if the file was missing or empty.
    """
    path = Path(block.analysis_path) / filename
    if not path.exists():
        print(f"No Kerr refs file found at: {path}")
        return False

    df = pd.read_csv(path)
    if df.empty:
        print(f"Kerr refs file is empty: {path}")
        return False

    row = df.iloc[0]

    # Helper to safely set attribute if value is finite
    def _set_attr(name):
        if name in row and pd.notna(row[name]):
            try:
                setattr(block, name, int(round(float(row[name]))))
            except (ValueError, TypeError):
                # keep existing value if conversion fails
                pass

    for col in ("kerr_ref_r_x", "kerr_ref_r_y", "kerr_ref_l_x", "kerr_ref_l_y"):
        _set_attr(col)

    print(f"Kerr refs loaded from: {path}")
    return True


@dataclass(frozen=True)
class KerrAnglePreview:
    """In-memory Kerr angle preview for a single eye (no disk writes)."""

    phi: np.ndarray
    theta: np.ndarray
    f_z: float
    ref_x: float
    ref_y: float
    n_input: int
    n_finite: int


def preview_kerr_angles(
    eye_df: pd.DataFrame,
    ref_x: float,
    ref_y: float,
) -> KerrAnglePreview:
    """
    Run ``BlockSync.kerr`` for one eye using a tentative reference point.

    Does not write CSVs or mutate the caller's dataframe. Missing
    ``major_ax``/``minor_ax`` are derived from ``width``/``height`` when needed.
    Placeholder ``OE_timestamp`` / ``ms_axis`` are filled only when absent so
    the static Kerr helper can assemble its output frame.
    """
    if eye_df is None or eye_df.empty:
        raise ValueError("eye_df is empty; cannot preview Kerr angles.")
    if not np.isfinite(ref_x) or not np.isfinite(ref_y):
        raise ValueError("Kerr reference must be a finite (x, y) point.")

    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync

    work = eye_df.copy()
    if "major_ax" not in work.columns or "minor_ax" not in work.columns:
        if not {"width", "height"}.issubset(work.columns):
            raise ValueError(
                "eye_df needs major_ax/minor_ax or width/height for Kerr preview."
            )
        work = BlockSync.get_maj_min_axes(work)
    if "eye_frame" not in work.columns:
        if "frame" in work.columns:
            work = work.rename(columns={"frame": "eye_frame"})
        else:
            work["eye_frame"] = np.arange(len(work), dtype=int)
    if "OE_timestamp" not in work.columns:
        work["OE_timestamp"] = np.nan
    if "ms_axis" not in work.columns:
        work["ms_axis"] = np.nan

    work["ratio2"] = work["minor_ax"] / work["major_ax"]
    if "phi" in work.columns:
        work["phi_ellipse"] = work["phi"]

    with np.errstate(invalid="ignore", divide="ignore"):
        f_z, angles = BlockSync.kerr(
            work, aEC=float(ref_x), bEC=float(ref_y)
        )

    if angles is None:
        raise RuntimeError("Kerr calculation returned no angles for this reference.")

    phi = np.asarray(angles["phi"], dtype=float)
    theta = np.asarray(angles["theta"], dtype=float)
    finite = np.isfinite(phi) & np.isfinite(theta)
    return KerrAnglePreview(
        phi=phi,
        theta=theta,
        f_z=float(f_z),
        ref_x=float(ref_x),
        ref_y=float(ref_y),
        n_input=int(len(work)),
        n_finite=int(np.count_nonzero(finite)),
    )


def _existing_kerr_angle_columns(columns) -> list[str]:
    """Return kerr angle / merge-suffix columns present in ``columns``."""
    return [c for c in columns if _KERR_ANGLE_COL_RE.match(str(c))]


def append_angle_data(eye_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append Kerr angle columns (``k_phi``, ``k_theta``) from ``new_df`` to ``eye_df``.

    Renames angle-CSV ``phi``/``theta`` to ``k_phi``/``k_theta``, then left-merges
    on ``OE_timestamp``. Idempotent: any existing ``k_phi``/``k_theta`` (including
    pandas merge suffixes ``k_phi_x`` / ``k_phi_y``) are dropped from a copy of
    ``eye_df`` before merging, so re-appending onto already-hydrated frames does
    not produce suffix columns. Does **not** reload eye data from disk — callers
    may pass filtered or alternate in-memory frames.
    """
    if eye_df is None:
        raise ValueError("eye_df is None; load or assign eye data before appending angles.")
    if new_df is None or new_df.empty:
        raise ValueError("Angle dataframe is empty; run calculate_kerr_angles first.")
    missing = [c for c in ("OE_timestamp", "phi", "theta") if c not in new_df.columns]
    if missing:
        raise ValueError(
            f"Angle dataframe missing required columns {missing}; "
            f"have {list(new_df.columns)}."
        )
    if "OE_timestamp" not in eye_df.columns:
        raise ValueError(
            "eye_df has no OE_timestamp column; cannot merge Kerr angles."
        )

    # Work on a copy so filtered/alternate in-memory frames stay intact for retry.
    base = eye_df.copy()
    drop_cols = _existing_kerr_angle_columns(base.columns)
    if drop_cols:
        base = base.drop(columns=drop_cols)

    angle_data = new_df[["OE_timestamp", "phi", "theta"]].rename(
        columns={"phi": "k_phi", "theta": "k_theta"}
    )
    return pd.merge(base, angle_data, on="OE_timestamp", how="left")


@dataclass
class AngleAppendCheck:
    """Passive post-append health check (columns + finite sample counts)."""

    side: str
    ok: bool
    columns_ok: bool
    n_rows: int
    n_k_phi: int
    n_k_theta: int
    n_src_phi: int
    n_src_theta: int
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "ok" if self.ok else "warn"
        return (
            f"{self.side}: {status} | cols={self.columns_ok} | "
            f"k_phi={self.n_k_phi}/{self.n_rows} k_theta={self.n_k_theta}/{self.n_rows} "
            f"(src phi={self.n_src_phi} theta={self.n_src_theta})"
            + (f" — {'; '.join(self.messages)}" if self.messages else "")
        )


def check_appended_angles(
    eye_df: pd.DataFrame,
    angle_df: pd.DataFrame,
    *,
    side: str = "eye",
    min_finite_fraction: float = 0.0,
) -> AngleAppendCheck:
    """
    Passive validation after ``append_angle_data``.

    Checks that canonical ``k_phi``/``k_theta`` exist (and no merge-suffix leftovers),
    and reports finite sample counts vs the source angle CSV. Does not mutate data
    or reload from disk. ``ok`` is False when columns are wrong or finite counts
    look suspicious relative to the source (or fall below ``min_finite_fraction``
    of eye rows when that threshold is > 0).
    """
    messages: list[str] = []
    n_rows = 0 if eye_df is None else len(eye_df)
    cols = list(eye_df.columns) if eye_df is not None else []

    has_phi = "k_phi" in cols
    has_theta = "k_theta" in cols
    suffix_cols = [
        c for c in cols if c in ("k_phi_x", "k_phi_y", "k_theta_x", "k_theta_y")
    ]
    columns_ok = has_phi and has_theta and not suffix_cols
    if not has_phi or not has_theta:
        messages.append(
            f"missing canonical columns "
            f"(have k_phi={has_phi}, k_theta={has_theta}; cols sample={cols[:12]}…)"
        )
    if suffix_cols:
        messages.append(f"merge-suffix leftovers present: {suffix_cols}")

    n_k_phi = int(eye_df["k_phi"].notna().sum()) if has_phi else 0
    n_k_theta = int(eye_df["k_theta"].notna().sum()) if has_theta else 0

    n_src_phi = (
        int(angle_df["phi"].notna().sum())
        if angle_df is not None and "phi" in angle_df.columns
        else 0
    )
    n_src_theta = (
        int(angle_df["theta"].notna().sum())
        if angle_df is not None and "theta" in angle_df.columns
        else 0
    )

    # Timestamp misalignment / empty merge: appended finite count far below source.
    if has_phi and n_src_phi > 0 and n_k_phi == 0:
        messages.append("k_phi is all-NaN but source phi has finite samples")
    elif has_phi and n_src_phi > 0 and n_k_phi < max(1, int(0.5 * n_src_phi)):
        messages.append(
            f"k_phi finite count ({n_k_phi}) << source phi ({n_src_phi}); "
            "possible OE_timestamp misalignment"
        )
    if has_theta and n_src_theta > 0 and n_k_theta == 0:
        messages.append("k_theta is all-NaN but source theta has finite samples")
    elif has_theta and n_src_theta > 0 and n_k_theta < max(1, int(0.5 * n_src_theta)):
        messages.append(
            f"k_theta finite count ({n_k_theta}) << source theta ({n_src_theta}); "
            "possible OE_timestamp misalignment"
        )

    if min_finite_fraction > 0 and n_rows > 0:
        need = int(np.ceil(min_finite_fraction * n_rows))
        if has_phi and n_k_phi < need:
            messages.append(
                f"k_phi finite {n_k_phi} < {min_finite_fraction:.0%} of {n_rows} rows"
            )
        if has_theta and n_k_theta < need:
            messages.append(
                f"k_theta finite {n_k_theta} < {min_finite_fraction:.0%} of {n_rows} rows"
            )

    return AngleAppendCheck(
        side=side,
        ok=columns_ok and not messages,
        columns_ok=columns_ok,
        n_rows=n_rows,
        n_k_phi=n_k_phi,
        n_k_theta=n_k_theta,
        n_src_phi=n_src_phi,
        n_src_theta=n_src_theta,
        messages=messages,
    )


def export_eye_data_w_angles(block, name_tag: str = 'default') -> None:
    """
    Export eye dataframes with angle data to CSV files.
    
    Parameters
    ----------
    block : BlockSync
        The BlockSync instance containing the eye data.
    name_tag : str, optional
        Tag to append to the output filenames. Default is 'default'.
    """
    block.right_eye_data.to_csv(block.analysis_path / f'right_eye_data_{name_tag}.csv')
    block.left_eye_data.to_csv(block.analysis_path / f'left_eye_data_{name_tag}.csv')
    print(f'Exported eye data with angles (tag: {name_tag}) for block {block.block_num}')


def calculate_kerr_angles_for_block(
    block,
    name_tag: str = 'default',
    kerr_refs_filename: str = "self_kerr_refs.csv",
    load_eye_data_flag: bool = True,
    export_flag: bool = True
) -> None:
    """
    Calculate Kerr angles for a single block and append to eye dataframes.
    
    This function performs the complete Kerr angle calculation pipeline:
    1. Loads eye data from CSV (if not already loaded)
    2. Loads Kerr reference coordinates
    3. Calculates Kerr angles using BlockSync.calculate_kerr_angles()
    4. Appends angle data (k_phi, k_theta) to eye dataframes
    5. Exports updated dataframes (optional)
    
    Parameters
    ----------
    block : BlockSync
        The BlockSync instance to process.
    name_tag : str, optional
        Tag for output files. Default is 'default'.
    kerr_refs_filename : str, optional
        Filename for Kerr reference coordinates CSV. Default is "self_kerr_refs.csv".
    load_eye_data_flag : bool, optional
        If True, load eye data from CSV files. If False, assume data is already loaded.
        Default is True.
    export_flag : bool, optional
        If True, export the updated dataframes. Default is True.
        
    Raises
    ------
    FileNotFoundError
        If eye data files are not found and load_eye_data_flag is True.
    AttributeError
        If Kerr references are not set after loading.
    """
    print(f'\n{"="*60}')
    print(f'Processing block {block.block_num}')
    print(f'{"="*60}')
    
    # Step 1: Load eye data if needed
    if load_eye_data_flag:
        try:
            load_eye_data(block)
        except FileNotFoundError:
            print(f'Skipping block {block.block_num} - eye data not found')
            return
    
    # Step 2: Load Kerr references
    refs_loaded = load_self_kerr_refs(block, filename=kerr_refs_filename)
    if not refs_loaded:
        print(f'Warning: Could not load Kerr references for block {block.block_num}. '
              f'Make sure {kerr_refs_filename} exists in {block.analysis_path}')
        return
    
    # Verify references are set
    if not hasattr(block, 'kerr_ref_l_x') or block.kerr_ref_l_x is None:
        print(f'Error: Kerr references not properly set for block {block.block_num}')
        return
    
    # Step 3: Calculate Kerr angles using BlockSync method
    try:
        block.calculate_kerr_angles(name_tag=name_tag)
    except Exception as e:
        print(f'Error calculating Kerr angles for block {block.block_num}: {e}')
        raise
    
    # Step 4: Load calculated angles and append to eye dataframes
    try:
        # Find the angle files
        left_angle_file = None
        right_angle_file = None
        
        for file in block.analysis_path.iterdir():
            if f'left_kerr_angle_{name_tag}.csv' in str(file):
                left_angle_file = file
            elif f'right_kerr_angle_{name_tag}.csv' in str(file):
                right_angle_file = file
        
        if left_angle_file is None or right_angle_file is None:
            raise FileNotFoundError(
                f'Kerr angle files not found for block {block.block_num} with tag {name_tag}'
            )
        
        left_angles = pd.read_csv(left_angle_file)
        right_angles = pd.read_csv(right_angle_file)
        
        # Append angle data to eye dataframes
        block.left_eye_data = append_angle_data(block.left_eye_data, left_angles)
        block.right_eye_data = append_angle_data(block.right_eye_data, right_angles)
        
        print(f'Successfully appended angle data to eye dataframes for block {block.block_num}')
        
    except FileNotFoundError as e:
        print(f'Error loading angle files for block {block.block_num}: {e}')
        raise
    except Exception as e:
        print(f'Error appending angle data for block {block.block_num}: {e}')
        raise
    
    # Step 5: Export updated dataframes
    if export_flag:
        export_eye_data_w_angles(block, name_tag=f'degrees_{name_tag}')
    
    print(f'Completed processing block {block.block_num}\n')


def calculate_kerr_angles_for_collection(
    block_collection: List,
    name_tag: str = 'default',
    kerr_refs_filename: str = "self_kerr_refs.csv",
    load_eye_data_flag: bool = True,
    export_flag: bool = True,
    continue_on_error: bool = True
) -> None:
    """
    Calculate Kerr angles for a collection of blocks.
    
    Parameters
    ----------
    block_collection : List[BlockSync]
        List of BlockSync instances to process.
    name_tag : str, optional
        Tag for output files. Default is 'default'.
    kerr_refs_filename : str, optional
        Filename for Kerr reference coordinates CSV. Default is "self_kerr_refs.csv".
    load_eye_data_flag : bool, optional
        If True, load eye data from CSV files. Default is True.
    export_flag : bool, optional
        If True, export the updated dataframes. Default is True.
    continue_on_error : bool, optional
        If True, continue processing remaining blocks if one fails. Default is True.
    """
    print(f'\n{"="*60}')
    print(f'Processing {len(block_collection)} blocks')
    print(f'{"="*60}\n')
    
    successful = 0
    failed = 0
    
    for block in block_collection:
        try:
            calculate_kerr_angles_for_block(
                block=block,
                name_tag=name_tag,
                kerr_refs_filename=kerr_refs_filename,
                load_eye_data_flag=load_eye_data_flag,
                export_flag=export_flag
            )
            successful += 1
        except Exception as e:
            failed += 1
            if continue_on_error:
                print(f'Error processing block {block.block_num}: {e}')
                print('Continuing with next block...\n')
            else:
                raise
    
    print(f'\n{"="*60}')
    print(f'Processing complete: {successful} successful, {failed} failed')
    print(f'{"="*60}\n')


if __name__ == "__main__":
    # Example usage when run as a script
    import sys
    from pathlib import Path
    
    # This would typically be called from a notebook or another script
    # Example:
    # from eye_tracking_system_tools.preprocessing import utility_functions as uf
    # from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import calculate_kerr_angles_for_collection
    # 
    # block_collection = uf.block_generator(...)
    # calculate_kerr_angles_for_collection(block_collection, name_tag='raw_verified')
    
    print("This script is designed to be imported and called from other scripts/notebooks.")
    print("See the module docstring for usage examples.")
