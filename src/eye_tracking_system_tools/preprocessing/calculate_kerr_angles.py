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

from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import numpy as np


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


def append_angle_data(eye_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append the angle columns (phi and theta) from new_df to eye_df.
    
    The function renames 'phi' to 'k_phi' and 'theta' to 'k_theta', then merges
    on the shared 'OE_timestamp' column.

    Parameters
    ----------
    eye_df : pd.DataFrame
        DataFrame containing the eye tracking data.
    new_df : pd.DataFrame
        DataFrame containing the new kinematics data with columns
        'phi' and 'theta' along with 'OE_timestamp'.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with angle data appended.
    """
    # Select the necessary columns and rename them
    angle_data = new_df[['OE_timestamp', 'phi', 'theta']].rename(
        columns={'phi': 'k_phi', 'theta': 'k_theta'}
    )

    # Merge on OE_timestamp using a left join to preserve all rows in eye_df
    merged_df = pd.merge(eye_df, angle_data, on='OE_timestamp', how='left')

    return merged_df


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
