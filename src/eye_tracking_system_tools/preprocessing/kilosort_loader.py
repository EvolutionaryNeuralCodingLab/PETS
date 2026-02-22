"""
Kilosort spike data loader and alignment utilities.

This module provides functions to load kilosort output files and align spike times
to the Open Ephys recording synchronization frame (ms from recording start).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


def load_kilosort_params(kilosort_path: Path) -> Dict:
    """
    Load kilosort params.py file and return as dictionary.
    
    Parameters
    ----------
    kilosort_path : Path
        Path to kilosort output folder (should contain params.py)
        
    Returns
    -------
    dict
        Dictionary containing kilosort parameters (sample_rate, n_channels_dat, etc.)
    """
    params_file = Path(kilosort_path) / 'params.py'
    if not params_file.exists():
        raise FileNotFoundError(f"params.py not found at {params_file}")
    
    # Read and execute params.py to get variables
    params_dict = {}
    with open(params_file, 'r') as f:
        content = f.read()
        # Execute in a safe namespace
        exec(content, {}, params_dict)
    
    return params_dict


def load_kilosort_spike_data(kilosort_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spike times and cluster assignments from kilosort output.
    
    Parameters
    ----------
    kilosort_path : Path
        Path to kilosort output folder
        
    Returns
    -------
    spike_times_samples : np.ndarray
        Spike times in samples (shape: n_spikes,)
    spike_clusters : np.ndarray
        Cluster ID for each spike (shape: n_spikes,)
    """
    kilosort_path = Path(kilosort_path)
    
    spike_times_file = kilosort_path / 'spike_times.npy'
    spike_clusters_file = kilosort_path / 'spike_clusters.npy'
    
    if not spike_times_file.exists():
        raise FileNotFoundError(f"spike_times.npy not found at {spike_times_file}")
    if not spike_clusters_file.exists():
        raise FileNotFoundError(f"spike_clusters.npy not found at {spike_clusters_file}")
    
    # Load numpy arrays
    # spike_times.npy is typically stored as (n_spikes, 1) but we want (n_spikes,)
    spike_times_samples = np.load(spike_times_file).flatten()
    spike_clusters = np.load(spike_clusters_file).flatten()
    
    if len(spike_times_samples) != len(spike_clusters):
        raise ValueError(
            f"Mismatch: spike_times has {len(spike_times_samples)} entries, "
            f"spike_clusters has {len(spike_clusters)} entries"
        )
    
    return spike_times_samples, spike_clusters


def load_cluster_groups(kilosort_path: Path) -> pd.DataFrame:
    """
    Load cluster group assignments (good, mua, noise, etc.) from cluster_group.tsv.
    
    Parameters
    ----------
    kilosort_path : Path
        Path to kilosort output folder
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cluster_id, group (e.g., 'good', 'mua', 'noise')
    """
    cluster_group_file = Path(kilosort_path) / 'cluster_group.tsv'
    
    if not cluster_group_file.exists():
        warnings.warn(f"cluster_group.tsv not found at {cluster_group_file}. "
                     f"All clusters will be treated as 'good'.")
        return pd.DataFrame(columns=['cluster_id', 'group'])
    
    df = pd.read_csv(cluster_group_file, sep='\t')
    
    # Handle different possible column names
    if 'KSLabel' in df.columns:
        df = df.rename(columns={'KSLabel': 'group'})
    elif 'group' not in df.columns:
        raise ValueError(f"cluster_group.tsv must have 'group' or 'KSLabel' column. Found: {df.columns.tolist()}")
    
    return df[['cluster_id', 'group']]


def get_good_cluster_ids(kilosort_path: Path) -> np.ndarray:
    """
    Get list of cluster IDs labeled as 'good'.
    
    Parameters
    ----------
    kilosort_path : Path
        Path to kilosort output folder
        
    Returns
    -------
    np.ndarray
        Array of cluster IDs labeled as 'good'
    """
    cluster_groups = load_cluster_groups(kilosort_path)
    
    if len(cluster_groups) == 0:
        warnings.warn("No cluster groups found. Returning empty array.")
        return np.array([], dtype=int)
    
    good_clusters = cluster_groups[cluster_groups['group'] == 'good']['cluster_id'].values
    return good_clusters.astype(int)


def align_spike_times_to_oe_recording(
    spike_times_samples: np.ndarray,
    kilosort_sample_rate: float,
    oe_recording_start_ms: float,
    oe_sample_rate: Optional[float] = None
) -> np.ndarray:
    """
    Align kilosort spike times to Open Ephys recording start (ms from start, starting at 0).
    
    Kilosort spike times are in samples at kilosort_sample_rate (typically 20kHz).
    The binary file processed by kilosort typically starts from the first OE recording sample
    (sample 0 in OE terms). This function converts spike times to milliseconds and aligns
    them to the OE recording synchronization frame (starting at 0 ms).
    
    Key assumption: The kilosort binary file (e.g., ch1_32.bin) starts from OE sample 0,
    so kilosort sample 0 corresponds to OE sample 0. No sample offset is needed.
    
    Parameters
    ----------
    spike_times_samples : np.ndarray
        Spike times in samples (at kilosort_sample_rate)
    kilosort_sample_rate : float
        Sampling rate used by kilosort (Hz), typically 20000
    oe_recording_start_ms : float
        Open Ephys globalStartTime_ms (the first timestamp in ms). This is used for
        verification but typically not needed if binary starts at sample 0.
    oe_sample_rate : float, optional
        Open Ephys sample rate (Hz). Currently not used but kept for future compatibility.
        If kilosort and OE rates differ, alignment may need adjustment.
        
    Returns
    -------
    spike_times_ms : np.ndarray
        Spike times in milliseconds from OE recording start (starting at 0)
    """
    # Convert spike times from samples to milliseconds
    # Kilosort sample 0 = OE sample 0, so direct conversion
    spike_times_ms = spike_times_samples / kilosort_sample_rate * 1000.0
    
    # Note: If the binary file starts at OE sample 0, spike times are already aligned.
    # The oe_recording_start_ms represents the timestamp offset, but since we're working
    # with sample indices from the start of the binary file, no offset is needed.
    # However, we verify that times are reasonable (non-negative, within recording duration)
    
    # Ensure no negative times (shouldn't happen if binary starts at sample 0)
    if np.any(spike_times_ms < -1.0):  # Allow small numerical errors
        warnings.warn(
            f"Found {np.sum(spike_times_ms < 0)} spikes with negative times. "
            f"Min time: {np.min(spike_times_ms):.3f} ms. "
            f"This may indicate the binary file doesn't start at OE sample 0."
        )
    
    return spike_times_ms


def load_and_align_kilosort_spikes(
    block,
    kilosort_path: Optional[Path] = None,
    filter_good_only: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load kilosort spike data and align to block's OE recording synchronization frame.
    
    This is the main convenience function that loads all kilosort data and aligns it
    to the same timebase as eye data and oe_rec data (ms from OE recording start).
    
    Parameters
    ----------
    block : BlockSync
        BlockSync object with initialized oe_rec attribute
    kilosort_path : Path, optional
        Path to kilosort output folder. If None, will look for spikeSorting/kilosort
        folder within the block's oe_path.
    filter_good_only : bool
        If True, only return spikes from clusters labeled as 'good'
        
    Returns
    -------
    spike_times_ms : np.ndarray
        Spike times in milliseconds from OE recording start (aligned, starting at 0)
    spike_clusters : np.ndarray
        Cluster IDs for each spike (filtered if filter_good_only=True)
    metadata : dict
        Dictionary containing:
        - 'kilosort_sample_rate': sampling rate used by kilosort (Hz)
        - 'oe_sample_rate': OE recording sample rate (Hz)
        - 'oe_recording_start_ms': OE globalStartTime_ms
        - 'n_spikes_total': total number of spikes before filtering
        - 'n_spikes_returned': number of spikes returned
        - 'n_good_clusters': number of 'good' clusters found
        - 'good_cluster_ids': array of good cluster IDs
    """
    if not hasattr(block, 'oe_rec') or block.oe_rec is None:
        raise ValueError("Block must have initialized oe_rec attribute")
    
    # Determine kilosort path
    if kilosort_path is None:
        # Look for spikeSorting/kilosort folder within oe_path
        spike_sorting_path = block.oe_path / 'spikeSorting' / 'kilosort'
        if not spike_sorting_path.exists():
            raise FileNotFoundError(
                f"Could not find kilosort folder. Tried: {spike_sorting_path}. "
                f"Please specify kilosort_path explicitly."
            )
        kilosort_path = spike_sorting_path
    else:
        kilosort_path = Path(kilosort_path)
    
    # Load kilosort parameters
    params = load_kilosort_params(kilosort_path)
    kilosort_sample_rate = params.get('sample_rate', 20000.0)
    
    # Load spike data
    spike_times_samples, spike_clusters = load_kilosort_spike_data(kilosort_path)
    
    # Get OE recording parameters
    oe_sample_rate = block.sample_rate
    oe_recording_start_ms = float(block.oe_rec.globalStartTime_ms)
    
    # Align spike times to OE recording start
    spike_times_ms = align_spike_times_to_oe_recording(
        spike_times_samples=spike_times_samples,
        kilosort_sample_rate=kilosort_sample_rate,
        oe_recording_start_ms=oe_recording_start_ms,
        oe_sample_rate=oe_sample_rate
    )
    
    # Filter for good clusters if requested
    n_spikes_total = len(spike_times_ms)
    if filter_good_only:
        good_cluster_ids = get_good_cluster_ids(kilosort_path)
        good_mask = np.isin(spike_clusters, good_cluster_ids)
        spike_times_ms = spike_times_ms[good_mask]
        spike_clusters = spike_clusters[good_mask]
        n_good_clusters = len(good_cluster_ids)
    else:
        n_good_clusters = len(np.unique(spike_clusters))
    
    metadata = {
        'kilosort_sample_rate': kilosort_sample_rate,
        'oe_sample_rate': oe_sample_rate,
        'oe_recording_start_ms': oe_recording_start_ms,
        'n_spikes_total': n_spikes_total,
        'n_spikes_returned': len(spike_times_ms),
        'n_good_clusters': n_good_clusters,
        'good_cluster_ids': get_good_cluster_ids(kilosort_path) if filter_good_only else np.unique(spike_clusters),
    }
    
    return spike_times_ms, spike_clusters, metadata
