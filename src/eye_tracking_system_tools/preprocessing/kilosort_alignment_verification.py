"""
Kilosort spike alignment verification by comparing with LFP traces.

This module provides functions to verify that kilosort spike times are correctly
aligned by overlaying them with LFP traces from oe_rec.get_data().
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, Span
from bokeh.layouts import column
from bokeh.io import output_notebook, reset_output

from .kilosort_loader import load_and_align_kilosort_spikes


def verify_spike_alignment_with_lfp(
    block,
    lfp_channels: List[int],
    time_range_ms: Tuple[float, float],
    kilosort_path: Optional[Path] = None,
    filter_good_only: bool = True,
    cluster_ids: Optional[np.ndarray] = None,
    width: int = 1400,
    height_per_channel: int = 200,
    to_browser: bool = True,
    downsample_lfp: int = 1,
) -> List[figure]:
    """
    Verify kilosort spike alignment by overlaying spikes on LFP traces.
    
    This function loads LFP traces for specified channels and overlays kilosort
    spike times as vertical lines. If alignment is correct, spikes should appear
    at the same times as the spike waveforms in the LFP traces.
    
    Parameters
    ----------
    block : BlockSync
        BlockSync object with initialized oe_rec attribute
    lfp_channels : list of int
        List of LFP channel numbers to plot (e.g., [18, 17, 9, 10])
    time_range_ms : tuple of (float, float)
        Time range to plot in milliseconds (start_ms, end_ms)
    kilosort_path : Path, optional
        Path to kilosort output folder. If None, auto-detects from block.oe_path
    filter_good_only : bool
        If True, only plot spikes from 'good' clusters
    cluster_ids : np.ndarray, optional
        Specific cluster IDs to plot. If None, plots all good clusters.
    width : int
        Plot width in pixels
    height_per_channel : int
        Height in pixels for each channel subplot
    to_browser : bool
        If True, shows plots in browser
    downsample_lfp : int
        Downsample factor for LFP traces (1 = no downsampling, 2 = every other sample, etc.)
        
    Returns
    -------
    list of figure
        List of Bokeh figure objects, one per channel
    """
    if not hasattr(block, 'oe_rec') or block.oe_rec is None:
        raise ValueError("Block must have initialized oe_rec attribute")
    
    start_ms, end_ms = time_range_ms
    window_ms = end_ms - start_ms
    
    # Load and align kilosort spike data
    spike_times_ms, spike_clusters, metadata = load_and_align_kilosort_spikes(
        block=block,
        kilosort_path=kilosort_path,
        filter_good_only=filter_good_only
    )
    
    # Filter by cluster IDs if specified
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        mask = np.isin(spike_clusters, cluster_ids)
        spike_times_ms = spike_times_ms[mask]
        spike_clusters = spike_clusters[mask]
    
    # Filter spikes to time range
    time_mask = (spike_times_ms >= start_ms) & (spike_times_ms <= end_ms)
    spike_times_in_range = spike_times_ms[time_mask]
    spike_clusters_in_range = spike_clusters[time_mask]
    
    print(f"Loaded {len(spike_times_in_range)} spikes in time range [{start_ms:.1f}, {end_ms:.1f}] ms")
    print(f"Number of unique clusters: {len(np.unique(spike_clusters_in_range))}")
    
    # Load LFP data for each channel
    start_arr = np.atleast_2d(np.array([start_ms], dtype=float))
    
    print(f"\nLoading LFP traces for channels {lfp_channels}...")
    try:
        lfp_data, lfp_timestamps = block.oe_rec.get_data(
            channels=lfp_channels,
            start_time_ms=start_arr,
            window_ms=window_ms,
            convert_microvolts=True,
            return_timestamps=True,
            repress_output=False,
        )
    except Exception as e:
        raise ValueError(f"Failed to load LFP data: {e}")
    
    if lfp_data is None or lfp_data.size == 0:
        raise ValueError("No LFP data returned")
    
    # lfp_data shape: [n_channels, n_windows, n_samples]
    # lfp_timestamps shape: [n_windows, n_samples]
    n_channels = lfp_data.shape[0]
    n_samples = lfp_data.shape[2]
    
    print(f"Loaded LFP data: {n_channels} channels, {n_samples} samples")
    
    # Downsample LFP if requested
    if downsample_lfp > 1:
        lfp_data = lfp_data[:, :, ::downsample_lfp]
        lfp_timestamps = lfp_timestamps[:, ::downsample_lfp]
        n_samples = lfp_data.shape[2]
        print(f"Downsampled LFP by factor {downsample_lfp}: {n_samples} samples")
    
    # Extract time vector (should be same for all channels)
    time_ms = lfp_timestamps[0, :]
    
    # Verify time alignment
    expected_start = start_ms
    expected_end = start_ms + window_ms
    actual_start = time_ms[0]
    actual_end = time_ms[-1]
    
    print(f"\nTime alignment check:")
    print(f"  Expected range: [{expected_start:.2f}, {expected_end:.2f}] ms")
    print(f"  Actual LFP range: [{actual_start:.2f}, {actual_end:.2f}] ms")
    print(f"  Difference: start={abs(actual_start - expected_start):.2f} ms, end={abs(actual_end - expected_end):.2f} ms")
    
    # Create plots
    reset_output()
    if to_browser:
        output_notebook()
    
    figures = []
    
    for ch_idx, channel_num in enumerate(lfp_channels):
        # Extract LFP trace for this channel
        lfp_trace = lfp_data[ch_idx, 0, :]  # [n_samples]
        
        # Create figure
        p = figure(
            width=width,
            height=height_per_channel,
            title=f"Channel {channel_num} LFP with Kilosort Spikes Overlaid",
            x_axis_label="Time (ms from OE recording start)",
            y_axis_label="Voltage (µV)",
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )
        
        # Plot LFP trace
        source_lfp = ColumnDataSource(data=dict(
            x=time_ms,
            y=lfp_trace,
            time_ms=time_ms,
            voltage=lfp_trace,
        ))
        
        p.line(
            x="x",
            y="y",
            line_width=1,
            color="black",
            alpha=0.7,
            legend_label=f"Channel {channel_num} LFP",
            source=source_lfp,
        )
        
        # Overlay spikes as vertical lines
        # Filter spikes to this time range (already done, but keep for clarity)
        spikes_for_plot = spike_times_in_range
        
        if len(spikes_for_plot) > 0:
            # Get y-range for spike lines
            y_min = np.min(lfp_trace)
            y_max = np.max(lfp_trace)
            y_range = y_max - y_min
            y_margin = y_range * 0.1
            
            # Plot spikes as vertical lines
            for spike_time in spikes_for_plot:
                p.line(
                    [spike_time, spike_time],
                    [y_min - y_margin, y_max + y_margin],
                    line_width=1,
                    color="red",
                    alpha=0.4,
                    line_dash="dashed",
                )
            
            # Also add scatter points at spike times on the trace
            # Find LFP values at spike times (interpolate if needed)
            spike_voltages = np.interp(spikes_for_plot, time_ms, lfp_trace)
            
            source_spikes = ColumnDataSource(data=dict(
                x=spikes_for_plot,
                y=spike_voltages,
                time_ms=spikes_for_plot,
                voltage=spike_voltages,
            ))
            
            p.scatter(
                x="x",
                y="y",
                size=4,
                color="red",
                alpha=0.8,
                legend_label=f"Kilosort Spikes ({len(spikes_for_plot)})",
                source=source_spikes,
            )
        
        # Add hover tool
        p.add_tools(HoverTool(
            tooltips=[
                ("Time (ms)", "@time_ms{0.2f}"),
                ("Voltage (µV)", "@voltage{0.2f}"),
            ],
            mode='vline'
        ))
        
        # Configure legend
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        
        figures.append(p)
    
    # Combine all plots vertically
    if len(figures) > 1:
        combined = column(*figures)
        if to_browser:
            show(combined)
        return figures
    else:
        if to_browser:
            show(figures[0])
        return figures


def verify_spike_alignment_summary(
    block,
    lfp_channels: List[int],
    time_range_ms: Tuple[float, float],
    kilosort_path: Optional[Path] = None,
    filter_good_only: bool = True,
) -> dict:
    """
    Generate a summary report of spike alignment verification.
    
    Parameters
    ----------
    block : BlockSync
        BlockSync object with initialized oe_rec attribute
    lfp_channels : list of int
        List of LFP channel numbers to check
    time_range_ms : tuple of (float, float)
        Time range to check (start_ms, end_ms)
    kilosort_path : Path, optional
        Path to kilosort output folder
    filter_good_only : bool
        If True, only check 'good' clusters
        
    Returns
    -------
    dict
        Summary dictionary with alignment statistics
    """
    start_ms, end_ms = time_range_ms
    window_ms = end_ms - start_ms
    
    # Load spike data
    spike_times_ms, spike_clusters, metadata = load_and_align_kilosort_spikes(
        block=block,
        kilosort_path=kilosort_path,
        filter_good_only=filter_good_only
    )
    
    # Filter to time range
    time_mask = (spike_times_ms >= start_ms) & (spike_times_ms <= end_ms)
    spike_times_in_range = spike_times_ms[time_mask]
    
    # Load LFP data
    start_arr = np.atleast_2d(np.array([start_ms], dtype=float))
    
    try:
        lfp_data, lfp_timestamps = block.oe_rec.get_data(
            channels=lfp_channels,
            start_time_ms=start_arr,
            window_ms=window_ms,
            convert_microvolts=True,
            return_timestamps=True,
            repress_output=True,
        )
    except Exception as e:
        return {"error": str(e)}
    
    lfp_time_ms = lfp_timestamps[0, :]
    
    # Calculate statistics
    summary = {
        "time_range_ms": time_range_ms,
        "lfp_channels": lfp_channels,
        "n_spikes_in_range": len(spike_times_in_range),
        "n_clusters": len(np.unique(spike_clusters[time_mask])),
        "spike_time_range": {
            "min": float(np.min(spike_times_in_range)) if len(spike_times_in_range) > 0 else None,
            "max": float(np.max(spike_times_in_range)) if len(spike_times_in_range) > 0 else None,
        },
        "lfp_time_range": {
            "min": float(np.min(lfp_time_ms)),
            "max": float(np.max(lfp_time_ms)),
        },
        "time_alignment_check": {
            "spike_min_vs_lfp_min": float(np.min(spike_times_in_range) - np.min(lfp_time_ms)) if len(spike_times_in_range) > 0 else None,
            "spike_max_vs_lfp_max": float(np.max(spike_times_in_range) - np.max(lfp_time_ms)) if len(spike_times_in_range) > 0 else None,
        },
        "oe_recording_info": {
            "globalStartTime_ms": float(block.oe_rec.globalStartTime_ms),
            "recordingDuration_ms": float(block.oe_rec.recordingDuration_ms),
            "sample_rate": float(block.sample_rate),
        },
        "kilosort_info": metadata,
    }
    
    return summary
