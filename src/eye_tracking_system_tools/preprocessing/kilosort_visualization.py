"""
Kilosort spike raster plot visualization.

This module provides functions to create raster plots of kilosort spike data
aligned to the Open Ephys recording synchronization frame.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category20, Category10
from bokeh.io import output_notebook, reset_output

from .kilosort_loader import load_and_align_kilosort_spikes


def plot_spike_raster(
    block,
    kilosort_path: Optional[Path] = None,
    filter_good_only: bool = True,
    time_range_ms: Optional[Tuple[float, float]] = None,
    max_spikes_per_cluster: Optional[int] = None,
    cluster_ids: Optional[np.ndarray] = None,
    width: int = 1200,
    height: int = 600,
    title: Optional[str] = None,
    to_browser: bool = True,
    show_cluster_labels: bool = True,
) -> figure:
    """
    Create a raster plot of kilosort spike times aligned to OE recording start.
    
    Parameters
    ----------
    block : BlockSync
        BlockSync object with initialized oe_rec attribute
    kilosort_path : Path, optional
        Path to kilosort output folder. If None, will look for spikeSorting/kilosort
        folder within the block's oe_path.
    filter_good_only : bool
        If True, only plot spikes from clusters labeled as 'good'
    time_range_ms : tuple of (float, float), optional
        Time range to plot in milliseconds (start_ms, end_ms). If None, plots all spikes.
    max_spikes_per_cluster : int, optional
        Maximum number of spikes to plot per cluster (for performance with large datasets).
        If None, plots all spikes.
    cluster_ids : np.ndarray, optional
        Specific cluster IDs to plot. If None, plots all good clusters (or all if filter_good_only=False).
    width : int
        Plot width in pixels
    height : int
        Plot height in pixels
    title : str, optional
        Plot title. If None, auto-generates from block info.
    to_browser : bool
        If True, shows plot in browser. If False, returns figure without showing.
    show_cluster_labels : bool
        If True, shows cluster ID labels in legend
        
    Returns
    -------
    figure
        Bokeh figure object
    """
    # Load and align spike data
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
    
    # Filter by time range if specified
    if time_range_ms is not None:
        start_ms, end_ms = time_range_ms
        mask = (spike_times_ms >= start_ms) & (spike_times_ms <= end_ms)
        spike_times_ms = spike_times_ms[mask]
        spike_clusters = spike_clusters[mask]
    
    if len(spike_times_ms) == 0:
        raise ValueError("No spikes found after filtering. Check time_range_ms and cluster_ids parameters.")
    
    # Get unique clusters and sort them
    unique_clusters = np.sort(np.unique(spike_clusters))
    n_clusters = len(unique_clusters)
    
    # Create color palette
    if n_clusters <= 10:
        palette = Category10[10]
    elif n_clusters <= 20:
        palette = Category20[20]
    else:
        # For more than 20 clusters, cycle through colors
        palette = Category20[20] * ((n_clusters // 20) + 1)
    
    # Prepare data for plotting
    # Each cluster gets a y-position (cluster index)
    cluster_to_ypos = {cid: i for i, cid in enumerate(unique_clusters)}
    
    # Subsample if requested
    if max_spikes_per_cluster is not None:
        np.random.seed(42)  # For reproducibility
        indices_to_plot = []
        for cid in unique_clusters:
            cluster_mask = spike_clusters == cid
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) > max_spikes_per_cluster:
                selected = np.random.choice(cluster_indices, size=max_spikes_per_cluster, replace=False)
                indices_to_plot.extend(selected)
            else:
                indices_to_plot.extend(cluster_indices)
        indices_to_plot = np.array(indices_to_plot)
        spike_times_ms_plot = spike_times_ms[indices_to_plot]
        spike_clusters_plot = spike_clusters[indices_to_plot]
    else:
        spike_times_ms_plot = spike_times_ms
        spike_clusters_plot = spike_clusters
    
    # Create y positions
    y_positions = np.array([cluster_to_ypos[cid] for cid in spike_clusters_plot])
    
    # Generate title if not provided
    if title is None:
        title = f"Spike Raster Plot - {block.animal_call} Block {block.block_num}"
        if filter_good_only:
            title += " (Good Units Only)"
        if time_range_ms is not None:
            title += f" [{time_range_ms[0]:.1f}-{time_range_ms[1]:.1f} ms]"
    
    # Create Bokeh figure
    reset_output()
    if to_browser:
        output_notebook()
    
    p = figure(
        width=width,
        height=height,
        title=title,
        x_axis_label="Time (ms from OE recording start)",
        y_axis_label="Cluster ID",
        tools="pan,box_zoom,wheel_zoom,reset,save",
    )
    
    # Plot spikes for each cluster
    for i, cid in enumerate(unique_clusters):
        cluster_mask = spike_clusters_plot == cid
        if not np.any(cluster_mask):
            continue
        
        cluster_times = spike_times_ms_plot[cluster_mask]
        cluster_y = y_positions[cluster_mask]
        
        # Create ColumnDataSource for hover tooltips
        source = ColumnDataSource(data=dict(
            x=cluster_times,
            y=cluster_y,
            time_ms=cluster_times,
            cluster_id=np.full(len(cluster_times), cid, dtype=int),
        ))
        
        cluster_label = f"Cluster {cid}" if show_cluster_labels else f"C{cid}"
        p.scatter(
            x="x",
            y="y",
            size=2,
            alpha=0.6,
            color=palette[i % len(palette)],
            legend_label=cluster_label,
            source=source,
        )
    
    # Add hover tool
    p.add_tools(HoverTool(
        tooltips=[
            ("Time (ms)", "@time_ms{0.2f}"),
            ("Cluster ID", "@cluster_id"),
        ],
        mode='mouse'
    ))
    
    # Set y-axis ticks to cluster IDs
    p.yaxis.ticker = list(range(n_clusters))
    p.yaxis.major_label_overrides = {i: str(cid) for i, cid in enumerate(unique_clusters)}
    
    # Configure legend
    p.legend.click_policy = "hide"
    p.legend.location = "top_right"
    
    if to_browser:
        show(p)
    
    return p


def plot_spike_raster_by_cluster(
    block,
    kilosort_path: Optional[Path] = None,
    filter_good_only: bool = True,
    time_range_ms: Optional[Tuple[float, float]] = None,
    cluster_ids: Optional[np.ndarray] = None,
    width: int = 1200,
    height_per_cluster: int = 100,
    title: Optional[str] = None,
    to_browser: bool = True,
) -> List[figure]:
    """
    Create separate raster plots for each cluster (one subplot per cluster).
    
    This is useful for detailed inspection of individual clusters.
    
    Parameters
    ----------
    block : BlockSync
        BlockSync object with initialized oe_rec attribute
    kilosort_path : Path, optional
        Path to kilosort output folder. If None, will look for spikeSorting/kilosort
        folder within the block's oe_path.
    filter_good_only : bool
        If True, only plot spikes from clusters labeled as 'good'
    time_range_ms : tuple of (float, float), optional
        Time range to plot in milliseconds (start_ms, end_ms). If None, plots all spikes.
    cluster_ids : np.ndarray, optional
        Specific cluster IDs to plot. If None, plots all good clusters.
    width : int
        Plot width in pixels
    height_per_cluster : int
        Height in pixels for each cluster subplot
    title : str, optional
        Base title for plots. Cluster ID will be appended.
    to_browser : bool
        If True, shows plots in browser. If False, returns figures without showing.
        
    Returns
    -------
    list of figure
        List of Bokeh figure objects, one per cluster
    """
    # Load and align spike data
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
    
    # Filter by time range if specified
    if time_range_ms is not None:
        start_ms, end_ms = time_range_ms
        mask = (spike_times_ms >= start_ms) & (spike_times_ms <= end_ms)
        spike_times_ms = spike_times_ms[mask]
        spike_clusters = spike_clusters[mask]
    
    if len(spike_times_ms) == 0:
        raise ValueError("No spikes found after filtering.")
    
    # Get unique clusters and sort them
    unique_clusters = np.sort(np.unique(spike_clusters))
    
    # Generate base title if not provided
    if title is None:
        title_base = f"Spike Raster - {block.animal_call} Block {block.block_num}"
    else:
        title_base = title
    
    reset_output()
    if to_browser:
        output_notebook()
    
    figures = []
    for cid in unique_clusters:
        cluster_mask = spike_clusters == cid
        cluster_times = spike_times_ms[cluster_mask]
        
        if len(cluster_times) == 0:
            continue
        
        p = figure(
            width=width,
            height=height_per_cluster,
            title=f"{title_base} - Cluster {cid} ({len(cluster_times)} spikes)",
            x_axis_label="Time (ms from OE recording start)",
            y_axis_label="",
            tools="pan,box_zoom,wheel_zoom,reset,save",
        )
        
        # Plot spikes as vertical lines (raster style)
        for t in cluster_times:
            p.line([t, t], [0, 1], line_width=1, color='black', alpha=0.7)
        
        # Add hover tool
        source = ColumnDataSource(data=dict(
            x=cluster_times,
            y=np.zeros(len(cluster_times)),
            time_ms=cluster_times,
            cluster_id=np.full(len(cluster_times), cid, dtype=int),
        ))
        
        p.scatter(
            x="x",
            y="y",
            size=3,
            alpha=0.8,
            color='black',
            source=source,
        )
        
        p.add_tools(HoverTool(
            tooltips=[
                ("Time (ms)", "@time_ms{0.2f}"),
                ("Cluster ID", "@cluster_id"),
            ],
            mode='mouse'
        ))
        
        p.yaxis.visible = False
        p.ygrid.visible = False
        
        figures.append(p)
    
    if to_browser:
        for p in figures:
            show(p)
    
    return figures
