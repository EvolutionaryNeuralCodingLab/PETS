"""
Reproduction script for Figure 2e: Relationship between saccade amplitude and peak angular speed.

This script loads the exported pickle bundle and recreates the linear fit plot.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams
from pathlib import Path

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


def main():
    """Main function to reproduce Figure 2e."""
    script_dir = Path(__file__).parent
    pickle_file = script_dir / "amplitude_velocity_linear_fit_bundle.pkl"
    output_path = script_dir / "figure_2e.pdf"
    
    try:
        with open(pickle_file, "rb") as f:
            bundle = pickle.load(f)
        print(f"Loaded bundle with linear fit data")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pickle_file}")
        print("Please place the 'amplitude_velocity_linear_fit_bundle.pkl' file in this directory")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get parameters from bundle
    params = bundle.get('params', {})
    per_animal_stats = bundle.get('per_animal_stats', {})
    global_fit = bundle.get('global_fit', {})
    edges = bundle.get('edges', np.array([]))
    
    # Create figure
    figsize = params.get('figsize', (1.5, 1.7))
    dpi = params.get('dpi', 300)
    lw = params.get('lw', 1.5)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Color cycle for animals
    colors = plt.cm.tab10(np.linspace(0, 1, len(per_animal_stats)))
    
    handles_anim = []
    labels_anim = []
    
    # Plot data for each animal
    for i, (animal, stats_df) in enumerate(per_animal_stats.items()):
        if stats_df is None or stats_df.empty:
            continue
        
        # Calculate bin centers
        amp_centers = (stats_df['amp_lo'].values + stats_df['amp_hi'].values) / 2
        mean_peak_v = stats_df['mean_peak_v'].values
        
        color = colors[i]
        line, = ax.plot(amp_centers, mean_peak_v, 'o-', color=color, markersize=3, linewidth=lw, label=animal)
        handles_anim.append(line)
        labels_anim.append(animal)
    
    # Plot global linear fit
    if global_fit:
        slope = global_fit.get('slope', 0)
        intercept = global_fit.get('intercept', 0)
        
        # Get x range from data
        all_amps = []
        for stats_df in per_animal_stats.values():
            if stats_df is not None and not stats_df.empty:
                all_amps.extend(stats_df['amp_lo'].values)
                all_amps.extend(stats_df['amp_hi'].values)
        
        if all_amps:
            x_fit = np.linspace(min(all_amps), max(all_amps), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, '--', color='k', linewidth=lw, label='Linear fit')
    
    ax.set_xlabel('Amplitude [deg]', fontsize=8)
    ax.set_ylabel('Peak V [deg/ms]', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    
    print(f"Figure 2e saved to {output_path}")


if __name__ == "__main__":
    main()
