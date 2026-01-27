"""
Reproduction script for Figure 2d: Mean angular-displacement traces.

This script loads the exported pickle bundle and recreates the position plot.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


def main():
    """Main function to reproduce Figure 2d."""
    script_dir = Path(__file__).parent
    pickle_file = script_dir / "pos_vel_by_amp_bins_bundle.pkl"
    output_path = script_dir / "figure_2d.pdf"
    
    try:
        with open(pickle_file, "rb") as f:
            bundle = pickle.load(f)
        print(f"Loaded bundle with {len(bundle.get('animals', {}))} animals")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pickle_file}")
        print("Please place the 'pos_vel_by_amp_bins_bundle.pkl' file in this directory")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get parameters from bundle
    params = bundle.get('params', {})
    animals = bundle.get('animals', {})
    
    # Create figure for position plot
    fig, ax = plt.subplots(figsize=params.get('fig_size', (1.8, 1.8)), dpi=params.get('dpi', 300))
    
    # Plot position traces for each amplitude bin
    for animal, animal_data in animals.items():
        series = animal_data.get('series', [])
        
        for series_item in series:
            label = series_item.get('label', '')
            color = series_item.get('color_rgba', (0, 0, 0, 1))
            pos_center = series_item.get('pos_center')
            pos_lo = series_item.get('pos_lo')
            pos_hi = series_item.get('pos_hi')
            
            if pos_center is not None:
                # Try to get time axis from bundle, or reconstruct it
                time_axis = series_item.get('time_axis') or series_item.get('mov_axis')
                if time_axis is None:
                    # Reconstruct time axis if not present (aligned to onset at 0)
                    time_axis = np.arange(len(pos_center))
                
                ax.plot(time_axis, pos_center, label=label, color=color, linewidth=params.get('lw', 1))
                
                if pos_lo is not None and pos_hi is not None:
                    ax.fill_between(time_axis, pos_lo, pos_hi, alpha=0.2, color=color)
    
    ax.set_xlabel("Time from onset (ms)", fontsize=8)
    position_calc = params.get('position_calc', 'mov_axis')
    if params.get('normalize_position_to_amp', False):
        ax.set_ylabel("Position (norm. amp)", fontsize=8)
    else:
        ax.set_ylabel("Position (deg)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=params.get('dpi', 300))
    plt.close(fig)
    
    print(f"Figure 2d saved to {output_path}")


if __name__ == "__main__":
    main()
