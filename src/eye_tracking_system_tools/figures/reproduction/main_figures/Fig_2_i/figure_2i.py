"""
Reproduction script for Figure 2i: Polar distributions of saccade directions.

This script loads the exported pickle file and recreates the polar histograms
showing saccade directions per eye and animal, rotated to align the dominant axis.

EXACT REPLICATION of create_saccade_polar_histogram_rotated_per_eye function.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# Color templates (EXACT from original)
COLOR_TEMPLATES = {
    # Colorblind-safe (Okabe–Ito)
    "okabeito": ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                 "#F0E442", "#56B4E9", "#E69F00", "#000000"],
    # Tableau-10 as an alternative
    "tableau10": [plt.get_cmap("tab10")(i) for i in range(10)],
}

def build_color_map(labels, template="okabeito", custom=None, order=None):
    """
    Return dict {label -> color}. To keep animal colors stable across figures,
    pass a canonical 'order' (list of all animals) or pass an explicit color_map.
    """
    labs = list(order) if order is not None else list(labels)
    base = list(custom) if custom is not None else list(COLOR_TEMPLATES.get(template, COLOR_TEMPLATES["okabeito"]))
    return {lab: base[i % len(base)] for i, lab in enumerate(labs)}


def main():
    """Main function to reproduce Figure 2i."""
    script_dir = Path(__file__).parent
    pickle_file = script_dir / "figure_2i.pickle"
    output_path = script_dir / "figure_2i.pdf"
    
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data for {data.get('figure_name', 'Figure 2i')}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pickle_file}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract data
    per_eye_histograms = data.get('per_eye_histograms', {})
    rotation_catalog = data.get('rotation_catalog', {})
    num_bins = data.get('num_bins', 36)
    baseline_prob = data.get('baseline_prob', 0.005)
    
    if not per_eye_histograms:
        print("Error: No histogram data found in pickle file")
        return
    
    # Animals (sorted for deterministic color assignment) - EXACT from original
    animals = np.array(sorted(per_eye_histograms.keys()))
    if animals.size == 0:
        print("Error: No animals found")
        return
    
    # Build color mapping using okabeito template (EXACT from original)
    cmap_built = build_color_map(animals, template="okabeito", custom=None, order=animals)
    
    # Matplotlib font defaults (EXACT from original)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Create figure with polar subplots (EXACT from original: figsize=(5, 4))
    fig, axs = plt.subplots(1, 2, figsize=(5, 4), dpi=300, subplot_kw=dict(projection='polar'))
    
    # Helper to process one eye (EXACT from original)
    def _process_eye(ax, eye_label):
        for animal in animals:
            if animal not in per_eye_histograms:
                continue
            if eye_label not in per_eye_histograms[animal]:
                continue
            
            eye_data = per_eye_histograms[animal][eye_label]
            
            # Get rotated histogram data
            rotated_centers = eye_data.get('rotated_centers')
            rotated_counts = eye_data.get('rotated_counts')
            n_saccades = eye_data.get('n_saccades', 0)
            
            if rotated_centers is None or rotated_counts is None:
                continue
            
            # Convert to numpy arrays
            rotated_centers = np.asarray(rotated_centers)
            rotated_counts = np.asarray(rotated_counts)
            
            # Close the loop for polar plotting (EXACT from original)
            theta = np.deg2rad(np.r_[rotated_centers, rotated_centers[0]])
            rho = np.r_[rotated_counts, rotated_counts[0]]
            
            ax.plot(theta, rho, lw=1.2, color=cmap_built[animal],
                   label=f"{animal} (n={n_saccades})")
        
        # Baseline shaded band (EXACT from original)
        theta_ring = np.linspace(0, 2 * np.pi, 512)
        ax.fill_between(theta_ring, 0, baseline_prob, alpha=0.15, color='gray', zorder=0)
        
        # Aesthetics (EXACT from original)
        ax.set_facecolor('white')
        ax.set_yticks([])  # cleaner polar
        ax.grid(False)
        ax.set_theta_zero_location('E')  # 0° to the right
        ax.set_theta_direction(-1)  # clockwise increases
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=6, frameon=False)
    
    # Right eye on left subplot; Left eye on right subplot (EXACT from original)
    _process_eye(axs[0], 'R')
    _process_eye(axs[1], 'L')
    axs[0].set_title("Right eye (rotated, axis-based)", fontsize=9, pad=10)
    axs[1].set_title("Left eye (rotated, axis-based)", fontsize=9, pad=10)
    
    plt.tight_layout()
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Figure 2i saved to {output_path}")
    
    # Print rotations (EXACT from original)
    if rotation_catalog:
        print("\n=== Axis-based rotation angles applied (degrees) ===")
        print(" (shortest rotation in [-90, 90] that aligns the dominant axis with 0°/180°)")
        for (animal, eye), rot in sorted(rotation_catalog.items(),
                                         key=lambda x: (x[0][0], x[0][1])):
            print(f"{animal:>10s}  Eye {eye}:  {rot:+7.2f}°")
        print("====================================================\n")
    
    print("Figure 2i reproduction complete!")


if __name__ == "__main__":
    main()
