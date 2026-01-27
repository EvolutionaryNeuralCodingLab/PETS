"""
Reproduction script for Figure 1e: Histogram of camera movements (jitter quantification)

This figure shows the histogram of camera movements, quantified as percentage of overall 
frames in each bin.

Data: distances.pkl contains a numpy array of displacement distances in micrometers.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# Set style to match paper
rcParams['pdf.fonttype'] = 42  # Ensure fonts are embedded and editable
rcParams['ps.fonttype'] = 42  # Ensure compatibility with vector outputs
plt.style.use('default')


def load_figure_data(pickle_path: Path):
    """
    Load pre-processed data for figure reproduction.
    
    Parameters
    ----------
    pickle_path : Path
        Path to the .pickle file containing figure data
        
    Returns
    -------
    np.ndarray
        Array of displacement distances in micrometers
    """
    with open(pickle_path, 'rb') as f:
        distances = pickle.load(f)
    
    if not isinstance(distances, np.ndarray):
        distances = np.array(distances)
    
    return distances


def create_figure(distances: np.ndarray, output_path: Path = None):
    """
    Create Figure 1e: Histogram of camera movements (jitter quantification).
    
    Parameters
    ----------
    distances : np.ndarray
        Array of displacement distances in micrometers
    output_path : Path, optional
        Where to save the figure. If None, displays instead.
        
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    # Create figure with specific size
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.6), dpi=150)
    
    # Plot the histogram
    # Bins: 15 bins from 0 to 500 μm
    bins = np.linspace(0, 500, 15)
    hist, bins = np.histogram(distances, bins=bins)
    percentage = (hist / len(distances)) * 100
    
    # Use 'gray' for bin fill and 'black' for edges
    ax.bar(bins[:-1], percentage, width=np.diff(bins), 
           color='gray', edgecolor='black', align='edge')
    
    # Set labels
    ax.set_xlabel('Displacement [$\mu$m]', fontsize=10)
    ax.set_ylabel('% frames', fontsize=10)
    
    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='y', which='both', length=3, color='black')
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.tick_params(axis='y', which='major', length=3, width=1, color='black')
    
    # Set white background and black text
    ax.set_facecolor('white')
    ax.title.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(colors='black')
    ax.grid(False)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='y', which='both', length=3.5, color='black')
    ax.tick_params(axis='x', which='both', length=3.5)
    
    # Set x-axis and y-axis limits
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 40)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, format='pdf', dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig, ax


def main():
    """Main function to reproduce Figure 1e."""
    # Get the directory of this script (all files are in the same folder)
    script_dir = Path(__file__).parent
    
    # Load data (pickle file should be in same directory as script)
    pickle_file = script_dir / "distances.pkl"
    
    if not pickle_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {pickle_file}\n"
            "Please ensure the distances.pkl file exists in the Fig_1_e directory."
        )
    
    # Load and create figure
    distances = load_figure_data(pickle_file)
    print(f"Loaded {len(distances)} displacement measurements")
    print(f"Distance range: {distances.min():.2f} - {distances.max():.2f} micrometers")
    
    # Create and save figure
    output_file = script_dir / "figure_1e.pdf"
    fig, ax = create_figure(distances, output_path=output_file)
    
    # Optionally display
    # plt.show()
    
    plt.close(fig)
    print("Figure 1e reproduction complete!")


if __name__ == "__main__":
    main()
