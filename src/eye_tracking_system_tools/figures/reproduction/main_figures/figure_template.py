"""
Template for figure reproduction scripts.

This template shows the structure for creating reproducible figures.
Each figure script should:
1. Load pre-processed data from the corresponding .pickle file
2. Create the exact figure as it appears in the paper
3. Save the figure to PDF (and optionally other formats)
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style to match paper
plt.style.use('seaborn-v0_8-paper')  # or your preferred style
plt.rcParams['pdf.fonttype'] = 42  # Ensure fonts are embedded
plt.rcParams['ps.fonttype'] = 42


def load_figure_data(pickle_path: Path):
    """
    Load pre-processed data for figure reproduction.
    
    Parameters
    ----------
    pickle_path : Path
        Path to the .pickle file containing figure data
        
    Returns
    -------
    dict
        Dictionary containing all data needed to reproduce the figure
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def create_figure(data: dict, output_path: Path = None):
    """
    Create the figure from pre-processed data.
    
    Parameters
    ----------
    data : dict
        Dictionary containing figure data (loaded from pickle)
    output_path : Path, optional
        Where to save the figure. If None, displays instead.
        
    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects
    """
    # TODO: Implement actual figure creation based on paper
    # This is a template - replace with actual plotting code
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Example: if data contains 'x' and 'y' arrays
    if 'x' in data and 'y' in data:
        ax.plot(data['x'], data['y'], label=data.get('label', 'Data'))
        ax.legend()
        ax.set_xlabel(data.get('xlabel', 'X'))
        ax.set_ylabel(data.get('ylabel', 'Y'))
        ax.set_title(data.get('title', 'Figure'))
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    
    return fig, ax


def main():
    """Main function to reproduce the figure."""
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Load data (pickle file should be in same directory)
    pickle_file = script_dir / f"{Path(__file__).stem}.pickle"
    
    if not pickle_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {pickle_file}\n"
            "Please ensure the .pickle file exists in the same directory."
        )
    
    # Load and create figure
    data = load_figure_data(pickle_file)
    fig, ax = create_figure(data)
    
    # Save figure
    output_file = script_dir / f"{Path(__file__).stem}.pdf"
    create_figure(data, output_path=output_file)
    
    # Optionally display
    # plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    main()
