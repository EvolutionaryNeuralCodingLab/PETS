"""
Reproduction script for Figure 2h: Heatmaps of eye-endpoint densities.

This script loads the exported pickle file and recreates the heatmaps comparing
synced vs non-synced saccade trajectories.

EXACT REPLICATION of plot_saccade_heatmaps_comparison_trajectories_angular function.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import rcParams
from pathlib import Path

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


def main():
    """Main function to reproduce Figure 2h."""
    script_dir = Path(__file__).parent
    pickle_file = script_dir / "figure_2h.pickle"
    output_path = script_dir / "figure_2h.pdf"
    colorbar_path = script_dir / "figure_2h_colorbar.pdf"
    
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data for {data.get('figure_name', 'Figure 2h')}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {pickle_file}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract data
    synced = data.get('synced', {})
    non_synced = data.get('non_synced', {})
    nbins = data.get('nbins', 200)
    global_min = data.get('global_min', -25)
    global_max = data.get('global_max', 25)
    
    # Extract trajectory data for each eye and condition
    r_x_s = synced.get('R', {}).get('x', np.array([]))
    r_y_s = synced.get('R', {}).get('y', np.array([]))
    l_x_s = synced.get('L', {}).get('x', np.array([]))
    l_y_s = synced.get('L', {}).get('y', np.array([]))
    
    r_x_ns = non_synced.get('R', {}).get('x', np.array([]))
    r_y_ns = non_synced.get('R', {}).get('y', np.array([]))
    l_x_ns = non_synced.get('L', {}).get('x', np.array([]))
    l_y_ns = non_synced.get('L', {}).get('y', np.array([]))
    
    # Convert to numpy arrays if needed
    def to_array(x):
        if isinstance(x, np.ndarray):
            return x
        return np.array(x) if x else np.array([])
    
    r_x_s = to_array(r_x_s)
    r_y_s = to_array(r_y_s)
    l_x_s = to_array(l_x_s)
    l_y_s = to_array(l_y_s)
    r_x_ns = to_array(r_x_ns)
    r_y_ns = to_array(r_y_ns)
    l_x_ns = to_array(l_x_ns)
    l_y_ns = to_array(l_y_ns)
    
    # Evaluate KDE (EXACT from original)
    def evaluate_kde(x, y):
        if len(x) == 0 or len(y) == 0:
            return None, None, None
        k = gaussian_kde(np.vstack([x, y]).astype(float))
        xi, yi = np.mgrid[global_min:global_max:nbins*1j,
                          global_min:global_max:nbins*1j]
        zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        zi /= np.sum(zi)  # Normalize to probability
        return xi, yi, zi
    
    xi_r_s, yi_r_s, zi_r_s = evaluate_kde(r_x_s, r_y_s)
    xi_l_s, yi_l_s, zi_l_s = evaluate_kde(l_x_s, l_y_s)
    xi_r_ns, yi_r_ns, zi_r_ns = evaluate_kde(r_x_ns, r_y_ns)
    xi_l_ns, yi_l_ns, zi_l_ns = evaluate_kde(l_x_ns, l_y_ns)
    
    # Shared color scale (EXACT from original)
    all_zi = [z for z in [zi_r_s, zi_l_s, zi_r_ns, zi_l_ns] if z is not None]
    vmin, vmax = (min(z.min() for z in all_zi), max(z.max() for z in all_zi)) if all_zi else (0, 1)
    
    # Create figure (EXACT from original: figsize=(4, 2.5))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4, 2.5), dpi=300, sharey=True, sharex=True)
    
    # Use turbo colormap (EXACT from original)
    cmap = plt.cm.turbo
    
    # Plot heatmap function (EXACT from original)
    def plot_heatmap(ax, xi, yi, zi, title=None):
        if zi is not None:
            ax.imshow(
                zi.T, extent=[global_min, global_max, global_min, global_max],
                origin='lower', cmap=cmap, vmin=vmin, vmax=vmax
            )
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=8, transform=ax.transAxes)
        if title:
            ax.set_title(title, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_aspect('equal', 'box')
    
    # Plot each panel (EXACT from original - no titles in original)
    plot_heatmap(axes[0, 0], xi_r_s, yi_r_s, zi_r_s)  # Right (Synced)
    plot_heatmap(axes[0, 1], xi_l_s, yi_l_s, zi_l_s)  # Left (Synced)
    plot_heatmap(axes[1, 0], xi_r_ns, yi_r_ns, zi_r_ns)  # Right (Non-Synced)
    plot_heatmap(axes[1, 1], xi_l_ns, yi_l_ns, zi_l_ns)  # Left (Non-Synced)
    
    # Note: Original code has commented-out axis labels, but we'll add minimal labels
    # for clarity while maintaining the exact plot structure
    
    plt.tight_layout()
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Create standalone colorbar (not in original but user requested)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label("Probability", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig_cbar.savefig(colorbar_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig_cbar)
    
    print(f"Figure 2h saved to {output_path}")
    print(f"Colorbar saved to {colorbar_path}")
    print("Figure 2h reproduction complete!")


if __name__ == "__main__":
    main()
