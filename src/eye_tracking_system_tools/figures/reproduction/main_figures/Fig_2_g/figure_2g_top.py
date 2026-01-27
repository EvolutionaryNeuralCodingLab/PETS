"""
Standalone reproduction script for:
plot_averaged_saccade_amplitude_distribution_angle()

This script loads the exported pickle:
    averaged_saccade_amplitude_angle_data.pkl
and recreates the plot exactly from the stored mean/SEM arrays + bins.

Usage:
- Place this script in the same folder as the .pkl, OR edit PICKLE_FILE below.
- Run: python reproduce_saccade_amp_hist_angle.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path


# --- Match your original global plotting settings ---
plt.style.use("default")
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


# --- Paths (edit if needed) ---
SCRIPT_DIR = Path(__file__).parent
PICKLE_FILE = SCRIPT_DIR / "averaged_saccade_amplitude_angle_data.pkl"
OUTPUT_PDF = SCRIPT_DIR / "fig_2g_top.pdf"


def main():
    # ---- load ----
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)

    # ---- extract (no recomputation) ----
    synced_mean = np.asarray(data["synced_mean"], dtype=np.float64)
    synced_sem = np.asarray(data["synced_sem"], dtype=np.float64)
    non_synced_mean = np.asarray(data["non_synced_mean"], dtype=np.float64)
    non_synced_sem = np.asarray(data["non_synced_sem"], dtype=np.float64)
    bins = np.asarray(data["bins"], dtype=np.float64)

    # Bin centers exactly like the original function
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # ---- reproduce figure ----
    # IMPORTANT:
    # Your original function's default is figure_size=(2.5, 1.7),
    # but in your example call you used (1.7, 1.2).
    # Since the pickle doesn't store figure_size, choose here.
    figure_size = (1.7, 1.2)  # change to (2.5, 1.7) if that was the exported one

    fig, ax = plt.subplots(figsize=figure_size, dpi=300)

    # Synced
    ax.plot(
        bin_centers, synced_mean,
        color="green", linestyle="-", linewidth=1.5,
        label="Synchronized"
    )
    ax.fill_between(
        bin_centers,
        synced_mean - synced_sem,
        synced_mean + synced_sem,
        color="green", alpha=0.3
    )

    # Non-synced
    ax.plot(
        bin_centers, non_synced_mean,
        color="blue", linestyle="-", linewidth=1.5,
        label="Monocular"
    )
    ax.fill_between(
        bin_centers,
        non_synced_mean - non_synced_sem,
        non_synced_mean + non_synced_sem,
        color="blue", alpha=0.3
    )

    # Styling (copied from your original)
    ax.set_title("Average Angular Saccade Amplitude Distribution", fontsize=12)
    ax.set_xlabel("Saccade Amplitude [deg]", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ---- save ----
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Loaded: {PICKLE_FILE}")
    print(f"Saved : {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
