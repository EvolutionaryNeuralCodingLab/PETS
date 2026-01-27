"""
Standalone reproduction script for:
plot_difference_saccade_amplitude_distribution_all_animals()

Loads:
    saccade_amplitude_difference_all_animals_data.pkl
and recreates the plot using the stored per-animal difference traces + bins.

Usage:
- Put this script in the same folder as the .pkl (or edit PICKLE_FILE below)
- Run: python reproduce_saccade_amp_diff_all_animals.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path


# --- Match your typical figure settings (safe + consistent) ---
plt.style.use("default")
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


# --- Paths (edit if needed) ---
SCRIPT_DIR = Path(__file__).parent
PICKLE_FILE = SCRIPT_DIR / "saccade_amplitude_difference_all_animals_data.pkl"
OUTPUT_PDF = SCRIPT_DIR / "fig_2g_bot.pdf"


def main():
    # ---- load ----
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)

    animal_diff_traces = data["animal_diff_traces"]  # dict: animal -> diff_trace array
    bins = np.asarray(data["bins"], dtype=np.float64)

    # Bin centers exactly like the original function
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # IMPORTANT:
    # The original function's default is figure_size=(2, 1.5),
    # but your example call used (1.7, 1).
    # The pickle does not store figure_size, so choose here to match the original export.
    figure_size = (1.7, 1.0)  # change to (2.0, 1.5) if that was used when exporting

    fig, ax = plt.subplots(figsize=figure_size, dpi=300)

    # Match matplotlib default color cycle usage in the original code
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot each animal's trace in the saved order (dict insertion order preserved in modern Python)
    for i, (animal, diff_trace) in enumerate(animal_diff_traces.items()):
        diff_trace = np.asarray(diff_trace, dtype=np.float64)

        ax.plot(
            bin_centers,
            diff_trace,
            color=colors[i % len(colors)],
            linestyle="-",
            linewidth=1.5,
            label=f"{animal}",
        )

    # Horizontal reference at zero (exact from original)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    # Styling (copied from your original)
    ax.set_xlabel("Amplitude [deg]", fontsize=10)
    ax.set_ylabel("Diff [probability]", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_xlim(left=0)

    # Note: original code had legend disabled; keep it disabled for exact match.
    # If you want it, uncomment:
    # ax.legend(fontsize=8)

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Loaded: {PICKLE_FILE}")
    print(f"Saved : {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
