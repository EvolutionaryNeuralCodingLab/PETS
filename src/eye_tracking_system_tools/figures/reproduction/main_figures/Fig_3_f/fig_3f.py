"""
Reproduce the plot created by compare_animals_difference_histogram_zscore_colored()
from the exported pickle: animal_diff_data_zscore.pkl

This pickle stores per-animal y-traces only, so this script rebuilds the x-axis
(bin centers) from x_range and num_bins (must match the export run).

It also recreates the separate legend PDF.

Usage:
- Put this script in the same folder as animal_diff_data_zscore.pkl
  (or edit PICKLE_FILE below).
- Edit NUM_BINS and X_RANGE to match the export call that generated the pickle.
- Run: python reproduce_pupil_diff_zscore.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from cycler import cycler


# --- Match your defaults ---
plt.style.use("default")
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


# -------------------------
# EDIT THESE
# -------------------------
SCRIPT_DIR = Path(__file__).parent
PICKLE_FILE = SCRIPT_DIR / "animal_diff_data_zscore.pkl"

OUTPUT_PDF = SCRIPT_DIR / "figure_3f.pdf"
OUTPUT_LEGEND_PDF = SCRIPT_DIR / "figure_3f_legend.pdf"

NUM_BINS = 15
X_RANGE = (-3, 3)

FIGURE_SIZE = (2.2, 1.7)
FONT_FAMILY = "Arial"

COLOR_TEMPLATE_OKABEITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                          "#F0E442", "#56B4E9", "#E69F00", "#000000"]


def build_color_map(labels, template_colors=None, order=None):
    labs = list(order) if order is not None else list(labels)
    base = list(template_colors) if template_colors is not None else list(COLOR_TEMPLATE_OKABEITO)
    return {lab: base[i % len(base)] for i, lab in enumerate(labs)}


def main():
    # ---- load y-traces ----
    with open(PICKLE_FILE, "rb") as f:
        animal_diffs = pickle.load(f)

    if not isinstance(animal_diffs, dict) or len(animal_diffs) == 0:
        raise ValueError("Pickle did not contain a non-empty dict of {animal: diff_trace}.")

    # ---- x-axis from bins ----
    bin_edges = np.linspace(float(X_RANGE[0]), float(X_RANGE[1]), int(NUM_BINS) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # ---- animal order (match original: sorted keys) ----
    animals = sorted(animal_diffs.keys())

    # ---- colors (match original default template behavior) ----
    cmap = build_color_map(animals, template_colors=COLOR_TEMPLATE_OKABEITO, order=animals)

    # ---- plotting ----
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [FONT_FAMILY]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=300)
    legend_handles, legend_labels = [], []

    for animal in animals:
        y = np.asarray(animal_diffs[animal], dtype=np.float64)
        h, = ax.plot(
            bin_centers, y,
            linewidth=1.5,
            color=cmap[animal],
            label=animal
        )
        legend_handles.append(h)
        legend_labels.append(animal)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.7)

    ax.set_xlim(float(X_RANGE[0]), float(X_RANGE[1]))
    ax.set_xlabel("Z-scored Pupil Diameter", fontsize=10)
    ax.set_ylabel("Probability Difference", fontsize=10)

    # spines / ticks exactly like original
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=True)
    ax.grid(False)

    plt.tight_layout()

    # ---- save main plot ----
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved main plot: {OUTPUT_PDF}")

    # ---- save separate legend ----
    if legend_handles:
        fig_leg = plt.figure(figsize=(2.0, 0.28 * max(1, len(legend_labels)) + 0.4), dpi=300)
        fig_leg.legend(
            legend_handles, legend_labels,
            loc="center",
            frameon=False,
            ncol=1,
            prop={"size": 8}
        )
        fig_leg.savefig(OUTPUT_LEGEND_PDF, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig_leg)
        print(f"Saved legend: {OUTPUT_LEGEND_PDF}")

    print(f"Loaded pickle: {PICKLE_FILE}")


if __name__ == "__main__":
    main()
