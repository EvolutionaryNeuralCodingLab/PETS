"""
Reproduce: export_inter_saccade_intervals_density_traces_from_blocks_LINEAR()
from the exported pickle: ISI_hist_linear_plotdata.pkl

This script uses ONLY the saved plot_data (no recomputation).
It recreates:
  1) Main ISI plot PDF
  2) Separate legend PDF (like the original exporter)

Place this script in the same folder as the pickle OR edit PICKLE_FILE below.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path


# --- Match your standard global settings ---
plt.style.use("default")
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


SCRIPT_DIR = Path(__file__).parent
PICKLE_FILE = SCRIPT_DIR / "ISI_hist_linear_plotdata.pkl"  

OUTPUT_PDF = SCRIPT_DIR / "ISI_hist_linear_10_300ms.pdf"
OUTPUT_LEGEND_PDF = SCRIPT_DIR / "legend_ISI_hist_linear_10_300ms.pdf"


def main():
    with open(PICKLE_FILE, "rb") as f:
        plot_data = pickle.load(f)

    # ---- Pull params (with sane fallbacks) ----
    params = plot_data.get("params", {}) if isinstance(plot_data, dict) else {}
    figure_size = tuple(params.get("figure_size", (2.0, 1.5)))
    xlim = params.get("xlim", (10.0, 300.0))
    ylim = params.get("ylim", None)
    font_family = params.get("font_family", "Arial")

    animals = list(plot_data.get("animals", []))
    cmap = plot_data.get("color_map", {})  # dict animal->color
    per_animal = plot_data.get("per_animal", {})
    combined = plot_data.get("combined", {})

    # ---- Match font settings used in the original plot ----
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family]

    # ---- Build figure ----
    fig, ax = plt.subplots(figsize=figure_size, dpi=300)

    legend_handles = []
    legend_labels = []

    # ---- Per-animal traces (in the exported order) ----
    for a in animals:
        if a not in per_animal:
            continue
        entry = per_animal[a]
        x = np.asarray(entry.get("x", []), dtype=np.float64)
        y = np.asarray(entry.get("y", []), dtype=np.float64)
        if x.size == 0 or y.size == 0:
            continue

        color = cmap.get(a, None)
        h, = ax.plot(
            x, y,
            linewidth=1.2,
            color=color,
            label=str(a)
        )
        legend_handles.append(h)
        legend_labels.append(str(a))

    # ---- Combined trace (if present) ----
    if isinstance(combined, dict) and combined.get("x", None) is not None and combined.get("y", None) is not None:
        x_all = np.asarray(combined.get("x", []), dtype=np.float64)
        y_all = np.asarray(combined.get("y", []), dtype=np.float64)
        label_all = combined.get("label", "All (combined)")

        if x_all.size and y_all.size:
            # These style defaults match your function defaults
            h_all, = ax.plot(
                x_all, y_all,
                "-", color="k", linewidth=2.0,
                label=label_all,
                zorder=10
            )
            legend_handles.insert(0, h_all)
            legend_labels.insert(0, label_all)

    # ---- Axes / cosmetics (copied from your function) ----
    ax.set_xscale("linear")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        # fallback to bin range if present
        bins = np.asarray(plot_data.get("bins", []), dtype=np.float64)
        if bins.size >= 2:
            ax.set_xlim(float(bins[0]), float(bins[-1]))

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    ax.set_xlabel("ISI [ms]", fontsize=10)
    ax.set_ylabel("Probability (per linear bin)", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=8, length=5, width=1, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()

    # ---- Save main plot ----
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved main plot: {OUTPUT_PDF}")

    # ---- Save separate legend (same sizing formula as your exporter) ----
    if legend_handles:
        fig_leg = plt.figure(figsize=(2.0, 0.28 * max(1, len(legend_labels)) + 0.4), dpi=300)
        fig_leg.legend(
            legend_handles,
            legend_labels,
            loc="center",
            frameon=False,
            ncol=1,
            prop={"size": 8},
        )
        fig_leg.savefig(OUTPUT_LEGEND_PDF, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig_leg)
        print(f"Saved legend: {OUTPUT_LEGEND_PDF}")

    print(f"Loaded pickle: {PICKLE_FILE}")


if __name__ == "__main__":
    main()
