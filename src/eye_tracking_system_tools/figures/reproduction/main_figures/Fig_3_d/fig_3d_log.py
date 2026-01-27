"""
Reproduce: export_inter_saccade_intervals_density_traces_from_blocks()
from exported pickle: ISI_histogram_plotdata.pkl

- Recreates:
  1) Main ISI log plot PDF
  2) Separate legend PDF

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

plt.style.use("default")
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


SCRIPT_DIR = Path(__file__).parent
PICKLE_FILE = SCRIPT_DIR / "ISI_histogram_plotdata.pkl" 

OUTPUT_PDF = SCRIPT_DIR / "ISI_histogram.pdf"
OUTPUT_LEGEND_PDF = SCRIPT_DIR / "legend_ISI_histogram.pdf"


def main():
    with open(PICKLE_FILE, "rb") as f:
        plot_data = pickle.load(f)

    params = plot_data.get("params", {}) if isinstance(plot_data, dict) else {}
    figure_size = tuple(params.get("figure_size", (1.8, 1.2)))
    xlim = params.get("xlim", None)
    ylim = params.get("ylim", None)  # note: your exporter doesn't store ylim in params; kept for compatibility
    font_family = params.get("font_family", "Arial")

    animals = list(plot_data.get("animals", []))
    cmap = plot_data.get("color_map", {})
    per_animal = plot_data.get("per_animal", {})
    combined = plot_data.get("combined", {})

    # Match font settings used in the original plot
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family]

    fig, ax = plt.subplots(figsize=figure_size, dpi=300)

    legend_handles = []
    legend_labels = []

    # --- per-animal traces ---
    for a in animals:
        if a not in per_animal:
            continue
        entry = per_animal[a]
        x = np.asarray(entry.get("x", []), dtype=np.float64)
        y = np.asarray(entry.get("y", []), dtype=np.float64)
        if x.size == 0 or y.size == 0:
            continue

        h, = ax.plot(
            x, y,
            linewidth=1.5,  # matches your log exporter
            color=cmap.get(a, None),
            label=str(a),
        )
        legend_handles.append(h)
        legend_labels.append(str(a))

    # --- combined trace ---
    if isinstance(combined, dict) and combined.get("x", None) is not None and combined.get("y", None) is not None:
        x_all = np.asarray(combined.get("x", []), dtype=np.float64)
        y_all = np.asarray(combined.get("y", []), dtype=np.float64)
        label_all = combined.get("label", "All (combined)")

        if x_all.size and y_all.size:
            # If you need non-default styles to be perfectly reproduced,
            # store avg_color/avg_linewidth/avg_linestyle in plot_data["params"].
            avg_color = params.get("avg_color", "k")
            avg_linewidth = float(params.get("avg_linewidth", 2.0))
            avg_linestyle = params.get("avg_linestyle", "-")

            h_all, = ax.plot(
                x_all, y_all,
                avg_linestyle,
                color=avg_color,
                linewidth=avg_linewidth,
                label=label_all,
                zorder=10,
            )
            legend_handles.insert(0, h_all)
            legend_labels.insert(0, label_all)

    # --- axes & cosmetics (match your log exporter) ---
    ax.set_xscale("log")

    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    else:
        bins = np.asarray(plot_data.get("bins", []), dtype=np.float64)
        if bins.size >= 2:
            ax.set_xlim(float(bins[0]), float(bins[-1]))

    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    ax.set_xlabel("ISI [ms]", fontsize=10)
    ax.set_ylabel("Probability", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=8, length=5, width=1, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0,0.15)
    plt.tight_layout()

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved main plot: {OUTPUT_PDF}")

    # --- separate legend (same sizing formula as your exporter) ---
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
