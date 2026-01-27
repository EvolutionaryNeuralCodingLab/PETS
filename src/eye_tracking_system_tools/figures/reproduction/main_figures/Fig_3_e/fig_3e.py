"""
Reproduce pupil diameter probability histograms (+ KDE) from:
    combined_aggregated_data.pkl

This script re-implements the SAME bin-edge logic and seaborn histplot call
as plot_combined_eye_probability_histograms(), but uses ONLY the exported pickle.

Put this script next to combined_aggregated_data.pkl (or edit PICKLE_FILE).
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import seaborn as sns


# --- Match your notebook defaults ---
plt.style.use("default")
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


# -------------------------
# Edit these if needed
# -------------------------
SCRIPT_DIR = Path(__file__).parent
PICKLE_FILE = SCRIPT_DIR / "combined_aggregated_data.pkl"
OUTPUT_PDF = SCRIPT_DIR / "figure_3e.pdf"

NUM_BINS = 40
X_RANGE = (1.0, 2.35)
OUTLIER_PERCENTILES = (0.001, 99.999)

FIGSIZE = (2.2, 1.5)
FIG_DPI = 200
SAVE_DPI = 300

COLOR_MAP = {"quiet": "blue", "active": "orange"}  # same as original


def _compute_bin_edges(all_data, num_bins, x_range, outlier_percentiles):
    """
    Copied logic from your function (with minimal refactoring),
    so bin edges match the original behavior.
    """
    if len(all_data) == 0:
        raise ValueError("No pupil data found; nothing to plot.")

    # restrict to x_range first
    all_data_in_range = [v for v in all_data if x_range[0] <= v <= x_range[1]]

    if len(all_data_in_range) == 0:
        raw_min = np.nanmin(all_data)
        raw_max = np.nanmax(all_data)
        print(
            f"[warn] No data within x_range {x_range}. "
            f"Global data span = [{raw_min:.4g}, {raw_max:.4g}]. "
            f"Expanding to a minimal epsilon around global range for plotting."
        )
        outlier_min, outlier_max = np.percentile(all_data, outlier_percentiles)
        start = min(outlier_min, outlier_max)
        stop = max(outlier_min, outlier_max)
        if not np.isfinite(start) or not np.isfinite(stop) or start == stop:
            start, stop = raw_min, raw_max
            if start == stop:
                stop = start + 1e-6
    else:
        outlier_min, outlier_max = np.percentile(all_data_in_range, outlier_percentiles)
        start = max(min(outlier_min, outlier_max), x_range[0])
        stop = min(max(outlier_min, outlier_max), x_range[1])
        if start >= stop:
            if start == stop:
                stop = start + 1e-6
            else:
                start, stop = sorted([start, stop])

    bin_edges = np.linspace(start, stop, num_bins + 1)

    if not np.all(bin_edges[1:] > bin_edges[:-1]):
        eps = np.finfo(float).eps * max(1.0, abs(stop))
        bin_edges = np.linspace(start, stop + eps * (num_bins + 1), num_bins + 1)

    print(
        f"[diag] Final bin range: [{bin_edges[0]:.4f}, {bin_edges[-1]:.4f}] "
        f"({len(bin_edges)-1} bins)."
    )
    print(
        f"[diag] Outlier percentiles used: {outlier_percentiles}. "
        f"Outlier min/max before intersection: {outlier_min:.4f} / {outlier_max:.4f}"
    )
    return bin_edges


def main():
    # ---- load combined_aggregated ----
    with open(PICKLE_FILE, "rb") as f:
        combined_aggregated = pickle.load(f)

    if not isinstance(combined_aggregated, dict):
        raise TypeError("Expected the pickle to contain a dict: {annotation: [values...]}")

    # ---- rebuild all_data exactly like the function (from all annotations) ----
    all_data = []
    for _, vals in combined_aggregated.items():
        if vals is None:
            continue
        # tolerate list/np array/pandas series-like
        try:
            for v in vals:
                if v is None:
                    continue
                if np.isfinite(v):
                    all_data.append(float(v))
        except TypeError:
            # non-iterable
            pass

    # ---- bin edges ----
    bin_edges = _compute_bin_edges(
        all_data=all_data,
        num_bins=NUM_BINS,
        x_range=X_RANGE,
        outlier_percentiles=OUTLIER_PERCENTILES,
    )

    # ---- plot ----
    sns.set(style="white")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)
    plotted_any = False

    for annotation, data in combined_aggregated.items():
        if not data:
            print(f"[info] '{annotation}': no samples; skipping.")
            continue

        filtered_data = [v for v in data if (v is not None and np.isfinite(v) and bin_edges[0] <= v <= bin_edges[-1])]
        if len(filtered_data) == 0:
            print(f"[info] '{annotation}': no samples within bin range; skipping.")
            continue

        sns.histplot(
            filtered_data,
            bins=bin_edges,
            stat="probability",
            element="bars",
            label=f"{annotation}",
            alpha=0.5,
            color=COLOR_MAP.get(annotation, "gray"),
            kde=True,
            ax=ax,
        )
        plotted_any = True
        print(f"[diag] '{annotation}': n={len(filtered_data)}")

    # cosmetics (copied from your function)
    ax.set_xlabel("Pupil diameter [mm]", fontsize=10)
    ax.set_ylabel("Likelihood", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", direction="out", which="major",
                   length=5, width=1, colors="black", bottom=True, left=True)
    ax.tick_params(axis="both", direction="out", which="minor",
                   length=5, width=1, colors="black", bottom=True, left=True)
    ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))

    if plotted_any:
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(
            0.5, 0.5, "No data to plot in selected range",
            transform=ax.transAxes, ha="center", va="center", fontsize=8
        )
        print("[info] No annotations produced plottable data in the final range.")

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", dpi=SAVE_DPI)
    plt.close(fig)

    print(f"Loaded: {PICKLE_FILE}")
    print(f"Saved : {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
