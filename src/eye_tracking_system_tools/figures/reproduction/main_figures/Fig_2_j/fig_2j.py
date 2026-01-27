"""
Reproduce the horizontal tuning per-animal plot by:
1) Loading a pickle that contains the original inputs
2) Unpacking into synced_df and non_synced_df
3) Running analyze_orientation_tuning()
4) Saving the PDF

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import datetime
from pathlib import Path

# Match your usual matplotlib defaults (optional but good practice)
plt.style.use("default")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def calculate_orientation_tuning(saccade_angles):
    saccade_angles = np.array(saccade_angles) % 360

    horizontal_ranges = [
        (315, 360),
        (0, 45),
        (135, 225),
    ]

    is_horizontal = np.logical_or.reduce([
        (saccade_angles >= low) & (saccade_angles <= high)
        if low < high else
        (saccade_angles >= low) | (saccade_angles <= high)
        for low, high in horizontal_ranges
    ])

    if len(saccade_angles) == 0:
        return np.nan

    p_horizontal = np.sum(is_horizontal) / len(saccade_angles)
    p_vertical = 1 - p_horizontal
    tuning_statistic = (p_horizontal - p_vertical) / (p_horizontal + p_vertical)  # denom = 1
    return float(tuning_statistic)


def analyze_orientation_tuning(synced_df, non_synced_df, export_path=None):
    tuning_results = {}
    unique_animals = synced_df["animal"].unique()

    for animal in unique_animals:
        synced_angles = synced_df.query("animal == @animal")["overall_angle_deg"].values
        non_synced_angles = non_synced_df.query("animal == @animal")["overall_angle_deg"].values

        synced_tuning = calculate_orientation_tuning(synced_angles)
        non_synced_tuning = calculate_orientation_tuning(non_synced_angles)

        tuning_results[animal] = (synced_tuning, non_synced_tuning)

    # Plot
    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)

    synced_tuning_values = [v[0] for v in tuning_results.values()]
    non_synced_tuning_values = [v[1] for v in tuning_results.values()]
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_animals)))

    for i, animal in enumerate(unique_animals):
        ax.scatter(
            synced_tuning_values[i],
            non_synced_tuning_values[i],
            color=colors[i],
            label=f"{animal}",
            s=30
        )

    ax.set_xlabel("Concurrent [A.U]", fontsize=8)
    ax.set_ylabel("Monocular [A.U]", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.7)

    all_values = synced_tuning_values + non_synced_tuning_values
    finite_vals = [v for v in all_values if np.isfinite(v)]
    axis_limit = (max(abs(min(finite_vals)), abs(max(finite_vals))) * 1.1) if finite_vals else 1.0
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)

    plt.tight_layout()

    # Export PDF only (no re-pickling)
    if export_path:
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = export_dir / f"figure_2j.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"Saved plot to: {pdf_path}")

    plt.show()
    return tuning_results


def unpack_inputs(obj):
    """
    Tries a few common pickle formats to recover (synced_df, non_synced_df).

    Supported:
    1) dict with keys: ('synced_df','non_synced_df') or ('synced','non_synced')
    2) tuple/list of length 2: (synced_df, non_synced_df)
    """
    if isinstance(obj, dict):
        if "synced_df" in obj and "non_synced_df" in obj:
            return obj["synced_df"], obj["non_synced_df"]
        if "synced" in obj and "non_synced" in obj:
            return obj["synced"], obj["non_synced"]

    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return obj[0], obj[1]

    raise TypeError(
        "Don't know how to unpack this pickle into (synced_df, non_synced_df). "
        "Expected dict with keys synced_df/non_synced_df (or synced/non_synced), "
        "or a (synced_df, non_synced_df) tuple/list."
    )


def main():
    SCRIPT_DIR = Path(__file__).parent
    PICKLE_FILE = SCRIPT_DIR / "saccade_angles_data.pkl"   
    OUTPUT_DIR = SCRIPT_DIR                                

    with open(PICKLE_FILE, "rb") as f:
        payload = pickle.load(f)

    synced_df, non_synced_df = unpack_inputs(payload)

    # Run the original analysis/plotting function
    analyze_orientation_tuning(synced_df, non_synced_df, export_path=str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
