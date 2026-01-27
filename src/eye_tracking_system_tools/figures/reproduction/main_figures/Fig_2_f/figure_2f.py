"""
Reproduction script for Figure 2f (no-downcast pickle).

- Uses float64 edges/counts from pickle
- Uses deterministic pcolormesh settings to avoid seams
- Optionally rasterizes only the heatmap (recommended for viewer consistency)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from pathlib import Path


plt.style.use("default")
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42


def _make_custom_turbo():
    turbo = plt.get_cmap("turbo", 256)
    turbo_colors = turbo(np.linspace(0, 1, 256))
    turbo_colors[0] = np.array([1, 1, 1, 1])  # white at zero
    return mcolors.ListedColormap(turbo_colors)


def _plot_panel(ax, hist, rng, title, n_ticks, tick_list, cmap, rasterize_mesh=True):
    xedges = np.asarray(hist["xedges"], dtype=np.float64)
    yedges = np.asarray(hist["yedges"], dtype=np.float64)
    norm_counts = np.asarray(hist["norm_counts"], dtype=np.float64)

    minv, maxv = rng
    vmax_panel = float(np.nanmax(norm_counts)) if norm_counts.size else 1.0
    if not np.isfinite(vmax_panel) or vmax_panel <= 0:
        vmax_panel = 1.0

    mesh = ax.pcolormesh(
        xedges, yedges, norm_counts.T,
        cmap=cmap,
        vmin=0, vmax=vmax_panel,
        shading="flat",          # deterministic (avoid 'auto' differences)
        antialiased=False,       # reduce seams
        edgecolors="none",
        linewidth=0,
    )
    if rasterize_mesh:
        # keeps axes/text/vector, but makes the heatmap robust against seam artifacts
        mesh.set_rasterized(True)

    ax.set_box_aspect(1)
    ax.set_xlim(minv, maxv)
    ax.set_ylim(minv, maxv)

    if tick_list is not None:
        ax.set_xticks(tick_list)
        ax.set_yticks(tick_list)
    else:
        ticks = np.linspace(minv, maxv, n_ticks)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.plot([minv, maxv], [minv, maxv], ls="--", color="gray", lw=1)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlabel("Right max V [deg/ms]", fontsize=9)
    ax.set_ylabel("Left max V [deg/ms]", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    return mesh


def main():
    script_dir = Path(__file__).parent
    pickle_file = script_dir / "figure_2f_nodowncast.pickle"
    output_path = script_dir / "figure_2f_nodowncast.pdf"
    colorbar_path = script_dir / "figure_2f_nodowncast_colorbar.pdf"

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded: {data.get('figure_name', 'figure_2f')}")
    if data.get("versions"):
        print("Saved with versions:", data["versions"])

    macro_hist = data["macro"]
    micro_hist = data["micro"]
    macro_range = tuple(data["macro_range"])
    micro_range = tuple(data["micro_range"])
    macro_n_ticks = int(data["macro_n_ticks"])
    micro_n_ticks = int(data["micro_n_ticks"])
    macro_tick_list = data.get("macro_tick_list")
    micro_tick_list = data.get("micro_tick_list")
    vmax_all = float(data.get("vmax_all", 1.0))

    cmap = _make_custom_turbo()

    fig, axs = plt.subplots(1, 2, figsize=(3, 1.7), dpi=300, constrained_layout=True)

    _plot_panel(
        axs[0], macro_hist, macro_range, "Macro",
        macro_n_ticks, macro_tick_list, cmap,
        rasterize_mesh=True,
    )
    _plot_panel(
        axs[1], micro_hist, micro_range, "Micro",
        micro_n_ticks, micro_tick_list, cmap,
        rasterize_mesh=True,
    )

    # Save main figure
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Standalone colorbar (matches your original pattern)
    if not np.isfinite(vmax_all) or vmax_all <= 0:
        vmax_all = 1.0

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax_all))
    sm.set_array([])

    fig_cbar = plt.figure(figsize=(1.2, 3.2), dpi=150)
    cax = fig_cbar.add_axes([0.35, 0.1, 0.2, 0.8])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("Probability", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig_cbar.savefig(colorbar_path, bbox_inches="tight", dpi=150)
    plt.close(fig_cbar)
    print(f"Saved: {colorbar_path}")


if __name__ == "__main__":
    main()
