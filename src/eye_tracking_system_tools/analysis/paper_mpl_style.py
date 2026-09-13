"""Paper matplotlib style: Type-42 fonts, Arial, Fig 2/3 sizes and colors.

Reproduction scripts under ``figures/reproduction/main_figures/`` set
``pdf.fonttype = 42`` so Adobe Illustrator can edit text. Matplotlib's default
Type 3 outlines glyphs into paths. Call :func:`apply_paper_style` before any
PDF save (and from standalone ``replot.py``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paper Fig 2f / 2c / 2e / colorbar canvases (inches).
FIGSIZE_2F = (3.0, 1.7)
FIGSIZE_2C_2D = (1.8, 1.8)
FIGSIZE_2E = (1.5, 1.7)
FIGSIZE_CBAR = (1.2, 3.2)
FIGSIZE_JITTER = (2.0, 1.6)
FIGSIZE_S3_PANEL = (1.7, 1.7)

TRACE_L = "#1f77b4"
TRACE_R = "#8c564b"
OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
]
SANS_SERIF = ["Arial", "Helvetica", "DejaVu Sans"]

TICK_SIZE_SMALL = 7
LABEL_SIZE_SMALL = 8
LABEL_SIZE_HEATMAP = 9
TITLE_SIZE_SMALL = 8
LABEL_SIZE_HIST = 10
TICK_SIZE_HIST = 8


def apply_paper_style() -> None:
    """Embed TrueType (Type 42) Arial so Illustrator can edit text."""
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = list(SANS_SERIF)
    plt.rcParams["axes.labelsize"] = LABEL_SIZE_SMALL
    plt.rcParams["xtick.labelsize"] = TICK_SIZE_SMALL
    plt.rcParams["ytick.labelsize"] = TICK_SIZE_SMALL
    plt.rcParams["axes.titlesize"] = TITLE_SIZE_SMALL
    plt.rcParams["legend.fontsize"] = LABEL_SIZE_SMALL
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"


def style_axes(
    ax,
    *,
    tick_size: float = TICK_SIZE_SMALL,
    hide_top_right: bool = True,
) -> None:
    """Paper spines and ticks: top/right off, remaining black, ticks out."""
    ax.tick_params(axis="both", labelsize=tick_size, direction="out")
    ax.grid(False)
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(colors="black")


def turbo_white0():
    """Turbo colormap with index 0 mapped to white (paper Fig 2f / S3)."""
    import matplotlib.colors as mcolors

    turbo = plt.get_cmap("turbo", 256)
    colors = turbo(np.linspace(0, 1, 256))
    colors[0] = np.array([1.0, 1.0, 1.0, 1.0])
    return mcolors.ListedColormap(colors)


def savefig_pdf(fig, path: Path | str, *, dpi: int | None = None, **kwargs) -> Path:
    """Save a PDF after re-applying Type 42 embedding."""
    apply_paper_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"format": "pdf", "bbox_inches": "tight"}
    if dpi is not None:
        kw["dpi"] = dpi
    kw.update(kwargs)
    fig.savefig(path, **kw)
    return path


def rasterize_heatmap_collections(ax) -> None:
    """Rasterize pcolormesh/image only; leave axes and text as vectors."""
    for coll in getattr(ax, "collections", []) or []:
        try:
            coll.set_rasterized(True)
        except AttributeError:
            continue
    for im in getattr(ax, "images", []) or []:
        try:
            im.set_rasterized(True)
        except AttributeError:
            continue
