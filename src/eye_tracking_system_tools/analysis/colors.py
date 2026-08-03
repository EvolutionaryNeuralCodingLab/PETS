"""Shared animal / label color maps (Okabe–Ito + Tableau)."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

COLOR_TEMPLATES: dict[str, list] = {
    "okabeito": [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#F0E442",
        "#56B4E9",
        "#E69F00",
        "#000000",
    ],
    "tableau10": [plt.get_cmap("tab10")(i) for i in range(10)],
}


def build_color_map(
    labels: Iterable,
    template: str = "okabeito",
    custom: Sequence | None = None,
    order: Sequence | None = None,
) -> dict:
    """Map each label to a color, cycling the chosen template."""
    labs = list(order) if order is not None else list(labels)
    base = (
        list(custom)
        if custom is not None
        else list(COLOR_TEMPLATES.get(template, COLOR_TEMPLATES["okabeito"]))
    )
    if not base:
        raise ValueError("Empty color template")
    return {lab: base[i % len(base)] for i, lab in enumerate(labs)}


def color_for(
    label,
    color_map: Mapping | None = None,
    *,
    template: str = "okabeito",
    index: int = 0,
):
    """Lookup in ``color_map`` or fall back to template by index."""
    if color_map is not None and label in color_map:
        return color_map[label]
    base = COLOR_TEMPLATES.get(template, COLOR_TEMPLATES["okabeito"])
    return base[index % len(base)]
