"""Shared helpers for saving / optionally displaying matplotlib figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def show_and_close(fig, show: bool = False) -> None:
    """Display ``fig`` in a notebook when ``show`` is True, then close it."""
    if show:
        from IPython.display import display

        display(fig)
    plt.close(fig)


def save_figure(fig, out_pdf: Path, *, show: bool = False, **savefig_kw) -> Path:
    """Save ``fig`` to ``out_pdf`` and optionally display it before closing."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    kw = {"format": "pdf", "bbox_inches": "tight"}
    kw.update(savefig_kw)
    if "format" not in savefig_kw and out_pdf.suffix.lower() == ".pdf":
        pass  # default already set
    fig.savefig(out_pdf, **kw)
    show_and_close(fig, show)
    return out_pdf
