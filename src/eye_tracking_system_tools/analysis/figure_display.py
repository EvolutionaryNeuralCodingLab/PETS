"""Shared helpers for saving / optionally displaying matplotlib figures."""

from __future__ import annotations

import io
from contextvars import ContextVar
from pathlib import Path

import matplotlib.pyplot as plt

# When set (notebook Build button), previews are collected here instead of
# published via IPython.display — widget click handlers otherwise fan the same
# display_data / stdout message out many times in the Output widget.
_preview_pngs: ContextVar[list[bytes] | None] = ContextVar(
    "figure_preview_pngs", default=None
)


def capture_figure_previews() -> list[bytes]:
    """Start collecting PNGs from :func:`show_and_close`; return the live list."""
    bucket: list[bytes] = []
    _preview_pngs.set(bucket)
    return bucket


def stop_capturing_figure_previews() -> None:
    _preview_pngs.set(None)


def _clear_inline_draw_flag() -> None:
    """Prevent matplotlib_inline ``flush_figures`` from re-emitting closed plots.

    ipywidgets button callbacks run inside a comm ``handle_msg`` that triggers
    IPython ``post_execute``, which calls ``flush_figures``.  With the inline
    backend in interactive mode, that re-displays every still-open pyplot
    figure — on top of any explicit ``display(fig)`` — producing duplicates.
    """
    try:
        from matplotlib_inline.backend_inline import show as inline_show

        inline_show._draw_called = False
        inline_show._to_draw = []
    except Exception:  # noqa: BLE001 — optional dependency / non-notebook use
        pass


def show_and_close(fig, show: bool = False) -> None:
    """Display ``fig`` in a notebook when ``show`` is True, then close it.

    Renders a static PNG and closes the pyplot figure *before* publishing so
    widget-driven ``post_execute`` / ``flush_figures`` cannot emit a second copy.
    Inside :func:`capture_figure_previews`, the PNG is stored for a single
    explicit append to the widget Output.
    """
    png: bytes | None = None
    if show:
        buf = io.BytesIO()
        dpi = getattr(fig, "dpi", None) or plt.rcParams.get("figure.dpi", 100)
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        png = buf.getvalue()

    plt.close(fig)
    _clear_inline_draw_flag()

    if png is None:
        return
    bucket = _preview_pngs.get()
    if bucket is not None:
        bucket.append(png)
        return
    from IPython.display import Image, display

    display(Image(data=png))


def save_figure(fig, out_pdf: Path, *, show: bool = False, **savefig_kw) -> Path:
    """Save ``fig`` to ``out_pdf`` and optionally display it before closing."""
    from eye_tracking_system_tools.analysis.paper_mpl_style import apply_paper_style

    apply_paper_style()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    kw = {"format": "pdf", "bbox_inches": "tight"}
    kw.update(savefig_kw)
    if "format" not in savefig_kw and out_pdf.suffix.lower() == ".pdf":
        pass  # default already set
    fig.savefig(out_pdf, **kw)
    show_and_close(fig, show)
    return out_pdf
