#!/usr/bin/env python3
"""S8 species traces, mouse 2c/2d/2e, and lizard–mouse 2f from frozen pickles.

Does not read recording blocks. S8a/c/e eye-video stills are static PNGs.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

PLOTS = ROOT / "plots"


def _copy_if(src: Path, dest: Path) -> None:
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


if __name__ == "__main__":
    for species, letter in (("lizard", "b"), ("mouse", "d"), ("turtle", "f")):
        run_replot(ROOT, "species_trace", f"trace_{species}_series.pkl")
        _copy_if(PLOTS / f"trace_{species}.pdf", PLOTS / f"S8{letter}_trace_{species}.pdf")

    iso = ROOT / "mouse_2c_2d_isolated"
    run_replot(iso, "pos_vel")
    for src_name, dest_name in (
        ("figure_2c_isolated.pdf", "S8g_figure_2c_isolated.pdf"),
        ("figure_2d_isolated.pdf", "S8h_figure_2d_isolated.pdf"),
        ("figure_2c.pdf", "S8g_figure_2c_isolated.pdf"),
        ("figure_2d.pdf", "S8h_figure_2d_isolated.pdf"),
    ):
        _copy_if(iso / "plots" / src_name, PLOTS / dest_name)

    e2 = ROOT / "mouse_figure_2e"
    run_replot(e2, "figure_2e")
    _copy_if(e2 / "plots" / "figure_2e.pdf", PLOTS / "S8i_figure_2e.pdf")

    s8j = ROOT / "S8j"
    run_replot(s8j, "figure_2f_lizard_mouse")
