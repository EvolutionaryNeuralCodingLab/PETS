#!/usr/bin/env python3
"""S10 Rayleigh noise cores (lizard / mouse / turtle) from metadata pickles.

Does not read recording blocks.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

PANELS = (
    ("S10a_lizard_rayleigh_noise_core.pkl", "S10a_lizard.pdf"),
    ("S10b_mouse_rayleigh_noise_core.pkl", "S10b_mouse.pdf"),
    ("S10c_turtle_rayleigh_noise_core.pkl", "S10c_turtle.pdf"),
)

if __name__ == "__main__":
    plots = ROOT / "plots"
    for pkl, dest_name in PANELS:
        run_replot(ROOT, "rayleigh_noise_core", pkl)
        src = plots / "rayleigh_noise_core.pdf"
        if src.is_file():
            shutil.copy2(src, plots / dest_name)
