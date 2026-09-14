#!/usr/bin/env python3
"""S1: simulation accuracy (S1a) plus leftover-error bars (S1b, S1c).

S1a is redrawn from metadata/ellipse_angle_mapping_correct_diameter_08mm_distance_13mm.csv.
S1b/S1c come from metadata/kerr_component_error.pkl. Does not read recording blocks.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

if __name__ == "__main__":
    from figure_S1a import main as draw_s1a

    draw_s1a()
    run_replot(ROOT, "kerr_component_error")
