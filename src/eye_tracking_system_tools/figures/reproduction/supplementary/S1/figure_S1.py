#!/usr/bin/env python3
"""S1 Kerr leftover-error bars from metadata/kerr_component_error.pkl.

Redraws S1b and S1c. S1a.pdf is a static Blender crop and is not regenerated.
Does not read recording blocks.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

if __name__ == "__main__":
    run_replot(ROOT, "kerr_component_error")
