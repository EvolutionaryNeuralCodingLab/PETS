#!/usr/bin/env python3
"""S12a back-and-forth large-saccade fraction from metadata/double_steps_move_only.pkl.

S12b_example.pdf is a static example trace and is not regenerated.
Does not read recording blocks.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

if __name__ == "__main__":
    run_replot(ROOT, "double_steps_move_only")
    src = ROOT / "plots" / "back_and_forth_overall.pdf"
    dest = ROOT / "plots" / "S12a_overall.pdf"
    if src.is_file():
        shutil.copy2(src, dest)
