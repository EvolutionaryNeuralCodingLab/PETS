#!/usr/bin/env python3
"""S9 unified jitter histogram from metadata/unified_jitter.pkl.

Does not read recording blocks.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

if __name__ == "__main__":
    run_replot(ROOT, "unified_jitter")
    src = ROOT / "plots" / "unified_jitter_quantification.pdf"
    dest = ROOT / "plots" / "S9a_unified_jitter.pdf"
    if src.is_file():
        shutil.copy2(src, dest)
