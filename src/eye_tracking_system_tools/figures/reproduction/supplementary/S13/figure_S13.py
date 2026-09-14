#!/usr/bin/env python3
"""S13 pre-onset amplitude main-sequence panels from metadata/s13_preonset_2e.pkl.

Does not read recording blocks.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

if __name__ == "__main__":
    run_replot(ROOT, "s13_preonset_2e")
