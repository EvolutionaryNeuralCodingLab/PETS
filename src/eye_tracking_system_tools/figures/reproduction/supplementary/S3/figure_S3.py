#!/usr/bin/env python3
"""S3 head-still / head-moving coupling histograms from metadata/figure_S3.pickle.

Does not read recording blocks.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from _run import run_replot

if __name__ == "__main__":
    run_replot(ROOT, "figure_s3")
