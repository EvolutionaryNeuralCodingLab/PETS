#!/usr/bin/env python
"""Count frames rejected by the interframe jitter-diff gate across a dataset.

Walks every block in the registry, loads ``analysis/jitter_report_dict.pkl``,
and counts samples where ``np.diff(top_correlation_dist) > 5`` pixels
(keep only interframe jitter diffs below 6 px). Writes a CSV with per-block,
per-animal, and all-animals totals.

Usage (repo root)::

    PYTHONPATH=src python scripts/overall_jitter_rejection_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from eye_tracking_system_tools.analysis.jitter_rejection_stats import main as stats_main

    return stats_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
