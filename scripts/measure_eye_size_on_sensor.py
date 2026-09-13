#!/usr/bin/env python
"""Measure eye size on the sensor for the camera-specs review reply.

Opens a Qt UI that wraps the preprocessing-GUI ROI picker. Mark the eye on one
representative block per species (lizard PV_126, mouse M_002, turtle T_18),
then press Calculate for pixel size and % of the 640×480 frame.

Usage (repo root, eye_repo_mac kernel)::

    PYTHONPATH=src python scripts/measure_eye_size_on_sensor.py

Or::

    PYTHONPATH=src python -m eye_tracking_system_tools.analysis.eye_size_gui
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from eye_tracking_system_tools.analysis.eye_size_gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
