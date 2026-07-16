#!/usr/bin/env python
"""Compute ``lizMov.mat`` from Open Ephys accelerometer data (Python pipeline).

Usage::

    python scripts/compute_lizard_movement.py \\
        --experiment-path /path/to/exp \\
        --animal PV_126 --date 2024_07_18 --block 006 \\
        --calibration-mat /path/to/calibration_results.mat \\
        --headstage HS3 \\
        [--overwrite]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-path", type=Path, required=True)
    parser.add_argument("--animal", type=str, required=True)
    parser.add_argument("--date", type=str, default=None, help="Experiment date yyyy_mm_dd")
    parser.add_argument("--block", type=str, required=True)
    parser.add_argument(
        "--calibration-mat",
        type=Path,
        default=None,
        help="Path to calibration_results.mat (recommended)",
    )
    parser.add_argument(
        "--headstage",
        type=str,
        default=None,
        help="Headstage ID inside calibration file (e.g. HS3)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from eye_tracking_system_tools.preprocessing.accel_calibration import (
        load_accel_calibration,
    )
    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
    from eye_tracking_system_tools.preprocessing.lizard_movement import (
        compute_and_save_lizard_movement,
    )

    block = BlockSync(
        args.animal,
        args.date,
        args.block,
        str(args.experiment_path),
    )
    calibration = None
    if args.calibration_mat is not None:
        if not args.headstage:
            print("[FAIL] --headstage is required when --calibration-mat is set.", file=sys.stderr)
            return 2
        calibration = load_accel_calibration(args.calibration_mat, args.headstage)
        print(f"[OK] Loaded calibration for {calibration.headstage_id}")

    out_path = compute_and_save_lizard_movement(
        block,
        overwrite=args.overwrite,
        calibration=calibration,
    )
    n = len(block.liz_mov_df) if block.liz_mov_df is not None else 0
    print(f"[OK] Wrote {out_path} ({n} movement samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
