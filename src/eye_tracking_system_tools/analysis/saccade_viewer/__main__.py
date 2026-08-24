"""CLI entry for the saccade verification viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from eye_tracking_system_tools.analysis.saccade_viewer.launch import (
    launch_saccade_viewer,
    sample_events_from_registry,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Saccade event verification viewer")
    p.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Block registry YAML (animal → block paths)",
    )
    p.add_argument(
        "--sample-per-block",
        type=int,
        default=3,
        help="When no events CSV given, load this many events per block",
    )
    p.add_argument(
        "--events-csv",
        type=Path,
        default=None,
        help="Optional pre-filtered events CSV instead of registry sampling",
    )
    p.add_argument("--pre-ms", type=float, default=250.0)
    p.add_argument("--post-ms", type=float, default=250.0)
    p.add_argument(
        "--auto-advance",
        action="store_true",
        help="Jump to next event after good/bad tag",
    )
    args = p.parse_args(argv)

    if args.events_csv is not None:
        import pandas as pd

        events = pd.read_csv(args.events_csv)
    else:
        events = sample_events_from_registry(
            args.registry,
            per_block=args.sample_per_block,
        )

    launch_saccade_viewer(
        events,
        registry_path=args.registry,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        auto_advance=args.auto_advance,
        block=True,
    )


if __name__ == "__main__":
    main()
