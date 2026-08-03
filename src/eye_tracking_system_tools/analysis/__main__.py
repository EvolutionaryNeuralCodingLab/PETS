"""CLI: build event tables and export figures from a block registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from eye_tracking_system_tools.analysis.pipeline import (
    run_from_event_pickle,
    run_from_registry,
)
from eye_tracking_system_tools.analysis.run_layout import resolve_run_dir


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Run figure exports from analyzed blocks or a frozen event pickle."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="YAML registry of animal → block paths (remount mode)",
    )
    parser.add_argument(
        "--event-pickle",
        type=Path,
        default=None,
        help="Fig-2j-style synced/non_synced pickle (paper event set)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=repo / "configs" / "analysis_params.yaml",
        help="YAML analysis/figure parameters",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=repo / "outputs",
        help="Parent of run folders (default: outputs/)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag → phase2_<tag>; empty overwrites phase2_latest",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Deprecated: explicit run directory (overrides --out-root/--tag)",
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        default=None,
        help="Optional subset e.g. 2c 2e 2f (default: all core 2c–2j)",
    )
    args = parser.parse_args(argv)

    if args.event_pickle is None and args.registry is None:
        args.registry = repo / "configs" / "sample_blocks.yaml"

    if args.out is not None:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        run = resolve_run_dir(args.out_root, args.tag or None)
        out_dir = run.run_dir

    print(f"params:   {args.params}")
    print(f"out:      {out_dir}")
    print(f"  figures:  {out_dir / 'figures'}")
    print(f"  metadata: {out_dir / 'metadata'}")
    if args.event_pickle is not None:
        print(f"event_pickle: {args.event_pickle}")
        written = run_from_event_pickle(
            args.event_pickle,
            args.params,
            out_dir,
            figures=args.figures,
        )
    else:
        print(f"registry: {args.registry}")
        written = run_from_registry(
            args.registry,
            args.params,
            out_dir,
            figures=args.figures,
        )
    print("\nWrote:")
    for name, path in written.items():
        if name.startswith("_"):
            continue
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
