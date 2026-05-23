"""
Average pupil-diameter traces aligned to Block Annotator events (CLI).

Example:
    python scripts/plots/plot_event_type_pupil_average.py \\
        --output-folder _annotator_out \\
        --event-type saccade \\
        --eye both \\
        --half-window-ms 100 \\
        --out _annotator_out/saccade_pupil_average.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from annotator_plot_core import (  # noqa: E402
    PlotRequest,
    PlotStyle,
    STREAM_L_PUPIL,
    STREAM_R_PUPIL,
    StderrLoadLog,
    build_plot_data,
    export_figures,
)
from eye_tracking_system_tools.annotation.event_explorer.catalog import (  # noqa: E402
    build_catalog,
    discover_annotation_files,
    event_types_in_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot event-type average pupil diameter (within-block z-scored)."
    )
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--annotation-json", type=Path, action="append", default=[])
    parser.add_argument("--event-type", action="append", default=[])
    parser.add_argument("--eye", choices=("left", "right", "both"), default="both")
    parser.add_argument("--half-window-ms", type=float, default=100.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--legend-out", type=Path, default=None)
    parser.add_argument("--figsize", type=float, nargs=2, default=(2.2, 1.8))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--normalization",
        choices=("within_block_zscore", "per_trial_zscore", "none"),
        default="within_block_zscore",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = StderrLoadLog()

    json_paths = discover_annotation_files(
        json_paths=[Path(p) for p in args.annotation_json],
        scan_dirs=[args.output_folder],
        recursive=False,
    )
    if not json_paths:
        print(f"No *_annotations.json found under {args.output_folder}", file=sys.stderr)
        return 0

    catalog = build_catalog(json_paths)
    if not catalog:
        print("No events in annotation files.", file=sys.stderr)
        return 0

    event_types = list(args.event_type) if args.event_type else event_types_in_catalog(catalog)

    if args.eye == "both":
        streams = [STREAM_L_PUPIL, STREAM_R_PUPIL]
    elif args.eye == "left":
        streams = [STREAM_L_PUPIL]
    else:
        streams = [STREAM_R_PUPIL]

    request = PlotRequest(
        annotation_paths=json_paths,
        event_types=event_types,
        stream_ids=streams,
        half_window_ms=args.half_window_ms,
        mode="average",
        normalization=args.normalization,
        style=PlotStyle(figsize=tuple(args.figsize), dpi=args.dpi),
    )

    series = build_plot_data(request, log)
    if not series:
        print("No plottable data.", file=sys.stderr)
        return 1

    main_path = args.out or (args.output_folder / "pupil_average.pdf")
    legend_path = args.legend_out or main_path.with_name(
        main_path.stem + "_legend.pdf"
    )
    export_figures(
        series,
        mode="average",
        normalization=request.normalization,
        style=request.style,
        main_path=main_path,
        legend_path=legend_path,
    )
    print(f"Saved {main_path}")
    print(f"Saved {legend_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
