"""
Average LFP traces aligned to detected saccades (left, right, and pooled).

Example:
    python pipelines/saccade_lfp_average/plot_saccade_lfp_average.py \\
        --experiment-path D:/data/experiments \\
        --animal PV_106 \\
        --blocks 015 016 \\
        --electrodes 1 2 3 \\
        --query "behavior == 'quiet'" \\
        --out _out/PV_106_saccade_lfp.pdf

GUI:
    python pipelines/saccade_lfp_average/saccade_lfp_gui.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from saccade_lfp_core import (  # noqa: E402
    PlotConfig,
    export_figures,
    normalize_block_numbers,
    resolve_export_paths,
    run_pipeline,
)


class StderrLog:
    def info(self, msg: str) -> None:
        print(msg, file=sys.stderr)

    def warn(self, msg: str) -> None:
        print(f"WARN: {msg}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average LFP aligned to saccades (L, R, and pooled)."
    )
    parser.add_argument(
        "--experiment-path",
        type=Path,
        required=True,
        help="Parent folder containing {animal}/{date}/block_xxx/ trees.",
    )
    parser.add_argument("--animal", required=True, help="Animal folder name, e.g. PV_106.")
    parser.add_argument(
        "--blocks",
        nargs="+",
        required=True,
        help="Block numbers to include, e.g. 015 016.",
    )
    parser.add_argument(
        "--electrodes",
        nargs="+",
        type=int,
        required=True,
        help="Open Ephys headstage channel numbers to plot.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional pandas query on the saccade table, e.g. \"behavior == 'quiet'\".",
    )
    parser.add_argument(
        "--half-window-ms",
        type=float,
        default=500.0,
        help="Half window around saccade onset (default 500 -> 1000 ms total).",
    )
    parser.add_argument(
        "--saccade-threshold",
        type=float,
        default=2.0,
        help="Automatic saccade threshold as median + k * std on pupil speed.",
    )
    parser.add_argument(
        "--min-saccade-frames",
        type=int,
        default=1,
        help="Minimum contiguous high-speed frames to count as a saccade.",
    )
    parser.add_argument(
        "--analysis-folder",
        type=Path,
        default=None,
        help="Analysis output root; with --plot-label saves under {folder}/{date}/.",
    )
    parser.add_argument(
        "--plot-label",
        default=None,
        help="Label appended to auto-generated PDF filenames.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF path (overrides --analysis-folder auto naming).",
    )
    parser.add_argument(
        "--legend-out",
        type=Path,
        default=None,
        help="Optional separate legend PDF.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figsize", type=float, nargs=2, default=(2.4, 1.8))
    parser.add_argument(
        "--ep-noise-std-k",
        type=float,
        default=0.0,
        help="Drop trials with snippet std > k × median(trial std); 0 = off.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Max saccades per OERecording.get_data call.",
    )
    parser.add_argument(
        "--detection-mode",
        choices=("velocity", "legacy"),
        default="velocity",
        help="Saccade detection: velocity (median+k×std) or legacy (speed_r threshold + sync).",
    )
    parser.add_argument(
        "--eye-data-source",
        choices=("auto", "eye_data_csv", "le_re_df"),
        default="auto",
        help="Eye CSV resolution: auto prefers left/right_eye_data.csv.",
    )
    parser.add_argument(
        "--legacy-speed-threshold",
        type=float,
        default=2.0,
        help="Legacy mode: speed_r threshold for saccade onset.",
    )
    parser.add_argument(
        "--legacy-sync-diff-ms",
        type=float,
        default=680.0,
        help="Legacy mode: max |L_on − R_on| (ms) to count as synced pair.",
    )
    parser.add_argument(
        "--legacy-magnitude-calib",
        type=float,
        default=1.0,
        help="Legacy mode: multiplier on integrated speed (magnitude).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PlotConfig(
        experiment_path=args.experiment_path,
        animal=args.animal,
        blocks=normalize_block_numbers(args.blocks),
        electrodes=args.electrodes,
        query=args.query,
        half_window_ms=args.half_window_ms,
        saccade_threshold=args.saccade_threshold,
        min_saccade_frames=args.min_saccade_frames,
        batch_size=args.batch_size,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        ep_noise_std_k=args.ep_noise_std_k,
        detection_mode=args.detection_mode,
        eye_data_source=args.eye_data_source,
        legacy_speed_threshold=args.legacy_speed_threshold,
        legacy_sync_diff_ms=args.legacy_sync_diff_ms,
        legacy_magnitude_calib=args.legacy_magnitude_calib,
    )

    block_labels = [f"{b:03d}" for b in config.blocks]
    log = StderrLog()
    try:
        result = run_pipeline(config, log)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.out is not None:
        out_path = args.out
        legend_path = args.legend_out
    elif args.analysis_folder is not None:
        resolved = resolve_export_paths(
            args.analysis_folder,
            args.animal,
            block_labels,
            plot_label=args.plot_label,
        )
        out_path = resolved.main_path
        legend_path = args.legend_out or resolved.legend_path
        if resolved.collision_warning:
            log.warn(resolved.collision_warning)
    else:
        tag = f"_{args.plot_label}" if args.plot_label else ""
        out_path = Path(f"saccade_lfp_{args.animal}{tag}.pdf")
        legend_path = args.legend_out

    export_figures(result, out_path, legend_path)
    print(f"Saved figure to {out_path}")
    if legend_path:
        print(f"Saved legend to {legend_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
