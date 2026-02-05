#!/usr/bin/env python
"""
Batch DLC + verification pipeline.

Reads the .txt log from a previous batch_block_synchronization.ipynb run, builds
block_collection only for successfully synced blocks, runs initialization and DLC
steps (jitter correction, LED cleanup, find_jittery_frames, create_eye_data, export),
then runs the data verification GUI for each block/eye with a terminal progress bar.

Usage:
  python -m eye_tracking_system_tools.preprocessing.run_batch_dlc_and_verification \\
      path/to/sync_log_batch_analysis_output_YYYY_MM_DD.txt [--no-verify] [--channeldict-json PATH]

If more than 10% of a block's frames are removed by find_jittery_frames, the block
is flagged as badly jittery and the user is prompted to inspect before continuing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from eye_tracking_system_tools.preprocessing import utility_functions as uf
from eye_tracking_system_tools.preprocessing.block_sync_core import (
    find_jittery_frames,
    load_final_sync_df,
    export_eye_data_2d,
)
from eye_tracking_system_tools.preprocessing.data_verification_utils import (
    interactive_eye_data_corrector_synced,
    export_corrected_eye_data,
)

# Jitter threshold: flag block when removed frames exceed this fraction
JITTER_REMOVAL_FRACTION_THRESHOLD = 0.10

# find_jittery_frames defaults (same as block_synchronization.ipynb)
FIND_JITTERY_MAX_DISTANCE = 60
FIND_JITTERY_DIFF_THRESHOLD = 5
FIND_JITTERY_GAP_TO_BRIDGE = 24


def parse_batch_sync_log(log_path: Path):
    """
    Parse the .txt log from batch_block_synchronization.ipynb.
    Returns: experiment_path (Path), animal (str), analysis_subfolder_name (str),
             synced_block_numbers (list of int).
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")

    m = re.search(r"Experiment:\s*(.+)", text)
    if not m:
        raise ValueError(f"Could not find 'Experiment:' line in {log_path}")
    experiment_path = Path(m.group(1).strip())

    m = re.search(r"Animal:\s*(\S+)\s+Blocks:", text)
    if not m:
        raise ValueError(f"Could not find 'Animal: ... Blocks:' in {log_path}")
    animal = m.group(1).strip()

    m = re.search(r"Batch sync report:\s*(.+)", text)
    if not m:
        raise ValueError(f"Could not find 'Batch sync report:' in {log_path}")
    analysis_subfolder_name = m.group(1).strip()

    # Synced blocks: lines "  Block N: path" after "Done. Synced"
    synced_block_numbers = []
    seen_done = False
    for line in text.splitlines():
        line = line.rstrip()
        if "Done. Synced" in line:
            seen_done = True
            continue
        if seen_done and line.strip().startswith("Block ") and ": " in line:
            # "  Block 6: path" or "  Block 006: path"
            part = line.strip()
            if part.startswith("Block "):
                rest = part[6:].split(":", 1)[0].strip()
                try:
                    synced_block_numbers.append(int(rest))
                except ValueError:
                    pass
        if seen_done and line.strip().startswith("***"):
            break

    if not synced_block_numbers:
        raise ValueError(
            f"No synced blocks found in {log_path}. "
            "Ensure the log contains 'Done. Synced N block(s).' followed by '  Block X: path' lines."
        )

    return experiment_path, animal, analysis_subfolder_name, synced_block_numbers


def create_run_folder_for_block(block, analysis_subfolder_name: str) -> Path:
    """Set block.analysis_path to block_path/analysis/<analysis_subfolder_name>."""
    base = block.block_path / "analysis"
    base.mkdir(parents=True, exist_ok=True)
    run_path = base / analysis_subfolder_name
    run_path.mkdir(parents=True, exist_ok=True)
    block.analysis_path = run_path
    return run_path


def load_channeldict_for_animal(animal: str, block_collection, channeldict_json_path: Path | None):
    """
    Get channeldict for animal: from JSON file if provided, else from first block's
    ttl_manual_mapping.json if present.
    """
    if channeldict_json_path and channeldict_json_path.exists():
        data = json.loads(channeldict_json_path.read_text())
        if isinstance(data.get("manual_line_map"), dict):
            # format { "LED_driver": 1, ... } -> { 1: "LED_driver", ... }
            name_to_line = data["manual_line_map"]
            return {int(line): name for name, line in name_to_line.items()}
        return {int(k): v for k, v in data.items()}

    # Try first block's TTL sidecar
    if not block_collection:
        return None
    first = block_collection[0]
    oe_ttl = first.block_path / "oe_files" / getattr(first, "oe_dirname", "")
    if not oe_ttl.exists():
        for d in (first.block_path / "oe_files").iterdir():
            if d.is_dir():
                oe_ttl = d
                break
    sidecar = oe_ttl / "ttl_manual_mapping.json"
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
        name_to_line = data.get("manual_line_map", data)
        return {int(line): name for name, line in name_to_line.items()}
    return None


def run_dlc_pipeline_for_block(block, overwrite_dlc: bool = False, force_apply_jitter: bool = False):
    """
    Run initialization + DLC + jitter + LED + find_jittery + create_eye_data for one block.
    If find_jittery_frames removes >10% of frames and force_apply_jitter is False,
    returns (block, False, fraction_removed, (vid_inds_l, vid_inds_r)) without applying removal.
    If force_apply_jitter is True, applies removal and create_eye_data anyway.
    """
    block.handle_eye_videos()
    block.parse_open_ephys_events()
    block.handle_arena_files()
    block.get_eye_brightness_vectors()
    load_final_sync_df(block)
    block.read_dlc_data(overwrite=overwrite_dlc, export=True)

    block.get_jitter_reports(export=True, overwrite=False, remove_led_blinks=False, sort_on_loading=True)
    block.correct_jitter()
    block.find_led_blink_frames(plot=False)
    block.remove_led_blinks_from_eye_df(export=True)

    df_inds_l, vid_inds_l = find_jittery_frames(
        block, "left",
        max_distance=FIND_JITTERY_MAX_DISTANCE,
        diff_threshold=FIND_JITTERY_DIFF_THRESHOLD,
        gap_to_bridge=FIND_JITTERY_GAP_TO_BRIDGE,
    )
    df_inds_r, vid_inds_r = find_jittery_frames(
        block, "right",
        max_distance=FIND_JITTERY_MAX_DISTANCE,
        diff_threshold=FIND_JITTERY_DIFF_THRESHOLD,
        gap_to_bridge=FIND_JITTERY_GAP_TO_BRIDGE,
    )

    # Total frames per eye (from jitter dict length)
    n_left = len(block.le_jitter_dict.get("top_correlation_dist", []))
    n_right = len(block.re_jitter_dict.get("top_correlation_dist", []))
    total_frames = n_left + n_right
    removed_frames = len(vid_inds_l) + len(vid_inds_r)
    fraction_removed = removed_frames / total_frames if total_frames else 0.0

    jitter_ok = fraction_removed <= JITTER_REMOVAL_FRACTION_THRESHOLD
    apply_removal = jitter_ok or force_apply_jitter
    if not apply_removal:
        return block, False, fraction_removed, (vid_inds_l, vid_inds_r)

    block.remove_eye_datapoints_based_on_video_frames("right", indices_to_nan=vid_inds_r)
    block.remove_eye_datapoints_based_on_video_frames("left", indices_to_nan=vid_inds_l)
    block.create_eye_data()
    export_eye_data_2d(block)
    return block, True, fraction_removed, (vid_inds_l, vid_inds_r)


def main():
    parser = argparse.ArgumentParser(
        description="Run batch DLC pipeline + verification from a batch sync log file.",
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the .txt log from batch_block_synchronization.ipynb (e.g. sync_log_batch_analysis_output_YYYY_MM_DD.txt)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the interactive verification step after DLC pipeline.",
    )
    parser.add_argument(
        "--channeldict-json",
        type=Path,
        default=None,
        help="Optional path to a JSON file with channel mapping (e.g. {\"manual_line_map\": {\"LED_driver\": 1, ...}}). "
             "If not set, script tries to load from first block's ttl_manual_mapping.json.",
    )
    parser.add_argument(
        "--overwrite-dlc",
        action="store_true",
        help="Overwrite existing DLC data when reading (read_dlc_data(overwrite=True)).",
    )
    parser.add_argument(
        "--skip-jittery-check",
        action="store_true",
        help="Do not stop on blocks with >10%% jittery frames; apply removal anyway (use with caution).",
    )
    args = parser.parse_args()

    log_path = args.log_path
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    print("Parsing batch sync log...")
    experiment_path, animal, analysis_subfolder_name, synced_block_numbers = parse_batch_sync_log(log_path)
    print(f"  Experiment: {experiment_path}")
    print(f"  Animal: {animal}")
    print(f"  Analysis subfolder: {analysis_subfolder_name}")
    print(f"  Synced blocks: {synced_block_numbers}")

    block_collection = list(uf.block_generator(
        block_numbers=synced_block_numbers,
        experiment_path=experiment_path,
        animal=animal,
        bad_blocks=[],
    ))
    if not block_collection:
        print("No blocks could be created. Check experiment_path and block numbers.", file=sys.stderr)
        sys.exit(1)

    channeldict = load_channeldict_for_animal(animal, block_collection, args.channeldict_json)
    if channeldict is None:
        print(
            "Warning: No channeldict found. Set --channeldict-json or ensure ttl_manual_mapping.json exists for a block.",
            file=sys.stderr,
        )
    else:
        for b in block_collection:
            b.channeldict = channeldict

    for b in block_collection:
        create_run_folder_for_block(b, analysis_subfolder_name)

    # ---- DLC pipeline ----
    print("\n--- Running DLC pipeline for each block ---")
    badly_jittery = []
    blocks_ready_for_verify = []
    for block in block_collection:
        bn = block.block_num
        print(f"\n[Block {bn}] Running init + DLC + jitter + LED + create_eye_data...")
        try:
            block, jitter_ok, frac, _ = run_dlc_pipeline_for_block(
                block,
                overwrite_dlc=args.overwrite_dlc,
                force_apply_jitter=args.skip_jittery_check,
            )
            if not jitter_ok:
                pct = frac * 100
                badly_jittery.append((block, pct))
                print(f"  [Block {bn}] BADLY JITTERY: {pct:.1f}% of frames removed (threshold {JITTER_REMOVAL_FRACTION_THRESHOLD*100:.0f}%).")
                if not args.skip_jittery_check:
                    print(
                        "  Please inspect this block (e.g. run block_synchronization.ipynb for this block or adjust jitter params). "
                        "Use --skip-jittery-check to apply removal anyway.",
                        file=sys.stderr,
                    )
                else:
                    print("  --skip-jittery-check set: removal applied; block ready for verification.")
                    blocks_ready_for_verify.append(block)
            else:
                print(f"  [Block {bn}] Done. Jitter removal fraction: {frac*100:.2f}%.")
                blocks_ready_for_verify.append(block)
        except Exception as e:
            print(f"  [Block {bn}] FAILED: {e}", file=sys.stderr)
            raise

    if badly_jittery and not args.skip_jittery_check:
        print("\nThe following blocks exceeded the jitter removal threshold and were not fully processed:")
        for block, pct in badly_jittery:
            print(f"  Block {block.block_num}: {pct:.1f}%")
        print("Fix or adjust parameters, then re-run. Or use --skip-jittery-check to force removal.")
        sys.exit(1)

    if args.no_verify:
        print("\n--no-verify: skipping interactive verification.")
        return

    # ---- Verification: one GUI per (block, eye) for blocks that have eye data ----
    tasks = []
    for block in blocks_ready_for_verify:
        tasks.append((block, "left"))
        tasks.append((block, "right"))

    total = len(tasks)
    print(f"\n--- Data verification: {total} videos (2 per block) ---")
    try:
        from tqdm import tqdm
        progress = tqdm(total=total, desc="Verification", unit="video")
    except ImportError:
        progress = None

    for i, (block, eye) in enumerate(tasks):
        if progress is None:
            print(f"\n[{i+1}/{total}] Block {block.block_num} — {eye} eye. Close the GUI to continue.")
        else:
            progress.set_description(f"Block {block.block_num} {eye}")
        interactive_eye_data_corrector_synced(block, eye, ref_point_xy=None)
        export_corrected_eye_data(block, include_rotation_pickle=False)
        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()
    print("\nBatch DLC + verification finished.")


if __name__ == "__main__":
    main()
