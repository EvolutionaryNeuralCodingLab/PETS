#!/usr/bin/env python
"""
Generate a block sync preprocessing report for given experiment and animals.

For each animal, discovers all block folders under the experiment path, scans
the full analysis tree for each block (analysis root + internal subfolders), and
flags blocks that are "ready" for further processing (have both eye brightness
pickle and jitter report pickle anywhere in that tree).

Usage:
  python -m eye_tracking_system_tools.batch_analysis.get_animal_block_report EXPERIMENT_PATH ANIMAL [ANIMAL ...]
  python -m eye_tracking_system_tools.batch_analysis.get_animal_block_report EXPERIMENT_PATH --animals AN1 AN2 -o report.csv

Output:
  - Full report CSV with one row per block and columns: animal, experiment_date,
    block_num, block_path, analysis_path, has_brightness_pickle,
    has_jitter_pickle, ready, brightness_found_at, jitter_found_at,
    files_found_count
  - Console summary of blocks that are ready (have both brightness and jitter
    pickles somewhere under analysis/).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


# Filenames used in the pipeline (BlockSync / block_sync_core)
BRIGHTNESS_PICKLE_NAMES = ("eye_brightness_values_dict.pkl", "eye_brightness.pickle")
JITTER_REPORT_PICKLE_NAME = "jitter_report_dict.pkl"


def discover_blocks(experiment_path: Path, animals: list[str]) -> list[dict]:
    """
    Discover all block folders for the given animals under experiment_path.

    Follows the same layout as utility_functions.block_generator:
      experiment_path / animal / date_folder / block_XXX / analysis [ / subfolder ]

    Returns a list of dicts with keys: animal, experiment_date, block_num, block_path.
    """
    experiment_path = Path(experiment_path)
    if not experiment_path.is_dir():
        return []

    rows = []
    for animal in animals:
        p = experiment_path / animal
        if not p.is_dir():
            continue
        # Date folders: directories whose name does not contain 'block'
        date_folders = [d for d in p.iterdir() if d.is_dir() and "block" not in d.name.lower()]
        for date_path in date_folders:
            date_name = date_path.name
            for item in date_path.iterdir():
                if not item.is_dir():
                    continue
                name_lower = item.name.lower()
                if "block" not in name_lower:
                    continue
                # Parse block number: block_6, block_006, etc.
                match = re.search(r"block[_\-]?(\d+)$", name_lower, re.IGNORECASE)
                block_num = match.group(1) if match else item.name.split("_")[-1]
                rows.append({
                    "animal": animal,
                    "experiment_date": date_name,
                    "block_num": block_num,
                    "block_path": item,
                })
    return rows


def check_ready(files: list[str]) -> tuple[bool, bool]:
    """Return (has_brightness_pickle, has_jitter_pickle)."""
    file_set = set(files)
    has_brightness = any(f in file_set for f in BRIGHTNESS_PICKLE_NAMES)
    has_jitter = JITTER_REPORT_PICKLE_NAME in file_set
    return has_brightness, has_jitter


def build_report_rows(experiment_path: Path, animals: list[str]) -> list[dict]:
    """Build full report: one row per block (recursive analysis scan)."""
    blocks = discover_blocks(experiment_path, animals)
    report = []
    for b in blocks:
        block_path = b["block_path"]
        analysis_path = block_path / "analysis"
        if not analysis_path.is_dir():
            report.append({
                "animal": b["animal"],
                "experiment_date": b["experiment_date"],
                "block_num": b["block_num"],
                "block_path": str(block_path),
                "analysis_path": "",
                "has_brightness_pickle": False,
                "has_jitter_pickle": False,
                "ready": False,
                "brightness_found_at": "",
                "jitter_found_at": "",
                "files_found_count": 0,
            })
            continue

        all_files = [p for p in analysis_path.rglob("*") if p.is_file()]
        file_names = [p.name for p in all_files]
        has_brightness, has_jitter = check_ready(file_names)

        brightness_matches = [
            str(p.relative_to(analysis_path))
            for p in all_files
            if p.name in BRIGHTNESS_PICKLE_NAMES
        ]
        jitter_matches = [
            str(p.relative_to(analysis_path))
            for p in all_files
            if p.name == JITTER_REPORT_PICKLE_NAME
        ]

        report.append({
            "animal": b["animal"],
            "experiment_date": b["experiment_date"],
            "block_num": b["block_num"],
            "block_path": str(block_path),
            "analysis_path": str(analysis_path),
            "has_brightness_pickle": has_brightness,
            "has_jitter_pickle": has_jitter,
            "ready": has_brightness and has_jitter,
            "brightness_found_at": "; ".join(brightness_matches),
            "jitter_found_at": "; ".join(jitter_matches),
            "files_found_count": len(all_files),
        })
    return report


def write_csv(report: list[dict], out_path: Path) -> None:
    """Write report to CSV."""
    if not report:
        return
    fieldnames = list(report[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in report:
            w.writerow(row)


def print_ready_summary(report: list[dict]) -> None:
    """Print which blocks are ready for further processing."""
    ready = [r for r in report if r["ready"]]
    not_ready = [r for r in report if not r["ready"]]

    print("\n" + "=" * 60)
    print("BLOCKS READY FOR FURTHER PROCESSING")
    print("(have both eye_brightness pickle and jitter_report_dict.pkl)")
    print("=" * 60)
    if not ready:
        print("None.")
    else:
        for r in ready:
            print(f"  {r['animal']}  date={r['experiment_date']}  block={r['block_num']}")
            print(f"    -> {r['analysis_path']}")
    print()

    if not_ready:
        print("Blocks not ready (missing brightness and/or jitter pickle anywhere under analysis/):")
        for r in not_ready:
            reason = []
            if not r.get("has_brightness_pickle"):
                reason.append("brightness")
            if not r.get("has_jitter_pickle"):
                reason.append("jitter")
            print(f"  {r['animal']}  date={r['experiment_date']}  block={r['block_num']}  (missing: {', '.join(reason)})")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate block sync preprocessing report: one row per block, recursive analysis scan."
    )
    parser.add_argument(
        "experiment_path",
        type=Path,
        help="Path to the experiment folder (contains one folder per animal).",
    )
    parser.add_argument(
        "animals",
        nargs="*",
        help="Animal folder names (e.g. PV_126). If not given, use --animals.",
    )
    parser.add_argument(
        "-a", "--animals",
        dest="animals_list",
        action="append",
        default=[],
        help="Add animal(s). Can be repeated. Alternative to positional animals.",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_csv",
        type=Path,
        default=None,
        help="Write full report to this CSV file (default: block_sync_preprocessing_report.csv in current dir).",
    )
    args = parser.parse_args()

    animals = list(args.animals) + (args.animals_list or [])
    if not animals:
        print("Error: provide at least one animal (positional or -a/--animals).", file=sys.stderr)
        return 1

    experiment_path = Path(args.experiment_path)
    if not experiment_path.is_dir():
        print(f"Error: experiment path is not a directory: {experiment_path}", file=sys.stderr)
        return 1

    report = build_report_rows(experiment_path, animals)
    if not report:
        print("No blocks found for the given experiment path and animals.")
        return 0

    csv_path = args.output_csv or Path("block_sync_preprocessing_report.csv")
    write_csv(report, csv_path)
    print(f"Full report written to: {csv_path}")

    print_ready_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
