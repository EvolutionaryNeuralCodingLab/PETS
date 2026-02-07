#!/usr/bin/env python
"""
Generate a block sync preprocessing report for given experiment and animals.

For each animal, discovers all block folders under the experiment path, lists
files in each block's analysis folder(s), and flags blocks that are "ready" for
further processing (have both eye brightness pickle and jitter report pickle).

Usage:
  python -m eye_tracking_system_tools.preprocessing.get_animal_block_report EXPERIMENT_PATH ANIMAL [ANIMAL ...]
  python -m eye_tracking_system_tools.preprocessing.get_animal_block_report EXPERIMENT_PATH --animals AN1 AN2 -o report.csv

Output:
  - Full report CSV with columns: animal, experiment_date, block_num, block_path, analysis_folder,
    analysis_path, files_in_analysis, has_brightness_pickle, has_jitter_pickle, ready
  - Console summary of blocks that are ready (have both brightness and jitter pickles).
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


def get_analysis_folders(block_path: Path) -> list[tuple[str, Path]]:
    """
    Return list of (folder_name, path) for each analysis folder to report.

    - If block_path/analysis does not exist, return [].
    - If block_path/analysis contains only files (no subdirs), return [(".", analysis_path)].
    - If block_path/analysis contains subdirs, return one entry per subdir (name, subdir_path).
    """
    analysis_base = block_path / "analysis"
    if not analysis_base.is_dir():
        return []

    entries = list(analysis_base.iterdir())
    subdirs = [e for e in entries if e.is_dir()]
    files_in_root = [e for e in entries if e.is_file()]

    if subdirs:
        return [(d.name, d) for d in subdirs]
    if files_in_root:
        return [(".", analysis_base)]
    return [(".", analysis_base)]


def list_analysis_files(analysis_path: Path) -> list[str]:
    """List names of all files in analysis_path (non-recursive)."""
    if not analysis_path.is_dir():
        return []
    return [p.name for p in analysis_path.iterdir() if p.is_file()]


def check_ready(files: list[str]) -> tuple[bool, bool]:
    """Return (has_brightness_pickle, has_jitter_pickle)."""
    file_set = set(files)
    has_brightness = any(f in file_set for f in BRIGHTNESS_PICKLE_NAMES)
    has_jitter = JITTER_REPORT_PICKLE_NAME in file_set
    return has_brightness, has_jitter


def build_report_rows(experiment_path: Path, animals: list[str]) -> list[dict]:
    """Build full report: one row per (block, analysis_folder)."""
    blocks = discover_blocks(experiment_path, animals)
    report = []
    for b in blocks:
        block_path = b["block_path"]
        analysis_folders = get_analysis_folders(block_path)
        if not analysis_folders:
            report.append({
                "animal": b["animal"],
                "experiment_date": b["experiment_date"],
                "block_num": b["block_num"],
                "block_path": str(block_path),
                "analysis_folder": "",
                "analysis_path": "",
                "files_in_analysis": "",
                "has_brightness_pickle": False,
                "has_jitter_pickle": False,
                "ready": False,
            })
            continue
        for folder_name, apath in analysis_folders:
            files = list_analysis_files(apath)
            files_str = "; ".join(sorted(files)) if files else ""
            has_brightness, has_jitter = check_ready(files)
            report.append({
                "animal": b["animal"],
                "experiment_date": b["experiment_date"],
                "block_num": b["block_num"],
                "block_path": str(block_path),
                "analysis_folder": folder_name,
                "analysis_path": str(apath),
                "files_in_analysis": files_str,
                "has_brightness_pickle": has_brightness,
                "has_jitter_pickle": has_jitter,
                "ready": has_brightness and has_jitter,
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
            sub = f" / {r['analysis_folder']}" if r["analysis_folder"] and r["analysis_folder"] != "." else ""
            print(f"  {r['animal']}  date={r['experiment_date']}  block={r['block_num']}{sub}")
            print(f"    -> {r['analysis_path']}")
    print()

    if not_ready:
        print("Blocks not ready (missing brightness and/or jitter pickle):")
        for r in not_ready:
            sub = f" / {r['analysis_folder']}" if r.get("analysis_folder") and r["analysis_folder"] != "." else ""
            reason = []
            if not r.get("has_brightness_pickle"):
                reason.append("brightness")
            if not r.get("has_jitter_pickle"):
                reason.append("jitter")
            print(f"  {r['animal']}  date={r['experiment_date']}  block={r['block_num']}{sub}  (missing: {', '.join(reason)})")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate block sync preprocessing report: list blocks and analysis files, flag ready blocks."
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
