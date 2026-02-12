#!/usr/bin/env python
"""
Promote latest heavy-analysis outputs from analysis subfolders to analysis root.

Why:
  Batch runs often create dated subfolders under each block's analysis folder
  (for example: analysis/batch_analysis_output_YYYY_MM_DD). Block initialization
  usually reads from analysis root, so this script copies the newest outputs
  (by file mtime) to analysis/ so they are auto-picked by BlockSync.

Default promoted outputs:
  - Brightness: newest of {eye_brightness_values_dict.pkl, eye_brightness.pickle}
    -> copied to analysis/eye_brightness_values_dict.pkl
  - Jitter: jitter_report_dict.pkl
    -> copied to analysis/jitter_report_dict.pkl
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eye_tracking_system_tools.batch_analysis.get_animal_block_report import discover_blocks


@dataclass(frozen=True)
class PromoteSpec:
    name: str
    candidates: tuple[str, ...]
    destination_name: str


DEFAULT_SPECS = (
    PromoteSpec(
        name="brightness",
        candidates=("eye_brightness_values_dict.pkl", "eye_brightness.pickle"),
        destination_name="eye_brightness_values_dict.pkl",
    ),
    PromoteSpec(
        name="jitter",
        candidates=("jitter_report_dict.pkl",),
        destination_name="jitter_report_dict.pkl",
    ),
)


def parse_animals(args: argparse.Namespace) -> list[str]:
    animals = list(args.animals) + (args.animals_list or [])
    return [a for a in animals if a]


def get_source_folders(analysis_root: Path, prefix: str | None) -> list[Path]:
    if not analysis_root.is_dir():
        return []
    subdirs = [p for p in analysis_root.iterdir() if p.is_dir()]
    if prefix:
        subdirs = [p for p in subdirs if p.name.startswith(prefix)]
    return sorted(subdirs, key=lambda p: p.name)


def choose_latest_candidate(source_folders: list[Path], candidate_names: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for folder in source_folders:
        for name in candidate_names:
            p = folder / name
            if p.is_file():
                candidates.append(p)
    if not candidates:
        return None
    # Highest mtime wins; tie-break by lexical path for deterministic behavior.
    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def backup_destination(dest: Path, backup_root: Path, dry_run: bool) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_path = backup_root / dest.name
    if dry_run:
        return backup_path
    shutil.copy2(dest, backup_path)
    return backup_path


def copy_promoted_file(src: Path, dest: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote latest batch outputs from analysis subfolders to analysis root.",
    )
    parser.add_argument(
        "experiment_path",
        type=Path,
        help="Path to experiment folder (contains animal folders).",
    )
    parser.add_argument(
        "animals",
        nargs="*",
        help="Animal names (e.g. PV_126). If omitted, use --animals.",
    )
    parser.add_argument(
        "-a",
        "--animals",
        dest="animals_list",
        action="append",
        default=[],
        help="Add animal name(s). Can be repeated.",
    )
    parser.add_argument(
        "--source-prefix",
        default="batch_analysis_output_",
        help=(
            "Only use analysis subfolders with this prefix as source "
            "(default: batch_analysis_output_). Use empty string to include all subfolders."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup existing analysis root files before replacing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes only; do not copy files.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for operation report.",
    )
    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)
    if not experiment_path.is_dir():
        print(f"Error: experiment path is not a directory: {experiment_path}", file=sys.stderr)
        return 1

    animals = parse_animals(args)
    if not animals:
        print("Error: provide at least one animal (positional or --animals).", file=sys.stderr)
        return 1

    blocks = discover_blocks(experiment_path, animals)
    if not blocks:
        print("No blocks found for the requested animals.")
        return 0

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, str]] = []
    promoted_count = 0
    skipped_count = 0

    print(f"Found {len(blocks)} block(s).")
    if args.dry_run:
        print("[DRY RUN] No files will be copied.")

    for block in blocks:
        block_path = Path(block["block_path"])
        analysis_root = block_path / "analysis"
        if not analysis_root.is_dir():
            rows.append(
                {
                    "animal": str(block["animal"]),
                    "date": str(block["experiment_date"]),
                    "block_num": str(block["block_num"]),
                    "status": "no_analysis_folder",
                    "target": "",
                    "source": "",
                    "note": "",
                }
            )
            skipped_count += 1
            continue

        prefix = args.source_prefix if args.source_prefix != "" else None
        source_folders = get_source_folders(analysis_root, prefix=prefix)
        if not source_folders:
            rows.append(
                {
                    "animal": str(block["animal"]),
                    "date": str(block["experiment_date"]),
                    "block_num": str(block["block_num"]),
                    "status": "no_source_subfolders",
                    "target": "",
                    "source": "",
                    "note": f"prefix={args.source_prefix!r}",
                }
            )
            skipped_count += 1
            continue

        for spec in DEFAULT_SPECS:
            src = choose_latest_candidate(source_folders, spec.candidates)
            dest = analysis_root / spec.destination_name

            if src is None:
                rows.append(
                    {
                        "animal": str(block["animal"]),
                        "date": str(block["experiment_date"]),
                        "block_num": str(block["block_num"]),
                        "status": "missing_source",
                        "target": str(dest),
                        "source": "",
                        "note": f"{spec.name}: candidates={','.join(spec.candidates)}",
                    }
                )
                skipped_count += 1
                continue

            src_resolved = src.resolve()
            dest_exists = dest.exists()
            dest_same = dest_exists and dest.resolve() == src_resolved

            if dest_same:
                rows.append(
                    {
                        "animal": str(block["animal"]),
                        "date": str(block["experiment_date"]),
                        "block_num": str(block["block_num"]),
                        "status": "already_latest",
                        "target": str(dest),
                        "source": str(src),
                        "note": spec.name,
                    }
                )
                skipped_count += 1
                continue

            backup_note = ""
            if dest_exists and not args.no_backup:
                backup_root = analysis_root / "__promote_backup__" / run_stamp
                backup_path = backup_destination(dest, backup_root, dry_run=args.dry_run)
                backup_note = f"backup={backup_path}"

            copy_promoted_file(src, dest, dry_run=args.dry_run)
            promoted_count += 1
            rows.append(
                {
                    "animal": str(block["animal"]),
                    "date": str(block["experiment_date"]),
                    "block_num": str(block["block_num"]),
                    "status": "promoted",
                    "target": str(dest),
                    "source": str(src),
                    "note": f"{spec.name}; {backup_note}".strip("; "),
                }
            )

    print("\nPromotion summary")
    print("-" * 60)
    print(f"Promoted: {promoted_count}")
    print(f"Skipped : {skipped_count}")
    print(f"Total rows: {len(rows)}")

    for r in rows:
        if r["status"] == "promoted":
            print(
                f"[{r['animal']} block {r['block_num']}] {r['status']}: "
                f"{Path(r['source']).name} -> {r['target']}"
            )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["animal", "date", "block_num", "status", "target", "source", "note"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDetailed CSV written to: {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

