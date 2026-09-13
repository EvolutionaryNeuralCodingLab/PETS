#!/usr/bin/env python3
"""Copy A+B paper+mouse block folders to offline SSD. COPY ONLY — never delete/move sources."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEST_ROOT = Path("/Volumes/Samsung_T5/experiments")

# A: paper cohort (Data-2) + B: mouse Jupyter blocks (Data)
BLOCK_PATHS: list[Path] = [
    # --- A paper ---
    Path("/Volumes/Data-2/Nimrod/experiments/PV_106/2025_08_06/block_008"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_106/2025_08_06/block_009"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_106/2025_08_06/block_010"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_106/2025_08_06/block_011"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_106/2025_08_06/block_012"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_143/2025_08_25/block_001"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_143/2025_08_25/block_002"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_143/2025_08_25/block_003"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_143/2025_08_25/block_004"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_62/2023_04_27/block_024"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_62/2023_04_27/block_026"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_62/2023_05_01/block_038"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_126/2024_07_18/block_007"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_126/2024_07_18/block_008"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_126/2024_07_18/block_009"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_126/2024_07_18/block_010"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_126/2024_08_13/block_011"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_126/2024_08_13/block_012"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_57/2024_11_25/block_007"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_57/2024_11_25/block_008"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_57/2024_11_25/block_009"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_57/2024_12_01/block_012"),
    Path("/Volumes/Data-2/Nimrod/experiments/PV_57/2024_12_01/block_013"),
    # --- B mouse ---
    Path("/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_012"),
    Path("/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_013"),
    Path("/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_014"),
    Path("/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_015"),
]

# Hard refuse any destructive rsync flags if someone edits the command later
FORBIDDEN_RSYNC_FLAGS = {
    "--delete",
    "--delete-before",
    "--delete-after",
    "--delete-during",
    "--delete-excluded",
    "--remove-source-files",
}


def relative_block_key(block_path: Path) -> Path:
    """Map .../experiments/Animal/date/block_xxx → Animal/date/block_xxx."""
    parts = block_path.resolve().parts
    try:
        i = parts.index("experiments")
    except ValueError as e:
        raise ValueError(f"path must contain 'experiments': {block_path}") from e
    rel = Path(*parts[i + 1 :])  # Animal/date/block
    if len(rel.parts) != 3:
        raise ValueError(f"expected Animal/date/block, got {rel}")
    return rel


def assert_safe_rsync_cmd(cmd: list[str]) -> None:
    for flag in FORBIDDEN_RSYNC_FLAGS:
        if flag in cmd:
            raise RuntimeError(f"refusing destructive rsync flag: {flag}")


def copy_block(src: Path, dest_root: Path, *, dry_run: bool) -> str:
    """
    Copy one block folder. Returns status: copied | skipped_exists | missing_src | error:...
    Never modifies or deletes anything under src.
    """
    if not src.is_dir():
        return "missing_src"

    rel = relative_block_key(src)
    dest = dest_root / rel

    # No overwrite: entire block destination must not exist
    if dest.exists():
        return "skipped_exists"

    # Trailing slash: copy contents into dest/; create dest via rsync
    # -a archive; --ignore-existing belt-and-suspenders (we already skip if dest exists)
    # macOS openrsync: use --progress (no --info=progress2)
    cmd = [
        "rsync",
        "-a",
        "--progress",
        "--ignore-existing",
        f"{src}/",
        f"{dest}/",
    ]
    assert_safe_rsync_cmd(cmd)

    if dry_run:
        print(f"DRY-RUN would copy:\n  {src}\n  -> {dest}")
        return "dry_run"

    # Create parent + empty dest dir then fill — never touches src
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=False)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"error:rsync_exit_{e.returncode}"
    return "copied"


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline SSD copy of A+B blocks (copy-only).")
    ap.add_argument("--dest-root", type=Path, default=DEST_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="required to actually copy")
    args = ap.parse_args()

    if not args.dry_run and not args.yes:
        print("Refusing to copy without --yes (use --dry-run to preview).", file=sys.stderr)
        return 2

    dest_root = args.dest_root
    if not dest_root.exists():
        print(f"dest root missing: {dest_root}", file=sys.stderr)
        return 1

    # Refuse writing into server mounts
    dest_resolved = str(dest_root.resolve())
    for banned in ("/Volumes/Data/", "/Volumes/Data-1/", "/Volumes/Data-2/"):
        if dest_resolved.startswith(banned.rstrip("/")) or dest_resolved == banned.rstrip("/"):
            print(f"refusing dest on server mount: {dest_root}", file=sys.stderr)
            return 1

    counts: dict[str, int] = {
        "copied": 0,
        "skipped_exists": 0,
        "missing_src": 0,
        "dry_run": 0,
        "error": 0,
    }
    for src in BLOCK_PATHS:
        status = copy_block(src, dest_root, dry_run=args.dry_run)
        key = "error" if status.startswith("error") else status
        counts[key] = counts.get(key, 0) + 1
        try:
            rel = relative_block_key(src) if src.is_dir() else "n/a"
        except ValueError:
            rel = "n/a"
        print(f"[{status}] {src.name}  ({rel})")

    print("\nSummary:", counts)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
