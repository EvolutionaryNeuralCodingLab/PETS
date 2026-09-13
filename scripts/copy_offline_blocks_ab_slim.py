#!/usr/bin/env python3
"""
Slim offline copy of A+B paper+mouse blocks → /Volumes/Samsung_T5/experiments.

COPY ONLY — never delete/move sources. Skip dest block if it already exists.

Policy:
  - analysis/, imu/, and non-video companions: copy as-is
  - arena_videos: at most one top_*.mp4 (never .avi); avi-only → skip arena
  - eye_videos: original .mp4/.h264 + docs; exclude DLC/labeled annotated videos
  - oe_files: one neural .continuous (prefer CH15, else any CH), events/xml/mat/csv;
    exclude other continuous, spikeSorting .bin, oe-embedded .avi
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEST_ROOT = Path("/Volumes/Samsung_T5/experiments")

BLOCK_PATHS: list[Path] = [
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
    Path("/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_012"),
    Path("/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_013"),
    Path("/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_014"),
    Path("/Volumes/Data/Nimrod/experiments/M_002/2026_07_28/block_015"),
]

VIDEO_EXTS = {".mp4", ".avi", ".h264", ".mkv", ".mov"}
EYE_KEEP_VIDEO_EXTS = {".mp4", ".h264"}
OE_KEEP_SUFFIXES = {
    ".events",
    ".xml",
    ".openephys",
    ".csv",
    ".mat",
    ".json",
    ".txt",
    ".log",
    ".timestamps",
    ".pkl",
    ".png",
    ".tsv",
    ".m",
}
# Never descend into these under oe_files (heavy / unused for offline remount)
OE_SKIP_DIR_NAMES = {
    "spikesorting",
    "kilosort",
    "avi_version",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def relative_block_key(block_path: Path) -> Path:
    parts = block_path.resolve().parts
    i = parts.index("experiments")
    rel = Path(*parts[i + 1 :])
    if len(rel.parts) != 3:
        raise ValueError(f"expected Animal/date/block, got {rel}")
    return rel


def is_dlc_annotated_video(path: Path) -> bool:
    n = path.name.lower()
    return ("dlc" in n) or ("labeled" in n)


def _walk_files(root: Path, *, skip_dir_names: set[str] | None = None):
    """Yield file Paths under root, pruning skip_dir_names (case-insensitive)."""
    skip = {n.lower() for n in (skip_dir_names or set())}
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        if skip:
            dirnames[:] = [d for d in dirnames if d.lower() not in skip]
        for name in filenames:
            if name.startswith("._") or name == ".DS_Store":
                continue
            yield Path(dirpath) / name


def pick_top_arena_mp4(block: Path) -> Path | None:
    """Prefer shallow videos/top_*.mp4; never .avi."""
    videos = block / "arena_videos" / "videos"
    candidates: list[Path] = []
    if videos.is_dir():
        for p in videos.iterdir():
            if p.is_file() and p.suffix.lower() == ".mp4" and not is_dlc_annotated_video(p):
                candidates.append(p)
    if not candidates:
        # fallback: any mp4 under arena_videos, skipping avi_version
        for p in _walk_files(block / "arena_videos", skip_dir_names={"avi_version"}):
            if p.suffix.lower() == ".mp4" and not is_dlc_annotated_video(p):
                candidates.append(p)
    if not candidates:
        return None

    def score(p: Path) -> tuple:
        n = p.name.lower()
        s = 0
        if n.startswith("top_") or n.startswith("top."):
            s += 100
        elif "top" in n:
            s += 50
        return (s, -p.stat().st_size)

    return sorted(candidates, key=score, reverse=True)[0]


def pick_oe_continuous(block: Path) -> Path | None:
    """Prefer CH15; else any neural .continuous; else any .continuous. Shallow Record Node only."""
    oe = block / "oe_files"
    if not oe.is_dir():
        return None
    cont: list[Path] = []
    # Prefer files directly under Record Node* (avoid deep spikeSorting walks)
    for exp in oe.iterdir():
        if not exp.is_dir():
            continue
        nodes = [p for p in exp.iterdir() if p.is_dir() and p.name.startswith("Record Node")]
        search_roots = nodes or [exp]
        for root in search_roots:
            try:
                for name in os.listdir(root):
                    if name.endswith(".continuous"):
                        cont.append(root / name)
            except OSError:
                continue
    if not cont:
        # rare flat layout / nested
        for p in _walk_files(oe, skip_dir_names=OE_SKIP_DIR_NAMES):
            if p.suffix == ".continuous":
                cont.append(p)
    if not cont:
        return None

    def is_ch15(p: Path) -> bool:
        n = p.name.upper()
        if "CH15" in n or "_CH15." in n:
            return True
        token = p.stem.split("_")[-1]
        return token == "15" and "AUX" not in n and "ADC" not in n

    def is_neural(p: Path) -> bool:
        n = p.name.upper()
        return ("ADC" not in n) and ("AUX" not in n)

    ch15 = [p for p in cont if is_ch15(p)]
    if ch15:
        return sorted(ch15, key=lambda p: p.name)[0]
    neural = [p for p in cont if is_neural(p)]
    pool = neural or cont
    return sorted(pool, key=lambda p: p.name)[0]


def should_copy_eye_file(path: Path) -> bool:
    if path.suffix.lower() in VIDEO_EXTS:
        if path.suffix.lower() not in EYE_KEEP_VIDEO_EXTS:
            return False
        if is_dlc_annotated_video(path):
            return False
        return True
    return True


def should_copy_oe_file(path: Path, kept_continuous: Path | None) -> bool:
    suf = path.suffix.lower()
    if suf == ".continuous":
        return kept_continuous is not None and path == kept_continuous
    if suf in {".bin", ".avi", ".mp4", ".h264", ".mkv", ".mov", ".npy"}:
        return False
    return suf in OE_KEEP_SUFFIXES or suf == ""


def iter_selected_files(block: Path) -> tuple[list[tuple[Path, Path]], Path | None, Path | None]:
    """
    Fast targeted selection. Returns (files, arena_pick, oe_pick).
    Does NOT walk entire block blindly; skips OE spikeSorting trees.
    """
    selected: list[tuple[Path, Path]] = []
    arena_pick = pick_top_arena_mp4(block)
    oe_pick = pick_oe_continuous(block)

    if arena_pick is not None:
        selected.append((arena_pick, arena_pick.relative_to(block)))

    for p in _walk_files(block / "eye_videos"):
        if should_copy_eye_file(p):
            selected.append((p, p.relative_to(block)))

    for p in _walk_files(block / "oe_files", skip_dir_names=OE_SKIP_DIR_NAMES):
        if should_copy_oe_file(p, oe_pick):
            selected.append((p, p.relative_to(block)))

    for top in ("analysis", "imu"):
        for p in _walk_files(block / top):
            selected.append((p, p.relative_to(block)))

    # any other top-level files (not dirs we already handled)
    handled = {"arena_videos", "eye_videos", "oe_files", "analysis", "imu"}
    try:
        for name in os.listdir(block):
            if name.startswith("."):
                continue
            p = block / name
            if p.is_file():
                selected.append((p, Path(name)))
            elif p.is_dir() and name not in handled and not name.startswith("span_"):
                for f in _walk_files(p):
                    selected.append((f, f.relative_to(block)))
    except OSError:
        pass

    return selected, arena_pick, oe_pick


def estimate_block(block: Path) -> dict:
    files, arena, oe = iter_selected_files(block)
    total = 0
    for p, _ in files:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return {
        "n_files": len(files),
        "bytes": total,
        "arena": arena.name if arena else None,
        "oe_cont": oe.name if oe else None,
        "files": files,
    }


def copy_block(src: Path, dest_root: Path, *, dry_run: bool, precomputed: dict | None = None) -> str:
    if not src.is_dir():
        return "missing_src"
    rel = relative_block_key(src)
    dest = dest_root / rel
    if dest.exists():
        return "skipped_exists"

    est = precomputed or estimate_block(src)
    files = est["files"]

    if dry_run:
        log(
            f"DRY-RUN {rel}: {est['n_files']} files, {est['bytes'] / 1024**3:.2f} GB | "
            f"arena={est['arena'] or 'NONE'} | oe={est['oe_cont'] or 'NONE'}"
        )
        return "dry_run"

    dest.mkdir(parents=True, exist_ok=False)
    for src_file, rel_file in files:
        out = dest / rel_file
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, out)
    return "copied"


def fmt_bytes(n: int) -> str:
    for u, d in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2)):
        if n >= d:
            return f"{n / d:.2f} {u}"
    return f"{n} B"


def main() -> int:
    ap = argparse.ArgumentParser(description="Slim offline SSD copy (copy-only).")
    ap.add_argument("--dest-root", type=Path, default=DEST_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate-only", action="store_true", help="print size estimate and exit")
    ap.add_argument("--yes", action="store_true", help="required for live copy")
    args = ap.parse_args()

    if not args.dry_run and not args.estimate_only and not args.yes:
        print("Refusing to copy without --yes (use --dry-run / --estimate-only).", file=sys.stderr)
        return 2

    dest_root = args.dest_root
    if args.yes or (args.dry_run and not args.estimate_only):
        if not dest_root.exists():
            print(f"dest root missing: {dest_root}", file=sys.stderr)
            return 1
        dest_resolved = str(dest_root.resolve())
        for banned in ("/Volumes/Data/", "/Volumes/Data-1/", "/Volumes/Data-2/"):
            if dest_resolved.startswith(banned.rstrip("/")):
                print(f"refusing dest on server mount: {dest_root}", file=sys.stderr)
                return 1

    total = 0
    counts = {"copied": 0, "skipped_exists": 0, "missing_src": 0, "dry_run": 0, "error": 0}

    for src in BLOCK_PATHS:
        key = relative_block_key(src) if src.is_dir() else str(src)
        try:
            est = estimate_block(src) if src.is_dir() else None
        except Exception as e:
            log(f"[error:estimate] {key}: {e}")
            counts["error"] += 1
            continue

        if est is None:
            log(f"[missing_src] {key}")
            counts["missing_src"] += 1
            continue

        total += est["bytes"]
        log(
            f"[{key}] {fmt_bytes(est['bytes']):>10}  "
            f"files={est['n_files']:<5} arena={est['arena'] or 'NONE':<40} oe={est['oe_cont'] or 'NONE'}"
        )

        if args.estimate_only:
            continue

        try:
            status = copy_block(src, dest_root, dry_run=args.dry_run, precomputed=est)
        except Exception as e:
            status = f"error:{e}"
        k = "error" if status.startswith("error") else status
        counts[k] = counts.get(k, 0) + 1
        if status.startswith("error") or status == "skipped_exists":
            log(f"  -> {status}")

    log(f"\nESTIMATED PORTABLE TOTAL: {fmt_bytes(total)}  ({total} bytes)")
    if not args.estimate_only:
        log(f"Summary: {counts}")
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
