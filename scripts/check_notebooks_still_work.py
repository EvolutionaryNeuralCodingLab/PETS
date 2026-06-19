#!/usr/bin/env python
"""Phase 0 exit-gate: re-run the five preprocessing notebooks on a sandbox copy.

The Preprocessing GUI plan (\u00a74.3 / \u00a76.3) requires that after the inline-helper
refactor we re-execute the five preprocessing notebooks end-to-end on the
sample block and confirm they still succeed with the same outputs. To avoid
clobbering ``D:\\sample_data_for_eye_repo`` we work on a sibling copy under the
repo root at ``_phase0_sample_copy/`` (gitignored).

What this script does:

1. Build/refresh ``_phase0_sample_copy/<animal>/<date>/block_<num>/`` by
   hardlinking the read-only input folders (``eye_videos``, ``arena_videos``,
   ``oe_files``, ``IMU``) and copying the ``analysis/`` folder so the
   notebooks can read existing artefacts but write any new ones into the
   sandbox without touching the original.
2. Generate temporary copies of the five notebooks under
   ``_phase0_sample_copy/notebooks/`` with the ``experiment_path = ...`` line
   patched to point at the sandbox.
3. Run each notebook via ``jupyter nbconvert --to notebook --execute``.
4. Compare ``analysis/`` outputs between the sandbox copy and the original
   sample block for the files the notebooks are expected to produce.

Use ``--smoke`` to skip the heavy execute step and only validate that
preparation succeeds (useful for fast iteration). Use ``--only NAME`` (may
be repeated) to run only specific notebooks.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = Path(r"D:\sample_data_for_eye_repo")
DEFAULT_COPY = REPO_ROOT / "_phase0_sample_copy"

NOTEBOOK_NAMES = [
    "block_synchronization.ipynb",
    "data_verification.ipynb",
    "kerr_degree_conversion.ipynb",
    "add_accelerometer_state_annotations.ipynb",
    "sync_free_eye_ellipse_pipeline.ipynb",
]

NOTEBOOK_DIR = (
    REPO_ROOT / "src" / "eye_tracking_system_tools" / "preprocessing"
)


def _mirror_block(
    src_block: Path,
    dst_block: Path,
    *,
    fresh_analysis: bool,
) -> None:
    """Hardlink input subfolders and copy the analysis folder.

    Hardlinking saves >1 GB of disk + IO since these are large MP4 / OE
    binary inputs that the notebooks only read. ``analysis/`` is fully
    copied so the notebook may overwrite without affecting the original.
    """

    if not src_block.is_dir():
        raise FileNotFoundError(f"Source block not found: {src_block}")
    dst_block.mkdir(parents=True, exist_ok=True)

    for child in src_block.iterdir():
        rel = child.name
        target = dst_block / rel
        if rel == "analysis":
            if fresh_analysis and target.exists():
                shutil.rmtree(target)
            if not target.exists():
                shutil.copytree(child, target)
            continue
        if child.is_dir():
            _hardlink_tree(child, target)
        else:
            _hardlink_file(child, target)


def _hardlink_tree(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        dst = dst_dir / src.name
        if src.is_dir():
            _hardlink_tree(src, dst)
        else:
            _hardlink_file(src, dst)


def _hardlink_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        # Different volume or permission issue: fall back to copy.
        shutil.copy2(src, dst)


def _patch_notebook(src_nb: Path, dst_nb: Path, new_experiment_path: Path) -> None:
    """Copy a notebook and patch the experiment-path string."""

    import json

    nb = json.loads(src_nb.read_text(encoding="utf-8"))

    target_literal_doublequote = f'Path(r"{new_experiment_path}")'
    target_literal_singlequote = f"Path(r'{new_experiment_path}')"

    sentinels = (
        'Path(r"D:\\sample_data_for_eye_repo")',
        "Path(r'D:\\sample_data_for_eye_repo')",
    )

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src_lines = cell.get("source") or []
        if isinstance(src_lines, str):
            src_lines = [src_lines]
        new_lines = []
        for line in src_lines:
            patched = line
            for sentinel in sentinels:
                if sentinel in patched:
                    if sentinel.startswith('Path(r"'):
                        patched = patched.replace(sentinel, target_literal_doublequote)
                    else:
                        patched = patched.replace(sentinel, target_literal_singlequote)
            new_lines.append(patched)
        cell["source"] = new_lines
        cell["outputs"] = []
        cell["execution_count"] = None

    dst_nb.parent.mkdir(parents=True, exist_ok=True)
    dst_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def _run_notebook(notebook: Path, timeout: int) -> tuple[int, str]:
    """Run a notebook with jupyter nbconvert --execute. Returns (rc, log_path)."""
    log_path = notebook.with_suffix(".log.txt")
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name=python3",
        str(notebook),
    ]
    print(f"[run] {' '.join(cmd)}")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[done] rc={proc.returncode} elapsed={dt:.1f}s log={log_path}")
    return proc.returncode, str(log_path)


def _compare_analysis(orig: Path, copy: Path, files: list[str]) -> dict[str, str]:
    """Return file -> verdict ('match', 'differs', 'missing_orig', 'missing_copy')."""
    out = {}
    for f in files:
        op = orig / f
        cp = copy / f
        if not op.exists() and not cp.exists():
            out[f] = "missing_both"
        elif not op.exists():
            out[f] = "missing_orig"
        elif not cp.exists():
            out[f] = "missing_copy"
        else:
            same = filecmp.cmp(op, cp, shallow=False)
            out[f] = "match" if same else "differs"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--copy-root",
        type=Path,
        default=DEFAULT_COPY,
        help="Sandbox root (gitignored).",
    )
    parser.add_argument("--animal", default="PV_106")
    parser.add_argument("--date", default="2025_09_04")
    parser.add_argument("--block", default="015")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Only run notebooks whose basename matches (may be repeated)",
    )
    parser.add_argument(
        "--fresh-analysis",
        action="store_true",
        help="Wipe sandbox analysis/ before re-running (compare from scratch).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only mirror + patch, do not run nbconvert.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-cell execution timeout (seconds). Default 3600.",
    )
    args = parser.parse_args(argv)

    src_block = args.sample_root / args.animal / args.date / f"block_{args.block}"
    dst_block = args.copy_root / args.animal / args.date / f"block_{args.block}"

    print(f"Source block: {src_block}")
    print(f"Sandbox block: {dst_block}")
    if not src_block.is_dir():
        print(f"[FAIL] sample block missing: {src_block}")
        return 1

    print("\n--- 1. Mirroring block (hardlinks for inputs, copy for analysis) ---")
    _mirror_block(src_block, dst_block, fresh_analysis=args.fresh_analysis)
    print("[OK] mirror complete")

    nb_out_dir = args.copy_root / "notebooks"
    nb_out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n--- 2. Patching experiment_path in notebooks (-> {args.copy_root}) ---"
    )
    patched_paths = []
    for nb_name in NOTEBOOK_NAMES:
        if args.only and nb_name not in args.only:
            continue
        src_nb = NOTEBOOK_DIR / nb_name
        if not src_nb.exists():
            print(f"[SKIP] missing notebook: {src_nb}")
            continue
        dst_nb = nb_out_dir / nb_name
        _patch_notebook(src_nb, dst_nb, args.copy_root)
        patched_paths.append(dst_nb)
        print(f"[OK] patched -> {dst_nb}")

    if args.smoke:
        print("\n[smoke] Skipping nbconvert execution.")
        return 0

    print("\n--- 3. Running notebooks via jupyter nbconvert --execute ---")
    failed = []
    for nb in patched_paths:
        rc, log = _run_notebook(nb, timeout=args.timeout)
        if rc != 0:
            failed.append((nb, log))

    print("\n--- 4. Comparing analysis/ outputs vs original ---")
    expected = [
        "eye_brightness_values_dict.pkl",
        "eye_left_simple_sync.csv",
        "eye_right_simple_sync.csv",
        "blocksync_df.csv",
        "final_sync_df.csv",
        "le_df.csv",
        "re_df.csv",
        "left_eye_data.csv",
        "right_eye_data.csv",
        "self_kerr_refs.csv",
        "left_kerr_angle_raw_verified.csv",
        "right_kerr_angle_raw_verified.csv",
    ]
    verdicts = _compare_analysis(
        src_block / "analysis",
        dst_block / "analysis",
        expected,
    )
    for f, v in verdicts.items():
        marker = {"match": "[OK]", "missing_both": "[--]"}.get(v, "[??]")
        print(f"  {marker} {f}: {v}")

    if failed:
        print(f"\n[FAIL] {len(failed)} notebook(s) errored during execution:")
        for nb, log in failed:
            print(f"  {nb} -- see {log}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
