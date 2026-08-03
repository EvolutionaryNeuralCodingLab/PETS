"""Copy reproduction baseline PDFs into a run's figures/repro_baseline/."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eye_tracking_system_tools.analysis.run_layout import RunDirs, assert_not_reproduction


def reproduction_root(repo: Path | None = None) -> Path:
    if repo is None:
        repo = Path(__file__).resolve().parents[3]
    return (
        repo
        / "src"
        / "eye_tracking_system_tools"
        / "figures"
        / "reproduction"
        / "main_figures"
    )


def discover_repro_scripts(root: Path | None = None) -> list[Path]:
    root = root or reproduction_root()
    scripts: list[Path] = []
    for fig_dir in sorted(root.glob("Fig_*")):
        if not fig_dir.is_dir():
            continue
        for pattern in ("figure_*.py", "fig_*.py"):
            for script in sorted(fig_dir.glob(pattern)):
                if script.name.startswith("_"):
                    continue
                scripts.append(script)
    return scripts


def _existing_pdfs(folder: Path) -> set[Path]:
    return {p.resolve() for p in folder.glob("*.pdf")}


def _run_script(script: Path, *, python: str | None = None) -> dict[str, Any]:
    """Execute a reproduction plotter; return status dict."""
    python = python or sys.executable
    before = _existing_pdfs(script.parent)
    env = os.environ.copy()
    # Prefer project mplconfig if present
    repo = Path(__file__).resolve().parents[3]
    mpl = repo / ".mplconfig"
    if mpl.is_dir():
        env["MPLCONFIGDIR"] = str(mpl)

    proc = subprocess.run(
        [python, str(script.name)],
        cwd=str(script.parent),
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    after = _existing_pdfs(script.parent)
    new_or_touched = sorted(after - before) + sorted(
        p for p in after & before  # may have been overwritten
    )
    # Prefer newly created; if none new, take all pdfs modified during run
    created = sorted(after - before)
    pdfs = created if created else sorted(script.parent.glob("*.pdf"))
    return {
        "script": str(script),
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
        "pdfs_in_script_dir": [str(p) for p in pdfs],
        "created_pdfs": [str(p) for p in created],
    }


def run_repro_baseline(
    run: RunDirs,
    *,
    repo: Path | None = None,
    python: str | None = None,
    scripts: list[Path] | None = None,
    cleanup_created: bool = True,
) -> Path:
    """
    Run archived ``figure_*.py`` / ``fig_*.py`` scripts and **copy** PDFs into
    ``run.figures_dir / repro_baseline / <Fig_id>/``.

    Never leaves intentional writes as the source of truth under reproduction/;
    newly created PDFs beside pickles are removed after copy when
    ``cleanup_created`` is True.
    """
    assert_not_reproduction(run.run_dir)
    run.ensure()
    root = reproduction_root(repo)
    scripts = scripts or discover_repro_scripts(root)
    dest_root = run.figures_dir / "repro_baseline"
    dest_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reproduction_root": str(root.resolve()),
        "figures_dest": str(dest_root.resolve()),
        "entries": [],
    }

    for script in scripts:
        fig_id = script.parent.name
        entry: dict[str, Any] = {
            "fig_id": fig_id,
            "script": str(script.resolve()),
            "pickle_siblings": [
                str(p.resolve())
                for p in sorted(script.parent.glob("*.pkl"))
                + sorted(script.parent.glob("*.pickle"))
            ],
        }
        before = _existing_pdfs(script.parent)
        try:
            result = _run_script(script, python=python)
            entry.update(
                {
                    "returncode": result["returncode"],
                    "ok": result["returncode"] == 0,
                    "stdout_tail": result["stdout"],
                    "stderr_tail": result["stderr"],
                }
            )
            after = _existing_pdfs(script.parent)
            created = after - before
            # Copy all PDFs present after the run (include pre-existing if script
            # overwrote in place — we still want a snapshot under outputs/).
            copied = []
            fig_dest = dest_root / fig_id
            fig_dest.mkdir(parents=True, exist_ok=True)
            for pdf in sorted(script.parent.glob("*.pdf")):
                target = fig_dest / pdf.name
                shutil.copy2(pdf, target)
                copied.append(str(target.resolve()))
            entry["copied_pdfs"] = copied

            if cleanup_created:
                for pdf in created:
                    try:
                        Path(pdf).unlink(missing_ok=True)
                    except OSError:
                        pass
                entry["cleaned_created_pdfs"] = [str(p) for p in sorted(created)]
        except Exception as exc:
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-2000:]
        manifest["entries"].append(entry)
        status = "ok" if entry.get("ok") else "FAIL"
        n_pdf = len(entry.get("copied_pdfs", []))
        print(f"[{status}] {fig_id}/{script.name} → {n_pdf} PDF(s)")

    manifest_path = run.metadata_dir / "repro_baseline_manifest.yaml"
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    from eye_tracking_system_tools.analysis.run_layout import resolve_run_dir

    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Run reproduction plotters and copy PDFs into a run folder."
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
        "--python",
        type=str,
        default=None,
        help="Python executable (default: current interpreter)",
    )
    parser.add_argument(
        "--keep-repro-pdfs",
        action="store_true",
        help="Do not delete PDFs newly written beside reproduction pickles",
    )
    args = parser.parse_args(argv)
    run = resolve_run_dir(args.out_root, args.tag or None)
    print(f"run: {run.run_dir}")
    run_repro_baseline(
        run,
        repo=repo,
        python=args.python,
        cleanup_created=not args.keep_repro_pdfs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
