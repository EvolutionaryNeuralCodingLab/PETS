"""Headless ellipse rotation-angle correction over a block registry."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eye_tracking_system_tools.analysis.block_registry import (
    BlockSpec,
    _infer_animal,
    _infer_block_num,
    read_paper_registry,
)
from eye_tracking_system_tools.preprocessing.conicoid.refined_io import (
    rotation_fixed_eye_csv_path,
    write_rotation_fixed_eye_table,
)
from eye_tracking_system_tools.preprocessing.conicoid.rotation_inputs import (
    SequentialVideoReader,
    discover_eye_video,
    jitter_report_path,
    load_jitter_eye,
    load_raw_dlc_table,
)
from eye_tracking_system_tools.preprocessing.conicoid.rotation_params import (
    RotationCorrectionParams,
    rotation_correction_params_path,
    try_read_rotation_params,
)
from eye_tracking_system_tools.preprocessing.conicoid.spin_max import (
    apply_jitter_to_spin_table,
    spin_maximize_eye_table,
)

ProgressFn = Callable[[str], None]

STATUS_OK = "ok"
STATUS_OK_JITTER_MISSING = "ok_jitter_missing"
STATUS_SKIPPED_EXISTS = "skipped_exists"
STATUS_SKIPPED_NO_PARAMS = "skipped_no_params"
STATUS_FAILED = "failed"


@dataclass
class BlockResult:
    animal: str
    block_path: Path
    status: str
    message: str = ""
    paths: dict[str, str] = field(default_factory=dict)
    n_ok: dict[str, int] = field(default_factory=dict)
    n: dict[str, int] = field(default_factory=dict)
    jitter_applied: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "animal": self.animal,
            "block": self.block_path.name,
            "block_path": str(self.block_path),
            "status": self.status,
            "message": self.message,
            "jitter_applied": self.jitter_applied,
            "left_path": self.paths.get("left", ""),
            "right_path": self.paths.get("right", ""),
            "left_n_ok": self.n_ok.get("left"),
            "right_n_ok": self.n_ok.get("right"),
        }


def load_rotation_registry(path: Path | str) -> list[BlockSpec]:
    """Load a paper ``animals:`` registry or a flat jitter ``blocks:`` list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if isinstance(data.get("animals"), dict):
        specs = read_paper_registry(path)
        if not specs:
            raise ValueError(f"{path}: animals mapping is empty")
        return specs
    rows = data.get("blocks")
    if not rows:
        raise ValueError(f"{path}: expected top-level 'animals' or 'blocks'")
    specs: list[BlockSpec] = []
    for row in rows:
        if not isinstance(row, dict) or "block_path" not in row:
            continue
        block_path = Path(row["block_path"]).expanduser()
        animal = str(row.get("animal") or _infer_animal(block_path))
        specs.append(
            BlockSpec(
                animal=animal,
                block_path=block_path,
                block_num=_infer_block_num(block_path),
            )
        )
    if not specs:
        raise ValueError(f"{path}: no usable block_path entries")
    return specs


def block_inventory(specs: Sequence[BlockSpec]) -> pd.DataFrame:
    """Mounted-block checklist for the notebook selector."""
    rows = []
    for spec in specs:
        analysis = spec.analysis_path
        left_ok = rotation_fixed_eye_csv_path(
            spec.block_path, "left", jitter_corrected=True
        )
        right_ok = rotation_fixed_eye_csv_path(
            spec.block_path, "right", jitter_corrected=True
        )
        left_nj = rotation_fixed_eye_csv_path(
            spec.block_path, "left", jitter_corrected=False
        )
        right_nj = rotation_fixed_eye_csv_path(
            spec.block_path, "right", jitter_corrected=False
        )
        rows.append(
            {
                "animal": spec.animal,
                "block": spec.block_path.name,
                "exists": spec.block_path.is_dir(),
                "params_yaml": rotation_correction_params_path(spec.block_path).is_file(),
                "jitter_pkl": jitter_report_path(spec.block_path).is_file(),
                "le_df": (analysis / "le_df.csv").is_file(),
                "re_df": (analysis / "re_df.csv").is_file(),
                "left_video": discover_eye_video(spec.block_path, "left") is not None,
                "right_video": discover_eye_video(spec.block_path, "right") is not None,
                "rotation_fixed": left_ok.is_file() and right_ok.is_file(),
                "jitter_not_corrected": left_nj.is_file() and right_nj.is_file(),
                "block_path": str(spec.block_path),
            }
        )
    return pd.DataFrame(rows)


def results_table(results: Sequence[BlockResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(
            columns=[
                "animal",
                "block",
                "status",
                "message",
                "jitter_applied",
                "left_n_ok",
                "right_n_ok",
                "block_path",
            ]
        )
    return pd.DataFrame([row.to_row() for row in results])


def _report(progress: ProgressFn | None, message: str) -> None:
    print(message, flush=True)
    if progress is not None:
        progress(message)


def _expected_output_paths(
    block_path: Path, *, jitter_corrected: bool
) -> dict[str, Path]:
    return {
        "left": rotation_fixed_eye_csv_path(
            block_path, "left", jitter_corrected=jitter_corrected
        ),
        "right": rotation_fixed_eye_csv_path(
            block_path, "right", jitter_corrected=jitter_corrected
        ),
    }


def _outputs_exist(paths: dict[str, Path]) -> bool:
    return all(path.is_file() for path in paths.values())


def _correct_one_eye(
    *,
    side: str,
    block_path: Path,
    params: RotationCorrectionParams,
    apply_jitter: bool,
    jitter_corrected: bool,
    progress: ProgressFn | None,
    show_tqdm: bool,
    job_title: str,
) -> tuple[Path, int, int]:
    settings = params.eye_settings()[side]
    raw = load_raw_dlc_table(block_path, side)
    video = discover_eye_video(block_path, side)
    if video is None:
        raise FileNotFoundError(
            f"{side} eye video not found under {Path(block_path) / 'eye_videos'}"
        )
    reader = SequentialVideoReader(video)
    try:
        if reader.nframes <= 0:
            raise RuntimeError(f"Could not open {side} eye video: {video}")
        if settings.frame_width is None:
            width = reader.frame_width()
            if width is not None:
                settings.frame_width = float(width)

        def grab(idx: int, _reader=reader) -> np.ndarray | None:
            return _reader.read_frame(int(idx), as_gray=True)

        def on_prog(cur: int, tot: int, _side: str = side) -> None:
            if progress is None:
                return
            progress(f"{_side} ellipse rotation: {cur}/{tot} frames")

        _report(
            progress,
            f"{job_title}, {side}:",
        )
        spun = spin_maximize_eye_table(
            raw,
            grab,
            settings=settings,
            progress=on_prog if progress is not None else None,
            progress_desc=side,
            show_tqdm=show_tqdm,
        )
        if apply_jitter:
            jitter = load_jitter_eye(block_path, side)
            if jitter is None:
                raise RuntimeError(
                    f"{side}: apply_jitter is True but jitter report is missing"
                )
            spun = apply_jitter_to_spin_table(spun, jitter)
        out_path = rotation_fixed_eye_csv_path(
            block_path, side, jitter_corrected=jitter_corrected
        )
        write_rotation_fixed_eye_table(spun, out_path)
        n_ok = int(np.isfinite(spun["phi_spin"].to_numpy(dtype=float)).sum())
        _report(
            progress,
            f"  {side} done: {n_ok}/{len(spun)} finite phi → {out_path}",
        )
        return out_path, n_ok, len(spun)
    finally:
        reader.close()


def run_block(
    block_path: Path | str,
    *,
    overwrite: bool = False,
    animal: str | None = None,
    progress: ProgressFn | None = None,
    show_tqdm: bool = True,
    block_index: int | None = None,
    n_blocks: int | None = None,
) -> BlockResult:
    """Correct both eyes for one block. Never aborts the caller on failure."""
    path = Path(block_path)
    animal_id = animal or _infer_animal(path)
    if block_index is not None and n_blocks is not None:
        job_title = (
            f"Working on block {block_index} out of {n_blocks}: "
            f"{animal_id} {path.name}"
        )
    else:
        job_title = f"Working on {animal_id} {path.name}"
    if not path.exists():
        _report(progress, job_title)
        return BlockResult(
            animal=animal_id,
            block_path=path,
            status=STATUS_FAILED,
            message=f"block path does not exist: {path}",
        )
    params_path = rotation_correction_params_path(path)
    params = try_read_rotation_params(path)
    if params is None:
        _report(progress, job_title)
        return BlockResult(
            animal=animal_id,
            block_path=path,
            status=STATUS_SKIPPED_NO_PARAMS,
            message=f"missing {params_path.name}",
        )

    jitter_left = load_jitter_eye(path, "left")
    jitter_right = load_jitter_eye(path, "right")
    jitter_available = jitter_left is not None and jitter_right is not None
    want_jitter = bool(params.apply_jitter)
    apply_jitter = want_jitter and jitter_available
    jitter_corrected = apply_jitter
    if want_jitter and not jitter_available:
        warning = (
            "WARNING: jitter_report_dict.pkl is missing or incomplete. "
            "Running ellipse rotation anyway; writing *_JitterNotCorrected.csv."
        )
        _report(progress, warning)
    elif not want_jitter:
        warning = (
            "apply_jitter is false in rotation_correction_params.yaml. "
            "Writing *_JitterNotCorrected.csv."
        )
        _report(progress, warning)
    else:
        warning = ""

    expected = _expected_output_paths(path, jitter_corrected=jitter_corrected)
    if not overwrite and _outputs_exist(expected):
        _report(progress, job_title)
        return BlockResult(
            animal=animal_id,
            block_path=path,
            status=STATUS_SKIPPED_EXISTS,
            message="output CSVs already exist (overwrite=False)",
            paths={side: str(p.resolve()) for side, p in expected.items()},
            jitter_applied=apply_jitter,
        )

    paths: dict[str, str] = {}
    n_ok: dict[str, int] = {}
    n_rows: dict[str, int] = {}
    errors: list[str] = []
    for side in ("left", "right"):
        try:
            out_path, ok_count, n_count = _correct_one_eye(
                side=side,
                block_path=path,
                params=params,
                apply_jitter=apply_jitter,
                jitter_corrected=jitter_corrected,
                progress=progress,
                show_tqdm=show_tqdm,
                job_title=job_title,
            )
            paths[side] = str(out_path.resolve())
            n_ok[side] = ok_count
            n_rows[side] = n_count
        except Exception as exc:
            errors.append(f"{side}: {exc}")
            _report(progress, f"  {side} failed: {exc}")

    if errors:
        return BlockResult(
            animal=animal_id,
            block_path=path,
            status=STATUS_FAILED,
            message="; ".join(errors),
            paths=paths,
            n_ok=n_ok,
            n=n_rows,
            jitter_applied=apply_jitter,
        )

    status = STATUS_OK if apply_jitter else STATUS_OK_JITTER_MISSING
    parts = [
        f"{side}: {n_ok[side]}/{n_rows[side]} finite phi → {paths[side]}"
        for side in ("left", "right")
        if side in paths
    ]
    message = " | ".join(parts)
    if warning:
        message = f"{warning} {message}"
    return BlockResult(
        animal=animal_id,
        block_path=path,
        status=status,
        message=message.strip(),
        paths=paths,
        n_ok=n_ok,
        n=n_rows,
        jitter_applied=apply_jitter,
    )


def run_registry(
    specs: Iterable[BlockSpec | Path | str],
    *,
    overwrite: bool = False,
    progress: ProgressFn | None = None,
    show_tqdm: bool = True,
) -> list[BlockResult]:
    """Run :func:`run_block` on each spec. Failures do not stop the list."""
    items: list[tuple[str | None, Path]] = []
    for spec in specs:
        if isinstance(spec, BlockSpec):
            items.append((spec.animal, spec.block_path))
        else:
            items.append((None, Path(spec)))

    n_blocks = len(items)
    results: list[BlockResult] = []
    for index, (animal, block_path) in enumerate(items, start=1):
        result = run_block(
            block_path,
            overwrite=overwrite,
            animal=animal,
            progress=progress,
            show_tqdm=show_tqdm,
            block_index=index,
            n_blocks=n_blocks,
        )
        if result.status in (STATUS_OK, STATUS_OK_JITTER_MISSING):
            _report(progress, f"  {result.status}")
        else:
            _report(progress, f"  {result.status}: {result.message}")
        results.append(result)
    return results


def _print_summary(results: Sequence[BlockResult]) -> None:
    table = results_table(results)
    if table.empty:
        print("No blocks processed.", flush=True)
        return
    counts = table["status"].value_counts().to_dict()
    print("\nRotation-correction summary", flush=True)
    print(table.to_string(index=False), flush=True)
    print("\nCounts:", counts, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch ellipse rotation-angle correction. "
            "Requires analysis/rotation_correction_params.yaml per block."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="YAML registry (animals: paper style, or blocks: jitter style).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing rotation_fixed CSVs (default: skip if present).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    specs = load_rotation_registry(args.registry)
    results = run_registry(specs, overwrite=bool(args.overwrite))
    _print_summary(results)
    n_failed = sum(1 for row in results if row.status == STATUS_FAILED)
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
