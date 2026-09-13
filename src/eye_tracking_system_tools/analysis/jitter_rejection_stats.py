"""Overall interframe-jitter rejection counts from existing jitter reports.

Counts video-frame samples that would be dropped by the preprocessing
``diff_threshold`` gate: keep frames whose interframe change in
``top_correlation_dist`` is below 6 pixels, i.e. reject when
``np.diff(top_correlation_dist) > 5`` (same signed test as
``find_jittery_frames``). This does **not** apply ``max_distance`` or
``gap_to_bridge``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from eye_tracking_system_tools.analysis.block_registry import (
    BlockSpec,
    read_paper_registry,
)
from eye_tracking_system_tools.analysis.jitter_epochs import (
    extract_amplitude_trace,
    infer_animal,
    load_jitter_report,
    read_registry_blocks,
)

DEFAULT_DIFF_THRESHOLD_PX = 5.0
EYES = ("left_eye", "right_eye")
CSV_NAME = "jitter_rejections.csv"
EXPERIMENT_VOLUME_ROOTS = (
    "/Volumes/Data-1/Nimrod/experiments",
    "/Volumes/Data-2/Nimrod/experiments",
    "/Volumes/Data/Nimrod/experiments",
)


@dataclass(frozen=True)
class EyeJitterStats:
    animal: str
    block: str
    block_path: Path
    eye: str
    n_frames: int
    n_frames_removed: int
    median_dist_px: float
    p95_dist_px: float
    max_dist_px: float
    median_abs_diff_px: float
    p95_abs_diff_px: float
    max_abs_diff_px: float
    status: str
    note: str = ""

    @property
    def pct_frames_removed(self) -> float:
        if self.n_frames <= 0:
            return float("nan")
        return 100.0 * self.n_frames_removed / self.n_frames


def load_dataset_specs(path: Path | str) -> list[BlockSpec]:
    """Load a paper ``animals:`` registry or a jitter ``blocks:`` registry."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data.get("animals"), dict):
        specs = read_paper_registry(path)
        if not specs:
            raise ValueError(f"{path}: animals mapping is empty")
        return specs
    if data.get("blocks"):
        jitter_specs = read_registry_blocks(path)
        return [
            BlockSpec(
                animal=js.animal,
                block_path=js.block_path,
                block_num=js.block_path.name,
            )
            for js in jitter_specs
        ]
    raise ValueError(f"{path}: expected top-level 'animals' or 'blocks'")


def resolve_block_path(path: Path | str) -> Path:
    """Use the registered path, or the same animal/date/block on a sibling volume."""
    path = Path(path)
    if path.exists():
        return path
    text = str(path)
    for src in EXPERIMENT_VOLUME_ROOTS:
        prefix = src.rstrip("/")
        if text == prefix or text.startswith(prefix + "/"):
            rest = text[len(prefix) :]
            for alt in EXPERIMENT_VOLUME_ROOTS:
                if alt == src:
                    continue
                candidate = Path(alt.rstrip("/") + rest)
                if candidate.exists():
                    return candidate
    return path


def rebase_block_path(path: Path | str, data_root: Path | str | None) -> Path:
    """If ``data_root`` is set, replace the ``.../experiments`` prefix with it."""
    path = Path(path)
    if data_root is None:
        return resolve_block_path(path)
    root = Path(data_root)
    parts = path.parts
    if "experiments" in parts:
        i = parts.index("experiments")
        candidate = root.joinpath(*parts[i + 1 :])
        if candidate.exists():
            return candidate
        return candidate
    return resolve_block_path(path)


def count_diff_rejections(
    dist: np.ndarray,
    *,
    threshold: float = DEFAULT_DIFF_THRESHOLD_PX,
) -> tuple[int, int]:
    """Return ``(n_frames, n_removed)`` for one ``top_correlation_dist`` trace.

    A sample is removed when the signed interframe difference exceeds
    ``threshold`` (default 5 px → keep only diffs below 6 px).
    """
    arr = np.asarray(dist, dtype=float)
    n_frames = int(arr.size)
    if n_frames < 2:
        return n_frames, 0
    diff = np.diff(arr)
    finite = np.isfinite(diff)
    n_removed = int(np.count_nonzero(finite & (diff > threshold)))
    return n_frames, n_removed


def _finite_summary(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(finite)),
        float(np.percentile(finite, 95)),
        float(np.max(finite)),
    )


def summarize_eye_trace(
    dist: np.ndarray,
    *,
    animal: str,
    block: str,
    block_path: Path,
    eye: str,
    threshold: float,
) -> EyeJitterStats:
    arr = np.asarray(dist, dtype=float)
    n_frames, n_removed = count_diff_rejections(arr, threshold=threshold)
    median_d, p95_d, max_d = _finite_summary(arr)
    if arr.size >= 2:
        abs_diff = np.abs(np.diff(arr))
    else:
        abs_diff = np.asarray([], dtype=float)
    median_ad, p95_ad, max_ad = _finite_summary(abs_diff)
    return EyeJitterStats(
        animal=animal,
        block=block,
        block_path=block_path,
        eye=eye,
        n_frames=n_frames,
        n_frames_removed=n_removed,
        median_dist_px=median_d,
        p95_dist_px=p95_d,
        max_dist_px=max_d,
        median_abs_diff_px=median_ad,
        p95_abs_diff_px=p95_ad,
        max_abs_diff_px=max_ad,
        status="ok",
    )


def _missing_eye(
    *,
    animal: str,
    block: str,
    block_path: Path,
    eye: str,
    status: str,
    note: str,
) -> EyeJitterStats:
    return EyeJitterStats(
        animal=animal,
        block=block,
        block_path=block_path,
        eye=eye,
        n_frames=0,
        n_frames_removed=0,
        median_dist_px=float("nan"),
        p95_dist_px=float("nan"),
        max_dist_px=float("nan"),
        median_abs_diff_px=float("nan"),
        p95_abs_diff_px=float("nan"),
        max_abs_diff_px=float("nan"),
        status=status,
        note=note,
    )


def collect_block_eye_stats(
    spec: BlockSpec,
    *,
    threshold: float = DEFAULT_DIFF_THRESHOLD_PX,
    data_root: Path | str | None = None,
) -> list[EyeJitterStats]:
    animal = spec.animal or infer_animal(spec.block_path)
    block_path = rebase_block_path(spec.block_path, data_root)
    block = block_path.name
    report_path = block_path / "analysis" / "jitter_report_dict.pkl"
    if not block_path.exists():
        return [
            _missing_eye(
                animal=animal,
                block=block,
                block_path=block_path,
                eye=eye,
                status="missing_block",
                note="block path does not exist",
            )
            for eye in EYES
        ]
    if not report_path.is_file():
        return [
            _missing_eye(
                animal=animal,
                block=block,
                block_path=block_path,
                eye=eye,
                status="missing_report",
                note="analysis/jitter_report_dict.pkl not found",
            )
            for eye in EYES
        ]
    try:
        report = load_jitter_report(report_path)
        traces = extract_amplitude_trace(report, eye="both")
    except Exception as exc:
        return [
            _missing_eye(
                animal=animal,
                block=block,
                block_path=block_path,
                eye=eye,
                status="load_error",
                note=str(exc),
            )
            for eye in EYES
        ]
    rows: list[EyeJitterStats] = []
    for eye in EYES:
        dist = traces.get(eye)
        if dist is None:
            rows.append(
                _missing_eye(
                    animal=animal,
                    block=block,
                    block_path=block_path,
                    eye=eye,
                    status="missing_eye",
                    note=f"{eye} missing in jitter report",
                )
            )
            continue
        rows.append(
            summarize_eye_trace(
                dist,
                animal=animal,
                block=block,
                block_path=block_path,
                eye=eye,
                threshold=threshold,
            )
        )
    return rows


def collect_dataset_eye_stats(
    specs: Sequence[BlockSpec],
    *,
    threshold: float = DEFAULT_DIFF_THRESHOLD_PX,
    data_root: Path | str | None = None,
    verbose: bool = False,
) -> list[EyeJitterStats]:
    rows: list[EyeJitterStats] = []
    n = len(specs)
    for i, spec in enumerate(specs, start=1):
        if verbose:
            print(f"[{i}/{n}] {spec.animal} {spec.block_path.name}", flush=True)
        rows.extend(
            collect_block_eye_stats(spec, threshold=threshold, data_root=data_root)
        )
    return rows


def _ok_rows(rows: Iterable[EyeJitterStats]) -> list[EyeJitterStats]:
    return [r for r in rows if r.status == "ok"]


def _pct(n_removed: int, n_frames: int) -> float:
    if n_frames <= 0:
        return float("nan")
    return 100.0 * n_removed / n_frames


def _aggregate_row(
    *,
    level: str,
    animal: str,
    block: str,
    block_path: str,
    rows: Sequence[EyeJitterStats],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ok = _ok_rows(rows)
    n_frames = int(sum(r.n_frames for r in ok))
    n_removed = int(sum(r.n_frames_removed for r in ok))
    median_d = float(np.nanmedian([r.median_dist_px for r in ok])) if ok else float("nan")
    p95_d = float(np.nanmedian([r.p95_dist_px for r in ok])) if ok else float("nan")
    max_d = float(np.nanmax([r.max_dist_px for r in ok])) if ok else float("nan")
    median_ad = float(np.nanmedian([r.median_abs_diff_px for r in ok])) if ok else float("nan")
    p95_ad = float(np.nanmedian([r.p95_abs_diff_px for r in ok])) if ok else float("nan")
    max_ad = float(np.nanmax([r.max_abs_diff_px for r in ok])) if ok else float("nan")
    n_missing = int(sum(1 for r in rows if r.status != "ok"))
    notes = sorted({r.note for r in rows if r.note})
    payload = {
        "level": level,
        "animal": animal,
        "block": block,
        "eye": "both",
        "block_path": block_path,
        "n_blocks": int(len({(r.animal, r.block) for r in rows})),
        "n_ok_eyes": len(ok),
        "n_missing_eyes": n_missing,
        "n_frames": n_frames,
        "n_frames_removed": n_removed,
        "pct_frames_removed": _pct(n_removed, n_frames),
        "median_dist_px": median_d,
        "p95_dist_px": p95_d,
        "max_dist_px": max_d,
        "median_abs_diff_px": median_ad,
        "p95_abs_diff_px": p95_ad,
        "max_abs_diff_px": max_ad,
        "status": "ok" if ok else (rows[0].status if rows else "empty"),
        "note": "; ".join(notes),
    }
    if extra:
        payload.update(extra)
    return payload


def build_rejection_table(
    rows: Sequence[EyeJitterStats],
    *,
    threshold: float = DEFAULT_DIFF_THRESHOLD_PX,
) -> pd.DataFrame:
    """One CSV with block, animal, and all-animals rows."""
    records: list[dict[str, Any]] = []

    by_block: dict[tuple[str, str], list[EyeJitterStats]] = {}
    for row in rows:
        by_block.setdefault((row.animal, row.block), []).append(row)
    for (animal, block), group in by_block.items():
        records.append(
            _aggregate_row(
                level="block",
                animal=animal,
                block=block,
                block_path=str(group[0].block_path),
                rows=group,
            )
        )

    by_animal: dict[str, list[EyeJitterStats]] = {}
    for row in rows:
        by_animal.setdefault(row.animal, []).append(row)
    for animal, group in by_animal.items():
        records.append(
            _aggregate_row(
                level="animal",
                animal=animal,
                block="",
                block_path="",
                rows=group,
            )
        )

    records.append(
        _aggregate_row(
            level="all",
            animal="ALL",
            block="",
            block_path="",
            rows=list(rows),
        )
    )

    df = pd.DataFrame.from_records(records)
    df["diff_threshold_px"] = float(threshold)
    level_order = {"block": 0, "animal": 1, "all": 2}
    df["_level_ord"] = df["level"].map(level_order).fillna(9)
    df = df.sort_values(["_level_ord", "animal", "block"], kind="mergesort")
    df = df.drop(columns=["_level_ord"])
    return df.reset_index(drop=True)


def write_rejection_csv(
    df: pd.DataFrame,
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def format_summary(df: pd.DataFrame) -> str:
    show = df[df["level"].isin(["animal", "all"])].copy()
    cols = [
        "level",
        "animal",
        "n_blocks",
        "n_frames",
        "n_frames_removed",
        "pct_frames_removed",
    ]
    view = show[cols]
    return view.to_string(
        index=False,
        formatters={"pct_frames_removed": lambda x: f"{x:.4f}" if pd.notna(x) else "nan"},
    )


def run_jitter_rejection_stats(
    *,
    registry: Path | str,
    out_csv: Path | str,
    threshold: float = DEFAULT_DIFF_THRESHOLD_PX,
    data_root: Path | str | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, Path]:
    specs = load_dataset_specs(registry)
    if not specs:
        raise ValueError(f"{registry}: no blocks listed")
    rows = collect_dataset_eye_stats(
        specs, threshold=threshold, data_root=data_root, verbose=verbose
    )
    df = build_rejection_table(rows, threshold=threshold)
    path = write_rejection_csv(df, out_csv)
    return df, path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Count frames rejected by the interframe jitter-diff gate "
            f"(diff > {DEFAULT_DIFF_THRESHOLD_PX:g} px) across a block registry."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo / "configs" / "paper_blocks.yaml",
        help="Paper animals: registry or jitter blocks: registry",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo / "outputs" / "jitter_rejection_overall" / CSV_NAME,
        help="Output CSV path",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=DEFAULT_DIFF_THRESHOLD_PX,
        help="Reject when np.diff(top_correlation_dist) exceeds this (pixels)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional local experiments root replacing .../experiments on registry paths",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    df, path = run_jitter_rejection_stats(
        registry=args.registry,
        out_csv=args.out,
        threshold=float(args.diff_threshold),
        data_root=args.data_root,
        verbose=True,
    )
    print(f"registry: {args.registry}")
    print(f"threshold: reject when np.diff(top_correlation_dist) > {args.diff_threshold:g} px")
    print()
    print(format_summary(df))
    print()
    print(f"wrote: {path}")
    missing = df[(df["level"] == "block") & (df["n_ok_eyes"] == 0)]
    if not missing.empty:
        print(f"warning: {len(missing)} block(s) had no usable jitter traces")
        for rec in missing.itertuples(index=False):
            print(f"  {rec.animal} {rec.block}: {rec.status} ({rec.note})")
    all_row = df[df["level"] == "all"].iloc[0]
    if int(all_row["n_frames"]) == 0:
        print(
            "error: no jitter traces loaded. Mount the experiment volume "
            "(/Volumes/Data-1, Data-2, or Data) or pass --data-root.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
