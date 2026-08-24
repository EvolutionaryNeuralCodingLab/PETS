"""
Data yield report: DLC likelihood histograms + ellipse completeness after removals.

For each registered block (modular / rigid / mouse / turtle):

1. Resolve the latest DeepLabCut CSV per eye (filtered-preferring) and pool
   Pupil/edge likelihood samples.
2. Load preferred ``*raw_verified*`` eye CSVs, apply ``noise_epochs_*`` and
   ``manual_event_annotations.csv`` in memory (never write disk), then measure
   fitted-ellipse yield and missing-data epoch structure.

Figures follow the jitter grouping: modular+rigid overlay; mouse alone; turtle alone.
"""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams

from eye_tracking_system_tools.analysis.export_meta import write_pickle_with_meta
from eye_tracking_system_tools.analysis.jitter_epochs import (
    JitterBlockSpec,
    infer_date,
    read_registry_blocks,
    write_jitter_registry,
)
from eye_tracking_system_tools.analysis.plot_bundle import begin_plot_bundle, finish_plot_bundle
from eye_tracking_system_tools.analysis.run_layout import (
    assert_not_reproduction,
)
from eye_tracking_system_tools.preprocessing.block_sync_core import (
    drop_pandas_index_artifact_columns,
)
from eye_tracking_system_tools.preprocessing.dlc_csv_io import (
    default_dlc_csv,
    likelihood_threshold_stats,
    list_dlc_csvs,
    load_dlc_likelihood_values,
    load_dlc_likelihood_values_many,
)
from eye_tracking_system_tools.preprocessing.noise_epochs import (
    GEOMETRY_COLS,
    mask_eye_df_by_epochs,
    read_noise_epochs,
    resolve_frame_col,
)

rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

YIELD_MOUNT_TYPES: tuple[str, ...] = ("modular", "rigid", "mouse", "turtle")
SHORT_GAP_MAX_FRAMES = 5
DEFAULT_LIKELIHOOD_THR = 0.95
MANUAL_ANNOTATIONS_NAME = "manual_event_annotations.csv"
MANUAL_TAG_COL = "manual_outlier_detected"
ANGLE_COLS = ("k_phi", "k_theta", "pupil_diameter")

COLOR_MODULAR = "#0072B2"
COLOR_RIGID = "#D55E00"
COLOR_MOUSE = "gray"
COLOR_TURTLE = "#009E73"

YIELD_REGISTRY_HEADER = """# Data yield report registry.
# Same schema as jitter_mount_blocks.yaml, plus turtle system condition.
#
# mount_type:
#   modular | rigid  -> pooled into *_modular_vs_rigid.pdf
#   mouse            -> dedicated *_mouse.pdf
#   turtle           -> dedicated *_turtle.pdf
#
# Edit by hand, or populate with:
#   development/data_yield_report_tool.ipynb
"""

BUNDLE_VERSION = 1
BUNDLE_NAME = "data_yield_report.pickle"
PER_BLOCK_CSV = "per_block_yield.csv"
POOL_SUMMARY_YAML = "yield_pool_summary.yaml"
MANIFEST_NAME = "manifest.yaml"
FOLDER_PREFIX = "yield_report"


# ---------------------------------------------------------------------------
# Paths / resolution
# ---------------------------------------------------------------------------


def resolve_eye_folder(block_path: Path | str, side: str) -> Path | None:
    """
    Locate LE/RE eye folder (same nested-subdir rule as BlockSync).

    ``side`` is ``left`` / ``right`` (or LE/RE aliases).
    """
    block_path = Path(block_path)
    side_lc = str(side).strip().lower()
    folder = "LE" if side_lc in ("left", "l", "le", "left_eye") else "RE"
    root = block_path / "eye_videos" / folder
    if not root.exists():
        return None
    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    return subdirs[0] if subdirs else root


def resolve_block_dlc_csvs(block_path: Path | str) -> dict[str, Path | None]:
    """Return ``{left, right}`` DLC CSV paths (None when missing)."""
    block_path = Path(block_path)
    out: dict[str, Path | None] = {"left": None, "right": None}
    for side in ("left", "right"):
        folder = resolve_eye_folder(block_path, side)
        if folder is None:
            continue
        cands = list_dlc_csvs(folder)
        if cands:
            out[side] = default_dlc_csv(cands)
    return out


def _eye_csv_candidates(analysis_path: Path, side: str) -> list[Path]:
    pattern = "left_eye_data*.csv" if side == "left" else "right_eye_data*.csv"
    return sorted(analysis_path.glob(pattern))


@dataclass(frozen=True)
class YieldEyeCsvChoice:
    side: str
    path: Path
    rule: str  # "raw_verified" | "newest"


def resolve_yield_eye_csv(analysis_path: Path | str, side: str) -> YieldEyeCsvChoice:
    """
    Prefer newest ``*raw_verified*`` eye CSV; else newest ``left/right_eye_data*.csv``.

    Unlike analysis ``eye_trace_io``, Kerr angles are not required.
    """
    analysis_path = Path(analysis_path)
    side_lc = "left" if str(side).lower().startswith("l") else "right"
    cands = _eye_csv_candidates(analysis_path, side_lc)
    if not cands:
        raise FileNotFoundError(
            f"{analysis_path}: no {side_lc} eye CSVs matching left/right_eye_data*.csv"
        )
    raw = [p for p in cands if "raw_verified" in p.name]
    if raw:
        chosen = max(raw, key=lambda p: p.stat().st_mtime)
        return YieldEyeCsvChoice(side=side_lc, path=chosen.resolve(), rule="raw_verified")
    chosen = max(cands, key=lambda p: p.stat().st_mtime)
    return YieldEyeCsvChoice(side=side_lc, path=chosen.resolve(), rule="newest")


def load_eye_dataframe(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return drop_pandas_index_artifact_columns(df.copy())


# ---------------------------------------------------------------------------
# In-memory removal masks
# ---------------------------------------------------------------------------


def _truthy_manual_flag(series: pd.Series) -> pd.Series:
    """Accept True / 'true' / 1 / '1' as flagged outliers."""
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def read_manual_outlier_intervals(
    block_path: Path | str,
    eye: str,
) -> list[tuple[float, float]]:
    """
    Return ``[(start_ms, end_ms), ...]`` for flagged manual outliers on one eye.

    Missing file → empty list. Rows without a truthy ``manual_outlier_detected``
    are ignored.
    """
    path = Path(block_path) / "analysis" / MANUAL_ANNOTATIONS_NAME
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    if df.empty or MANUAL_TAG_COL not in df.columns:
        return []
    eye_lc = "left" if str(eye).lower().startswith("l") else "right"
    eye_col = df["eye"].astype(str).str.strip().str.lower() if "eye" in df.columns else None
    if eye_col is None:
        return []
    eye_match = eye_col.isin(
        {eye_lc, eye_lc[0], f"{eye_lc}_eye", "l" if eye_lc == "left" else "r",
         "le" if eye_lc == "left" else "re"}
    )
    flagged = _truthy_manual_flag(df[MANUAL_TAG_COL])
    keep = df.loc[eye_match & flagged]
    if keep.empty or "start_ms" not in keep.columns or "end_ms" not in keep.columns:
        return []
    out: list[tuple[float, float]] = []
    for _, row in keep.iterrows():
        s = float(pd.to_numeric(row["start_ms"], errors="coerce"))
        e = float(pd.to_numeric(row["end_ms"], errors="coerce"))
        if np.isfinite(s) and np.isfinite(e) and e >= s:
            out.append((s, e))
    return out


def mask_eye_df_by_manual_ms(
    df: pd.DataFrame,
    intervals: Iterable[tuple[float, float]],
    *,
    ms_col: str = "ms_axis",
    extra_cols: Collection[str] = ANGLE_COLS,
) -> tuple[pd.DataFrame, int]:
    """
    NaN geometry (+ optional angle cols) where ``ms_col`` falls in any interval.

    ``n_hit`` counts rows that had finite ``center_x`` (or any target col) and matched.
    """
    intervals = list(intervals)
    if df is None or df.empty or not intervals:
        return (df.copy() if df is not None else pd.DataFrame()), 0
    if ms_col not in df.columns:
        return df.copy(), 0

    out = df.copy()
    ms = pd.to_numeric(out[ms_col], errors="coerce")
    hit = pd.Series(False, index=out.index)
    for start_ms, end_ms in intervals:
        hit |= (ms >= float(start_ms)) & (ms <= float(end_ms))

    cols = [c for c in (*GEOMETRY_COLS, *extra_cols) if c in out.columns]
    if "center_x" in out.columns:
        has_geom = out["center_x"].notna()
        n_hit = int((hit & has_geom).sum())
    else:
        n_hit = int(hit.sum())
    if cols and hit.any():
        out.loc[hit, cols] = np.nan
    return out, n_hit


def apply_cached_removals(
    df: pd.DataFrame,
    eye: str,
    *,
    noise_epochs: pd.DataFrame | None = None,
    manual_intervals: Iterable[tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Apply already-loaded noise epochs + manual intervals (no disk I/O).

    Returns ``(masked_df, {"n_noise_hit", "n_manual_hit"})``.
    """
    eye_lc = "left" if str(eye).lower().startswith("l") else "right"
    report = {"n_noise_hit": 0, "n_manual_hit": 0}
    out = df.copy()

    if noise_epochs is not None and not noise_epochs.empty:
        try:
            frame_col = resolve_frame_col(out, eye_lc)
        except KeyError:
            frame_col = None
        if frame_col is not None:
            out, n_noise = mask_eye_df_by_epochs(
                out, noise_epochs, frame_col=frame_col, categories=None
            )
            report["n_noise_hit"] = int(n_noise)

    intervals = list(manual_intervals or ())
    if intervals:
        out, n_manual = mask_eye_df_by_manual_ms(out, intervals)
        report["n_manual_hit"] = int(n_manual)
    return out, report


def apply_yield_removals_in_memory(
    df: pd.DataFrame,
    block_path: Path | str,
    eye: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Apply all noise-epoch categories then manual outlier intervals (in memory).

    Returns ``(masked_df, {"n_noise_hit", "n_manual_hit"})``.
    """
    block_path = Path(block_path)
    eye_lc = "left" if str(eye).lower().startswith("l") else "right"
    return apply_cached_removals(
        df,
        eye_lc,
        noise_epochs=read_noise_epochs(block_path, eye_lc),
        manual_intervals=read_manual_outlier_intervals(block_path, eye_lc),
    )


# ---------------------------------------------------------------------------
# Fitted / missing stats
# ---------------------------------------------------------------------------


def fitted_mask(df: pd.DataFrame) -> np.ndarray:
    """True where the frame has a fitted ellipse (finite ``center_x``)."""
    if df is None or df.empty:
        return np.asarray([], dtype=bool)
    if "center_x" in df.columns:
        return np.isfinite(pd.to_numeric(df["center_x"], errors="coerce").to_numpy(dtype=float))
    for col in GEOMETRY_COLS:
        if col in df.columns:
            return np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
    return np.zeros(len(df), dtype=bool)


def missing_epoch_lengths(missing: np.ndarray) -> np.ndarray:
    """Run lengths of contiguous True runs in a boolean missing mask."""
    missing = np.asarray(missing, dtype=bool)
    if missing.size == 0 or not missing.any():
        return np.asarray([], dtype=int)
    # Pad with False edges so run boundaries are clean.
    padded = np.concatenate([[False], missing, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return (ends - starts).astype(int)


@dataclass
class MissingEpochStats:
    n_missing_frames: int = 0
    n_epochs: int = 0
    n_short_epochs: int = 0
    n_long_epochs: int = 0
    n_short_frames: int = 0
    n_long_frames: int = 0
    pct_missing_frames_short: float = float("nan")
    pct_missing_frames_long: float = float("nan")
    median_epoch_frames: float = float("nan")
    epoch_lengths: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_missing_frames": self.n_missing_frames,
            "n_epochs": self.n_epochs,
            "n_short_epochs": self.n_short_epochs,
            "n_long_epochs": self.n_long_epochs,
            "n_short_frames": self.n_short_frames,
            "n_long_frames": self.n_long_frames,
            "pct_missing_frames_short": self.pct_missing_frames_short,
            "pct_missing_frames_long": self.pct_missing_frames_long,
            "median_epoch_frames": self.median_epoch_frames,
        }


def summarize_missing_epochs(
    missing: np.ndarray,
    *,
    short_max: int = SHORT_GAP_MAX_FRAMES,
) -> MissingEpochStats:
    """
    Summarize contiguous missing runs.

    Short = length ≤ ``short_max`` frames. Percentages are of **missing frames**
    (not of all frames).
    """
    lengths = missing_epoch_lengths(missing)
    n_missing = int(np.asarray(missing, dtype=bool).sum())
    if lengths.size == 0:
        return MissingEpochStats(n_missing_frames=n_missing, epoch_lengths=lengths)

    short = lengths <= int(short_max)
    n_short_frames = int(lengths[short].sum())
    n_long_frames = int(lengths[~short].sum())
    pct_short = (100.0 * n_short_frames / n_missing) if n_missing else float("nan")
    pct_long = (100.0 * n_long_frames / n_missing) if n_missing else float("nan")
    return MissingEpochStats(
        n_missing_frames=n_missing,
        n_epochs=int(lengths.size),
        n_short_epochs=int(short.sum()),
        n_long_epochs=int((~short).sum()),
        n_short_frames=n_short_frames,
        n_long_frames=n_long_frames,
        pct_missing_frames_short=pct_short,
        pct_missing_frames_long=pct_long,
        median_epoch_frames=float(np.median(lengths)),
        epoch_lengths=lengths,
    )


@dataclass
class EyeYieldRecord:
    block_key: str
    animal: str
    date: str
    block: str
    mount_type: str
    eye: str
    block_path: str
    eye_csv: str
    eye_csv_rule: str
    dlc_csv: str | None
    n_frames: int
    n_fitted: int
    yield_pct: float
    n_noise_hit: int
    n_manual_hit: int
    missing: MissingEpochStats
    likelihood_n: int = 0
    likelihood_mean: float = float("nan")
    likelihood_frac_kept: float = float("nan")

    def as_row(self) -> dict[str, Any]:
        row = {
            "block_key": self.block_key,
            "animal": self.animal,
            "date": self.date,
            "block": self.block,
            "mount_type": self.mount_type,
            "eye": self.eye,
            "block_path": self.block_path,
            "eye_csv": self.eye_csv,
            "eye_csv_rule": self.eye_csv_rule,
            "dlc_csv": self.dlc_csv,
            "n_frames": self.n_frames,
            "n_fitted": self.n_fitted,
            "yield_pct": self.yield_pct,
            "n_noise_hit": self.n_noise_hit,
            "n_manual_hit": self.n_manual_hit,
            "likelihood_n": self.likelihood_n,
            "likelihood_mean": self.likelihood_mean,
            "likelihood_frac_kept": self.likelihood_frac_kept,
        }
        row.update({f"missing_{k}": v for k, v in self.missing.as_dict().items()})
        return row


# ---------------------------------------------------------------------------
# In-memory dataset (one network pull, reuse everywhere)
# ---------------------------------------------------------------------------

_EYE_KEEP_EXTRA = ("ms_axis", "eye_frame", "L_eye_frame", "R_eye_frame", "frame")


def _slim_eye_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed for yield / removal masking."""
    keep = [c for c in (*GEOMETRY_COLS, *ANGLE_COLS, *_EYE_KEEP_EXTRA) if c in df.columns]
    if not keep:
        return df.copy()
    return df.loc[:, keep].copy()


@dataclass
class LoadedEyePayload:
    """Per-eye artefacts loaded once from disk/network."""

    eye: str
    eye_csv: str | None = None
    eye_csv_rule: str | None = None
    dlc_csv: str | None = None
    eye_df: pd.DataFrame | None = None
    likelihood: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    noise_epochs: pd.DataFrame = field(default_factory=pd.DataFrame)
    manual_intervals: list[tuple[float, float]] = field(default_factory=list)
    has_noise_file: bool = False
    has_manual_file: bool = False
    error: str | None = None


@dataclass
class LoadedBlockPayload:
    spec: JitterBlockSpec
    left: LoadedEyePayload = field(default_factory=lambda: LoadedEyePayload(eye="left"))
    right: LoadedEyePayload = field(default_factory=lambda: LoadedEyePayload(eye="right"))
    error: str | None = None

    @property
    def block_key(self) -> str:
        return self.spec.block_key

    def eye_payload(self, eye: str) -> LoadedEyePayload:
        return self.left if str(eye).lower().startswith("l") else self.right


@dataclass
class YieldDataset:
    """
    RAM-resident yield inputs for a registry.

    Built once by :func:`load_yield_dataset`; QC / pool / finalize reuse it so
    large eye + DLC CSVs are not re-read from network storage.
    """

    specs: list[JitterBlockSpec]
    blocks: dict[str, LoadedBlockPayload] = field(default_factory=dict)
    loaded_utc: str = ""
    notes: str = ""

    def __len__(self) -> int:
        return len(self.blocks)

    def summarize(self) -> dict[str, Any]:
        n_eyes = 0
        n_eye_rows = 0
        n_lik = 0
        n_err = 0
        for blk in self.blocks.values():
            if blk.error:
                n_err += 1
            for eye in (blk.left, blk.right):
                if eye.error:
                    n_err += 1
                if eye.eye_df is not None:
                    n_eyes += 1
                    n_eye_rows += len(eye.eye_df)
                n_lik += int(eye.likelihood.size)
        return {
            "n_blocks": len(self.blocks),
            "n_eyes_loaded": n_eyes,
            "n_eye_rows": n_eye_rows,
            "n_likelihood_samples": n_lik,
            "n_errors": n_err,
            "loaded_utc": self.loaded_utc,
        }


def _load_one_eye(block_path: Path, eye: str) -> LoadedEyePayload:
    eye_lc = "left" if str(eye).lower().startswith("l") else "right"
    analysis = block_path / "analysis"
    payload = LoadedEyePayload(eye=eye_lc)
    payload.has_noise_file = (analysis / f"noise_epochs_{eye_lc}.csv").is_file()
    payload.has_manual_file = (analysis / MANUAL_ANNOTATIONS_NAME).is_file()
    try:
        payload.noise_epochs = read_noise_epochs(block_path, eye_lc)
    except Exception as exc:  # noqa: BLE001
        payload.error = f"noise_epochs: {exc}"
        payload.noise_epochs = pd.DataFrame()
    try:
        payload.manual_intervals = read_manual_outlier_intervals(block_path, eye_lc)
    except Exception as exc:  # noqa: BLE001
        payload.error = (payload.error + "; " if payload.error else "") + f"manual: {exc}"
        payload.manual_intervals = []

    dlc_map = resolve_block_dlc_csvs(block_path)
    dlc = dlc_map.get(eye_lc)
    if dlc is not None:
        payload.dlc_csv = dlc.name
        try:
            payload.likelihood = load_dlc_likelihood_values(dlc)
        except Exception as exc:  # noqa: BLE001
            payload.error = (payload.error + "; " if payload.error else "") + f"dlc: {exc}"
            payload.likelihood = np.asarray([], dtype=float)

    try:
        choice = resolve_yield_eye_csv(analysis, eye_lc)
        payload.eye_csv = choice.path.name
        payload.eye_csv_rule = choice.rule
        payload.eye_df = _slim_eye_dataframe(load_eye_dataframe(choice.path))
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        payload.error = (payload.error + "; " if payload.error else "") + f"eye_csv: {exc}"
    return payload


def load_yield_dataset(
    specs: list[JitterBlockSpec],
    *,
    verbose: bool = True,
) -> YieldDataset:
    """
    Pull all eye CSVs, DLC likelihoods, and removal catalogs into RAM once.

    Subsequent QC / pool / plot / finalize should pass the returned
    :class:`YieldDataset` so they never re-hit network storage.
    """
    dataset = YieldDataset(
        specs=list(specs),
        loaded_utc=datetime.now(timezone.utc).isoformat(),
    )
    n = len(specs)
    for i, spec in enumerate(specs, start=1):
        if verbose:
            print(f"[{i}/{n}] loading {spec.block_key} [{spec.mount_type}] …", flush=True)
        blk = LoadedBlockPayload(spec=spec)
        try:
            blk.left = _load_one_eye(Path(spec.block_path), "left")
            blk.right = _load_one_eye(Path(spec.block_path), "right")
        except Exception as exc:  # noqa: BLE001
            blk.error = str(exc)
            if verbose:
                print(f"  ERROR: {exc}", flush=True)
        dataset.blocks[spec.block_key] = blk
        if verbose:
            eyes_ok = sum(
                1
                for e in (blk.left, blk.right)
                if e.eye_df is not None
            )
            lik_n = int(blk.left.likelihood.size + blk.right.likelihood.size)
            print(
                f"  eyes={eyes_ok}/2  likelihood_samples={lik_n}"
                + (f"  err={blk.error}" if blk.error else ""),
                flush=True,
            )
    if verbose:
        print("load complete:", dataset.summarize(), flush=True)
    return dataset


def eye_yield_from_payload(
    spec: JitterBlockSpec,
    payload: LoadedEyePayload,
    *,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
) -> EyeYieldRecord | None:
    """Compute one eye's yield record from a cached payload (no disk I/O)."""
    if payload.eye_df is None:
        return None
    masked, hits = apply_cached_removals(
        payload.eye_df,
        payload.eye,
        noise_epochs=payload.noise_epochs,
        manual_intervals=payload.manual_intervals,
    )
    fitted = fitted_mask(masked)
    n_frames = int(fitted.size)
    n_fitted = int(fitted.sum())
    yield_pct = (100.0 * n_fitted / n_frames) if n_frames else float("nan")
    missing = summarize_missing_epochs(~fitted)

    vals = np.asarray(payload.likelihood, dtype=float)
    vals = vals[np.isfinite(vals)]
    lik_n = int(vals.size)
    lik_mean = float(np.mean(vals)) if lik_n else float("nan")
    lik_frac = (
        float(likelihood_threshold_stats(vals, likelihood_threshold)["frac_kept"])
        if lik_n
        else float("nan")
    )
    return EyeYieldRecord(
        block_key=spec.block_key,
        animal=spec.animal,
        date=infer_date(spec.block_path),
        block=Path(spec.block_path).name,
        mount_type=spec.mount_type,
        eye=payload.eye,
        block_path=str(Path(spec.block_path)),
        eye_csv=payload.eye_csv or "",
        eye_csv_rule=payload.eye_csv_rule or "",
        dlc_csv=payload.dlc_csv,
        n_frames=n_frames,
        n_fitted=n_fitted,
        yield_pct=yield_pct,
        n_noise_hit=hits["n_noise_hit"],
        n_manual_hit=hits["n_manual_hit"],
        missing=missing,
        likelihood_n=lik_n,
        likelihood_mean=lik_mean,
        likelihood_frac_kept=lik_frac,
    )


def collect_eye_yield_from_dataset(
    dataset: YieldDataset,
    *,
    include: Collection[str] | None = None,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
    verbose: bool = False,
) -> list[EyeYieldRecord]:
    keys = None if include is None else {str(k) for k in include}
    out: list[EyeYieldRecord] = []
    for spec in dataset.specs:
        if keys is not None and spec.block_key not in keys:
            continue
        blk = dataset.blocks.get(spec.block_key)
        if blk is None:
            continue
        recs: list[EyeYieldRecord] = []
        for eye_payload in (blk.left, blk.right):
            rec = eye_yield_from_payload(
                spec, eye_payload, likelihood_threshold=likelihood_threshold
            )
            if rec is not None:
                recs.append(rec)
        if verbose:
            print(f"{spec.block_key} [{spec.mount_type}]: {len(recs)} eye(s)")
        out.extend(recs)
    return out


def collect_likelihood_pools_from_dataset(
    dataset: YieldDataset,
    *,
    include: Collection[str] | None = None,
) -> dict[str, np.ndarray]:
    keys = None if include is None else {str(k) for k in include}
    buckets: dict[str, list[np.ndarray]] = {m: [] for m in YIELD_MOUNT_TYPES}
    for spec in dataset.specs:
        if keys is not None and spec.block_key not in keys:
            continue
        mt = str(spec.mount_type)
        if mt not in buckets:
            continue
        blk = dataset.blocks.get(spec.block_key)
        if blk is None:
            continue
        for eye_payload in (blk.left, blk.right):
            vals = np.asarray(eye_payload.likelihood, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                buckets[mt].append(vals)
    return {
        k: (np.concatenate(v) if v else np.asarray([], dtype=float))
        for k, v in buckets.items()
    }


def yield_registry_table_from_dataset(
    dataset: YieldDataset,
    *,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
) -> pd.DataFrame:
    """QC table built entirely from a loaded :class:`YieldDataset` (no disk I/O)."""
    rows: list[dict[str, object]] = []
    for spec in dataset.specs:
        blk = dataset.blocks.get(spec.block_key)
        if blk is None:
            rows.append(
                {
                    "animal": spec.animal,
                    "block": Path(spec.block_path).name,
                    "mount_type": spec.mount_type,
                    "block_path": str(spec.block_path),
                    "yield_note": "not in dataset",
                }
            )
            continue
        row: dict[str, object] = {
            "animal": spec.animal,
            "block": Path(spec.block_path).name,
            "mount_type": spec.mount_type,
            "dlc_left": blk.left.dlc_csv,
            "dlc_right": blk.right.dlc_csv,
            "eye_left": blk.left.eye_csv,
            "eye_left_rule": blk.left.eye_csv_rule,
            "eye_right": blk.right.eye_csv,
            "eye_right_rule": blk.right.eye_csv_rule,
            "noise_left": blk.left.has_noise_file,
            "noise_right": blk.right.has_noise_file,
            "manual_annotations": blk.left.has_manual_file or blk.right.has_manual_file,
            "yield_left_pct": None,
            "yield_right_pct": None,
            "likelihood_left_n": int(blk.left.likelihood.size),
            "likelihood_right_n": int(blk.right.likelihood.size),
            "block_path": str(spec.block_path),
        }
        notes = [e for e in (blk.error, blk.left.error, blk.right.error) if e]
        if notes:
            row["yield_note"] = "; ".join(notes)
        for eye_payload, key in ((blk.left, "yield_left_pct"), (blk.right, "yield_right_pct")):
            rec = eye_yield_from_payload(
                spec, eye_payload, likelihood_threshold=likelihood_threshold
            )
            if rec is not None:
                row[key] = round(rec.yield_pct, 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def collect_eye_yield(
    spec: JitterBlockSpec,
    *,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
) -> list[EyeYieldRecord]:
    """Compute per-eye yield records for one block (skips missing eyes)."""
    # Disk path — prefer :func:`load_yield_dataset` + ``eye_yield_from_payload`` in notebooks.
    analysis = Path(spec.block_path) / "analysis"
    dlc_paths = resolve_block_dlc_csvs(spec.block_path)
    records: list[EyeYieldRecord] = []
    for eye in ("left", "right"):
        try:
            choice = resolve_yield_eye_csv(analysis, eye)
        except FileNotFoundError:
            continue
        df = load_eye_dataframe(choice.path)
        masked, hits = apply_yield_removals_in_memory(df, spec.block_path, eye)
        fitted = fitted_mask(masked)
        n_frames = int(fitted.size)
        n_fitted = int(fitted.sum())
        yield_pct = (100.0 * n_fitted / n_frames) if n_frames else float("nan")
        missing = summarize_missing_epochs(~fitted)

        lik_n = 0
        lik_mean = float("nan")
        lik_frac = float("nan")
        dlc = dlc_paths.get(eye)
        if dlc is not None and dlc.is_file():
            vals = load_dlc_likelihood_values(dlc)
            lik_n = int(vals.size)
            if lik_n:
                lik_mean = float(np.mean(vals))
                lik_frac = float(
                    likelihood_threshold_stats(vals, likelihood_threshold)["frac_kept"]
                )

        records.append(
            EyeYieldRecord(
                block_key=spec.block_key,
                animal=spec.animal,
                date=infer_date(spec.block_path),
                block=Path(spec.block_path).name,
                mount_type=spec.mount_type,
                eye=eye,
                block_path=str(Path(spec.block_path)),
                eye_csv=choice.path.name,
                eye_csv_rule=choice.rule,
                dlc_csv=dlc.name if dlc is not None else None,
                n_frames=n_frames,
                n_fitted=n_fitted,
                yield_pct=yield_pct,
                n_noise_hit=hits["n_noise_hit"],
                n_manual_hit=hits["n_manual_hit"],
                missing=missing,
                likelihood_n=lik_n,
                likelihood_mean=lik_mean,
                likelihood_frac_kept=lik_frac,
            )
        )
    return records


def collect_all_eye_yields(
    specs: list[JitterBlockSpec],
    *,
    include: Collection[str] | None = None,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
    verbose: bool = False,
    dataset: YieldDataset | None = None,
) -> list[EyeYieldRecord]:
    if dataset is not None:
        return collect_eye_yield_from_dataset(
            dataset,
            include=include,
            likelihood_threshold=likelihood_threshold,
            verbose=verbose,
        )
    keys = None if include is None else {str(k) for k in include}
    out: list[EyeYieldRecord] = []
    for spec in specs:
        if keys is not None and spec.block_key not in keys:
            continue
        recs = collect_eye_yield(spec, likelihood_threshold=likelihood_threshold)
        if verbose:
            print(f"{spec.block_key} [{spec.mount_type}]: {len(recs)} eye(s)")
        out.extend(recs)
    return out


def collect_likelihood_pools(
    specs: list[JitterBlockSpec],
    *,
    include: Collection[str] | None = None,
    dataset: YieldDataset | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate DLC likelihood samples by mount type."""
    if dataset is not None:
        return collect_likelihood_pools_from_dataset(dataset, include=include)
    keys = None if include is None else {str(k) for k in include}
    buckets: dict[str, list[np.ndarray]] = {m: [] for m in YIELD_MOUNT_TYPES}
    for spec in specs:
        if keys is not None and spec.block_key not in keys:
            continue
        mt = str(spec.mount_type)
        if mt not in buckets:
            continue
        paths = [p for p in resolve_block_dlc_csvs(spec.block_path).values() if p is not None]
        if not paths:
            continue
        vals = load_dlc_likelihood_values_many(paths)
        if vals.size:
            buckets[mt].append(vals)
    return {
        k: (np.concatenate(v) if v else np.asarray([], dtype=float))
        for k, v in buckets.items()
    }


def records_to_dataframe(records: list[EyeYieldRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([r.as_row() for r in records])


def pool_yield_summary(
    records: list[EyeYieldRecord],
    *,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
    likelihood_pools: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Frame-weighted yield + missing-epoch aggregates per mount type."""
    df = records_to_dataframe(records)
    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "likelihood_threshold": float(likelihood_threshold),
        "short_gap_max_frames": SHORT_GAP_MAX_FRAMES,
        "groups": {},
        "blocks_used": sorted({r.block_key for r in records}),
    }
    for mount in YIELD_MOUNT_TYPES:
        sub = df[df["mount_type"] == mount] if not df.empty else df
        if sub.empty:
            summary["groups"][mount] = {
                "n_block_eyes": 0,
                "n_frames": 0,
                "n_fitted": 0,
                "yield_pct_weighted": float("nan"),
                "yield_pct_mean_unweighted": float("nan"),
                "pct_missing_frames_short": float("nan"),
                "pct_missing_frames_long": float("nan"),
                "median_epoch_frames": float("nan"),
                "likelihood_n": 0,
                "likelihood_mean": float("nan"),
                "likelihood_frac_kept": float("nan"),
            }
            continue
        n_frames = int(sub["n_frames"].sum())
        n_fitted = int(sub["n_fitted"].sum())
        n_missing = int(sub["missing_n_missing_frames"].sum())
        n_short = int(sub["missing_n_short_frames"].sum())
        n_long = int(sub["missing_n_long_frames"].sum())
        # Recompute median across concatenated epoch lengths when available.
        all_lengths = np.concatenate(
            [r.missing.epoch_lengths for r in records if r.mount_type == mount]
        )
        median_ep = float(np.median(all_lengths)) if all_lengths.size else float("nan")
        lik = (
            likelihood_pools.get(mount, np.asarray([], dtype=float))
            if likelihood_pools is not None
            else np.asarray([], dtype=float)
        )
        lik = lik[np.isfinite(lik)]
        lik_stats = likelihood_threshold_stats(lik, likelihood_threshold) if lik.size else {
            "frac_kept": float("nan")
        }
        summary["groups"][mount] = {
            "n_block_eyes": int(len(sub)),
            "n_blocks": int(sub["block_key"].nunique()),
            "n_frames": n_frames,
            "n_fitted": n_fitted,
            "yield_pct_weighted": (100.0 * n_fitted / n_frames) if n_frames else float("nan"),
            "yield_pct_mean_unweighted": float(sub["yield_pct"].mean()),
            "pct_missing_frames_short": (100.0 * n_short / n_missing) if n_missing else float("nan"),
            "pct_missing_frames_long": (100.0 * n_long / n_missing) if n_missing else float("nan"),
            "median_epoch_frames": median_ep,
            "n_missing_frames": n_missing,
            "likelihood_n": int(lik.size),
            "likelihood_mean": float(np.mean(lik)) if lik.size else float("nan"),
            "likelihood_frac_kept": float(lik_stats["frac_kept"]),
        }
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _hist_percent(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return bins[:-1], np.zeros(len(bins) - 1)
    hist, edges = np.histogram(values, bins=bins)
    return edges[:-1], (hist / values.size) * 100.0


def figure_likelihood_modular_vs_rigid(
    pools: dict[str, np.ndarray],
    *,
    n_bins: int = 50,
    xmin: float = 0.0,
    xmax: float = 1.0,
):
    mod = pools.get("modular", np.asarray([]))
    rig = pools.get("rigid", np.asarray([]))
    finite = np.concatenate([a[np.isfinite(a)] for a in (mod, rig) if a.size])
    if finite.size == 0:
        raise ValueError("No modular/rigid likelihood samples to plot")
    bins = np.linspace(xmin, xmax, n_bins + 1)
    fig, ax = plt.subplots(1, 1, figsize=(2.4, 1.8), dpi=150)
    width = np.diff(bins)
    x0, y0 = _hist_percent(mod, bins)
    x1, y1 = _hist_percent(rig, bins)
    ax.bar(x0, y0, width=width, align="edge", color=COLOR_MODULAR, edgecolor="black",
           alpha=0.55, label=f"modular (n={int(np.isfinite(mod).sum())})")
    ax.bar(x1, y1, width=width, align="edge", color=COLOR_RIGID, edgecolor="black",
           alpha=0.55, label=f"rigid (n={int(np.isfinite(rig).sum())})")
    ax.set_xlabel("DLC likelihood", fontsize=10)
    ax.set_ylabel("% samples", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def figure_likelihood_single(
    pools: dict[str, np.ndarray],
    mount: str,
    *,
    color: str = COLOR_MOUSE,
    n_bins: int = 50,
    xmin: float = 0.0,
    xmax: float = 1.0,
):
    vals = pools.get(mount, np.asarray([]))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"No {mount} likelihood samples to plot")
    bins = np.linspace(xmin, xmax, n_bins + 1)
    x, y = _hist_percent(vals, bins)
    fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.7), dpi=150)
    ax.bar(x, y, width=np.diff(bins), align="edge", color=color, edgecolor="black",
           label=f"{mount} (n={vals.size})")
    ax.set_xlabel("DLC likelihood", fontsize=10)
    ax.set_ylabel("% samples", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


ELLIPSE_METRIC_LABELS = ("yield", "miss ≤5 fr", "miss >5 fr")
ELLIPSE_METRIC_KEYS = (
    "yield_pct_weighted",
    "pct_missing_frames_short",
    "pct_missing_frames_long",
)
ELLIPSE_METRIC_EXPLAIN = (
    "yield: % frames with a fitted ellipse (finite center_x) after removals.\n"
    "miss ≤5 fr: of missing frames, % in short gaps (≤5 contiguous frames).\n"
    "miss >5 fr: of missing frames, % in longer missing epochs."
)
MOUNT_COLORS = {
    "modular": COLOR_MODULAR,
    "rigid": COLOR_RIGID,
    "mouse": COLOR_MOUSE,
    "turtle": COLOR_TURTLE,
}


def _group_metric_values(group: dict[str, Any]) -> list[float]:
    return [float(group.get(k, np.nan)) for k in ELLIPSE_METRIC_KEYS]


def _annotate_bars(ax, containers, *, fmt: str = "{:.1f}", fontsize: int = 6) -> None:
    """Write one-decimal labels centered on each bar."""
    for container in containers:
        for patch in container:
            height = patch.get_height()
            if not np.isfinite(height):
                continue
            ax.text(
                patch.get_x() + patch.get_width() / 2.0,
                float(height) + 1.2,
                fmt.format(float(height)),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                clip_on=False,
            )


def _add_metric_footnote(fig, *, y: float = 0.015) -> None:
    fig.text(
        0.5,
        y,
        ELLIPSE_METRIC_EXPLAIN,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#333333",
        linespacing=1.35,
    )


def figure_ellipse_yield_modular_vs_rigid(summary: dict[str, Any]):
    groups = summary.get("groups") or {}
    mod = groups.get("modular") or {}
    rig = groups.get("rigid") or {}
    mod_vals = _group_metric_values(mod)
    rig_vals = _group_metric_values(rig)
    if all(not np.isfinite(v) for v in (*mod_vals, *rig_vals)):
        raise ValueError("No modular/rigid yield stats to plot")

    x = np.arange(len(ELLIPSE_METRIC_LABELS))
    width = 0.36
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.6), dpi=150)
    c0 = ax.bar(
        x - width / 2,
        mod_vals,
        width,
        color=COLOR_MODULAR,
        edgecolor="black",
        alpha=0.8,
        label="modular",
    )
    c1 = ax.bar(
        x + width / 2,
        rig_vals,
        width,
        color=COLOR_RIGID,
        edgecolor="black",
        alpha=0.8,
        label="rigid",
    )
    _annotate_bars(ax, (c0, c1))
    ax.set_xticks(x)
    ax.set_xticklabels(list(ELLIPSE_METRIC_LABELS), fontsize=8)
    ax.set_ylabel("%", fontsize=10)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_title("Ellipse yield — modular vs rigid", fontsize=9)
    fig.subplots_adjust(bottom=0.38)
    _add_metric_footnote(fig)
    return fig


def figure_ellipse_yield_single(summary: dict[str, Any], mount: str, *, color: str):
    g = (summary.get("groups") or {}).get(mount) or {}
    vals = _group_metric_values(g)
    if all(not np.isfinite(v) for v in vals):
        raise ValueError(f"No {mount} yield stats to plot")
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6), dpi=150)
    x = np.arange(len(ELLIPSE_METRIC_LABELS))
    container = ax.bar(
        x,
        vals,
        color=color,
        edgecolor="black",
        alpha=0.85,
        width=0.65,
    )
    _annotate_bars(ax, (container,))
    ax.set_xticks(x)
    ax.set_xticklabels(list(ELLIPSE_METRIC_LABELS), fontsize=8, rotation=0)
    ax.set_ylabel("%", fontsize=10)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"Ellipse yield — {mount}", fontsize=9)
    fig.subplots_adjust(bottom=0.38)
    _add_metric_footnote(fig)
    return fig


def figure_ellipse_yield_by_condition(summary: dict[str, Any]):
    """
    Single barplot of frame-weighted ellipse yield % for all system conditions.

    Conditions with no finite yield are omitted.
    """
    groups = summary.get("groups") or {}
    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    for mount in YIELD_MOUNT_TYPES:
        g = groups.get(mount) or {}
        v = float(g.get("yield_pct_weighted", np.nan))
        if not np.isfinite(v):
            continue
        labels.append(mount)
        vals.append(v)
        colors.append(MOUNT_COLORS.get(mount, "gray"))
    if not vals:
        raise ValueError("No condition yield stats to plot")

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.4), dpi=150)
    x = np.arange(len(labels))
    container = ax.bar(x, vals, color=colors, edgecolor="black", alpha=0.85, width=0.7)
    _annotate_bars(ax, (container,))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("yield %", fontsize=10)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Ellipse yield by system condition", fontsize=9)
    fig.text(
        0.5,
        0.02,
        "yield: % frames with a fitted ellipse (finite center_x) after removals.",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#333333",
    )
    fig.subplots_adjust(bottom=0.22)
    return fig


def retag_dataset_block(
    dataset: YieldDataset,
    block_key: str,
    mount_type: str,
) -> JitterBlockSpec:
    """
    Change ``mount_type`` on one cached block in RAM (no reload from disk).

    Also updates the matching entry in ``dataset.specs``. Persist by writing the
    registry afterward (browser save or :func:`write_yield_registry`).
    """
    mt = str(mount_type).strip().lower()
    if mt not in YIELD_MOUNT_TYPES:
        raise ValueError(
            f"mount_type must be one of {YIELD_MOUNT_TYPES}, got {mount_type!r}"
        )
    blk = dataset.blocks.get(block_key)
    if blk is None:
        raise KeyError(f"block_key not in dataset: {block_key!r}")
    new_spec = JitterBlockSpec(
        animal=blk.spec.animal,
        block_path=blk.spec.block_path,
        mount_type=mt,  # type: ignore[arg-type]
    )
    blk.spec = new_spec
    for i, spec in enumerate(dataset.specs):
        if spec.block_key == block_key:
            dataset.specs[i] = new_spec
            break
    return new_spec

def _save_figure(fig, out_pdf: Path, *, show: bool = False) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    if show:
        from IPython.display import display

        display(fig)
    plt.close(fig)
    return out_pdf


def write_per_block_csv(records: list[EyeYieldRecord], metadata_dir: Path) -> Path:
    path = Path(metadata_dir) / PER_BLOCK_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    records_to_dataframe(records).to_csv(path, index=False)
    return path


def write_pool_summary_yaml(summary: dict[str, Any], metadata_dir: Path) -> Path:
    path = Path(metadata_dir) / POOL_SUMMARY_YAML
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    return path


def run_yield_report(
    specs: list[JitterBlockSpec],
    figures_dir: Path,
    metadata_dir: Path,
    *,
    include: Collection[str] | None = None,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
    n_bins: int = 50,
    show: bool = False,
    verbose: bool = True,
    dataset: YieldDataset | None = None,
) -> dict[str, Path]:
    """Compute stats, write metadata + figures. Returns written path map."""
    figures_dir = Path(figures_dir)
    metadata_dir = Path(metadata_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    assert_not_reproduction(figures_dir)
    assert_not_reproduction(metadata_dir)

    records = collect_all_eye_yields(
        specs,
        include=include,
        likelihood_threshold=likelihood_threshold,
        verbose=verbose,
        dataset=dataset,
    )
    lik_pools = collect_likelihood_pools(specs, include=include, dataset=dataset)
    summary = pool_yield_summary(
        records,
        likelihood_threshold=likelihood_threshold,
        likelihood_pools=lik_pools,
    )

    written: dict[str, Path] = {}
    written["per_block_csv"] = write_per_block_csv(records, metadata_dir)
    written["pool_summary"] = write_pool_summary_yaml(summary, metadata_dir)

    run_dir = Path(figures_dir)
    if run_dir.name in {"figures", "plots"}:
        run_dir = run_dir.parent

    def _yield_bundle(plot_id: str, pdf_name: str, fig, values) -> Path:
        if "modular_vs_rigid" in plot_id:
            mount_type, mounts = "modular_vs_rigid", {"modular", "rigid"}
        elif plot_id.endswith("_mouse"):
            mount_type, mounts = "mouse", {"mouse"}
        elif plot_id.endswith("_turtle"):
            mount_type, mounts = "turtle", {"turtle"}
        else:
            mount_type, mounts = plot_id, set()
        keys = [r.block_key for r in records if r.mount_type in mounts] if mounts else []
        animals = sorted({r.animal for r in records if r.mount_type in mounts}) if mounts else []
        bundle = begin_plot_bundle(
            run_dir,
            plot_id,
            kind="yield_histogram",
            logic_key="yield_histogram",
            cohort={
                "cohort": mount_type,
                "animals": animals,
                "block_keys": keys,
                "rule": "registry_mount_type",
                "mount_type": mount_type,
            },
        )
        pdf = _save_figure(fig, bundle.plots_dir / pdf_name, show=show)
        arr = np.asarray(values, dtype=float) if values is not None else np.array([])
        write_pickle_with_meta(
            {"values": arr, "n_bins": int(n_bins), "pdf_name": pdf_name, "units": "likelihood"},
            bundle.metadata_dir / "yield_values.pkl",
            meta={"plot_id": plot_id, "mount_type": mount_type},
            entrypoint="eye_tracking_system_tools.analysis.data_yield.run_yield_report",
        )
        finite = arr[np.isfinite(arr)] if arr.size else arr
        slice_summary = {
            "mount_type": mount_type,
            "n_samples": int(finite.size),
            "block_keys": keys,
        }
        with open(bundle.metadata_dir / "yield_pool_summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(slice_summary, f, sort_keys=False)
        finish_plot_bundle(bundle)
        return pdf

    # Likelihood figures
    try:
        fig = figure_likelihood_modular_vs_rigid(lik_pools, n_bins=n_bins)
        vals = np.concatenate(
            [np.asarray(lik_pools.get(m, []), dtype=float) for m in ("modular", "rigid") if len(lik_pools.get(m, []))]
        ) if lik_pools else np.array([])
        written["dlc_modular_vs_rigid"] = _yield_bundle(
            "yield_dlc_likelihood_modular_vs_rigid",
            "yield_dlc_likelihood_modular_vs_rigid.pdf",
            fig,
            vals,
        )
    except ValueError as exc:
        if verbose:
            print(f"skip DLC modular/rigid: {exc}")

    for mount, color, name in (
        ("mouse", COLOR_MOUSE, "yield_dlc_likelihood_mouse.pdf"),
        ("turtle", COLOR_TURTLE, "yield_dlc_likelihood_turtle.pdf"),
    ):
        try:
            fig = figure_likelihood_single(lik_pools, mount, color=color, n_bins=n_bins)
            written[f"dlc_{mount}"] = _yield_bundle(
                f"yield_dlc_likelihood_{mount}",
                name,
                fig,
                lik_pools.get(mount, []),
            )
        except ValueError as exc:
            if verbose:
                print(f"skip DLC {mount}: {exc}")

    # Ellipse yield figures
    try:
        fig = figure_ellipse_yield_modular_vs_rigid(summary)
        written["ellipse_modular_vs_rigid"] = _yield_bundle(
            "yield_ellipse_modular_vs_rigid",
            "yield_ellipse_modular_vs_rigid.pdf",
            fig,
            np.array([]),
        )
    except ValueError as exc:
        if verbose:
            print(f"skip ellipse modular/rigid: {exc}")

    for mount, color, name in (
        ("mouse", COLOR_MOUSE, "yield_ellipse_mouse.pdf"),
        ("turtle", COLOR_TURTLE, "yield_ellipse_turtle.pdf"),
    ):
        try:
            fig = figure_ellipse_yield_single(summary, mount, color=color)
            written[f"ellipse_{mount}"] = _yield_bundle(
                f"yield_ellipse_{mount}",
                name,
                fig,
                np.array([]),
            )
        except ValueError as exc:
            if verbose:
                print(f"skip ellipse {mount}: {exc}")

    try:
        fig = figure_ellipse_yield_by_condition(summary)
        written["ellipse_by_condition"] = _yield_bundle(
            "yield_ellipse_by_condition",
            "yield_ellipse_by_condition.pdf",
            fig,
            np.array([]),
        )
    except ValueError as exc:
        if verbose:
            print(f"skip ellipse by-condition: {exc}")

    # Bundle for redraw / finalize
    bundle = {
        "version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "records": records_to_dataframe(records),
        "likelihood_pools": {k: v for k, v in lik_pools.items()},
        "likelihood_threshold": float(likelihood_threshold),
        "n_bins": int(n_bins),
    }
    bundle_path = metadata_dir / BUNDLE_NAME
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    written["bundle"] = bundle_path
    return written


@dataclass
class YieldExportResult:
    export_dir: Path
    bundle_path: Path
    per_block_csv: Path
    summary_path: Path
    manifest_path: Path
    figures: dict[str, Path] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"export: {self.export_dir}"]
        for name, path in self.figures.items():
            lines.append(f"  {name}: {path.name}")
        lines.append(f"  data: {self.bundle_path.name}")
        return "\n".join(lines)


def export_folder_name(tag: str = "", *, when: datetime | None = None) -> str:
    when = when or datetime.now()
    safe = "".join(c if (c.isalnum() or c in "-.") else "_" for c in str(tag).strip())
    safe = safe.strip("_")
    parts = [FOLDER_PREFIX, safe, when.strftime("%Y%m%d"), when.strftime("%H"), when.strftime("%M")]
    return "_".join(p for p in parts if p)


def finalize_yield_export(
    specs: list[JitterBlockSpec],
    metadata_dir: Path,
    out_root: Path,
    *,
    tag: str = "",
    include: Collection[str] | None = None,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THR,
    n_bins: int = 50,
    notes: str = "",
    dataset: YieldDataset | None = None,
) -> YieldExportResult:
    """Write a dated self-contained yield report folder under ``out_root``."""
    out_root = Path(out_root)
    export_dir = out_root / export_folder_name(tag)
    export_dir.mkdir(parents=True, exist_ok=True)
    assert_not_reproduction(export_dir)

    fig_dir = export_dir / "figures"
    meta_dir = export_dir / "metadata"
    written = run_yield_report(
        specs,
        fig_dir,
        meta_dir,
        include=include,
        likelihood_threshold=likelihood_threshold,
        n_bins=n_bins,
        show=False,
        verbose=False,
        dataset=dataset,
    )

    figures: dict[str, Path] = {}
    for key, path in written.items():
        if key.startswith("dlc_") or key.startswith("ellipse_"):
            dest = export_dir / path.name
            if path.resolve() != dest.resolve():
                shutil.copy2(path, dest)
            figures[key] = dest

    for src_key, name in (
        ("per_block_csv", PER_BLOCK_CSV),
        ("pool_summary", POOL_SUMMARY_YAML),
        ("bundle", BUNDLE_NAME),
    ):
        src = written[src_key]
        dest = export_dir / name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)

    # metadata_dir arg kept for API symmetry with jitter finalize; artifacts
    # are regenerated into the export folder (not copied from working run).
    _ = metadata_dir

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "notes": notes,
        "n_specs": len(specs),
        "include": list(include) if include is not None else None,
        "likelihood_threshold": float(likelihood_threshold),
        "figures": {k: str(p.name) for k, p in figures.items()},
        "files": {
            "per_block_csv": PER_BLOCK_CSV,
            "pool_summary": POOL_SUMMARY_YAML,
            "bundle": BUNDLE_NAME,
        },
    }
    manifest_path = export_dir / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return YieldExportResult(
        export_dir=export_dir,
        bundle_path=export_dir / BUNDLE_NAME,
        per_block_csv=export_dir / PER_BLOCK_CSV,
        summary_path=export_dir / POOL_SUMMARY_YAML,
        manifest_path=manifest_path,
        figures=figures,
    )


def seed_yield_registry_from_jitter(
    yield_registry: Path | str,
    jitter_registry: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Copy jitter mount blocks into the yield registry.

    If the yield registry already has blocks and ``overwrite`` is False, leave it.
    """
    yield_registry = Path(yield_registry)
    existing = read_registry_blocks(yield_registry)
    if existing and not overwrite:
        return yield_registry
    specs = read_registry_blocks(jitter_registry)
    return write_yield_registry(yield_registry, specs)


def write_yield_registry(path: Path | str, specs: list[JitterBlockSpec]) -> Path:
    """Write yield registry with yield-specific header comment."""
    path = Path(path)
    # Temporarily use write_jitter_registry then replace header.
    write_jitter_registry(path, specs, header=False)
    body = path.read_text(encoding="utf-8")
    path.write_text(YIELD_REGISTRY_HEADER + body, encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_LIKELIHOOD_THR",
    "SHORT_GAP_MAX_FRAMES",
    "YIELD_MOUNT_TYPES",
    "EyeYieldRecord",
    "LoadedBlockPayload",
    "LoadedEyePayload",
    "MissingEpochStats",
    "YieldDataset",
    "YieldExportResult",
    "YieldEyeCsvChoice",
    "apply_cached_removals",
    "apply_yield_removals_in_memory",
    "collect_all_eye_yields",
    "collect_eye_yield",
    "collect_eye_yield_from_dataset",
    "collect_likelihood_pools",
    "collect_likelihood_pools_from_dataset",
    "eye_yield_from_payload",
    "figure_ellipse_yield_by_condition",
    "finalize_yield_export",
    "fitted_mask",
    "load_yield_dataset",
    "missing_epoch_lengths",
    "pool_yield_summary",
    "read_manual_outlier_intervals",
    "records_to_dataframe",
    "resolve_block_dlc_csvs",
    "resolve_eye_folder",
    "resolve_yield_eye_csv",
    "retag_dataset_block",
    "run_yield_report",
    "seed_yield_registry_from_jitter",
    "summarize_missing_epochs",
    "write_yield_registry",
    "yield_registry_table_from_dataset",
]
