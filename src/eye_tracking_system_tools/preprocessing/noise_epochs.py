"""Noise-epoch catalogs: bad frame ranges by category (no auto-NaN).

Each eye has ``analysis/noise_epochs_{left,right}.csv`` with columns:

* ``start_frame`` / ``end_frame`` — inclusive eye-video frame indices
* ``category`` — source tag (``pupil_perimeter``, ``led_blink``, ``jitter_outlier``, …)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Iterable

import numpy as np
import pandas as pd

CATEGORY_PUPIL_PERIMETER = "pupil_perimeter"
CATEGORY_LED_BLINK = "led_blink"
CATEGORY_JITTER_OUTLIER = "jitter_outlier"

EPOCH_COLUMNS = ("start_frame", "end_frame", "category")

GEOMETRY_COLS = (
    "center_x",
    "center_y",
    "center_x_corrected",
    "center_y_corrected",
    "width",
    "height",
    "phi",
    "major_ax",
    "minor_ax",
    "ratio",
    "ellipse_size",
)


def noise_epochs_filename(eye: str) -> str:
    eye_lc = _normalize_eye(eye)
    return f"noise_epochs_{eye_lc}.csv"


def noise_epochs_path(block_path: Path | str, eye: str) -> Path:
    return Path(block_path) / "analysis" / noise_epochs_filename(eye)


def _normalize_eye(eye: str) -> str:
    eye_lc = str(eye).strip().lower()
    if eye_lc in ("l", "le", "left_eye"):
        return "left"
    if eye_lc in ("r", "re", "right_eye"):
        return "right"
    if eye_lc not in ("left", "right"):
        raise ValueError(f"eye must be left/right, got {eye!r}")
    return eye_lc


def _empty_epochs() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EPOCH_COLUMNS))


def frames_to_epochs(frame_indices: Iterable[Any]) -> pd.DataFrame:
    """Compress sorted unique frame indices into contiguous ``[start, end]`` rows."""
    frames = np.asarray(list(frame_indices), dtype=float)
    frames = frames[np.isfinite(frames)]
    if frames.size == 0:
        return pd.DataFrame(columns=["start_frame", "end_frame"])
    frames = np.unique(frames.astype(int))
    frames.sort()
    starts: list[int] = []
    ends: list[int] = []
    run_start = int(frames[0])
    prev = run_start
    for f in frames[1:]:
        fi = int(f)
        if fi == prev + 1:
            prev = fi
            continue
        starts.append(run_start)
        ends.append(prev)
        run_start = fi
        prev = fi
    starts.append(run_start)
    ends.append(prev)
    return pd.DataFrame({"start_frame": starts, "end_frame": ends})


def epochs_to_frame_set(
    epochs: pd.DataFrame | None,
    categories: Collection[str] | None = None,
) -> set[int]:
    """Expand epoch rows to a set of inclusive frame indices."""
    if epochs is None or epochs.empty:
        return set()
    work = epochs
    if categories is not None:
        cats = {str(c) for c in categories}
        if "category" in work.columns:
            work = work[work["category"].astype(str).isin(cats)]
    out: set[int] = set()
    for _, row in work.iterrows():
        a = int(row["start_frame"])
        b = int(row["end_frame"])
        if b < a:
            a, b = b, a
        out.update(range(a, b + 1))
    return out


def read_noise_epochs(block_path: Path | str, eye: str) -> pd.DataFrame:
    path = noise_epochs_path(block_path, eye)
    if not path.is_file():
        return _empty_epochs()
    df = pd.read_csv(path)
    for col in EPOCH_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"{path}: missing column {col}")
    out = df[list(EPOCH_COLUMNS)].copy()
    out["start_frame"] = out["start_frame"].astype(int)
    out["end_frame"] = out["end_frame"].astype(int)
    out["category"] = out["category"].astype(str)
    return out.reset_index(drop=True)


def write_noise_epochs(block_path: Path | str, eye: str, epochs: pd.DataFrame) -> Path:
    path = noise_epochs_path(block_path, eye)
    path.parent.mkdir(parents=True, exist_ok=True)
    if epochs is None or epochs.empty:
        _empty_epochs().to_csv(path, index=False)
    else:
        out = epochs[list(EPOCH_COLUMNS)].copy()
        out["start_frame"] = out["start_frame"].astype(int)
        out["end_frame"] = out["end_frame"].astype(int)
        out["category"] = out["category"].astype(str)
        out = out.sort_values(["category", "start_frame", "end_frame"]).reset_index(
            drop=True
        )
        out.to_csv(path, index=False)
    return path


def list_categories(block_path: Path | str, eye: str | None = None) -> list[str]:
    """Unique categories present for one or both eyes."""
    eyes = ("left", "right") if eye is None else (_normalize_eye(eye),)
    cats: set[str] = set()
    for e in eyes:
        df = read_noise_epochs(block_path, e)
        if not df.empty:
            cats.update(df["category"].astype(str).tolist())
    return sorted(cats)


def append_epochs(
    block_path: Path | str,
    eye: str,
    *,
    category: str,
    frames: Iterable[Any] | None = None,
    epochs: pd.DataFrame | None = None,
    replace_category: bool = True,
) -> tuple[Path, int, int]:
    """
    Append (or replace) epochs for ``category``.

    Provide either ``frames`` (compressed automatically) or an epochs frame
    with ``start_frame``/``end_frame``.

    Returns ``(path, n_frames, n_epochs)``.
    """
    eye_lc = _normalize_eye(eye)
    category = str(category).strip()
    if not category:
        raise ValueError("category must be non-empty")

    if frames is not None:
        ep = frames_to_epochs(frames)
    elif epochs is not None:
        ep = epochs.copy()
        if "start_frame" not in ep.columns or "end_frame" not in ep.columns:
            raise ValueError("epochs must have start_frame and end_frame")
        ep = ep[["start_frame", "end_frame"]].copy()
    else:
        raise ValueError("Provide frames= or epochs=")

    if ep.empty:
        existing = read_noise_epochs(block_path, eye_lc)
        if replace_category and not existing.empty:
            existing = existing[existing["category"] != category]
            write_noise_epochs(block_path, eye_lc, existing)
        return noise_epochs_path(block_path, eye_lc), 0, 0

    ep["category"] = category
    n_frames = int(
        (ep["end_frame"].astype(int) - ep["start_frame"].astype(int) + 1).sum()
    )
    n_epochs = int(len(ep))

    existing = read_noise_epochs(block_path, eye_lc)
    if replace_category and not existing.empty:
        existing = existing[existing["category"] != category]
    merged = (
        pd.concat([existing, ep[list(EPOCH_COLUMNS)]], ignore_index=True)
        if not existing.empty
        else ep[list(EPOCH_COLUMNS)]
    )
    path = write_noise_epochs(block_path, eye_lc, merged)
    return path, n_frames, n_epochs


def mask_eye_df_by_epochs(
    df: pd.DataFrame,
    epochs: pd.DataFrame | None,
    *,
    frame_col: str,
    categories: Collection[str] | None = None,
    geometry_cols: Collection[str] = GEOMETRY_COLS,
) -> tuple[pd.DataFrame, int]:
    """
    Return a copy of ``df`` with geometry NaN'd on frames covered by epochs.

    Does not write to disk. ``n_hit`` is the number of rows that had a finite
    center (or any geometry) and matched a bad frame.
    """
    if df is None or df.empty:
        return (df.copy() if df is not None else pd.DataFrame()), 0
    if frame_col not in df.columns:
        raise KeyError(f"DataFrame missing frame column {frame_col!r}")

    bad = epochs_to_frame_set(epochs, categories=categories)
    if not bad:
        return df.copy(), 0

    out = df.copy()
    frame_vals = pd.to_numeric(out[frame_col], errors="coerce")
    hit = frame_vals.isin(bad)
    # Prefer counting rows that currently have a center
    if "center_x" in out.columns:
        has_geom = out["center_x"].notna()
        n_hit = int((hit & has_geom).sum())
    else:
        n_hit = int(hit.sum())
    cols = [c for c in geometry_cols if c in out.columns]
    if cols and hit.any():
        out.loc[hit, cols] = np.nan
    return out, n_hit


def resolve_frame_col(df: pd.DataFrame, eye: str) -> str:
    """Pick the eye-video frame column for a dataframe."""
    eye_lc = _normalize_eye(eye)
    candidates = (
        ("L_eye_frame", "R_eye_frame")
        if eye_lc == "left"
        else ("R_eye_frame", "L_eye_frame")
    )
    for c in (*candidates, "eye_frame", "frame"):
        if c in df.columns:
            return c
    raise KeyError(f"No eye-frame column found for {eye_lc} in {list(df.columns)}")


def apply_noise_epochs_to_block_csvs(
    block_path: Path | str,
    *,
    categories: Collection[str],
    eyes: Collection[str] = ("left", "right"),
    update_le_re_df: bool = True,
    update_eye_data: bool = True,
) -> dict[str, Any]:
    """
    Confirmed write: NaN geometry in on-disk eye tables for selected categories.

    Updates ``le_df.csv``/``re_df.csv`` and/or ``left/right_eye_data.csv`` when
    those files exist.
    """
    block_path = Path(block_path)
    analysis = block_path / "analysis"
    cats = [str(c) for c in categories]
    report: dict[str, Any] = {"categories": cats, "eyes": {}}

    for eye in eyes:
        eye_lc = _normalize_eye(eye)
        epochs = read_noise_epochs(block_path, eye_lc)
        eye_report: dict[str, Any] = {"n_epochs": 0, "files": {}}
        if epochs.empty:
            report["eyes"][eye_lc] = eye_report
            continue
        eye_report["n_epochs"] = int(
            len(epochs[epochs["category"].isin(cats)]) if cats else len(epochs)
        )

        if update_le_re_df:
            le_re = analysis / ("le_df.csv" if eye_lc == "left" else "re_df.csv")
            if le_re.is_file():
                df = pd.read_csv(le_re, index_col=0)
                frame_col = resolve_frame_col(df, eye_lc)
                masked, n_hit = mask_eye_df_by_epochs(
                    df, epochs, frame_col=frame_col, categories=cats
                )
                masked.to_csv(le_re)
                eye_report["files"][le_re.name] = n_hit

        if update_eye_data:
            eye_csv = analysis / (
                "left_eye_data.csv" if eye_lc == "left" else "right_eye_data.csv"
            )
            if eye_csv.is_file():
                df = pd.read_csv(eye_csv, index_col=0)
                frame_col = resolve_frame_col(df, eye_lc)
                masked, n_hit = mask_eye_df_by_epochs(
                    df, epochs, frame_col=frame_col, categories=cats
                )
                masked.to_csv(eye_csv)
                eye_report["files"][eye_csv.name] = n_hit

        report["eyes"][eye_lc] = eye_report
    return report


def frames_outside_perimeter(
    df: pd.DataFrame,
    perimeter: dict[str, Any] | None,
    *,
    frame_col: str | None = None,
) -> list[int]:
    """Eye-video frames whose ellipse center lies outside ``perimeter``."""
    from eye_tracking_system_tools.preprocessing.pupil_perimeter import (
        normalize_perimeter,
        point_in_perimeter,
    )

    peri = normalize_perimeter(perimeter)
    if peri is None or df is None or df.empty:
        return []
    if "center_x" not in df.columns or "center_y" not in df.columns:
        return []
    col = frame_col
    if col is None:
        for c in ("eye_frame", "frame", "L_eye_frame", "R_eye_frame"):
            if c in df.columns:
                col = c
                break
    if col is None:
        return []

    cx = df["center_x"].to_numpy(dtype=float)
    cy = df["center_y"].to_numpy(dtype=float)
    fr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    has = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(fr)
    inside = point_in_perimeter(cx, cy, peri)
    if isinstance(inside, bool):
        inside = np.full(len(df), inside, dtype=bool)
    bad = has & ~inside
    return sorted({int(f) for f in fr[bad]})
