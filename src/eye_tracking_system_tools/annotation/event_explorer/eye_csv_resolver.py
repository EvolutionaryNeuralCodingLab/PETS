"""Resolve newest left/right eye CSV and auto-detect plot columns."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from eye_tracking_system_tools.annotation.event_explorer.models import ColumnMap

_LEFT_GLOB = "left_eye_data*.csv"
_RIGHT_GLOB = "right_eye_data*.csv"


def glob_eye_candidates(analysis_path: Path, side: str) -> list[Path]:
    analysis_path = Path(analysis_path)
    pattern = _LEFT_GLOB if side == "left" else _RIGHT_GLOB
    return sorted(analysis_path.glob(pattern))


def pick_newest_csv(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_eye_csv(
    analysis_path: Path, side: str
) -> tuple[Path | None, list[Path]]:
    """Return (chosen_path, all_candidates) — chosen is newest by mtime."""
    candidates = glob_eye_candidates(analysis_path, side)
    return pick_newest_csv(candidates), candidates


def load_eye_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def frame_column_name(df: pd.DataFrame) -> str | None:
    for name in (
        "eye_frame",
        "Eye_frame",
        "L_eye_frame",
        "R_eye_frame",
        "arena_frame",
    ):
        if name in df.columns:
            return name
    for col in df.columns:
        if "frame" in col.lower():
            return col
    return None


def _match_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    cols_lower = {c: c.lower() for c in df.columns}
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for col, low in cols_lower.items():
            if rx.search(low):
                return col
    return None


def detect_pupil_column(df: pd.DataFrame) -> str | None:
    w = _match_column(df, [r"width", r"pupil.*width"])
    h = _match_column(df, [r"height", r"pupil.*height"])
    if w and h:
        return f"__diameter__:{w}:{h}"
    return _match_column(df, [r"pupil.*diam", r"diameter"])


def detect_degrees_column(df: pd.DataFrame, side: str) -> str | None:
    side_prefix = "l" if side == "left" else "r"
    patterns = [
        rf"^{side_prefix}_?degrees$",
        rf"^{side_prefix}_deg",
        r"k_theta",
        r"k_phi",
        r"kerr.*theta",
        r"kerr.*phi",
        r".*degrees.*",
        r".*_deg$",
    ]
    col = _match_column(df, patterns)
    if col:
        return col
    if "k_theta" in df.columns:
        return "k_theta"
    if "k_phi" in df.columns:
        return "k_phi"
    return None


def pupil_values(df: pd.DataFrame, spec: str) -> pd.Series | None:
    if spec.startswith("__diameter__:"):
        _, w_col, h_col = spec.split(":", 2)
        if w_col not in df.columns or h_col not in df.columns:
            return None
        w = df[w_col].astype(float)
        h = df[h_col].astype(float)
        return 2.0 * (w * h).pow(0.5)
    if spec in df.columns:
        return df[spec].astype(float)
    return None


def build_column_map(
    le_df: pd.DataFrame | None,
    re_df: pd.DataFrame | None,
    overrides: ColumnMap | None = None,
) -> ColumnMap:
    ov = overrides or ColumnMap()
    pupil = ov.pupil
    l_deg = ov.l_degrees
    r_deg = ov.r_degrees

    if pupil is None and le_df is not None:
        pupil = detect_pupil_column(le_df)
    if pupil is None and re_df is not None:
        pupil = detect_pupil_column(re_df)

    if l_deg is None and le_df is not None:
        l_deg = detect_degrees_column(le_df, "left")
    if r_deg is None and re_df is not None:
        r_deg = detect_degrees_column(re_df, "right")

    return ColumnMap(pupil=pupil, l_degrees=l_deg, r_degrees=r_deg)
