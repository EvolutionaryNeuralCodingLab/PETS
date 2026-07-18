"""Full-block series catalog for the Data Exploration tab.

Eye metrics are paired (left + right on one plot). Electrophysiology streams are
listed separately for multi-select. Uses Event Explorer column resolution and
Annotator OE stream helpers — not event-snippet extractors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eye_tracking_system_tools.annotation.block_annotator.models import compute_ms_axis
from eye_tracking_system_tools.annotation.block_annotator.oe_streams import (
    OEStream,
    list_oe_streams,
    load_overview_trace,
)
from eye_tracking_system_tools.annotation.event_explorer.eye_csv_resolver import (
    detect_pupil_column,
    load_eye_dataframe,
    pupil_values,
    resolve_eye_csv,
)

# Paired eye metrics (one picker entry → L+R on the same axes).
EYE_PUPIL_SIZE = "pupil_size"
EYE_K_PHI = "k_phi"
EYE_K_THETA = "k_theta"

EYE_METRIC_ORDER = (EYE_PUPIL_SIZE, EYE_K_PHI, EYE_K_THETA)

EYE_METRIC_LABELS = {
    EYE_PUPIL_SIZE: "pupil_size",
    EYE_K_PHI: "k_phi",
    EYE_K_THETA: "k_theta",
}

# Column aliases: prefer Kerr names, then bare phi/theta (when both present).
_ANGLE_ALIASES = {
    EYE_K_PHI: ("k_phi", "k_phi_corr", "phi"),
    EYE_K_THETA: ("k_theta", "k_theta_corr", "theta"),
}

# Untagged base files use this sentinel tag.
EYE_VERSION_BASE = ""

# Default-on when available.
DEFAULT_EYE_ENABLED = {EYE_PUPIL_SIZE, EYE_K_PHI, EYE_K_THETA}

COLOR_LEFT = "#1f77b4"  # blue
COLOR_RIGHT = "#d62728"  # red

EP_COLORS = [
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#ff7f0e",
    "#2ca02c",
]

_LEFT_TAGGED_RE = re.compile(r"^left_eye_data(?:_(.+))?\.csv$", re.IGNORECASE)
_RIGHT_TAGGED_RE = re.compile(r"^right_eye_data(?:_(.+))?\.csv$", re.IGNORECASE)


@dataclass
class SeriesTrace:
    series_id: str
    label: str
    time_ms: np.ndarray
    values: np.ndarray
    available: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EyeMetricPair:
    """One eye metric with optional left/right traces (same plot)."""

    metric_id: str
    label: str
    left: SeriesTrace | None = None
    right: SeriesTrace | None = None

    @property
    def available(self) -> bool:
        return (self.left is not None and self.left.available) or (
            self.right is not None and self.right.available
        )


@dataclass(frozen=True)
class EyeDataVersion:
    """One on-disk eye_data pair (base or Kerr ``name_tag``)."""

    tag: str
    left_path: Path | None
    right_path: Path | None

    @property
    def label(self) -> str:
        if self.tag == EYE_VERSION_BASE:
            return "base (left/right_eye_data.csv)"
        return self.tag

    @property
    def mtime(self) -> float:
        times = []
        for p in (self.left_path, self.right_path):
            if p is not None and p.is_file():
                times.append(p.stat().st_mtime)
        return max(times) if times else 0.0


@dataclass
class ExploreCatalog:
    """Loaded traces aligned to absolute block time (ms)."""

    ms_axis: np.ndarray
    sample_rate_hz: float
    eye_metrics: dict[str, EyeMetricPair] = field(default_factory=dict)
    oe_rec: Any = None
    oe_streams: list[OEStream] = field(default_factory=list)
    stale_sync: bool = False
    le_csv_path: Path | None = None
    re_csv_path: Path | None = None
    eye_version_tag: str = EYE_VERSION_BASE

    @property
    def ms_min(self) -> float:
        return float(self.ms_axis[0]) if len(self.ms_axis) else 0.0

    @property
    def ms_max(self) -> float:
        return float(self.ms_axis[-1]) if len(self.ms_axis) else 0.0


def eye_csvs_are_stale(analysis_path: Path) -> bool:
    """True when ``final_sync_df.csv`` is newer than either eye-data CSV."""
    analysis = Path(analysis_path)
    final_sync = analysis / "final_sync_df.csv"
    if not final_sync.is_file():
        return False
    sync_mtime = final_sync.stat().st_mtime
    for side in ("left", "right"):
        path, _ = resolve_eye_csv(analysis, side)
        if path is not None and path.is_file() and path.stat().st_mtime < sync_mtime:
            return True
    return False


def discover_eye_data_versions(analysis_path: Path) -> list[EyeDataVersion]:
    """List base + tagged ``*_eye_data_*.csv`` pairs under ``analysis/``."""
    analysis = Path(analysis_path)
    if not analysis.is_dir():
        return []

    left_by_tag: dict[str, Path] = {}
    right_by_tag: dict[str, Path] = {}
    for path in analysis.glob("left_eye_data*.csv"):
        m = _LEFT_TAGGED_RE.match(path.name)
        if not m:
            continue
        tag = m.group(1) or EYE_VERSION_BASE
        left_by_tag[tag] = path
    for path in analysis.glob("right_eye_data*.csv"):
        m = _RIGHT_TAGGED_RE.match(path.name)
        if not m:
            continue
        tag = m.group(1) or EYE_VERSION_BASE
        right_by_tag[tag] = path

    tags = set(left_by_tag) | set(right_by_tag)
    versions = [
        EyeDataVersion(
            tag=tag,
            left_path=left_by_tag.get(tag),
            right_path=right_by_tag.get(tag),
        )
        for tag in tags
    ]
    base = [v for v in versions if v.tag == EYE_VERSION_BASE]
    tagged = sorted(
        [v for v in versions if v.tag != EYE_VERSION_BASE],
        key=lambda v: v.mtime,
        reverse=True,
    )
    return base + tagged


def prefer_eye_data_version(versions: list[EyeDataVersion]) -> EyeDataVersion | None:
    """Prefer newest version that has Kerr angle columns; else newest overall."""
    if not versions:
        return None

    def _has_angles(v: EyeDataVersion) -> bool:
        for path in (v.left_path, v.right_path):
            if path is None or not path.is_file():
                continue
            try:
                cols = {c.lower() for c in pd.read_csv(path, nrows=0).columns}
            except OSError:
                continue
            if {"k_phi", "k_theta"} & cols or ({"phi", "theta"} <= cols):
                return True
        return False

    with_angles = [v for v in versions if _has_angles(v)]
    pool = with_angles or versions
    return max(pool, key=lambda v: v.mtime)


def load_eye_data_version(
    analysis_path: Path,
    tag: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, Path | None, Path | None]:
    """Load left/right eye CSVs for a specific version tag (``\"\"`` = base)."""
    versions = {v.tag: v for v in discover_eye_data_versions(analysis_path)}
    version = versions.get(tag)
    if version is None:
        return None, None, None, None
    le_df = load_eye_dataframe(version.left_path) if version.left_path else None
    re_df = load_eye_dataframe(version.right_path) if version.right_path else None
    return le_df, re_df, version.left_path, version.right_path


def load_eye_dataframes(
    analysis_path: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, Path | None, Path | None]:
    """Load preferred eye-data version (angles if available, else newest)."""
    versions = discover_eye_data_versions(analysis_path)
    preferred = prefer_eye_data_version(versions)
    if preferred is None:
        return None, None, None, None
    return load_eye_data_version(analysis_path, preferred.tag)


def _finite_xy(time_ms: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_ms, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if t.shape != y.shape:
        n = min(t.size, y.size)
        t = t[:n]
        y = y[:n]
    mask = np.isfinite(t) & np.isfinite(y)
    return t[mask], y[mask]


def _eye_time_ms(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float,
    sync_ms: np.ndarray | None = None,
) -> np.ndarray:
    if "ms_axis" in df.columns:
        return df["ms_axis"].to_numpy(dtype=np.float64)
    if "OE_timestamp" in df.columns:
        return df["OE_timestamp"].to_numpy(dtype=np.float64) / (sample_rate_hz / 1000.0)
    if "Arena_TTL" in df.columns:
        return df["Arena_TTL"].to_numpy(dtype=np.float64) / (sample_rate_hz / 1000.0)
    if sync_ms is not None and len(sync_ms):
        return np.linspace(
            float(sync_ms[0]), float(sync_ms[-1]), len(df), dtype=np.float64
        )
    return np.arange(len(df), dtype=np.float64)


def _find_column(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    lower = {c.lower(): c for c in df.columns}
    return lower.get(name.lower())


def _has_column(df: pd.DataFrame, name: str) -> bool:
    return _find_column(df, name) is not None


def _angle_series(df: pd.DataFrame | None, metric_id: str) -> pd.Series | None:
    """Resolve k_phi / k_theta with aliases; avoid ellipse-only ``phi``."""
    if df is None or metric_id not in _ANGLE_ALIASES:
        return None
    aliases = _ANGLE_ALIASES[metric_id]
    has_theta_like = _has_column(df, "k_theta") or _has_column(df, "theta")
    has_phi_like = _has_column(df, "k_phi") or _has_column(df, "phi")
    for alias in aliases:
        col = _find_column(df, alias)
        if col is None:
            continue
        # Bare phi/theta only when the pair looks Kerr-like (both present).
        if alias in ("phi", "theta") and not (has_phi_like and has_theta_like):
            continue
        return df[col]
    return None


def _side_trace(
    *,
    metric_id: str,
    side: str,
    df: pd.DataFrame | None,
    value_series: pd.Series | None,
    sample_rate_hz: float,
    sync_ms: np.ndarray,
) -> SeriesTrace | None:
    if df is None or value_series is None:
        return None
    t = _eye_time_ms(df, sample_rate_hz=sample_rate_hz, sync_ms=sync_ms)
    t, y = _finite_xy(t, value_series.to_numpy(dtype=np.float64))
    label = f"{side[0].upper()} {metric_id}"
    return SeriesTrace(
        series_id=f"{side}_{metric_id}",
        label=label,
        time_ms=t,
        values=y,
        available=t.size > 0,
        meta={"side": side, "metric_id": metric_id},
    )


def _pupil_series(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None:
        return None
    spec = detect_pupil_column(df)
    if spec is None:
        return None
    return pupil_values(df, spec)


def _build_eye_metric(
    metric_id: str,
    *,
    le_df: pd.DataFrame | None,
    re_df: pd.DataFrame | None,
    sample_rate_hz: float,
    sync_ms: np.ndarray,
) -> EyeMetricPair | None:
    if metric_id == EYE_PUPIL_SIZE:
        left_vals = _pupil_series(le_df)
        right_vals = _pupil_series(re_df)
    else:
        left_vals = _angle_series(le_df, metric_id)
        right_vals = _angle_series(re_df, metric_id)

    left = _side_trace(
        metric_id=metric_id,
        side="left",
        df=le_df,
        value_series=left_vals,
        sample_rate_hz=sample_rate_hz,
        sync_ms=sync_ms,
    )
    right = _side_trace(
        metric_id=metric_id,
        side="right",
        df=re_df,
        value_series=right_vals,
        sample_rate_hz=sample_rate_hz,
        sync_ms=sync_ms,
    )
    pair = EyeMetricPair(
        metric_id=metric_id,
        label=EYE_METRIC_LABELS[metric_id],
        left=left,
        right=right,
    )
    return pair if pair.available else None


def build_explore_catalog(
    final_sync_df: pd.DataFrame,
    sample_rate_hz: float,
    *,
    le_df: pd.DataFrame | None = None,
    re_df: pd.DataFrame | None = None,
    oe_rec: Any = None,
    stale_sync: bool = False,
    le_csv_path: Path | None = None,
    re_csv_path: Path | None = None,
    eye_version_tag: str = EYE_VERSION_BASE,
) -> ExploreCatalog:
    """Assemble paired eye metrics and OE stream list from loaded tables."""
    if final_sync_df is None or final_sync_df.empty:
        raise ValueError("final_sync_df is required and must be non-empty.")

    ms_axis = compute_ms_axis(final_sync_df, float(sample_rate_hz))
    eye_metrics: dict[str, EyeMetricPair] = {}
    for metric_id in EYE_METRIC_ORDER:
        pair = _build_eye_metric(
            metric_id,
            le_df=le_df,
            re_df=re_df,
            sample_rate_hz=float(sample_rate_hz),
            sync_ms=ms_axis,
        )
        if pair is not None:
            eye_metrics[metric_id] = pair

    streams = list_oe_streams(oe_rec) if oe_rec is not None else []

    return ExploreCatalog(
        ms_axis=ms_axis,
        sample_rate_hz=float(sample_rate_hz),
        eye_metrics=eye_metrics,
        oe_rec=oe_rec,
        oe_streams=streams,
        stale_sync=stale_sync,
        le_csv_path=le_csv_path,
        re_csv_path=re_csv_path,
        eye_version_tag=eye_version_tag,
    )


def load_ep_overview(
    catalog: ExploreCatalog,
    stream: OEStream,
    *,
    downsample: int = 50,
) -> SeriesTrace | None:
    """Decimated OE overview for the full block span."""
    if catalog.oe_rec is None or len(catalog.ms_axis) == 0:
        return None
    result = load_overview_trace(
        catalog.oe_rec,
        stream,
        catalog.ms_axis,
        downsample=int(downsample),
    )
    if result is None:
        return None
    t, y = result
    t, y = _finite_xy(t, y)
    return SeriesTrace(
        series_id=f"ep:{stream.kind}:{stream.channel}",
        label=stream.label,
        time_ms=t,
        values=y,
        available=t.size > 0,
        meta={
            "stream_label": stream.label,
            "kind": stream.kind,
            "channel": stream.channel,
        },
    )


def available_eye_metric_ids(catalog: ExploreCatalog) -> list[str]:
    """Stable order of available paired eye metrics."""
    return [
        mid
        for mid in EYE_METRIC_ORDER
        if mid in catalog.eye_metrics and catalog.eye_metrics[mid].available
    ]


def ep_stream_key(stream: OEStream) -> str:
    return f"{stream.kind}:{stream.channel}"
