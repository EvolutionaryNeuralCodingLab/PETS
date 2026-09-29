"""Prepare eye traces and run saccade detection for the Preprocessing GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.eye_trace_io import csv_choices_meta, load_block_eyes
from eye_tracking_system_tools.analysis.saccade_events import (
    create_saccade_events_with_direction_segmentation_robust,
)

Mode = Literal["deg", "px"]


@dataclass
class TraceBundle:
    """One block's left/right traces with speed columns precomputed."""

    spec: BlockSpec
    left: pd.DataFrame
    right: pd.DataFrame
    csv_meta: dict = field(default_factory=dict)

    def eye(self, which: str) -> pd.DataFrame:
        w = which.strip().upper()
        if w in {"L", "LEFT"}:
            return self.left
        if w in {"R", "RIGHT"}:
            return self.right
        raise KeyError(f"eye must be L or R, got {which!r}")


def prepare_traces(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure pixel and angular speed columns exist (same formulas as the detector)."""
    out = df.copy()
    if "speed_x" not in out.columns and "center_x" in out.columns:
        out["speed_x"] = out["center_x"].diff()
    if "speed_y" not in out.columns and "center_y" in out.columns:
        out["speed_y"] = out["center_y"].diff()
    if "speed_r" not in out.columns and {"speed_x", "speed_y"}.issubset(out.columns):
        out["speed_r"] = np.sqrt(out["speed_x"] ** 2 + out["speed_y"] ** 2)
    if "angular_speed_phi" not in out.columns and "k_phi" in out.columns:
        out["angular_speed_phi"] = out["k_phi"].diff()
    if "angular_speed_theta" not in out.columns and "k_theta" in out.columns:
        out["angular_speed_theta"] = out["k_theta"].diff()
    if "angular_speed_r" not in out.columns and {
        "angular_speed_phi",
        "angular_speed_theta",
    }.issubset(out.columns):
        out["angular_speed_r"] = np.sqrt(
            out["angular_speed_phi"] ** 2 + out["angular_speed_theta"] ** 2
        )
    return out


def load_trace_bundle(spec: BlockSpec, *, log: bool = True) -> TraceBundle:
    loaded = load_block_eyes(spec, log=log)
    return TraceBundle(
        spec=spec,
        left=prepare_traces(loaded.left),
        right=prepare_traces(loaded.right),
        csv_meta=csv_choices_meta(loaded),
    )


def saccade_params_from_dict(params: dict[str, Any]) -> dict[str, Any]:
    """Map the YAML ``saccade:`` section to detector kwargs."""
    s = params.get("saccade", {}) if params else {}
    return {
        "speed_threshold": float(s.get("speed_threshold_deg_per_frame", 0.8)),
        "directional_delta_threshold_deg": float(
            s.get("directional_delta_threshold_deg", 90.0)
        ),
        "min_subsaccade_samples": int(s.get("min_subsaccade_samples", 2)),
        "min_net_disp": float(s.get("min_net_disp_deg", 0.5)),
        "speed_profile": bool(s.get("speed_profile", True)),
    }


def detect_eye(
    df: pd.DataFrame,
    saccade_params: dict[str, Any] | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the production angular saccade detector; return (enriched_df, events)."""
    if saccade_params is None:
        saccade_params = saccade_params_from_dict(params or {})
    prepared = prepare_traces(df)
    return create_saccade_events_with_direction_segmentation_robust(
        prepared, **saccade_params
    )


def raw_threshold_runs(
    df: pd.DataFrame,
    *,
    mode: Mode = "deg",
    threshold: float,
) -> list[tuple[float, float]]:
    """
    Contiguous above-threshold runs as ``(on_ms, off_ms)`` before length/disp filters.

    Used as a light overlay so the speed gate alone is visible.
    """
    speed_col = "angular_speed_r" if mode == "deg" else "speed_r"
    if speed_col not in df.columns or "ms_axis" not in df.columns:
        return []
    speed = df[speed_col].to_numpy(dtype=float)
    ms = df["ms_axis"].to_numpy(dtype=float)
    above = np.isfinite(speed) & (speed > float(threshold))
    if not above.any():
        return []
    runs: list[tuple[float, float]] = []
    in_run = False
    start_i = 0
    for i, flag in enumerate(above):
        if flag and not in_run:
            in_run = True
            start_i = i
        elif not flag and in_run:
            in_run = False
            runs.append((float(ms[start_i]), float(ms[i - 1])))
    if in_run:
        runs.append((float(ms[start_i]), float(ms[len(above) - 1])))
    return runs
