"""Tests for saccade viewer event subset selectors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eye_tracking_system_tools.analysis.saccade_viewer.selectors.common import (
    enrich_events_for_viewer,
    numeric_event_columns,
    unique_events_by_identity,
)
from eye_tracking_system_tools.analysis.saccade_viewer.selectors.threshold_selector import (
    ThresholdRule,
    apply_threshold_rules,
)


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["M_002", "M_002", "M_002"],
            "block": ["012", "012", "013"],
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [1000.0, 2000.0, 1500.0],
            "net_angular_disp": [5.0, 12.0, 3.0],
            "head_movement": [False, True, False],
            "speed_profile_angular": [[1.0, 2.0], [3.0], [0.5]],
        }
    )


def test_numeric_event_columns_skips_profiles():
    cols = numeric_event_columns(_sample_events())
    assert "net_angular_disp" in cols
    assert "speed_profile_angular" not in cols
    assert "saccade_on_ms" in cols


def test_apply_threshold_rules_min_max():
    df = _sample_events()
    rules = [
        ThresholdRule("net_angular_disp", min_val=4.0),
        ThresholdRule("net_angular_disp", max_val=10.0),
    ]
    out = apply_threshold_rules(df, rules)
    assert len(out) == 1
    assert out.iloc[0]["net_angular_disp"] == 5.0


def test_enrich_events_for_viewer(tmp_path: Path):
    df = _sample_events()
    subset = df.iloc[[0]].copy()
    subset["right_peak_v"] = 0.1
    path = str(tmp_path / "block_012")
    out = enrich_events_for_viewer(subset, {"M_002_block_012": path})
    assert "block_path" in out.columns
    assert out.iloc[0]["block_path"] == path
    assert "right_peak_v" not in out.columns


def test_unique_events_by_identity():
    df = pd.concat([_sample_events(), _sample_events().iloc[[0]]], ignore_index=True)
    out = unique_events_by_identity(df)
    assert len(out) == 3
