"""Tests for saccade LFP pipeline eye loading and legacy detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PIPELINE = Path(__file__).resolve().parents[1] / "pipelines" / "saccade_lfp_average"


@pytest.fixture(scope="module")
def sample_analysis_path() -> Path | None:
    p = Path(r"D:/sample_data_for_eye_repo/PV_106/2025_09_04/block_015/analysis")
    if not p.is_dir():
        pytest.skip("Sample block analysis folder not available")
    return p


def test_resolve_eye_csv_auto_prefers_eye_data(sample_analysis_path: Path) -> None:
    import sys

    sys.path.insert(0, str(_PIPELINE))
    from eye_data_loader import resolve_eye_csv_pair

    left, right, desc = resolve_eye_csv_pair(sample_analysis_path, source="auto")
    assert left is not None and right is not None
    assert "eye_data" in left.name or "left" in left.name.lower()
    assert "fallback" not in desc


def test_legacy_detection_produces_sync_status(sample_analysis_path: Path) -> None:
    import sys

    sys.path.insert(0, str(_PIPELINE))
    from saccade_detection_legacy import annotate_events_sync_status, create_saccade_events_df

    left = pd.read_csv(sample_analysis_path / "left_eye_data.csv", index_col=0)
    right = pd.read_csv(sample_analysis_path / "right_eye_data.csv", index_col=0)
    _, l_ev = create_saccade_events_df(left, speed_threshold=2.0, use_pupil_diameter=False)
    _, r_ev = create_saccade_events_df(right, speed_threshold=2.0, use_pupil_diameter=False)
    events = pd.concat(
        [
            l_ev.assign(eye="L", block="015", saccade_start_ms=l_ev["saccade_on_ms"]),
            r_ev.assign(eye="R", block="015", saccade_start_ms=r_ev["saccade_on_ms"]),
        ],
        ignore_index=True,
    )
    annotated = annotate_events_sync_status(events, sync_diff_ms=680)
    assert "sync_status" in annotated.columns
    assert set(annotated["sync_status"]) <= {"synced", "non_synced"}
    assert annotated["sync_status"].eq("synced").sum() % 2 == 0
