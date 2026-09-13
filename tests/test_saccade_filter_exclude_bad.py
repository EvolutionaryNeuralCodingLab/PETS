"""Tests for SaccadeFilter.exclude_bad via apply_saccade_filter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eye_tracking_system_tools.analysis.block_registry import BlockSpec
from eye_tracking_system_tools.analysis.pipeline import (
    BlockBundle,
    EventTables,
    SaccadeFilter,
    apply_saccade_filter,
)
from eye_tracking_system_tools.analysis.saccade_viewer.artifacts import save_verification_tags
from eye_tracking_system_tools.analysis.saccade_viewer.models import (
    VerificationEvent,
    compute_event_id,
)


def _tables_with_tags(tmp_path: Path) -> EventTables:
    block_path = tmp_path / "block_012"
    block_path.mkdir(parents=True)
    spec = BlockSpec(animal="M_002", block_path=block_path, block_num="012")

    events = pd.DataFrame(
        {
            "animal": ["M_002", "M_002", "M_002"],
            "block": ["012", "012", "012"],
            "eye": ["L", "R", "L"],
            "saccade_on_ms": [1000.0, 2000.0, 3000.0],
            "saccade_off_ms": [1030.0, 2030.0, 3030.0],
            "net_angular_disp": [1.0, 2.0, 3.0],
        }
    )
    empty = events.iloc[0:0].copy()
    bundle = BlockBundle(
        spec=spec,
        left=pd.DataFrame(),
        right=pd.DataFrame(),
        left_csv_meta={},
        right_csv_meta={},
        l_saccades=events.loc[events["eye"] == "L"].reset_index(drop=True),
        r_saccades=events.loc[events["eye"] == "R"].reset_index(drop=True),
        all_saccades=events.copy(),
    )
    tables = EventTables(
        blocks=[bundle],
        all_saccades=events.copy(),
        synced=empty.copy(),
        non_synced=events.copy(),
        params={},
    )

    bad_id = compute_event_id(
        animal="M_002", block="012", eye="L", onset_ms=1000.0, off_ms=1030.0
    )
    good_id = compute_event_id(
        animal="M_002", block="012", eye="R", onset_ms=2000.0, off_ms=2030.0
    )
    save_verification_tags(
        block_path,
        [
            VerificationEvent(
                event_id=bad_id,
                animal="M_002",
                block="012",
                block_key="M_002_block_012",
                block_path=block_path,
                eye="L",
                onset_ms=1000.0,
                off_ms=1030.0,
                verification_status="bad",
            ),
            VerificationEvent(
                event_id=good_id,
                animal="M_002",
                block="012",
                block_key="M_002_block_012",
                block_path=block_path,
                eye="R",
                onset_ms=2000.0,
                off_ms=2030.0,
                verification_status="good",
            ),
        ],
    )
    return tables


def test_saccade_filter_exclude_bad_roundtrip():
    filt = SaccadeFilter(exclude_bad=True)
    assert filt.is_active()
    assert filt.to_dict()["exclude_bad"] is True
    restored = SaccadeFilter.from_mapping(filt.to_dict())
    assert restored is not None
    assert restored.exclude_bad is True
    assert "exclude verification-bad" in restored.describe()


def test_apply_saccade_filter_exclude_bad(tmp_path: Path):
    tables = _tables_with_tags(tmp_path)
    assert len(tables.all_saccades) == 3

    kept = apply_saccade_filter(tables, {"exclude_bad": True})
    # bad L@1000 dropped; good R@2000 + unset L@3000 kept
    assert len(kept.all_saccades) == 2
    onsets = set(kept.all_saccades["saccade_on_ms"].astype(float))
    assert onsets == {2000.0, 3000.0}
    assert len(kept.non_synced) == 2
    assert len(kept.blocks[0].all_saccades) == 2

    untouched = apply_saccade_filter(tables, SaccadeFilter())
    assert len(untouched.all_saccades) == 3
