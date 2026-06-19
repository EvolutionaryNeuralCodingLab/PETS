"""Shared pytest fixtures for the Preprocessing GUI tests.

The ``sample_block`` fixture provides a ready-to-use :class:`BlockSync`
instance pointing at the reference sample block on the developer's machine
(``D:\\sample_data_for_eye_repo\\PV_106\\2025_09_04\\block_015``). Tests that
need it should request it; on machines without the sample data they will be
skipped automatically so CI does not break.

The ``qapp`` fixture provides a single :class:`QApplication` per test
session so we can instantiate Qt widgets headlessly without ``pytest-qt``.
If ``pytest-qt`` is installed pytest's own ``qtbot``/``qapp`` will still
work; this fixture is for code paths that just need a live QApplication.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


SAMPLE_BLOCK_PATH = Path(r"D:\sample_data_for_eye_repo\PV_106\2025_09_04\block_015")


def _sample_block_available() -> bool:
    return (SAMPLE_BLOCK_PATH / "analysis" / "final_sync_df.csv").exists()


@pytest.fixture(scope="session")
def sample_block_path() -> Path:
    if not _sample_block_available():
        pytest.skip(
            f"Sample block not found at {SAMPLE_BLOCK_PATH}; "
            "skipping tests that depend on the reference data."
        )
    return SAMPLE_BLOCK_PATH


@pytest.fixture(scope="session")
def sample_block(sample_block_path: Path):
    """Build a BlockSync for the sample block and load final_sync_df."""
    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
    from eye_tracking_system_tools.preprocessing.block_sync_core import (
        load_final_sync_df,
    )

    block = BlockSync(
        animal_call="PV_106",
        experiment_date="2025_09_04",
        block_num="015",
        path_to_animal_folder=str(sample_block_path.parents[2]),
    )
    load_final_sync_df(block, verbose=False)
    return block


@pytest.fixture(scope="session")
def qapp_session():
    """Session-wide QApplication using the offscreen QPA when no display.

    This avoids spawning a window during CI. If pytest-qt is installed it
    provides its own ``qapp`` -- we don't override that.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
