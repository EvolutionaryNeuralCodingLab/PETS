"""Data models for the Preprocessing GUI.

These dataclasses are the single source of state shared between the Sync,
Verify, Kerr, Behavior and Sync-free tabs. They intentionally do not hold any
Qt objects so they remain easy to test headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StageStatus(Enum):
    """Per-tab status icon.

    NOT_STARTED  - none of this stage's outputs exist on disk yet.
    PARTIAL      - some but not all of this stage's outputs exist.
    COMPLETE     - all of this stage's outputs exist and are up-to-date.
    STALE        - outputs exist but an upstream signature is newer.
    """

    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    COMPLETE = "complete"
    STALE = "stale"


@dataclass
class BlockHandle:
    """Lightweight, picklable reference to one block on disk.

    The actual BlockSync object is built lazily by the tabs so we don't pay
    its startup cost up front. ``block_path`` always points to
    ``<animal>/<date>/block_NNN`` (or ``<animal>/block_NNN`` when there is no
    date level).
    """

    animal_call: str
    experiment_date: str | None
    block_num: str
    block_path: Path
    path_to_animal_folder: Path
    channeldict: dict[int, str] | None = None

    @property
    def analysis_path(self) -> Path:
        return self.block_path / "analysis"

    @property
    def display_label(self) -> str:
        if self.experiment_date:
            return f"{self.animal_call} / {self.experiment_date} / block_{self.block_num}"
        return f"{self.animal_call} / block_{self.block_num}"


@dataclass
class GuiState:
    """Shared mutable state across tabs.

    Populated by the startup dialog and the block picker. Tabs read from it
    and write incremental results back (e.g. ``df_left_simple_sync``).
    """

    experiment_path: Path | None = None
    blocks: list[BlockHandle] = field(default_factory=list)
    current_index: int = 0
    output_folder: Path | None = None

    # Cached pipeline artefacts for the current block (cleared when block changes).
    df_left_simple_sync: Any = None
    df_right_simple_sync: Any = None
    arena_grid_df: Any = None
    final_sync_df: Any = None

    @property
    def current_block(self) -> BlockHandle | None:
        if not self.blocks:
            return None
        idx = max(0, min(self.current_index, len(self.blocks) - 1))
        return self.blocks[idx]

    def clear_cached_artefacts(self) -> None:
        self.df_left_simple_sync = None
        self.df_right_simple_sync = None
        self.arena_grid_df = None
        self.final_sync_df = None
