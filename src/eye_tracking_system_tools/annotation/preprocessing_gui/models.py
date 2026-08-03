"""Data models for the Preprocessing GUI.

These dataclasses are the single source of state shared between the Sync,
Verify, Kerr, Calibration, Saccades, Behavior and Sync-free tabs. They intentionally do not hold any
Qt objects so they remain easy to test headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eye_tracking_system_tools.annotation.preprocessing_gui.block_session import (
        BlockSyncSession,
    )


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
    session: BlockSyncSession | None = None

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

    def ensure_session(self) -> BlockSyncSession:
        if self.session is None:
            from eye_tracking_system_tools.annotation.preprocessing_gui.block_session import (
                BlockSyncSession,
            )

            self.session = BlockSyncSession()
        return self.session

    def index_for_path(self, block_path: Path) -> int | None:
        target = Path(block_path).resolve()
        for i, handle in enumerate(self.blocks):
            if Path(handle.block_path).resolve() == target:
                return i
        return None

    def add_blocks(self, handles: list[BlockHandle]) -> int:
        """Append blocks not already in session (dedup by block_path)."""
        existing = {Path(b.block_path).resolve() for b in self.blocks}
        added = 0
        for handle in handles:
            key = Path(handle.block_path).resolve()
            if key in existing:
                continue
            self.blocks.append(handle)
            existing.add(key)
            added += 1
        if self.blocks and self.current_index >= len(self.blocks):
            self.current_index = len(self.blocks) - 1
        return added

    def remove_block_at(self, index: int) -> BlockHandle | None:
        if not (0 <= index < len(self.blocks)):
            return None
        removed = self.blocks.pop(index)
        if not self.blocks:
            self.current_index = 0
        elif index < self.current_index:
            self.current_index -= 1
        elif self.current_index >= len(self.blocks):
            self.current_index = len(self.blocks) - 1
        return removed
