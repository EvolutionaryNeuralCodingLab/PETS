"""Shared BlockSync cache for the preprocessing GUI."""

from __future__ import annotations

from pathlib import Path

from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.ttl_mapping import (
    normalize_channeldict,
)
from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
from eye_tracking_system_tools.preprocessing.block_sync_core import (
    drop_pandas_index_artifact_columns,
)


class BlockSyncSession:
    """One cached :class:`BlockSync` per ``block_path`` for the GUI session."""

    def __init__(self) -> None:
        self._cache: dict[str, BlockSync] = {}
        self.last_channeldict: dict[int, str] | None = None

    @staticmethod
    def _key(block: BlockHandle) -> str:
        return str(Path(block.block_path).resolve())

    def resolve_channeldict(self, block: BlockHandle) -> dict[int, str] | None:
        if block.channeldict is not None:
            return normalize_channeldict(block.channeldict)
        if self.last_channeldict is not None:
            return dict(self.last_channeldict)
        return None

    def remember_channeldict(self, channeldict: dict[int, str] | dict[str, str]) -> None:
        normalized = normalize_channeldict(channeldict)
        if normalized:
            self.last_channeldict = normalized

    def get(self, block: BlockHandle) -> BlockSync:
        key = self._key(block)
        if key not in self._cache:
            bs = BlockSync(
                block.animal_call,
                block.experiment_date,
                block.block_num,
                block.path_to_animal_folder,
                channeldict=self.resolve_channeldict(block),
            )
            self.sanitize(bs)
            self._cache[key] = bs
        return self._cache[key]

    def invalidate(self, block: BlockHandle | None = None) -> None:
        if block is None:
            self._cache.clear()
            return
        self._cache.pop(self._key(block), None)

    def release(self, block: BlockHandle) -> None:
        """Drop cached instance for a block removed from the GUI session."""
        key = self._key(block)
        self._cache.pop(key, None)

    def has(self, block: BlockHandle) -> bool:
        return self._key(block) in self._cache

    @staticmethod
    def sanitize(blocksync: BlockSync) -> None:
        """Strip stale ``level_0`` / ``index`` columns from loaded analysis tables."""
        if getattr(blocksync, "oe_events", None) is not None:
            blocksync.oe_events = drop_pandas_index_artifact_columns(blocksync.oe_events)
        if getattr(blocksync, "final_sync_df", None) is not None:
            blocksync.final_sync_df = drop_pandas_index_artifact_columns(
                blocksync.final_sync_df
            )
        if getattr(blocksync, "blocksync_df", None) is not None:
            blocksync.blocksync_df = drop_pandas_index_artifact_columns(
                blocksync.blocksync_df
            )
        if getattr(blocksync, "le_df", None) is not None:
            blocksync.le_df = drop_pandas_index_artifact_columns(blocksync.le_df)
        if getattr(blocksync, "re_df", None) is not None:
            blocksync.re_df = drop_pandas_index_artifact_columns(blocksync.re_df)

    @staticmethod
    def ensure_eye_videos(blocksync: BlockSync) -> None:
        if getattr(blocksync, "_gui_eye_videos_prepared", False):
            return
        if getattr(blocksync, "le_videos", None) and getattr(blocksync, "re_videos", None):
            blocksync._gui_eye_videos_prepared = True
            return
        blocksync.handle_eye_videos()
        blocksync._gui_eye_videos_prepared = True
