"""Lazy per-block load: sync, eye CSV (newest glob), OE, stale-sync guard."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
    infer_metadata,
    load_block_session,
)
from eye_tracking_system_tools.annotation.block_annotator.models import AnnotatorConfig
from eye_tracking_system_tools.annotation.event_explorer.eye_csv_resolver import (
    build_column_map,
    load_eye_dataframe,
    resolve_eye_csv,
)
from eye_tracking_system_tools.annotation.event_explorer.load_log import LoadLog
from eye_tracking_system_tools.annotation.event_explorer.models import (
    BlockDataCache,
    BlockStatus,
    ColumnMap,
    EventRecord,
)
from eye_tracking_system_tools.annotation.event_explorer.remap_dialog import RemapBlockDialog
from eye_tracking_system_tools.annotation.event_explorer.stale_sync_dialog import (
    StaleSyncDialog,
)


class BlockDataManager:
    def __init__(
        self,
        load_log: LoadLog,
        *,
        column_overrides: ColumnMap | None = None,
        parent_widget=None,
    ) -> None:
        self._log = load_log
        self._cache: dict[Path, BlockDataCache] = {}
        self._remap: dict[Path, Path] = {}
        self._column_overrides = column_overrides or ColumnMap()
        self._parent = parent_widget
        self._stale_ack: set[Path] = set()
        self._skipped: set[Path] = set()

    @property
    def remap_table(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self._remap.items()}

    def set_remap_table(self, table: dict[str, str]) -> None:
        self._remap = {Path(k): Path(v) for k, v in table.items()}

    def set_column_overrides(self, overrides: ColumnMap) -> None:
        self._column_overrides = overrides
        for cache in self._cache.values():
            cache.column_map = build_column_map(
                cache.le_df, cache.re_df, self._column_overrides
            )

    def resolve_block_path(self, record: EventRecord) -> Path:
        original = Path(record.block_path)
        if original in self._remap:
            return self._remap[original]
        if original.exists():
            return original.resolve()
        return original

    def ensure_block(
        self,
        record: EventRecord,
        *,
        ask_remap: Callable[[], bool] = lambda: True,
        ask_stale: Callable[[], bool] = lambda: True,
    ) -> BlockDataCache | None:
        block_path = self.resolve_block_path(record)
        key = block_path.resolve() if block_path.exists() else block_path

        if key in self._skipped:
            return None
        if key in self._cache:
            return self._cache[key]

        if not block_path.exists():
            self._log.warn(f"block_path missing: {block_path} (event {record.event_id})")
            if not ask_remap():
                self._skipped.add(key)
                return None
            dlg = RemapBlockDialog(
                block_path, record.animal_call, record.block_num, self._parent
            )
            if dlg.exec() != dlg.DialogCode.Accepted or dlg.chosen_path is None:
                self._skipped.add(key)
                self._log.warn(f"User declined remap for {block_path}")
                return None
            new_path = dlg.chosen_path
            self._remap[Path(record.block_path)] = new_path
            block_path = new_path
            key = block_path.resolve()
            self._log.info(f"Remapped block_path -> {block_path}")

        try:
            cache = self._load_block(block_path, record)
        except Exception as exc:
            self._log.warn(f"Block load failed {block_path}: {exc}")
            self._cache[key] = BlockDataCache(
                block_path=block_path,
                final_sync_df=pd.DataFrame(),
                ms_axis=__import__("numpy").array([]),
                sample_rate_hz=30000.0,
                skipped=True,
            )
            return self._cache[key]

        if cache.skipped:
            self._skipped.add(key)
            return None

        self._cache[key] = cache
        return cache

    def _load_block(self, block_path: Path, record: EventRecord) -> BlockDataCache:
        block_path = Path(block_path).resolve()
        self._log.info(
            f"Loading block {block_path} (annotation {record.annotation_path.name})"
        )

        tmp_out = block_path / "analysis"
        session = load_block_session(
            block_path,
            tmp_out,
            AnnotatorConfig(),
            animal_call=record.animal_call,
            experiment_date=record.experiment_date,
            block_num=record.block_num,
        )

        analysis = block_path / "analysis"
        sync_path = analysis / "final_sync_df.csv"
        if not sync_path.exists():
            sync_path = analysis / "blocksync_df.csv"

        le_path, le_cands = resolve_eye_csv(analysis, "left")
        re_path, re_cands = resolve_eye_csv(analysis, "right")

        self._log.info(
            f"Eye CSV candidates L ({len(le_cands)}): "
            + ", ".join(p.name for p in le_cands) or "(none)"
        )
        self._log.info(
            f"Eye CSV candidates R ({len(re_cands)}): "
            + ", ".join(p.name for p in re_cands) or "(none)"
        )
        if le_path:
            self._log.info(f"Selected L eye CSV: {le_path.name} (newest mtime)")
        else:
            self._log.warn("No left_eye_data*.csv found")
        if re_path:
            self._log.info(f"Selected R eye CSV: {re_path.name} (newest mtime)")
        else:
            self._log.warn("No right_eye_data*.csv found")

        le_df = load_eye_dataframe(le_path) if le_path else None
        re_df = load_eye_dataframe(re_path) if re_path else None

        stale = False
        if sync_path.exists():
            sync_mtime = sync_path.stat().st_mtime
            for label, ep in (("L", le_path), ("R", re_path)):
                if ep and ep.exists() and sync_mtime > ep.stat().st_mtime:
                    stale = True
                    self._log.warn(
                        f"Stale sync: {sync_path.name} newer than {label} eye CSV {ep.name}"
                    )

        stale_ack = block_path in self._stale_ack
        if stale and not stale_ack:
            eye_paths = [p for p in (le_path, re_path) if p]
            dlg = StaleSyncDialog(block_path, sync_path, eye_paths, self._parent)
            if dlg.exec() != dlg.DialogCode.Accepted:
                self._log.warn(f"User skipped block due to stale sync: {block_path}")
                return BlockDataCache(
                    block_path=block_path,
                    final_sync_df=session.final_sync_df,
                    ms_axis=session.ms_axis,
                    sample_rate_hz=session.sample_rate_hz,
                    skipped=True,
                )
            self._stale_ack.add(block_path)
            stale_ack = True

        col_map = build_column_map(le_df, re_df, self._column_overrides)
        le_degrees_df = None
        re_degrees_df = None
        from eye_tracking_system_tools.annotation.event_explorer.eye_csv_resolver import (
            detect_degrees_column,
        )

        if col_map.l_degrees is None or (
            le_df is not None and col_map.l_degrees not in le_df.columns
        ):
            for cand in le_cands:
                if le_path and cand == le_path:
                    continue
                try:
                    extra = load_eye_dataframe(cand)
                    deg = detect_degrees_column(extra, "left")
                    if deg:
                        col_map.l_degrees = deg
                        le_degrees_df = extra
                        self._log.info(
                            f"L degrees column '{deg}' from alternate {cand.name}"
                        )
                        break
                except OSError:
                    pass
        if col_map.r_degrees is None or (
            re_df is not None and col_map.r_degrees not in re_df.columns
        ):
            for cand in re_cands:
                if re_path and cand == re_path:
                    continue
                try:
                    extra = load_eye_dataframe(cand)
                    deg = detect_degrees_column(extra, "right")
                    if deg:
                        col_map.r_degrees = deg
                        re_degrees_df = extra
                        self._log.info(
                            f"R degrees column '{deg}' from alternate {cand.name}"
                        )
                        break
                except OSError:
                    pass
        if session.oe_rec is not None:
            from eye_tracking_system_tools.annotation.block_annotator.oe_streams import (
                list_oe_streams,
            )

            streams = list_oe_streams(session.oe_rec)
            self._log.info(
                "OE streams: " + ", ".join(s.label for s in streams[:12])
                + (" …" if len(streams) > 12 else "")
            )
        else:
            self._log.info("No OERecording for block")

        return BlockDataCache(
            block_path=block_path,
            final_sync_df=session.final_sync_df,
            ms_axis=session.ms_axis,
            sample_rate_hz=session.sample_rate_hz,
            le_csv_path=le_path,
            re_csv_path=re_path,
            le_df=le_df,
            re_df=re_df,
            le_degrees_df=le_degrees_df,
            re_degrees_df=re_degrees_df,
            oe_rec=session.oe_rec,
            column_map=col_map,
            stale_sync_warn=stale,
            stale_sync_acknowledged=stale_ack,
        )

    def status_for_record(self, record: EventRecord) -> BlockStatus:
        block_path = self.resolve_block_path(record)
        key = block_path.resolve() if block_path.exists() else block_path
        if key in self._skipped:
            return BlockStatus.SKIPPED | BlockStatus.LOAD_FAILED
        cache = self._cache.get(key)
        if cache is None:
            return BlockStatus.NONE
        if cache.skipped:
            return BlockStatus.SKIPPED
        flags = BlockStatus.NONE
        if cache.le_df is not None:
            flags |= BlockStatus.HAS_LE
        if cache.re_df is not None:
            flags |= BlockStatus.HAS_RE
        if cache.oe_rec is not None:
            flags |= BlockStatus.HAS_OE
        if cache.stale_sync_warn:
            flags |= BlockStatus.STALE_SYNC
        return flags

    def update_record_statuses(self, catalog: list[EventRecord]) -> None:
        for rec in catalog:
            rec.block_status = self.status_for_record(rec)

    def clear_cache(self) -> None:
        """Drop loaded blocks so the next preview reloads data from disk."""
        self._cache.clear()
        self._skipped.clear()
