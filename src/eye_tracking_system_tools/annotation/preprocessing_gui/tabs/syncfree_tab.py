"""Stage 5 -- Sync-free eye ellipse pipeline.

Phase 0 skeleton. Real controls in Phase 7.
"""

from __future__ import annotations

from pathlib import Path

from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab


class SyncFreeTab(BaseTab):
    tab_id = "syncfree"
    tab_label = "Sync-free"

    def status_signature(self, block: BlockHandle) -> list[Path]:
        tag = self._config.syncfree_artifact_tag
        ap = block.analysis_path
        return [
            ap / f"left_eye_degrees_from_syncfree_{tag}.csv",
            ap / f"right_eye_degrees_from_syncfree_{tag}.csv",
        ]
