"""Stage 3 -- Kerr-degree conversion.

Phase 0 skeleton. Real controls in Phase 5.
"""

from __future__ import annotations

from pathlib import Path

from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab


class KerrTab(BaseTab):
    tab_id = "kerr"
    tab_label = "Kerr"

    def status_signature(self, block: BlockHandle) -> list[Path]:
        tag = self._config.kerr_name_tag
        ap = block.analysis_path
        return [
            ap / f"left_kerr_angle_{tag}.csv",
            ap / f"right_kerr_angle_{tag}.csv",
        ]
