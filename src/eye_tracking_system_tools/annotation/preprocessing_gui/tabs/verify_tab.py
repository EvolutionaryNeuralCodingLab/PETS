"""Stage 2 -- Data verification (Kerr-ref pick + ellipse review).

Phase 0 skeleton. The full ``EllipseVerifierWidget`` ships in Phase 4.
"""

from __future__ import annotations

from pathlib import Path

from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab


class VerifyTab(BaseTab):
    tab_id = "verify"
    tab_label = "Verify"

    def status_signature(self, block: BlockHandle) -> list[Path]:
        return [block.analysis_path / "self_kerr_refs.csv"]
