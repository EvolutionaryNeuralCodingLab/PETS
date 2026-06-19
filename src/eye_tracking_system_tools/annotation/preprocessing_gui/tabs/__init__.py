"""Per-stage tabs for the Preprocessing GUI.

Each tab subclasses :class:`BaseTab` and declares:

* ``tab_id`` (stable string used by the status bus)
* ``tab_label`` (display name)
* ``status_signature(block)`` (list of files whose existence implies done)

Phase 0 ships skeletons only; the actual controls land in Phases 1-7.
"""

from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.sync_tab import SyncTab
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.verify_tab import (
    VerifyTab,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.kerr_tab import KerrTab
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.behavior_tab import (
    BehaviorTab,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.syncfree_tab import (
    SyncFreeTab,
)

__all__ = [
    "BaseTab",
    "SyncTab",
    "VerifyTab",
    "KerrTab",
    "BehaviorTab",
    "SyncFreeTab",
]
