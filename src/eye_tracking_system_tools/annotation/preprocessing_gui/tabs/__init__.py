"""Per-stage tabs for the Preprocessing GUI.

Each tab subclasses :class:`BaseTab` and declares:

* ``tab_id`` (stable string used by the status bus)
* ``tab_label`` (display name)
* ``status_signature(block)`` (list of files whose existence implies done)

Explore (Data Exploration) is a post-sync inspection tab.
Calibration writes ``LR_pix_size.csv``.
Saccades (after Kerr) covers velocity thresholding, detection, and finalize
to ``analysis/saccades/``.

Imports are lazy so ``tabs.spin_max_dialog`` (used by the Jupyter tuner) does
not pull in Refine / the rest of the GUI and create a circular import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseTab": ".base",
    "SyncTab": ".sync_tab",
    "VerifyTab": ".verify_tab",
    "RefineTab": ".refine_tab",
    "ConicoidTab": ".conicoid_tab",
    "KerrTab": ".kerr_tab",
    "CalibrationTab": ".calibration_tab",
    "SaccadesTab": ".saccades_tab",
    "BehaviorTab": ".behavior_tab",
    "SyncFreeTab": ".syncfree_tab",
    "ExploreTab": ".explore_tab",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    modname = _EXPORTS.get(name)
    if modname is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modname, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
