"""Launch helpers for the saccade verification viewer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.analysis.block_registry import BlockSpec, load_registry
from eye_tracking_system_tools.analysis.saccade_viewer.app import (
    SaccadeViewerWindow,
    build_viewer_from_dataframe,
)


def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        return get_ipython() is not None
    except ImportError:
        return False


def _ensure_qapplication() -> QtWidgets.QApplication:
    """Return a process-wide QApplication that survives window close (Jupyter-safe)."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        # Avoid passing Jupyter's sys.argv — it can confuse Qt platform plugins.
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def _block_until_window_closed(win: SaccadeViewerWindow, app: QtWidgets.QApplication) -> None:
    """
    Wait for the viewer to close without tearing down the QApplication.

    ``app.exec()`` quits the whole application when the last window closes,
    which segfaults the Jupyter kernel on macOS. A local event loop tied to
    ``win.closed`` keeps Qt alive for later notebook cells.
    """
    if _running_in_ipython():
        loop = QtCore.QEventLoop()
        win.closed.connect(loop.quit)
        loop.exec()
        return
    app.exec()


def launch_saccade_viewer(
    events: pd.DataFrame,
    *,
    registry_path: Path | str | None = None,
    specs: list[BlockSpec] | None = None,
    pre_ms: float = 250.0,
    post_ms: float = 250.0,
    auto_advance: bool = False,
    block: bool = True,
) -> SaccadeViewerWindow | None:
    """
    Open the saccade verification window.

    Parameters
    ----------
    events
        Event table with ``animal``, ``block``, onset ms, and optionally ``eye``.
    registry_path
        YAML registry used to resolve ``block_path`` when not on the table.
    block
        When True (default), block until the user closes the window.
    """
    app = _ensure_qapplication()

    win = build_viewer_from_dataframe(
        events,
        registry_path=registry_path,
        specs=specs,
        pre_ms=pre_ms,
        post_ms=post_ms,
        auto_advance=auto_advance,
    )
    win.show()
    win.raise_()
    win.activateWindow()

    if block:
        _block_until_window_closed(win, app)
    return win


def sample_events_from_registry(
    registry_path: Path | str,
    *,
    per_block: int = 3,
    prefer_finalized: bool = True,
) -> pd.DataFrame:
    """Load a small per-block subsample for smoke testing."""
    from eye_tracking_system_tools.analysis.saccade_export import read_finalized_saccades

    specs = load_registry(registry_path)
    parts: list[pd.DataFrame] = []
    for spec in specs:
        if prefer_finalized:
            try:
                fin = read_finalized_saccades(spec.block_path)
                part = fin.all_saccades.head(per_block)
            except Exception:
                part = pd.DataFrame()
        else:
            part = pd.DataFrame()
        if part.empty:
            continue
        part = part.copy()
        part["block_path"] = str(spec.block_path)
        parts.append(part)
    if not parts:
        raise RuntimeError(f"No events loaded from registry {registry_path}")
    return pd.concat(parts, ignore_index=True)


__all__ = ["launch_saccade_viewer", "sample_events_from_registry"]
