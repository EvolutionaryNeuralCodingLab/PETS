"""Helpers for opening Bokeh sync plots in the browser."""

from __future__ import annotations

from eye_tracking_system_tools.preprocessing.notebook_helpers import (
    plot_simple_sync_bokeh,
)


def open_shift_plot(block, df_left, df_right, *, show_led: bool = True) -> None:
    """Open the notebook's shift-correction plot in the default browser."""
    plot_simple_sync_bokeh(
        block,
        df_left=df_left,
        df_right=df_right,
        show_led=show_led,
        to_browser=True,
    )
