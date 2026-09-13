"""Tests for notebook figure display helpers (no duplicate inline paints)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eye_tracking_system_tools.analysis.figure_display import (
    _clear_inline_draw_flag,
    capture_figure_previews,
    show_and_close,
    stop_capturing_figure_previews,
)


def test_show_and_close_closes_figure_and_clears_inline_flag() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    assert fig.number in plt.get_fignums()

    with patch(
        "eye_tracking_system_tools.analysis.figure_display._clear_inline_draw_flag"
    ) as clear_flag:
        show_and_close(fig, show=False)
        clear_flag.assert_called_once()

    assert fig.number not in plt.get_fignums()


def test_show_and_close_publishes_png_not_live_figure() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    displayed: list[object] = []

    def _capture(obj, **_kwargs):
        displayed.append(obj)

    fake_image_cls = MagicMock(side_effect=lambda data: ("Image", data[:16]))

    with (
        patch("IPython.display.display", _capture),
        patch("IPython.display.Image", fake_image_cls),
    ):
        show_and_close(fig, show=True)

    assert fig.number not in plt.get_fignums()
    assert len(displayed) == 1
    assert displayed[0][0] == "Image"
    assert isinstance(displayed[0][1], (bytes, bytearray))
    fake_image_cls.assert_called_once()


def test_clear_inline_draw_flag_is_safe_without_matplotlib_inline() -> None:
    # Should not raise even if matplotlib_inline is absent / unused.
    _clear_inline_draw_flag()


def test_show_and_close_can_collect_previews_without_display() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    bucket = capture_figure_previews()
    try:
        with patch("IPython.display.display") as disp:
            show_and_close(fig, show=True)
            disp.assert_not_called()
    finally:
        stop_capturing_figure_previews()
    assert fig.number not in plt.get_fignums()
    assert len(bucket) == 1
    assert isinstance(bucket[0], (bytes, bytearray))
    assert bucket[0][:8] == b"\x89PNG\r\n\x1a\n"
