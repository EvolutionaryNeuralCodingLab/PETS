"""Event-scoped video panel with clip bounds and single-event highlight."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.explore_video_panel import (
    ExploreVideoPanel,
)
from eye_tracking_system_tools.analysis.saccade_viewer.event_clip import (
    ClipBounds,
    bounds_for_event,
    clamp_index,
    ms_to_index,
)
from eye_tracking_system_tools.analysis.saccade_viewer.models import VerificationEvent


class EventVideoPanel(QtWidgets.QWidget):
    """Wraps :class:`ExploreVideoPanel` with clip-scoped scrubbing."""

    time_changed = QtCore.pyqtSignal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._clip: ClipBounds | None = None
        self._pre_ms = 250.0
        self._post_ms = 250.0
        self._current_event: VerificationEvent | None = None
        self._clip_end_reached = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._panel = ExploreVideoPanel(self)
        layout.addWidget(self._panel, stretch=1)

        self._panel.time_changed.connect(self._on_panel_time_changed)
        self._panel._playback.index_changed.connect(self._on_index_changed)

    @property
    def panel(self) -> ExploreVideoPanel:
        return self._panel

    def clear(self) -> None:
        self._clip = None
        self._current_event = None
        self._panel.clear()

    def bind_session(self, session) -> None:
        self._panel._bind_session(session)

    def set_pre_post_ms(self, pre_ms: float, post_ms: float) -> None:
        self._pre_ms = max(0.0, float(pre_ms))
        self._post_ms = max(0.0, float(post_ms))
        if self._current_event is not None and self._panel._session is not None:
            self.set_event(self._current_event)

    def set_event(self, event: VerificationEvent | None) -> None:
        self._current_event = event
        session = self._panel._session
        if event is None or session is None:
            self._clip = None
            self._panel.clear_saccade_events()
            return

        self._clip = bounds_for_event(
            session.ms_axis,
            event,
            pre_ms=self._pre_ms,
            post_ms=self._post_ms,
        )
        self._highlight_event(event)
        self._panel._playback.pause()
        self._panel._playback.set_index(self._clip.i_start)
        self._clip_end_reached = False

    def _highlight_event(self, event: VerificationEvent) -> None:
        row = {
            "eye": event.eye,
            "saccade_on_ms": event.onset_ms,
            "saccade_off_ms": event.off_ms,
        }
        self._panel.set_saccade_events(pd.DataFrame([row]))

    def seek_ms(self, ms: float, *, emit: bool = False) -> None:
        session = self._panel._session
        if session is None:
            return
        idx = ms_to_index(session.ms_axis, ms)
        if self._clip is not None:
            idx = clamp_index(idx, self._clip)
        self._panel._playback.pause()
        self._panel._suppress_time_emit = not emit
        try:
            self._panel._playback.set_index(idx)
        finally:
            self._panel._suppress_time_emit = False

    def current_ms(self) -> float | None:
        return self._panel.current_ms()

    def toggle_play(self) -> None:
        if self._panel._playback.playing:
            self._panel._playback.pause()
        else:
            self._clip_end_reached = False
            self._panel._playback.play(forward=True)

    def play_from_start(self) -> None:
        """Seek to clip start and play forward."""
        if self._clip is None:
            return
        self._panel._playback.pause()
        self._panel._playback.set_index(self._clip.i_start)
        self._clip_end_reached = False
        self._panel._playback.play(forward=True)
        session = self._panel._session
        if session is not None:
            self.time_changed.emit(float(session.ms_at(self._clip.i_start)))

    def step_frame(self, delta: int) -> None:
        session = self._panel._session
        if session is None or self._clip is None:
            self._panel._playback.step(delta)
            return
        self._panel._playback.pause()
        idx = clamp_index(self._panel._playback.index + delta, self._clip)
        self._panel._playback.set_index(idx)
        self.time_changed.emit(float(session.ms_at(idx)))

    def jump_clip_start(self) -> None:
        if self._clip is None:
            return
        self.seek_ms(self._clip.t_start_ms, emit=True)

    def jump_clip_end(self) -> None:
        if self._clip is None:
            return
        self.seek_ms(self._clip.t_end_ms, emit=True)

    def release_resources(self) -> None:
        self._panel.clear()

    def _on_index_changed(self, index: int) -> None:
        if self._clip is None:
            return
        session = self._panel._session
        if session is None:
            return
        if index > self._clip.i_end:
            self._panel._playback.pause()
            self._panel._playback.set_index(self._clip.i_end, emit=False)
            self._clip_end_reached = True
        elif index < self._clip.i_start:
            self._panel._playback.pause()
            self._panel._playback.set_index(self._clip.i_start, emit=False)

    def _on_panel_time_changed(self, ms: float) -> None:
        self.time_changed.emit(float(ms))
