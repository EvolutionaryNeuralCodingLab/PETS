"""Playback timing: 60 Hz default, speed multiplier, row vs time advance."""

from __future__ import annotations

import time

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class PlaybackController(QObject):
    index_changed = pyqtSignal(int)
    playing_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n = 0
        self._ms_axis: np.ndarray | None = None
        self._index = 0
        self._playing = False
        self._forward = True
        self._speed = 1.0
        self._fps = 60.0
        self._step_rows = 1
        self._play_start_wall: float | None = None
        self._play_start_idx = 0
        self._play_start_ms = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)

    def set_timeline(self, n: int, ms_axis: np.ndarray) -> None:
        self._n = int(n)
        self._ms_axis = np.asarray(ms_axis, dtype=np.float64)
        self.set_index(min(self._index, max(0, self._n - 1)))

    def set_fps(self, fps: float) -> None:
        self._fps = max(1.0, float(fps))
        self._update_interval()

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.05, float(speed))
        self._update_interval()
        self._reset_play_anchor()

    def set_step_rows(self, step: int) -> None:
        self._step_rows = max(1, int(step))

    @property
    def index(self) -> int:
        return self._index

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def forward(self) -> bool:
        return self._forward

    def set_index(self, index: int, *, emit: bool = True) -> None:
        if self._n <= 0:
            self._index = 0
            return
        self._index = int(max(0, min(index, self._n - 1)))
        if emit:
            self.index_changed.emit(self._index)

    def toggle_play(self, *, forward: bool = True) -> None:
        if self._playing and self._forward == forward:
            self.pause()
            return
        if self._playing:
            self.pause()
        self.play(forward=forward)

    def play(self, *, forward: bool = True) -> None:
        if self._n <= 0:
            return
        self._forward = bool(forward)
        self._playing = True
        self._reset_play_anchor()
        self._update_interval()
        self._timer.start()
        self.playing_changed.emit(True)

    def pause(self) -> None:
        self._playing = False
        self._timer.stop()
        self._play_start_wall = None
        self.playing_changed.emit(False)

    def step(self, delta_rows: int) -> None:
        self.pause()
        self.set_index(self._index + delta_rows)

    def _update_interval(self) -> None:
        interval = int(round(1000.0 / self._fps / self._speed))
        self._timer.setInterval(max(1, interval))

    def _reset_play_anchor(self) -> None:
        self._play_start_wall = None

    def _on_timer(self) -> None:
        if self._n <= 0 or self._ms_axis is None:
            return

        step = 1 if self._forward else -1

        if self._speed <= 1.0:
            nxt = self._index + step
            if nxt < 0:
                self.set_index(0)
                self.pause()
                return
            if nxt >= self._n:
                self.set_index(self._n - 1)
                self.pause()
                return
            if nxt != self._index:
                self.set_index(nxt)
            return

        now = time.perf_counter()
        if self._play_start_wall is None:
            self._play_start_wall = now
            self._play_start_idx = self._index
            self._play_start_ms = float(self._ms_axis[self._index])

        elapsed = (now - self._play_start_wall) * self._speed
        if self._forward:
            target_ms = self._play_start_ms + elapsed * 1000.0
            nxt = int(np.searchsorted(self._ms_axis, target_ms, side="right") - 1)
            nxt = max(0, min(nxt, self._n - 1))
            if nxt >= self._n - 1 and target_ms >= float(self._ms_axis[-1]):
                self.set_index(self._n - 1)
                self.pause()
                return
        else:
            target_ms = self._play_start_ms - elapsed * 1000.0
            nxt = int(np.searchsorted(self._ms_axis, target_ms, side="left"))
            nxt = max(0, min(nxt, self._n - 1))
            if nxt <= 0 and target_ms <= float(self._ms_axis[0]):
                self.set_index(0)
                self.pause()
                return

        if nxt != self._index:
            self.set_index(nxt)
