"""Synced LE / Arena / RE video panel for Data Exploration (Annotator reuse)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.block_annotator.models import (
    AnnotatorConfig,
    BlockSession,
    first_timeline_index_with_frame,
)
from eye_tracking_system_tools.annotation.block_annotator.playback_controller import (
    PlaybackController,
)
from eye_tracking_system_tools.annotation.block_annotator.video_widget import VideoPanel
from eye_tracking_system_tools.annotation.block_annotator.block_loader import (
    load_block_session,
)


def _ellipse_frame_col(df: pd.DataFrame | None, preferred: str) -> str:
    if df is None:
        return preferred
    for name in (preferred, "eye_frame", "Eye_frame", "L_eye_frame", "R_eye_frame"):
        if name in df.columns:
            return name
    return preferred


class ExploreVideoPanel(QtWidgets.QWidget):
    """Three-way synced video with Annotator-style transport."""

    time_changed = QtCore.pyqtSignal(float)  # ms — emitted when scrub/play moves

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: BlockSession | None = None
        self._arena_index = 0
        self._playback = PlaybackController(self)
        self._suppress_time_emit = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        transport = QtWidgets.QHBoxLayout()
        self._btn_play = QtWidgets.QPushButton("▶ Play")
        self._btn_rev = QtWidgets.QPushButton("◀ Rev")
        self._btn_step_back = QtWidgets.QPushButton("⟨")
        self._btn_step_fwd = QtWidgets.QPushButton("⟩")
        self._speed = QtWidgets.QDoubleSpinBox()
        self._speed.setRange(0.05, 20.0)
        self._speed.setSingleStep(0.25)
        self._speed.setValue(1.0)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._hud = QtWidgets.QLabel("—")
        self._hud.setMinimumWidth(180)
        transport.addWidget(self._btn_play)
        transport.addWidget(self._btn_rev)
        transport.addWidget(self._btn_step_back)
        transport.addWidget(self._btn_step_fwd)
        transport.addWidget(QtWidgets.QLabel("Speed"))
        transport.addWidget(self._speed)
        transport.addWidget(self._slider, stretch=1)
        transport.addWidget(self._hud)
        layout.addLayout(transport)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._left_panel = VideoPanel("Left eye", min_width=120, min_height=100)
        self._arena_panel = VideoPanel("Arena", min_width=180, min_height=120)
        self._right_panel = VideoPanel("Right eye", min_width=120, min_height=100)
        splitter.addWidget(self._left_panel)
        splitter.addWidget(self._arena_panel)
        splitter.addWidget(self._right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        layout.addWidget(splitter, stretch=1)

        arena_row = QtWidgets.QHBoxLayout()
        self._arena_combo = QtWidgets.QComboBox()
        arena_row.addWidget(QtWidgets.QLabel("Arena:"))
        arena_row.addWidget(self._arena_combo, stretch=1)
        layout.addLayout(arena_row)

        # Eye display options on their own row so toggles stay visible.
        eye_opts = QtWidgets.QHBoxLayout()
        self._le_ann = QtWidgets.QCheckBox("L ellipse")
        self._re_ann = QtWidgets.QCheckBox("R ellipse")
        self._le_flip_v = QtWidgets.QCheckBox("L flip V")
        self._re_flip_v = QtWidgets.QCheckBox("R flip V")
        self._le_flip_h = QtWidgets.QCheckBox("L flip H")
        self._re_flip_h = QtWidgets.QCheckBox("R flip H")
        eye_opts.addWidget(self._le_ann)
        eye_opts.addWidget(self._re_ann)
        eye_opts.addSpacing(12)
        eye_opts.addWidget(self._le_flip_v)
        eye_opts.addWidget(self._re_flip_v)
        eye_opts.addSpacing(8)
        eye_opts.addWidget(self._le_flip_h)
        eye_opts.addWidget(self._re_flip_h)
        eye_opts.addStretch(1)
        layout.addLayout(eye_opts)

        self._placeholder = QtWidgets.QLabel(
            "Load analysis to sync Left / Arena / Right videos to the playhead."
        )
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #555; padding: 8px;")
        layout.addWidget(self._placeholder)

        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_rev.clicked.connect(self._toggle_rev)
        self._btn_step_back.clicked.connect(lambda: self._playback.step(-1))
        self._btn_step_fwd.clicked.connect(lambda: self._playback.step(1))
        self._speed.valueChanged.connect(self._playback.set_speed)
        self._slider.valueChanged.connect(self._on_slider)
        self._playback.index_changed.connect(self._on_index)
        self._playback.playing_changed.connect(self._on_playing_changed)
        self._arena_combo.currentIndexChanged.connect(self._on_arena_changed)
        for w in (
            self._le_ann,
            self._re_ann,
            self._le_flip_v,
            self._re_flip_v,
            self._le_flip_h,
            self._re_flip_h,
        ):
            w.toggled.connect(self._update_frame)

        self._set_enabled(False)

    def clear(self) -> None:
        self._playback.pause()
        self._session = None
        self._arena_panel.set_video(None)
        self._left_panel.set_video(None)
        self._right_panel.set_video(None)
        self._arena_combo.clear()
        self._slider.setMaximum(0)
        self._hud.setText("—")
        self._placeholder.show()
        self._set_enabled(False)

    def load_block(
        self,
        block_path: Path,
        output_folder: Path,
        *,
        animal_call: str | None = None,
        experiment_date: str | None = None,
        block_num: str | None = None,
        le_ellipse_df: pd.DataFrame | None = None,
        re_ellipse_df: pd.DataFrame | None = None,
    ) -> BlockSession:
        """Load videos + timeline via Annotator ``load_block_session``."""
        self.clear()
        session = load_block_session(
            Path(block_path),
            Path(output_folder),
            AnnotatorConfig(),
            animal_call=animal_call,
            experiment_date=experiment_date,
            block_num=block_num,
        )
        if le_ellipse_df is not None:
            session.le_ellipse_df = le_ellipse_df
        if re_ellipse_df is not None:
            session.re_ellipse_df = re_ellipse_df
        self._bind_session(session)
        return session

    def set_ellipse_data(
        self,
        le_df: pd.DataFrame | None,
        re_df: pd.DataFrame | None,
    ) -> None:
        if self._session is None:
            return
        self._session.le_ellipse_df = le_df
        self._session.re_ellipse_df = re_df
        self._left_panel.set_ellipse_data(
            le_df, _ellipse_frame_col(le_df, "L_eye_frame")
        )
        self._right_panel.set_ellipse_data(
            re_df, _ellipse_frame_col(re_df, "R_eye_frame")
        )
        self._update_frame()

    def seek_ms(self, ms: float, *, emit: bool = False) -> None:
        if self._session is None or self._session.n <= 0:
            return
        idx = int(np.searchsorted(self._session.ms_axis, ms, side="left"))
        idx = max(0, min(idx, self._session.n - 1))
        self._playback.pause()
        self._suppress_time_emit = not emit
        try:
            self._playback.set_index(idx)
        finally:
            self._suppress_time_emit = False

    def current_ms(self) -> float | None:
        if self._session is None or self._session.n <= 0:
            return None
        return float(self._session.ms_at(self._playback.index))

    def _set_enabled(self, enabled: bool) -> None:
        for w in (
            self._btn_play,
            self._btn_rev,
            self._btn_step_back,
            self._btn_step_fwd,
            self._speed,
            self._slider,
            self._arena_combo,
            self._le_ann,
            self._re_ann,
            self._le_flip_v,
            self._re_flip_v,
            self._le_flip_h,
            self._re_flip_h,
        ):
            w.setEnabled(enabled)

    def _bind_session(self, session: BlockSession) -> None:
        self._session = session
        self._arena_index = 0
        self._arena_combo.blockSignals(True)
        self._arena_combo.clear()
        for p in session.arena_videos:
            self._arena_combo.addItem(p.name, str(p))
        self._arena_combo.blockSignals(False)

        if session.arena_videos:
            self._arena_panel.set_video(session.arena_videos[0])
        if session.le_videos:
            self._left_panel.set_video(session.le_videos[0])
        if session.re_videos:
            self._right_panel.set_video(session.re_videos[0])

        self._left_panel.set_ellipse_data(
            session.le_ellipse_df,
            _ellipse_frame_col(session.le_ellipse_df, "L_eye_frame"),
        )
        self._right_panel.set_ellipse_data(
            session.re_ellipse_df,
            _ellipse_frame_col(session.re_ellipse_df, "R_eye_frame"),
        )
        self._left_panel.set_display_limits(360, 0)
        self._right_panel.set_display_limits(360, 0)
        self._arena_panel.set_display_limits(640, 0)

        self._playback.set_timeline(session.n, session.ms_axis)
        self._slider.setMaximum(max(0, session.n - 1))
        start = first_timeline_index_with_frame(session.final_sync_df)
        self._placeholder.hide()
        self._set_enabled(True)
        missing = []
        if not session.arena_videos:
            missing.append("arena")
        if not session.le_videos:
            missing.append("left eye")
        if not session.re_videos:
            missing.append("right eye")
        if missing:
            self._placeholder.setText(
                "Missing video(s): " + ", ".join(missing) + ". Timeline still scrubbable."
            )
            self._placeholder.show()
        self._playback.set_index(start)

    def _toggle_play(self) -> None:
        self._playback.toggle_play(forward=True)

    def _toggle_rev(self) -> None:
        self._playback.toggle_play(forward=False)

    def _on_playing_changed(self, playing: bool) -> None:
        if playing and self._playback.forward:
            self._btn_play.setText("⏸ Pause")
            self._btn_rev.setText("◀ Rev")
        elif playing and not self._playback.forward:
            self._btn_play.setText("▶ Play")
            self._btn_rev.setText("⏸ Pause")
        else:
            self._btn_play.setText("▶ Play")
            self._btn_rev.setText("◀ Rev")

    def _on_slider(self, value: int) -> None:
        self._playback.pause()
        self._playback.set_index(int(value))

    def _on_arena_changed(self, index: int) -> None:
        self._arena_index = index
        if self._session and 0 <= index < len(self._session.arena_videos):
            self._arena_panel.set_video(self._session.arena_videos[index])
            self._update_frame()

    def _on_index(self, index: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._update_frame()
        if self._session is not None and not self._suppress_time_emit:
            self.time_changed.emit(float(self._session.ms_at(index)))

    def _update_frame(self) -> None:
        if self._session is None:
            return
        i = self._playback.index
        arena_f, le_f, re_f = self._session.frame_ids_at(i)
        ms = self._session.ms_at(i)
        self._hud.setText(
            f"i={i}/{max(0, self._session.n - 1)} | {ms:.1f} ms | "
            f"A={arena_f} L={le_f} R={re_f}"
        )
        self._left_panel.set_flip_horizontal(self._le_flip_h.isChecked())
        self._right_panel.set_flip_horizontal(self._re_flip_h.isChecked())
        self._left_panel.set_flip_vertical(self._le_flip_v.isChecked())
        self._right_panel.set_flip_vertical(self._re_flip_v.isChecked())
        self._left_panel.set_show_annotations(self._le_ann.isChecked())
        self._right_panel.set_show_annotations(self._re_ann.isChecked())
        self._arena_panel.show_frame(arena_f, f"frame {arena_f}")
        self._left_panel.show_frame(le_f, f"frame {le_f}")
        self._right_panel.show_frame(re_f, f"frame {re_f}")
