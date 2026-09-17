"""Build and walk the ellipse-rotation parameter dialog (Jupyter-safe)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.analysis.block_registry import BlockSpec, _infer_animal
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.spin_max_dialog import (
    SpinMaximizerDialog,
)
from eye_tracking_system_tools.preprocessing.conicoid.rotation_inputs import (
    SequentialVideoReader,
    discover_eye_video,
    load_raw_dlc_table,
)
from eye_tracking_system_tools.preprocessing.conicoid.rotation_params import (
    try_read_rotation_params,
    write_rotation_params,
)


def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        return get_ipython() is not None
    except ImportError:
        return False


def _ensure_qapplication() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def _exec_dialog(dialog: QtWidgets.QDialog, app: QtWidgets.QApplication) -> int:
    if _running_in_ipython():
        loop = QtCore.QEventLoop()
        dialog.finished.connect(loop.quit)
        dialog.open()
        loop.exec()
        return int(dialog.result())
    return int(dialog.exec())


@dataclass
class TunerSession:
    block_path: Path
    left_df: pd.DataFrame
    right_df: pd.DataFrame
    left_reader: SequentialVideoReader
    right_reader: SequentialVideoReader
    left_width: float
    right_width: float
    params: RotationCorrectionParams | None

    def close(self) -> None:
        self.left_reader.close()
        self.right_reader.close()


def open_tuner_session(block_path: Path | str) -> TunerSession:
    path = Path(block_path)
    left_video = discover_eye_video(path, "left")
    right_video = discover_eye_video(path, "right")
    if left_video is None or right_video is None:
        raise FileNotFoundError(f"Eye videos missing under {path / 'eye_videos'}")
    left_df = load_raw_dlc_table(path, "left")
    right_df = load_raw_dlc_table(path, "right")
    left_reader = SequentialVideoReader(left_video)
    right_reader = SequentialVideoReader(right_video)
    left_width = left_reader.frame_width() or 0.0
    right_width = right_reader.frame_width() or 0.0
    if left_width <= 0:
        frame = left_reader.read_frame(0)
        left_width = float(frame.shape[1]) if frame is not None else 0.0
    if right_width <= 0:
        frame = right_reader.read_frame(0)
        right_width = float(frame.shape[1]) if frame is not None else 0.0
    return TunerSession(
        block_path=path,
        left_df=left_df,
        right_df=right_df,
        left_reader=left_reader,
        right_reader=right_reader,
        left_width=float(left_width),
        right_width=float(right_width),
        params=try_read_rotation_params(path),
    )


def build_spin_dialog(
    session: TunerSession,
    *,
    parent: QtWidgets.QWidget | None = None,
    left_frame0: int = 0,
    right_frame0: int = 0,
    block_label: str | None = None,
) -> SpinMaximizerDialog:
    existing = session.params
    return SpinMaximizerDialog(
        left_df=session.left_df,
        right_df=session.right_df,
        left_frame_fn=session.left_reader.read_frame,
        right_frame_fn=session.right_reader.read_frame,
        left_nframes=max(session.left_reader.nframes, 1),
        right_nframes=max(session.right_reader.nframes, 1),
        left_frame0=left_frame0,
        right_frame0=right_frame0,
        left_width=session.left_width,
        right_width=session.right_width,
        parent=parent,
        initial_left=None if existing is None else existing.left,
        initial_right=None if existing is None else existing.right,
        initial_apply_jitter=True if existing is None else existing.apply_jitter,
        block_label=block_label,
    )


def save_dialog_params(
    block_path: Path | str, dialog: SpinMaximizerDialog
) -> Path:
    return write_rotation_params(block_path, dialog.correction_params())


def launch_rotation_param_tuner(
    blocks: Iterable[BlockSpec | Path | str],
    *,
    parent: QtWidgets.QWidget | None = None,
) -> list[dict[str, str]]:
    """Walk selected blocks one dialog at a time. Cancel skips that block."""
    app = _ensure_qapplication()
    outcomes: list[dict[str, str]] = []
    items: list[tuple[str, Path]] = []
    for spec in blocks:
        if isinstance(spec, BlockSpec):
            items.append((spec.block_key, spec.block_path))
        else:
            path = Path(spec)
            items.append((f"{_infer_animal(path)}_{path.name}", path))

    for label, block_path in items:
        session = None
        try:
            session = open_tuner_session(block_path)
            dialog = build_spin_dialog(
                session, parent=parent, block_label=label
            )
            code = _exec_dialog(dialog, app)
            if code == int(QtWidgets.QDialog.DialogCode.Accepted):
                path = save_dialog_params(block_path, dialog)
                outcomes.append(
                    {
                        "block": label,
                        "status": "saved",
                        "path": str(path.resolve()),
                    }
                )
                print(f"Saved {path}", flush=True)
            else:
                outcomes.append(
                    {"block": label, "status": "skipped", "path": ""}
                )
                print(f"Skipped {label}", flush=True)
        except Exception as exc:
            outcomes.append(
                {"block": label, "status": "failed", "path": "", "message": str(exc)}
            )
            print(f"Failed {label}: {exc}", flush=True)
        finally:
            if session is not None:
                session.close()
    return outcomes
