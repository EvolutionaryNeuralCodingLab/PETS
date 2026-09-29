"""Block discovery + selection widgets for the Preprocessing GUI."""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle

_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
_BLOCK_RE = re.compile(r"^block_(\d+)$", re.IGNORECASE)


def discover_blocks(
    experiment_path: Path,
    animal: str | None = None,
    block_filter: list[str] | None = None,
) -> list[BlockHandle]:
    """Walk an experiment folder and return every block that looks valid."""
    experiment_path = Path(experiment_path)
    if not experiment_path.is_dir():
        return []

    out: list[BlockHandle] = []

    animals = (
        [animal]
        if animal
        else sorted(p.name for p in experiment_path.iterdir() if p.is_dir())
    )

    for a in animals:
        animal_dir = experiment_path / a
        if not animal_dir.is_dir():
            continue
        for child in sorted(animal_dir.iterdir()):
            if not child.is_dir():
                continue
            if _DATE_RE.match(child.name):
                for block_dir in sorted(child.iterdir()):
                    if not block_dir.is_dir():
                        continue
                    m = _BLOCK_RE.match(block_dir.name)
                    if not m:
                        continue
                    block_num = m.group(1)
                    if block_filter and not _matches_filter(block_num, block_filter):
                        continue
                    out.append(
                        BlockHandle(
                            animal_call=a,
                            experiment_date=child.name,
                            block_num=block_num,
                            block_path=block_dir,
                            path_to_animal_folder=experiment_path,
                        )
                    )
            else:
                m = _BLOCK_RE.match(child.name)
                if not m:
                    continue
                block_num = m.group(1)
                if block_filter and not _matches_filter(block_num, block_filter):
                    continue
                out.append(
                    BlockHandle(
                        animal_call=a,
                        experiment_date=None,
                        block_num=block_num,
                        block_path=child,
                        path_to_animal_folder=experiment_path,
                    )
                )

    return out


def _matches_filter(block_num: str, block_filter: list[str]) -> bool:
    nums_int = set()
    raw_strs = set()
    for s in block_filter:
        s = str(s).strip()
        if not s:
            continue
        raw_strs.add(s)
        raw_strs.add(s.zfill(3))
        try:
            nums_int.add(int(s))
        except ValueError:
            pass
    try:
        bn_int = int(block_num)
    except ValueError:
        bn_int = None
    return (block_num in raw_strs) or (bn_int is not None and bn_int in nums_int)


def infer_fields_from_block_folder(
    block_folder: Path,
) -> tuple[Path, str, str] | None:
    """Infer (experiment_path, animal, block_num) from a block folder path."""
    p = Path(block_folder)
    if not p.name.lower().startswith("block_"):
        return None
    block_num = p.name.split("_", 1)[1].strip()
    if not block_num:
        return None
    if p.parent.name.count("_") == 2 and len(p.parent.name) == 10:
        animal = p.parent.parent.name
        experiment_path = p.parent.parent.parent
    else:
        animal = p.parent.name
        experiment_path = p.parent.parent
    if not animal or not experiment_path.exists():
        return None
    return (experiment_path, animal, block_num)


class BlockPicker(QtWidgets.QWidget):
    """Top-of-window strip showing loaded blocks and session controls."""

    block_changed = QtCore.pyqtSignal(int)
    add_blocks_requested = QtCore.pyqtSignal()
    release_block_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._blocks: list[BlockHandle] = []

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        self._label = QtWidgets.QLabel("No block loaded")
        self._combo = QtWidgets.QComboBox()
        self._combo.setMinimumWidth(320)
        self._btn_add = QtWidgets.QPushButton("Add block(s)…")
        self._btn_release = QtWidgets.QPushButton("Release block")
        self._btn_release.setEnabled(False)
        self._reload_btn = QtWidgets.QPushButton("Reload")

        layout.addWidget(QtWidgets.QLabel("Active block:"))
        layout.addWidget(self._combo, stretch=1)
        layout.addWidget(self._btn_add)
        layout.addWidget(self._btn_release)
        layout.addWidget(self._reload_btn)
        layout.addStretch(1)
        layout.addWidget(self._label)

        self._combo.currentIndexChanged.connect(self.block_changed.emit)
        self._btn_add.clicked.connect(self.add_blocks_requested.emit)
        self._btn_release.clicked.connect(self.release_block_requested.emit)

    def set_blocks(self, blocks: list[BlockHandle], current_index: int = 0) -> None:
        self._blocks = list(blocks)
        self._combo.blockSignals(True)
        self._combo.clear()
        for b in self._blocks:
            self._combo.addItem(b.display_label, b)
        if self._blocks:
            idx = max(0, min(current_index, len(self._blocks) - 1))
            self._combo.setCurrentIndex(idx)
        else:
            self._combo.setCurrentIndex(-1)
        self._combo.blockSignals(False)
        self._reload_btn.setEnabled(bool(self._blocks))
        self._btn_release.setEnabled(bool(self._blocks))
        self._refresh_label()
        if self._blocks:
            self.block_changed.emit(self._combo.currentIndex())

    def reload_button(self) -> QtWidgets.QPushButton:
        return self._reload_btn

    def current_index(self) -> int:
        return self._combo.currentIndex()

    def current_block(self) -> BlockHandle | None:
        i = self._combo.currentIndex()
        if 0 <= i < len(self._blocks):
            return self._blocks[i]
        return None

    def set_session_busy(self, busy: bool) -> None:
        self._btn_add.setEnabled(not busy)
        self._btn_release.setEnabled(not busy and bool(self._blocks))

    def _refresh_label(self) -> None:
        b = self.current_block()
        if b is None:
            self._label.setText("No block loaded")
        else:
            self._label.setText(f"{b.block_path}")
