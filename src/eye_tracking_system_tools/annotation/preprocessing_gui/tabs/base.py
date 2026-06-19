"""Base class for Preprocessing GUI tabs.

Each tab subclasses :class:`BaseTab` and overrides:

* :pyattr:`tab_id`, :pyattr:`tab_label`
* :py:meth:`status_signature` -- which output files this tab produces
* :py:meth:`set_block` -- called by the main window when the active block changes
* :py:meth:`build_ui` -- create child widgets (called from ``__init__``)

In Phase 0 the tabs only show a "coming in Phase N" placeholder; the
real controls are added in later phases.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
    PreprocConfig,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import (
    BlockHandle,
    GuiState,
)


class BaseTab(QtWidgets.QWidget):
    tab_id: str = "base"
    tab_label: str = "Tab"

    def __init__(
        self,
        state: GuiState,
        config: PreprocConfig,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._state = state
        self._config = config
        self._placeholder_label: QtWidgets.QLabel | None = None
        self.build_ui()

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._placeholder_label = QtWidgets.QLabel(
            f"<i>{self.tab_label}</i> &mdash; controls land in a later phase."
        )
        self._placeholder_label.setTextFormat(QtWidgets.QLabel().textFormat())
        layout.addWidget(self._placeholder_label)
        layout.addStretch(1)

    def set_block(self, block: BlockHandle | None) -> None:
        if self._placeholder_label is not None:
            if block is None:
                self._placeholder_label.setText(
                    f"<i>{self.tab_label}</i> &mdash; no block loaded."
                )
            else:
                self._placeholder_label.setText(
                    f"<i>{self.tab_label}</i> &mdash; controls land in a later "
                    f"phase.<br>Active block: <b>{block.display_label}</b>"
                )

    def status_signature(self, block: BlockHandle) -> list[Path]:
        """Files whose existence implies this stage is done.

        Phase 0 returns an empty list so the status bus reports NOT_STARTED.
        Subclasses override.
        """
        return []
