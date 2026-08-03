"""Stage 2 -- Data verification (Kerr-ref pick + pupil perimeter + ellipse review)."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets

from eye_tracking_system_tools.annotation.preprocessing_gui.analysis_artifacts import (
    VERIFY_ARTIFACT_PROFILE,
    LoadReport,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.ellipse_verifier import (
    EllipseVerifierWidget,
)
from eye_tracking_system_tools.annotation.preprocessing_gui.models import BlockHandle
from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.base import BaseTab
from eye_tracking_system_tools.preprocessing.data_verification_utils import (
    export_corrected_eye_data,
    export_current_kerr_refs,
    load_eye_data,
)
from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
    load_self_kerr_refs,
)
from eye_tracking_system_tools.preprocessing.pupil_perimeter import (
    PERIMETERS_FILENAME,
    read_pupil_perimeters,
    write_pupil_perimeters,
)


def eye_data_is_stale(block: BlockHandle) -> bool:
    """True when ``final_sync_df.csv`` is newer than either eye-data CSV."""
    analysis = block.analysis_path
    final_sync = analysis / "final_sync_df.csv"
    left_eye = analysis / "left_eye_data.csv"
    right_eye = analysis / "right_eye_data.csv"
    if not final_sync.is_file():
        return False
    if not left_eye.is_file() or not right_eye.is_file():
        return False
    sync_mtime = final_sync.stat().st_mtime
    return left_eye.stat().st_mtime < sync_mtime or right_eye.stat().st_mtime < sync_mtime


class VerifyTab(BaseTab):
    tab_id = "verify"
    tab_label = "Verify"

    def __init__(self, state, config, parent=None):
        self._left_verifier: EllipseVerifierWidget | None = None
        self._right_verifier: EllipseVerifierWidget | None = None
        super().__init__(state, config, parent)

    def artifact_profile(self):
        return VERIFY_ARTIFACT_PROFILE

    def build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_artifact_panel())

        self._stale_banner = QtWidgets.QLabel()
        self._stale_banner.setWordWrap(True)
        self._stale_banner.setStyleSheet(
            "background-color: #fff3cd; color: #664d03; padding: 8px; border-radius: 4px;"
        )
        self._stale_banner.hide()
        layout.addWidget(self._stale_banner)

        self._info = QtWidgets.QLabel(
            "Review ellipses, pick Kerr refs, and optionally draw a pupil perimeter "
            "(physiological bound). Use Compute angles span on each eye to preview "
            "φ/θ histograms for the current red-dot reference before saving. "
            "Commit bad datapoints writes noise-epoch catalogs "
            "(does not NaN eye CSVs). Zoom region + image filters are per-eye display "
            "aids. Re-run Sync → Read DLC to refit with DLC keypoint masking."
        )
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        self._verifier_host = QtWidgets.QWidget()
        self._verifier_layout = QtWidgets.QHBoxLayout(self._verifier_host)
        self._verifier_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._verifier_host, stretch=1)

        save_row = QtWidgets.QHBoxLayout()
        save_row.addStretch(1)
        self._btn_save = QtWidgets.QPushButton("Save & export (both eyes)")
        self._btn_save.setEnabled(False)
        save_row.addWidget(self._btn_save)
        layout.addLayout(save_row)

        self._status = QtWidgets.QLabel("")
        layout.addWidget(self._status)

        self._btn_save.clicked.connect(self._save_all)

    def status_signature(self, block: BlockHandle) -> list[Path]:
        # Kerr refs remain the primary completion signal; perimeter is optional.
        return [block.analysis_path / "self_kerr_refs.csv"]

    def set_block(self, block: BlockHandle | None) -> None:
        self._clear_verifiers()
        self._btn_save.setEnabled(False)
        if block is None:
            self._info.setText("No block loaded.")
            self._stale_banner.hide()
            self._status.setText("")
        else:
            self._info.setText(
                f"Active block: {block.display_label}. "
                "Kerr / Perimeter / Zoom on each eye; Commit bad datapoints catalogs "
                f"pupil_perimeter epochs. Save exports eye CSVs, self_kerr_refs.csv, "
                f"and {PERIMETERS_FILENAME}."
            )
            self._update_stale_banner(block)
            self._status.setText(
                "Use 'Load prev analysis' when eye data exists on disk, "
                "or complete Sync tab step 5 first."
            )
        super().set_block(block)

    def _after_load_artifacts(self, report: LoadReport) -> None:
        if self._block is None:
            return
        try:
            self._load_verifiers(self._block)
            self._btn_save.setEnabled(True)
            self._status.setText(
                "Loaded verification data. Adjust both eyes (Kerr / perimeter / zoom), "
                "then Save once."
            )
        except Exception as e:
            self._info.setText(f"Cannot load verification data: {e}")
            self._status.setText("")

    def _update_stale_banner(self, block: BlockHandle) -> None:
        if eye_data_is_stale(block):
            self._stale_banner.setText(
                "Eye-data CSVs are older than final_sync_df.csv — upstream sync may have "
                "changed. Re-run Sync finalize or re-export eye data before verifying."
            )
            self._stale_banner.show()
        else:
            self._stale_banner.hide()

    def _clear_verifiers(self) -> None:
        while self._verifier_layout.count():
            item = self._verifier_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._left_verifier = None
        self._right_verifier = None

    def _load_verifiers(self, block: BlockHandle) -> None:
        left_path = block.analysis_path / "left_eye_data.csv"
        right_path = block.analysis_path / "right_eye_data.csv"
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(
                "left_eye_data.csv / right_eye_data.csv are missing. "
                "Complete Sync tab step 5 first."
            )

        blocksync = self._require_blocksync()
        self._session.ensure_eye_videos(blocksync)
        load_eye_data(blocksync)
        load_self_kerr_refs(blocksync)
        perimeters = read_pupil_perimeters(block.block_path)

        ref_l = None
        ref_r = None
        if getattr(blocksync, "kerr_ref_l_x", None) is not None and getattr(
            blocksync, "kerr_ref_l_y", None
        ) is not None:
            ref_l = (int(blocksync.kerr_ref_l_x), int(blocksync.kerr_ref_l_y))
        if getattr(blocksync, "kerr_ref_r_x", None) is not None and getattr(
            blocksync, "kerr_ref_r_y", None
        ) is not None:
            ref_r = (int(blocksync.kerr_ref_r_x), int(blocksync.kerr_ref_r_y))

        if not blocksync.le_videos or not blocksync.re_videos:
            raise RuntimeError("Eye videos not found — run Prepare data on Sync tab.")

        self._clear_verifiers()
        self._left_verifier = EllipseVerifierWidget(
            blocksync.left_eye_data,
            blocksync.le_videos[0],
            "left",
            ref_point_xy=ref_l,
            parent=self,
            perimeter=perimeters.get("left"),
            block_path=block.block_path,
        )
        self._right_verifier = EllipseVerifierWidget(
            blocksync.right_eye_data,
            blocksync.re_videos[0],
            "right",
            ref_point_xy=ref_r,
            parent=self,
            perimeter=perimeters.get("right"),
            block_path=block.block_path,
        )
        self._verifier_layout.addWidget(self._left_verifier)
        self._verifier_layout.addWidget(self._right_verifier)

    def _save_all(self) -> None:
        if self._left_verifier is None or self._right_verifier is None:
            return

        blocksync = self._require_blocksync()
        left_df = self._left_verifier.df()
        right_df = self._right_verifier.df()
        left_ref = self._left_verifier.ref_xy()
        right_ref = self._right_verifier.ref_xy()

        blocksync.left_eye_data = left_df
        blocksync.right_eye_data = right_df
        if left_ref is not None:
            blocksync.kerr_ref_l_x = int(round(left_ref[0]))
            blocksync.kerr_ref_l_y = int(round(left_ref[1]))
        if right_ref is not None:
            blocksync.kerr_ref_r_x = int(round(right_ref[0]))
            blocksync.kerr_ref_r_y = int(round(right_ref[1]))

        export_corrected_eye_data(blocksync)
        export_current_kerr_refs(blocksync)
        peri_path = write_pupil_perimeters(
            Path(blocksync.block_path),
            self._left_verifier.perimeter(),
            self._right_verifier.perimeter(),
        )
        self._status.setText(
            "Saved both eyes → left/right_eye_data.csv, self_kerr_refs.csv, "
            f"and {peri_path.name}."
        )
