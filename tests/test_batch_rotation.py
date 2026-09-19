"""Batch ellipse rotation correction: YAML params, skip/overwrite, jitter tagging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from eye_tracking_system_tools.preprocessing.conicoid.batch_rotation import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_OK_JITTER_MISSING,
    STATUS_SKIPPED_EXISTS,
    STATUS_SKIPPED_NO_PARAMS,
    load_rotation_registry,
    run_block,
    run_registry,
)
from eye_tracking_system_tools.preprocessing.conicoid.refined_io import (
    rotation_fixed_eye_csv_path,
)
from eye_tracking_system_tools.preprocessing.conicoid.rotation_params import (
    RotationCorrectionParams,
    read_rotation_params,
    settings_from_mapping,
    settings_to_mapping,
    write_rotation_params,
)
from eye_tracking_system_tools.preprocessing.conicoid.spin_max import SpinMaxSettings


def _dlc_csv(path: Path, side: str, n: int = 3) -> None:
    frame_col = "L_eye_frame" if side == "left" else "R_eye_frame"
    pd.DataFrame(
        {
            frame_col: np.arange(n),
            "center_x": np.full(n, 8.0),
            "center_y": np.full(n, 8.0),
            "width": np.full(n, 4.0),
            "height": np.full(n, 3.0),
            "phi": np.full(n, 0.2),
        }
    ).to_csv(path, index=False)


def _params() -> RotationCorrectionParams:
    return RotationCorrectionParams(
        left=SpinMaxSettings(threshold=80, roi_mult=1.3, x_flip=True, frame_width=16.0),
        right=SpinMaxSettings(threshold=None, roi_mult=1.1, x_flip=False, frame_width=16.0),
        apply_jitter=True,
    )


def _fake_reader_cls(image: np.ndarray):
    class FakeReader:
        nframes = 8

        def __init__(self, *args, **kwargs):
            pass

        def close(self) -> None:
            return None

        def read_frame(self, idx: int | None, as_gray: bool = False):
            if as_gray and image.ndim == 3:
                return image[..., 0]
            return image

        def frame_width(self) -> float:
            return float(image.shape[1])

    return FakeReader


def _patch_videos(monkeypatch, image: np.ndarray) -> None:
    from eye_tracking_system_tools.preprocessing.conicoid import batch_rotation as br

    fake = _fake_reader_cls(image)
    monkeypatch.setattr(br, "SequentialVideoReader", fake)
    monkeypatch.setattr(br, "discover_eye_video", lambda *_a, **_k: Path("dummy.mp4"))


def test_settings_yaml_roundtrip_otsu_and_xflip():
    left = SpinMaxSettings(
        threshold=None, roi_mult=1.4, x_flip=True, frame_width=640.0, median_k=7
    )
    payload = settings_to_mapping(left)
    assert payload["threshold"] is None
    assert payload["x_flip"] is True
    restored = settings_from_mapping(payload)
    assert restored.threshold is None
    assert restored.roi_mult == pytest.approx(1.4)
    assert restored.x_flip is True
    assert restored.frame_width == pytest.approx(640.0)
    assert restored.median_k == 7


def test_write_read_rotation_params(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    written = write_rotation_params(tmp_path, _params())
    assert written.name == "rotation_correction_params.yaml"
    loaded = read_rotation_params(tmp_path)
    assert loaded.apply_jitter is True
    assert loaded.left.threshold == 80
    assert loaded.left.x_flip is True
    assert loaded.left.roi_mult == pytest.approx(1.3)
    assert loaded.right.threshold is None
    raw = yaml.safe_load(written.read_text())
    assert raw["left"]["threshold"] == 80
    assert raw["right"]["threshold"] is None


def test_run_block_skips_without_params(tmp_path: Path):
    (tmp_path / "analysis").mkdir()
    result = run_block(tmp_path, show_tqdm=False)
    assert result.status == STATUS_SKIPPED_NO_PARAMS


def test_run_block_fails_without_dlc_or_videos(tmp_path: Path):
    (tmp_path / "analysis").mkdir()
    write_rotation_params(tmp_path, _params())
    result = run_block(tmp_path, show_tqdm=False)
    assert result.status == STATUS_FAILED
    assert "le_df" in result.message or "video" in result.message.lower()


def test_run_block_missing_jitter_writes_tagged_csv(tmp_path: Path, monkeypatch):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    write_rotation_params(tmp_path, _params())
    _dlc_csv(analysis / "le_df.csv", "left")
    _dlc_csv(analysis / "re_df.csv", "right")
    img = np.full((16, 16, 3), 180, dtype=np.uint8)
    _patch_videos(monkeypatch, img)

    result = run_block(tmp_path, overwrite=True, show_tqdm=False)
    assert result.status == STATUS_OK_JITTER_MISSING
    left = rotation_fixed_eye_csv_path(tmp_path, "left", jitter_corrected=False)
    right = rotation_fixed_eye_csv_path(tmp_path, "right", jitter_corrected=False)
    assert left.is_file()
    assert right.is_file()
    assert "JitterNotCorrected" in left.name
    assert not rotation_fixed_eye_csv_path(
        tmp_path, "left", jitter_corrected=True
    ).is_file()


def test_run_block_with_jitter_writes_untagged_csv(tmp_path: Path, monkeypatch):
    import pickle

    analysis = tmp_path / "analysis"
    analysis.mkdir()
    write_rotation_params(tmp_path, _params())
    _dlc_csv(analysis / "le_df.csv", "left")
    _dlc_csv(analysis / "re_df.csv", "right")
    jitter = {
        "left_eye": {
            "x_displacement": np.zeros(8),
            "y_displacement": np.zeros(8),
        },
        "right_eye": {
            "x_displacement": np.zeros(8),
            "y_displacement": np.zeros(8),
        },
    }
    with open(analysis / "jitter_report_dict.pkl", "wb") as handle:
        pickle.dump(jitter, handle)
    img = np.full((16, 16, 3), 180, dtype=np.uint8)
    _patch_videos(monkeypatch, img)

    result = run_block(tmp_path, overwrite=True, show_tqdm=False)
    assert result.status == STATUS_OK
    assert result.jitter_applied is True
    left = rotation_fixed_eye_csv_path(tmp_path, "left", jitter_corrected=True)
    assert left.is_file()
    assert "JitterNotCorrected" not in left.name


def test_overwrite_false_skips_existing(tmp_path: Path, monkeypatch):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    write_rotation_params(tmp_path, _params())
    _dlc_csv(analysis / "le_df.csv", "left")
    _dlc_csv(analysis / "re_df.csv", "right")
    img = np.full((16, 16, 3), 180, dtype=np.uint8)
    _patch_videos(monkeypatch, img)
    first = run_block(tmp_path, overwrite=True, show_tqdm=False)
    assert first.status == STATUS_OK_JITTER_MISSING
    left = rotation_fixed_eye_csv_path(tmp_path, "left", jitter_corrected=False)
    stamp = left.read_text()
    second = run_block(tmp_path, overwrite=False, show_tqdm=False)
    assert second.status == STATUS_SKIPPED_EXISTS
    assert left.read_text() == stamp


def test_load_rotation_registry_animals_and_blocks(tmp_path: Path):
    animals = tmp_path / "paper.yaml"
    animals.write_text(
        "animals:\n  PV_1:\n    - /does/not/need/to/exist/block_001\n",
        encoding="utf-8",
    )
    specs = load_rotation_registry(animals)
    assert len(specs) == 1
    assert specs[0].animal == "PV_1"

    flat = tmp_path / "jitter.yaml"
    flat.write_text(
        "blocks:\n- animal: PV_2\n  block_path: /tmp/block_002\n  mount_type: rigid\n",
        encoding="utf-8",
    )
    specs2 = load_rotation_registry(flat)
    assert specs2[0].animal == "PV_2"


def test_rotation_tuner_import_does_not_cycle():
    pytest.importorskip("PyQt6")
    from eye_tracking_system_tools.annotation.preprocessing_gui.rotation_tuner import (
        launch_rotation_param_tuner,
        save_dialog_params,
    )

    assert callable(save_dialog_params)
    assert callable(launch_rotation_param_tuner)


def test_dialog_ok_saves_params_label():
    pytest.importorskip("PyQt6")
    from PyQt6 import QtWidgets

    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.spin_max_dialog import (
        SpinMaximizerDialog,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    empty = pd.DataFrame(
        columns=["eye_frame", "center_x", "center_y", "width", "height", "phi"]
    )
    dialog = SpinMaximizerDialog(
        left_df=empty,
        right_df=empty,
        left_frame_fn=lambda _i: None,
        right_frame_fn=lambda _i: None,
        left_nframes=1,
        right_nframes=1,
        left_frame0=0,
        right_frame0=0,
        left_width=16.0,
        right_width=16.0,
    )
    ok = dialog.findChildren(QtWidgets.QPushButton)
    labels = [b.text() for b in ok]
    assert any("Save rotation-correction parameters" in t for t in labels)
    params = dialog.correction_params()
    assert params.apply_jitter in (True, False)
    dialog.close()


def test_run_registry_continues_after_failure(tmp_path: Path):
    missing = tmp_path / "missing_block"
    missing.mkdir()
    (missing / "analysis").mkdir()
    ok_block = tmp_path / "ok_block"
    ok_block.mkdir()
    (ok_block / "analysis").mkdir()
    write_rotation_params(ok_block, _params())
    from eye_tracking_system_tools.analysis.block_registry import BlockSpec

    results = run_registry(
        [
            BlockSpec("A", missing, "001"),
            BlockSpec("B", ok_block, "002"),
        ],
        show_tqdm=False,
    )
    assert results[0].status == STATUS_SKIPPED_NO_PARAMS
    assert results[1].status == STATUS_FAILED


def test_run_registry_prints_block_and_eye_titles(
    tmp_path: Path, monkeypatch, capsys
):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    write_rotation_params(tmp_path, _params())
    _dlc_csv(analysis / "le_df.csv", "left")
    _dlc_csv(analysis / "re_df.csv", "right")
    img = np.full((16, 16, 3), 180, dtype=np.uint8)
    _patch_videos(monkeypatch, img)
    run_registry([tmp_path], overwrite=True, show_tqdm=False)
    out = capsys.readouterr().out
    assert "Working on block 1 out of 1" in out
    assert ", left:" in out
    assert ", right:" in out
    assert "left done:" in out
    assert "right done:" in out
