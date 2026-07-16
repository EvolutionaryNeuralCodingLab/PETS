"""Tests for Python lizMov (accelerometer) pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from eye_tracking_system_tools.preprocessing.accel_calibration import (
    AccelCalibration,
    list_headstages,
    load_accel_calibration,
    matlab_default_calibration,
)
from eye_tracking_system_tools.preprocessing.lizard_movement import (
    LizMovResult,
    _buffer_nodelay,
    _peak_envelope,
    _process_chunk,
    compute_lizard_movement,
    oe_rec_has_accel_channels,
    resolve_liz_mov_output_path,
    write_liz_mov_mat,
)

CALIB_EXAMPLE = Path(
    "/Volumes/Data-1/accelerometer_calibrations/headtagse_cali_recs/calibration_results.mat"
)


@pytest.fixture
def calib_example_available() -> bool:
    return CALIB_EXAMPLE.is_file()


def test_list_headstages_example(calib_example_available):
    if not calib_example_available:
        pytest.skip("Example calibration_results.mat not mounted")
    hs = list_headstages(CALIB_EXAMPLE)
    assert "HS3" in hs
    assert len(hs) >= 5


def test_load_accel_calibration_example(calib_example_available):
    if not calib_example_available:
        pytest.skip("Example calibration_results.mat not mounted")
    cal = load_accel_calibration(CALIB_EXAMPLE, "HS3")
    assert cal.headstage_id == "HS3"
    assert cal.zero_g_bias_uv.shape == (3,)
    assert cal.sensitivity_uv_per_g.shape == (3,)
    assert np.all(np.isfinite(cal.zero_g_bias_uv))


def test_matlab_default_calibration_shapes():
    cal = matlab_default_calibration()
    assert cal.zero_g_bias_uv.shape == (3,)
    assert cal.sensitivity_uv_per_g.shape == (3,)


def test_buffer_nodelay():
    x = np.arange(20, dtype=float)
    buf = _buffer_nodelay(x, 5)
    assert buf.shape == (5, 4)
    np.testing.assert_array_equal(buf[:, 0], [0, 1, 2, 3, 4])


def test_peak_envelope_positive_amplitude():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    env = _peak_envelope(x, 15)
    assert env.shape == x.shape
    assert np.all(env >= 0)


def test_process_chunk_produces_movement_events():
    n = 5000
    t_ms = np.arange(n) * (1000.0 / 30000.0)
    raw = np.zeros((3, n))
    raw[0, 1000:1100] = 500000.0
    params = __import__(
        "eye_tracking_system_tools.preprocessing.lizard_movement", fromlist=["LizMovParams"]
    ).LizMovParams()
    cal = matlab_default_calibration()
    parts = _process_chunk(
        raw,
        t_ms,
        0.0,
        params,
        cal.zero_g_bias_uv,
        cal.sensitivity_uv_per_g,
        static_samples=250,
    )
    t_mov = np.concatenate(parts[0])
    assert len(t_mov) > 0


def test_resolve_liz_mov_output_path_creates_analysis(tmp_path):
    from types import SimpleNamespace

    oe_path = tmp_path / "oe_files" / "exp" / "Record Node"
    oe_path.mkdir(parents=True)
    bs = SimpleNamespace(oe_path=oe_path, animal_call="PV_228")
    out = resolve_liz_mov_output_path(bs)
    assert out.parent.name == "PV_228"
    assert (oe_path / "analysis" / "PV_228").is_dir()
    assert out == oe_path / "analysis" / "PV_228" / "lizMov.mat"


def test_oe_rec_has_accel_channels():
    oe_aux = MagicMock()
    oe_aux.accel_files = ["a", "b", "c"]
    oe_aux.analogChannelNumbers = np.array([])
    assert oe_rec_has_accel_channels(oe_aux)

    oe_analog = MagicMock()
    oe_analog.accel_files = []
    oe_analog.analogChannelNumbers = np.array([1, 2, 3])
    assert oe_rec_has_accel_channels(oe_analog)

    oe_none = MagicMock()
    oe_none.accel_files = []
    oe_none.analogChannelNumbers = np.array([])
    assert not oe_rec_has_accel_channels(oe_none)


def test_write_liz_mov_mat_roundtrip(tmp_path):
    result = LizMovResult(
        t_mov_ms=np.array([1.0, 2.0]),
        movAll=np.array([0.1, 0.2]),
        t_static_ms=np.array([3.0]),
        staticAll=np.array([0.05]),
        dotProductsXYZ=np.array([[1.0], [2.0], [3.0]]),
        angles=np.array([[0.1], [0.2]]),
        par_liz_mov={"staticWin": 1000.0},
    )
    path = tmp_path / "lizMov.mat"
    write_liz_mov_mat(path, result)
    with h5py.File(path, "r") as f:
        t_mov = f["t_mov_ms"][:].squeeze()
        mov = f["movAll"][:].squeeze()
    np.testing.assert_allclose(t_mov, [1.0, 2.0])
    np.testing.assert_allclose(mov, [0.1, 0.2])


def test_compute_lizard_movement_mock_oe_rec():
    n = 30000
    fs = 30000.0
    sample_ms = 1000.0 / fs
    raw = np.random.default_rng(0).normal(0, 1000, (3, n))
    oe = MagicMock()
    oe.recordingDuration_ms = n * sample_ms
    oe.accel_files = ["a.continuous", "b.continuous", "c.continuous"]
    oe.analogChannelNumbers = []
    timestamps = np.tile(np.arange(n) * sample_ms, (1, 1))

    def fake_accel(channels, start_arr, window_ms, return_timestamps=True, **kwargs):
        win = int(round(window_ms / sample_ms))
        data = raw[:, np.newaxis, :win]
        ts = timestamps[:, :win]
        return data, ts

    oe.get_accel_data = fake_accel
    cal = matlab_default_calibration()
    result = compute_lizard_movement(oe, calibration=cal, win_ms=5000.0)
    assert isinstance(result.t_mov_ms, np.ndarray)
    assert isinstance(result.movAll, np.ndarray)


def test_parity_vs_matlab_lizmov(behavior_sample_block_path, calib_example_available):
    """Regression against existing MATLAB lizMov.mat on PV_126 block_006."""
    if not calib_example_available:
        pytest.skip("Calibration file not available")
    from eye_tracking_system_tools.preprocessing.BlockSync_class import BlockSync
    from eye_tracking_system_tools.preprocessing.lizard_movement import (
        resolve_liz_mov_output_path,
    )

    mat_path = None
    for p in behavior_sample_block_path.rglob("lizMov.mat"):
        if "analysis" in p.parts:
            mat_path = p
            break
    if mat_path is None or not mat_path.is_file():
        pytest.skip("Reference lizMov.mat not on behavior sample block")

    block = BlockSync(
        animal_call="PV_126",
        experiment_date="2024_07_18",
        block_num="006",
        path_to_animal_folder=str(behavior_sample_block_path.parents[2]),
    )
    with h5py.File(mat_path, "r") as ref:
        ref_t = ref["t_mov_ms"][:].squeeze()
        ref_mov = ref["movAll"][:].squeeze()

    # Try headstages until correlation is acceptable (user selects manually in GUI).
    best_corr = -1.0
    best_hs = None
    for hs in list_headstages(CALIB_EXAMPLE):
        cal = load_accel_calibration(CALIB_EXAMPLE, hs)
        py = compute_lizard_movement(block.oe_rec, calibration=cal)
        if len(py.t_mov_ms) < 10 or len(ref_t) < 10:
            continue
        n = min(len(py.t_mov_ms), len(ref_t), 5000)
        corr = np.corrcoef(py.movAll[:n], ref_mov[:n])[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_hs = hs

    if best_hs is None:
        pytest.skip("Could not compare movement traces (insufficient data)")

    # Parity threshold — envelope approximation may not be bit-exact.
    assert best_corr > 0.85, (
        f"Best headstage {best_hs} correlation {best_corr:.3f} below threshold"
    )


def test_behavior_tab_has_compute_button(qapp_session, tmp_path):
    from eye_tracking_system_tools.annotation.preprocessing_gui.config_io import (
        ensure_config_template,
        load_config,
    )
    from eye_tracking_system_tools.annotation.preprocessing_gui.models import GuiState
    from eye_tracking_system_tools.annotation.preprocessing_gui.tabs.behavior_tab import (
        BehaviorTab,
    )

    ensure_config_template(tmp_path)
    config = load_config(None, tmp_path)
    state = GuiState(output_folder=tmp_path)
    tab = BehaviorTab(state, config)
    assert hasattr(tab, "_btn_compute")
    assert hasattr(tab, "_calib_path_edit")
    assert hasattr(tab, "_headstage_combo")
