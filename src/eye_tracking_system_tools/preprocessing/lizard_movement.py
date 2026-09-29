"""Python port of MATLAB ``sleepAnalysis.getLizardMovements`` — writes ``lizMov.mat``."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.stats import kurtosis as scipy_kurtosis

from eye_tracking_system_tools.annotation.block_annotator.oe_streams import (
    list_oe_streams,
)
from eye_tracking_system_tools.preprocessing.accel_calibration import (
    AccelCalibration,
    matlab_default_calibration,
)
from eye_tracking_system_tools.preprocessing.OERecording import OERecording

logger = logging.getLogger(__name__)

TARGET_FS_HZ = 250.0
HP_CUTOFF_HZ = 1.0


@dataclass(frozen=True)
class LizMovParams:
    acc_channels: tuple[int, int, int] = (1, 2, 3)
    envelop_window: int = 15
    kurtosis_noise_threshold: float = 3.25
    event_detection_threshold_std: float = 3.0
    static_detection_threshold_std: float = 2.0
    mov_long_win_ms: float = 1000 * 60 * 30
    mov_long_ol_ms: float = 1000.0
    static_win_ms: float = 1000.0
    t_start_ms: float = 0.0
    win_ms: float = 0.0
    apply_notch: bool = False
    zero_g_bias: np.ndarray | None = None
    sensitivity: np.ndarray | None = None
    noise_buffer_samples: int = 1500
    noise_buffer_fallback: int = 500


@dataclass
class LizMovResult:
    t_mov_ms: np.ndarray
    movAll: np.ndarray
    t_static_ms: np.ndarray
    staticAll: np.ndarray
    dotProductsXYZ: np.ndarray
    angles: np.ndarray
    par_liz_mov: dict[str, Any] = field(default_factory=dict)

    def to_liz_mov_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"t_mov_ms": self.t_mov_ms.ravel(), "movAll": self.movAll.ravel()}
        )


def resolve_liz_mov_output_path(blocksync) -> Path:
    """Return target ``lizMov.mat`` path; create ``analysis/`` subtree if needed."""
    p = Path(blocksync.oe_path) / "analysis"
    p.mkdir(parents=True, exist_ok=True)
    try:
        entries = os.listdir(p)
    except OSError:
        entries = []
    matches = [name for name in entries if blocksync.animal_call in name]
    if not matches:
        sub = p / blocksync.animal_call
    else:
        sub = p / matches[0]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / "lizMov.mat"


def oe_rec_has_accel_channels(oe_rec: OERecording | None) -> bool:
    """True when the recording exposes at least 3 AUX or analog accelerometer channels."""
    if oe_rec is None:
        return False
    accel_files = getattr(oe_rec, "accel_files", None) or []
    if len(accel_files) >= 3:
        return True
    analog = getattr(oe_rec, "analogChannelNumbers", None)
    if analog is None:
        return False
    return len(np.atleast_1d(analog)) >= 3


def _resolve_calibration_vectors(
    params: LizMovParams,
    calibration: AccelCalibration | None,
) -> tuple[np.ndarray, np.ndarray]:
    if calibration is not None:
        return calibration.zero_g_bias_uv.copy(), calibration.sensitivity_uv_per_g.copy()
    if params.zero_g_bias is not None and params.sensitivity is not None:
        return (
            np.asarray(params.zero_g_bias, dtype=np.float64).reshape(3),
            np.asarray(params.sensitivity, dtype=np.float64).reshape(3),
        )
    warnings.warn(
        "No accelerometer calibration provided; using MATLAB getLizardMovements "
        "hardcoded defaults.",
        stacklevel=2,
    )
    fallback = matlab_default_calibration()
    return fallback.zero_g_bias_uv.copy(), fallback.sensitivity_uv_per_g.copy()


def _use_analog_channels(oe_rec: OERecording, acc_channels: tuple[int, ...]) -> bool:
    accel_files = getattr(oe_rec, "accel_files", None) or []
    if len(accel_files) >= 3:
        return False
    analog = getattr(oe_rec, "analogChannelNumbers", None)
    if analog is None:
        return False
    analog_list = [int(c) for c in np.atleast_1d(analog).tolist()]
    if len(analog_list) < 3:
        return False
    return all(c in analog_list for c in acc_channels)


def _fetch_accel_chunk(
    oe_rec: OERecording,
    acc_channels: tuple[int, int, int],
    start_ms: float,
    window_ms: float,
    *,
    read_from_analog: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw data ``[3, nSamples]`` in µV and per-sample ``t_ms`` from chunk start."""
    start_arr = np.atleast_2d(np.array([start_ms], dtype=np.float64))
    channels = list(acc_channels)
    if read_from_analog:
        data, timestamps = oe_rec.get_analog_data(
            channels, start_arr, window_ms, return_timestamps=True
        )
        # shape [n_ch, n_win, n_samples]; µV
        raw = data[:, 0, :]
    else:
        data, timestamps = oe_rec.get_accel_data(
            channels, start_arr, window_ms, return_timestamps=True
        )
        # mV → µV
        raw = data[:, 0, :] * 1000.0
    t_ms = timestamps[0, :] - start_ms
    return raw, t_ms


def _decimate_to_250hz(raw_uv: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Low-pass decimate each axis to 250 Hz."""
    factor = max(1, int(round(fs_hz / TARGET_FS_HZ)))
    if factor <= 1:
        return raw_uv, np.arange(raw_uv.shape[1]) * (1000.0 / fs_hz)
    decimated = np.vstack(
        [signal.decimate(raw_uv[i], factor, ftype="fir", zero_phase=True) for i in range(3)]
    )
    t_ms = np.arange(decimated.shape[1]) * (1000.0 / TARGET_FS_HZ)
    return decimated, t_ms


def _highpass_1hz(x: np.ndarray, fs_hz: float = TARGET_FS_HZ) -> np.ndarray:
    sos = signal.butter(4, HP_CUTOFF_HZ, btype="high", fs=fs_hz, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _peak_envelope(x: np.ndarray, span: int) -> np.ndarray:
    """Approximate MATLAB ``envelope(x, span, 'peak')`` as upper - lower."""
    x = np.asarray(x, dtype=np.float64).ravel()
    size = max(3, int(span) | 1)
    upper = maximum_filter1d(x, size=size, mode="nearest")
    lower = minimum_filter1d(x, size=size, mode="nearest")
    return upper - lower


def _buffer_nodelay(x: np.ndarray, n: int) -> np.ndarray:
    """MATLAB ``buffer(x, n, 0, 'nodelay')`` → shape ``(n, n_columns)``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(n)
    if n <= 0:
        raise ValueError("buffer length must be positive")
    n_cols = len(x) // n
    if n_cols == 0:
        return np.empty((n, 0))
    trimmed = x[: n_cols * n]
    return trimmed.reshape(n_cols, n).T


def _process_chunk(
    raw_uv: np.ndarray,
    t_ms: np.ndarray,
    chunk_start_ms: float,
    params: LizMovParams,
    zero_g: np.ndarray,
    sensitivity: np.ndarray,
    static_samples: int,
) -> tuple[list, list, list, list, list, list]:
    fs_hz = 1000.0 / (t_ms[1] - t_ms[0]) if len(t_ms) > 1 else TARGET_FS_HZ
    filtered, t_filt = _decimate_to_250hz(raw_uv, fs_hz)

    calibrated = (filtered - zero_g.reshape(3, 1)) / sensitivity.reshape(3, 1)
    mag = np.sqrt(np.sum(calibrated**2, axis=0))
    all_axes = _peak_envelope(mag, params.envelop_window)
    all_axes_hp = _highpass_1hz(all_axes)

    buf_hp = _buffer_nodelay(all_axes_hp, params.noise_buffer_samples)
    buf_env = _buffer_nodelay(all_axes, params.noise_buffer_samples)
    if buf_hp.shape[1] == 0:
        buf_hp = _buffer_nodelay(all_axes_hp, params.noise_buffer_fallback)
        buf_env = _buffer_nodelay(all_axes, params.noise_buffer_fallback)
    if buf_hp.shape[1] == 0:
        noise_samples = buf_env
    else:
        k_vals = scipy_kurtosis(buf_hp, axis=0, fisher=False, nan_policy="omit")
        noise_cols = k_vals < params.kurtosis_noise_threshold
        if not np.any(noise_cols):
            warnings.warn(
                "Noise could not be estimated; trying smaller buffer window.",
                stacklevel=2,
            )
            buf_hp = _buffer_nodelay(all_axes_hp, params.noise_buffer_fallback)
            buf_env = _buffer_nodelay(all_axes, params.noise_buffer_fallback)
            if buf_hp.shape[1] > 0:
                k_vals = scipy_kurtosis(buf_hp, axis=0, fisher=False, nan_policy="omit")
                noise_cols = k_vals < params.kurtosis_noise_threshold
            else:
                noise_cols = np.array([], dtype=bool)
        if buf_env.shape[1] == 0 or not np.any(noise_cols):
            noise_samples = buf_env
        else:
            noise_samples = buf_env[:, noise_cols]

    noise_std = float(np.std(noise_samples)) if noise_samples.size else 0.0
    noise_mean = float(np.mean(noise_samples)) if noise_samples.size else 0.0
    th_mov = noise_mean + params.event_detection_threshold_std * noise_std
    th_static = noise_mean + params.static_detection_threshold_std * noise_std

    mov_mask = all_axes > th_mov
    t_mov = chunk_start_ms + t_filt[mov_mask]
    mov_vals = all_axes[mov_mask]

    static_binary = np.abs(all_axes) < th_static
    if static_samples > 1:
        # MATLAB conv(...,'same') keeps len(static_binary); np.convolve does not when
        # kernel is longer than the signal.
        static_sum = (
            uniform_filter1d(
                static_binary.astype(np.float64),
                size=static_samples,
                mode="nearest",
            )
            * static_samples
        )
        static_mask = static_sum >= static_samples - 1e-9
    else:
        static_mask = static_binary

    t_static = chunk_start_ms + t_filt[static_mask]
    static_vals = all_axes[static_mask]
    dots = calibrated[:, static_mask]
    pitch = np.arctan(dots[2, :] / np.sqrt(dots[0, :] ** 2 + dots[1, :] ** 2 + 1e-30))
    roll = np.arctan(dots[0, :] / (dots[1, :] + 1e-30))
    ang = np.vstack([roll, pitch])

    return (
        [t_mov],
        [mov_vals],
        [t_static],
        [static_vals],
        [dots],
        [ang],
    )


def compute_lizard_movement(
    oe_rec: OERecording,
    *,
    params: LizMovParams | None = None,
    calibration: AccelCalibration | None = None,
    t_start_ms: float | None = None,
    win_ms: float | None = None,
    read_from_analog: bool | None = None,
) -> LizMovResult:
    """Run the full getLizardMovements pipeline on one OERecording."""
    params = params or LizMovParams()
    zero_g, sensitivity = _resolve_calibration_vectors(params, calibration)

    if read_from_analog is None:
        read_from_analog = _use_analog_channels(oe_rec, params.acc_channels)
    if not read_from_analog and len(getattr(oe_rec, "accel_files", []) or []) < 3:
        streams = list_oe_streams(oe_rec)
        labels = [s.label for s in streams]
        raise ValueError(
            "Need 3 AUX accelerometer channels or 3 analog channels. "
            f"Available OE streams: {labels or 'none'}"
        )

    t_start = float(t_start_ms if t_start_ms is not None else params.t_start_ms)
    duration = float(oe_rec.recordingDuration_ms)
    win = float(win_ms if win_ms is not None else params.win_ms)
    if win <= 0:
        win = duration - t_start
    end_time = min(t_start + win, duration)

    chunk_starts = np.arange(
        t_start,
        end_time,
        params.mov_long_win_ms - params.mov_long_ol_ms,
    )
    if len(chunk_starts) == 0:
        chunk_starts = np.array([t_start])

    static_samples = int(round(TARGET_FS_HZ * params.static_win_ms / 1000.0))

    all_t_mov: list[np.ndarray] = []
    all_mov: list[np.ndarray] = []
    all_t_static: list[np.ndarray] = []
    all_static: list[np.ndarray] = []
    all_dots: list[np.ndarray] = []
    all_angles: list[np.ndarray] = []

    for i, start in enumerate(chunk_starts):
        chunk_win = min(params.mov_long_win_ms, end_time - start)
        if chunk_win <= 0:
            continue
        raw, t_ms = _fetch_accel_chunk(
            oe_rec,
            params.acc_channels,
            float(start),
            float(chunk_win),
            read_from_analog=read_from_analog,
        )
        parts = _process_chunk(
            raw,
            t_ms,
            float(start),
            params,
            zero_g,
            sensitivity,
            static_samples,
        )
        all_t_mov.extend(parts[0])
        all_mov.extend(parts[1])
        all_t_static.extend(parts[2])
        all_static.extend(parts[3])
        all_dots.extend(parts[4])
        all_angles.extend(parts[5])
        logger.info("Accelerometer chunk %d/%d", i + 1, len(chunk_starts))

    def _cat(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.array([])
        return np.concatenate([p.ravel() for p in parts])

    t_mov_ms = _cat(all_t_mov)
    mov_all = _cat(all_mov)
    t_static_ms = _cat(all_t_static)
    static_all = _cat(all_static)
    if all_dots:
        dot_products = np.hstack(all_dots)
    else:
        dot_products = np.empty((3, 0))
    if all_angles:
        angles = np.hstack(all_angles)
    else:
        angles = np.empty((2, 0))

    par = {
        "accCh": list(params.acc_channels),
        "envelopWindow": params.envelop_window,
        "kurtosisNoiseThreshold": params.kurtosis_noise_threshold,
        "eventDetectionThresholdStd": params.event_detection_threshold_std,
        "staticDetectionThresholdStd": params.static_detection_threshold_std,
        "movLongWin": params.mov_long_win_ms,
        "movLongOL": params.mov_long_ol_ms,
        "zeroGBias": zero_g.tolist(),
        "sensitivity": sensitivity.tolist(),
        "staticWin": params.static_win_ms,
        "tStart": t_start,
        "win": win,
        "applyNotch": int(params.apply_notch),
        "readFromAnalogCh": int(read_from_analog),
    }

    return LizMovResult(
        t_mov_ms=t_mov_ms,
        movAll=mov_all,
        t_static_ms=t_static_ms,
        staticAll=static_all,
        dotProductsXYZ=dot_products,
        angles=angles,
        par_liz_mov=par,
    )


def write_liz_mov_mat(path: Path, result: LizMovResult) -> None:
    """Write MATLAB v7.3 HDF5 ``lizMov.mat`` readable by ``block_get_lizard_movement``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def colvec(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64).reshape(-1, 1)

    with h5py.File(path, "w") as f:
        f.create_dataset("t_mov_ms", data=colvec(result.t_mov_ms))
        f.create_dataset("movAll", data=colvec(result.movAll))
        f.create_dataset("t_static_ms", data=colvec(result.t_static_ms))
        f.create_dataset("staticAll", data=colvec(result.staticAll))
        f.create_dataset("dotProductsXYZ", data=np.asarray(result.dotProductsXYZ, dtype=np.float64))
        f.create_dataset("angles", data=np.asarray(result.angles, dtype=np.float64))
        par_grp = f.create_group("parLizMov")
        for key, val in result.par_liz_mov.items():
            if isinstance(val, (list, tuple)):
                par_grp.create_dataset(key, data=np.array(val, dtype=np.float64))
            elif isinstance(val, (int, float)):
                par_grp.create_dataset(key, data=np.array([[float(val)]]))
            else:
                par_grp.attrs[key] = str(val)


def compute_and_save_lizard_movement(
    blocksync,
    *,
    overwrite: bool = False,
    params: LizMovParams | None = None,
    calibration: AccelCalibration | None = None,
) -> Path:
    """Compute lizMov for a BlockSync instance and write ``lizMov.mat``."""
    if blocksync.oe_rec is None:
        raise RuntimeError("BlockSync.oe_rec is not initialized.")
    out_path = resolve_liz_mov_output_path(blocksync)
    if out_path.is_file() and not overwrite:
        logger.info("lizMov.mat already exists at %s (overwrite=False)", out_path)
        return out_path
    result = compute_lizard_movement(
        blocksync.oe_rec,
        params=params,
        calibration=calibration,
    )
    write_liz_mov_mat(out_path, result)
    blocksync.liz_mov_df = result.to_liz_mov_df()
    logger.info(
        "Wrote lizMov.mat (%d movement samples) to %s",
        len(result.t_mov_ms),
        out_path,
    )
    return out_path
