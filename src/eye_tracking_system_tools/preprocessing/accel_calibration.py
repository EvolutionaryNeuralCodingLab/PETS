"""Load per-headstage accelerometer calibration from MATLAB ``calibration_results.mat``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# MATLAB getLizardMovements hardcoded fallbacks (µV and µV/g).
MATLAB_DEFAULT_ZERO_G_BIAS_UV = np.array(
    [472971.375013507, 494947.777368294, 493554.966062353], dtype=np.float64
)
MATLAB_DEFAULT_SENSITIVITY_UV_PER_G = np.array(
    [-34735.1293215227, 34954.7241622606, 34589.4195371905], dtype=np.float64
)

_ZERO_G_FIELDS = ("zeroGbais", "zeroGBias", "zero_g_bias")
_SENS_FIELDS = ("sensetivity", "sensitivity", "sensitivity_uv_per_g")


@dataclass(frozen=True)
class AccelCalibration:
    headstage_id: str
    zero_g_bias_uv: np.ndarray
    sensitivity_uv_per_g: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "zero_g_bias_uv", np.asarray(self.zero_g_bias_uv, dtype=np.float64).reshape(3)
        )
        object.__setattr__(
            self,
            "sensitivity_uv_per_g",
            np.asarray(self.sensitivity_uv_per_g, dtype=np.float64).reshape(3),
        )


def _read_matlab_struct_group(grp: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    zero_g = None
    sens = None
    for key in grp.keys():
        kl = key.lower()
        arr = np.array(grp[key]).squeeze().astype(np.float64)
        if kl in {f.lower() for f in _ZERO_G_FIELDS}:
            zero_g = arr
        elif kl in {f.lower() for f in _SENS_FIELDS}:
            sens = arr
    if zero_g is None or sens is None:
        raise KeyError(
            f"Calibration struct missing zeroG/sensitivity fields; found {list(grp.keys())}"
        )
    return zero_g.reshape(3), sens.reshape(3)


def list_headstages(calibration_mat: Path) -> list[str]:
    """Return sorted headstage IDs stored under ``cali_result``."""
    path = Path(calibration_mat)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with h5py.File(path, "r") as f:
        if "cali_result" not in f:
            raise KeyError(f"'cali_result' not found in {path}")
        return sorted(f["cali_result"].keys())


def load_accel_calibration(
    calibration_mat: Path,
    headstage_id: str,
) -> AccelCalibration:
    """Load zero-G bias and sensitivity for one headstage from a v7.3 .mat file."""
    path = Path(calibration_mat)
    with h5py.File(path, "r") as f:
        cr = f["cali_result"]
        if headstage_id not in cr:
            available = sorted(cr.keys())
            raise KeyError(
                f"Headstage {headstage_id!r} not in {path.name}; "
                f"available: {available}"
            )
        ref = cr[headstage_id]
        if isinstance(ref, h5py.Dataset):
            grp = f[ref[0, 0]] if ref.ndim == 2 else f[ref[0]]
        else:
            grp = ref
        zero_g, sens = _read_matlab_struct_group(grp)
    return AccelCalibration(headstage_id, zero_g, sens)


def matlab_default_calibration() -> AccelCalibration:
    """Fallback matching hardcoded MATLAB ``getLizardMovements`` parameters."""
    return AccelCalibration(
        "MATLAB_default",
        MATLAB_DEFAULT_ZERO_G_BIAS_UV.copy(),
        MATLAB_DEFAULT_SENSITIVITY_UV_PER_G.copy(),
    )
