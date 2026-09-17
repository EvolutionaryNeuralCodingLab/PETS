"""Optional Dierkes 2019 refraction maps ``Reye`` / ``Rpupil``.

Geometric unprojection ignores the cornea. Dierkes et al. correct the
eyeball centre and pupil with polynomial maps trained on ray-traced
LeGrand images. PETS does **not** ship those images; this module applies
coefficients from ``analysis/conicoid_refraction.npz`` when present.

The aqueous refractive index used by Dierkes is ``n_ref = 1.3375``.
Maps are **off by default** — call :func:`apply_refraction_to_fit` explicitly.

NPZ keys
--------
``reye_coef`` : (3, n_features) correction added to ``E_hat``
``rpupil_n_coef`` : (3, n_features) correction added to gaze ``n``
``rpupil_r_coef`` : (n_features,) correction added to pupil radius
``feature_names`` : optional sequence of monomial names (see
:func:`refraction_features`)
``n_ref`` : float, default 1.3375

Identity maps (all-zero coefficients) leave the geometric solve unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .eye_model import EyeModelFit, Sphere, snap_pupils_to_sphere
from .geometry import Circle3D

N_REF = 1.3375
REFRACTION_NPZ_NAME = "conicoid_refraction.npz"

_DEFAULT_FEATURES = ("1", "Ex", "Ey", "Ez", "nx", "ny", "nz", "n_ref")


@dataclass(frozen=True)
class RefractionMaps:
    """Polynomial refraction maps. Missing pupil maps are treated as identity."""

    reye_coef: np.ndarray
    rpupil_n_coef: np.ndarray | None = None
    rpupil_r_coef: np.ndarray | None = None
    feature_names: tuple[str, ...] = _DEFAULT_FEATURES
    n_ref: float = N_REF
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reye_coef", np.asarray(self.reye_coef, dtype=float)
        )
        if self.rpupil_n_coef is not None:
            object.__setattr__(
                self,
                "rpupil_n_coef",
                np.asarray(self.rpupil_n_coef, dtype=float),
            )
        if self.rpupil_r_coef is not None:
            object.__setattr__(
                self,
                "rpupil_r_coef",
                np.asarray(self.rpupil_r_coef, dtype=float).reshape(-1),
            )


def refraction_npz_path(block_path: Path | str) -> Path:
    path = Path(block_path)
    analysis = path if path.name == "analysis" else path / "analysis"
    return analysis / REFRACTION_NPZ_NAME


def refraction_features(
    eyeball: np.ndarray,
    normal: np.ndarray,
    *,
    n_ref: float = N_REF,
    names: tuple[str, ...] = _DEFAULT_FEATURES,
) -> np.ndarray:
    """Evaluate the monomial feature vector used by the polynomial maps."""
    e = np.asarray(eyeball, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)
    lookup = {
        "1": 1.0,
        "Ex": float(e[0]),
        "Ey": float(e[1]),
        "Ez": float(e[2]),
        "nx": float(n[0]),
        "ny": float(n[1]),
        "nz": float(n[2]),
        "n_ref": float(n_ref),
    }
    out = np.empty(len(names), dtype=float)
    for i, name in enumerate(names):
        if name not in lookup:
            raise KeyError(f"Unknown refraction feature {name!r}")
        out[i] = lookup[name]
    return out


def load_refraction_maps(path: Path | str) -> RefractionMaps:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    names = tuple(str(n) for n in data["feature_names"]) if "feature_names" in data else _DEFAULT_FEATURES
    n_ref = float(data["n_ref"]) if "n_ref" in data else N_REF
    return RefractionMaps(
        reye_coef=data["reye_coef"],
        rpupil_n_coef=data["rpupil_n_coef"] if "rpupil_n_coef" in data else None,
        rpupil_r_coef=data["rpupil_r_coef"] if "rpupil_r_coef" in data else None,
        feature_names=names,
        n_ref=n_ref,
        source=str(path),
    )


def save_refraction_maps(path: Path | str, maps: RefractionMaps) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "reye_coef": np.asarray(maps.reye_coef, dtype=float),
        "feature_names": np.asarray(maps.feature_names, dtype=object),
        "n_ref": np.asarray(maps.n_ref, dtype=float),
    }
    if maps.rpupil_n_coef is not None:
        payload["rpupil_n_coef"] = np.asarray(maps.rpupil_n_coef, dtype=float)
    if maps.rpupil_r_coef is not None:
        payload["rpupil_r_coef"] = np.asarray(maps.rpupil_r_coef, dtype=float)
    np.savez(path, **payload)
    return path


def identity_refraction_maps(*, n_features: int | None = None) -> RefractionMaps:
    n = int(n_features) if n_features is not None else len(_DEFAULT_FEATURES)
    return RefractionMaps(
        reye_coef=np.zeros((3, n)),
        rpupil_n_coef=np.zeros((3, n)),
        rpupil_r_coef=np.zeros(n),
        feature_names=_DEFAULT_FEATURES[:n] if n <= len(_DEFAULT_FEATURES) else _DEFAULT_FEATURES,
        n_ref=N_REF,
        source="identity",
    )


def apply_reye(
    eyeball: np.ndarray,
    normal: np.ndarray,
    maps: RefractionMaps,
    *,
    n_ref: float | None = None,
) -> np.ndarray:
    """``E_corr = E_hat + Reye(E_hat, n)``."""
    n_use = maps.n_ref if n_ref is None else float(n_ref)
    feat = refraction_features(
        eyeball, normal, n_ref=n_use, names=maps.feature_names
    )
    coef = np.asarray(maps.reye_coef, dtype=float)
    if coef.ndim != 2 or coef.shape[0] != 3 or coef.shape[1] != feat.size:
        raise ValueError(
            f"reye_coef shape {coef.shape} does not match features {feat.size}"
        )
    return np.asarray(eyeball, dtype=float).reshape(3) + coef @ feat


def apply_rpupil(
    circle: Circle3D,
    eyeball: np.ndarray,
    maps: RefractionMaps,
    *,
    n_ref: float | None = None,
) -> Circle3D:
    """Correct gaze / radius after ``Reye``. Identity if pupil maps are missing."""
    n_use = maps.n_ref if n_ref is None else float(n_ref)
    n = np.asarray(circle.normal, dtype=float).reshape(3)
    feat = refraction_features(
        eyeball, n, n_ref=n_use, names=maps.feature_names
    )
    if maps.rpupil_n_coef is not None:
        n = n + np.asarray(maps.rpupil_n_coef, dtype=float) @ feat
        nrm = float(np.linalg.norm(n))
        if nrm > 0:
            n = n / nrm
    radius = float(circle.radius)
    if maps.rpupil_r_coef is not None:
        radius = radius + float(np.dot(maps.rpupil_r_coef, feat))
        radius = max(radius, 1e-6)
    return Circle3D(
        centre=np.asarray(circle.centre, dtype=float),
        normal=n,
        radius=radius,
    )


def apply_refraction_to_fit(
    fit: EyeModelFit,
    maps: RefractionMaps,
    *,
    n_ref: float | None = None,
) -> EyeModelFit:
    """Correct sphere centre then optionally pupil gaze/radius; re-snap.

    A representative inlier normal is used as the ``n`` argument of ``Reye``
    (Dierkes evaluate the map per recording, not per frame, for the eyeball).
    """
    n_use = maps.n_ref if n_ref is None else float(n_ref)
    ref_n = None
    for obs in fit.pupils:
        if obs.circle is not None and obs.inlier:
            ref_n = np.asarray(obs.circle.normal, dtype=float)
            break
    if ref_n is None:
        ref_n = np.array([0.0, 0.0, -1.0])
    e_corr = apply_reye(fit.sphere.centre, ref_n, maps, n_ref=n_use)
    fit.sphere = Sphere(centre=e_corr, radius=fit.sphere.radius)
    if maps.rpupil_n_coef is not None or maps.rpupil_r_coef is not None:
        for obs in fit.pupils:
            if obs.circle is None:
                continue
            obs.circle = apply_rpupil(obs.circle, e_corr, maps, n_ref=n_use)
    return snap_pupils_to_sphere(fit)


def try_load_refraction_maps(block_path: Path | str) -> RefractionMaps | None:
    path = refraction_npz_path(block_path)
    if not path.is_file():
        return None
    return load_refraction_maps(path)
