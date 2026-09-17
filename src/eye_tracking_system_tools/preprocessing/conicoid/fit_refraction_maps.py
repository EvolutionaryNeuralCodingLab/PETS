"""Stub: fit Dierkes ``Reye`` / ``Rpupil`` polynomials from ray-traced pairs.

Generating the training set needs LeGrand (or equivalent) cornea ray-traced
images that are **not** in this repository. The Blender ellipse CSV used for
synthetic gaze tests is a sphere without a cornea, so it cannot train these
maps.

Expected training columns (once a generator exists)
---------------------------------------------------
``Ex_hat, Ey_hat, Ez_hat`` — geometric eyeball centre
``nx, ny, nz`` — geometric gaze
``Ex_true, Ey_true, Ez_true`` — ground-truth centre after refraction
``nx_true, ny_true, nz_true, r_true`` — ground-truth pupil

Until then this module only writes identity maps (useful for wiring tests)
or least-squares coefficients if the caller already has a table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .refraction import (
    N_REF,
    RefractionMaps,
    identity_refraction_maps,
    refraction_features,
    save_refraction_maps,
)


def fit_maps_from_table(
    table: pd.DataFrame,
    *,
    n_ref: float = N_REF,
) -> RefractionMaps:
    """Least-squares ``Reye`` / ``Rpupil`` from a labelled table.

    Requires ``Ex_hat``… and ``Ex_true``… columns. Raises if they are missing
    so callers do not silently train on the wrong schema.
    """
    needed = (
        "Ex_hat",
        "Ey_hat",
        "Ez_hat",
        "nx",
        "ny",
        "nz",
        "Ex_true",
        "Ey_true",
        "Ez_true",
    )
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise ValueError(
            "Refraction-map training needs LeGrand ray-traced pairs; "
            f"missing columns {missing}. See this module's docstring."
        )
    feats = np.stack(
        [
            refraction_features(
                [row.Ex_hat, row.Ey_hat, row.Ez_hat],
                [row.nx, row.ny, row.nz],
                n_ref=n_ref,
            )
            for row in table.itertuples(index=False)
        ]
    )
    dE = table[["Ex_true", "Ey_true", "Ez_true"]].to_numpy(dtype=float) - table[
        ["Ex_hat", "Ey_hat", "Ez_hat"]
    ].to_numpy(dtype=float)
    reye_coef, *_ = np.linalg.lstsq(feats, dE, rcond=None)
    rpupil_n = None
    rpupil_r = None
    if {"nx_true", "ny_true", "nz_true"}.issubset(table.columns):
        dN = table[["nx_true", "ny_true", "nz_true"]].to_numpy(dtype=float) - table[
            ["nx", "ny", "nz"]
        ].to_numpy(dtype=float)
        rpupil_n, *_ = np.linalg.lstsq(feats, dN, rcond=None)
        rpupil_n = rpupil_n.T
    if "r_true" in table.columns and "r_hat" in table.columns:
        dr = table["r_true"].to_numpy(dtype=float) - table["r_hat"].to_numpy(
            dtype=float
        )
        rpupil_r, *_ = np.linalg.lstsq(feats, dr, rcond=None)
        rpupil_r = rpupil_r.reshape(-1)
    return RefractionMaps(
        reye_coef=reye_coef.T,
        rpupil_n_coef=rpupil_n,
        rpupil_r_coef=rpupil_r,
        n_ref=n_ref,
        source="lstsq",
    )


def write_identity_maps(path: Path | str) -> Path:
    """Write an all-zero npz so the GUI refraction checkbox can be exercised."""
    return save_refraction_maps(path, identity_refraction_maps())
