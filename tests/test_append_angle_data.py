"""Idempotent Kerr angle append + passive column/count checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.calculate_kerr_angles import (
    KerrAnglePreview,
    append_angle_data,
    check_appended_angles,
    preview_kerr_angles,
)


def _eye(n: int = 5, *, with_angles: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "OE_timestamp": np.arange(n, dtype=np.int64) * 100,
            "phi": np.linspace(0.1, 0.5, n),
            "major_ax": np.ones(n),
            "minor_ax": np.ones(n) * 0.8,
        }
    )
    if with_angles:
        df["k_phi"] = np.linspace(1.0, 2.0, n)
        df["k_theta"] = np.linspace(-1.0, 1.0, n)
    return df


def _angles(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OE_timestamp": np.arange(n, dtype=np.int64) * 100,
            "phi": np.linspace(10.0, 20.0, n),
            "theta": np.linspace(-5.0, 5.0, n),
            "eye_frame": np.arange(n),
        }
    )


def _preview_eye_df(n: int = 40) -> pd.DataFrame:
    """Synthetic pupil ellipses clustered near a Kerr ref for preview tests."""
    rng = np.random.default_rng(0)
    cx = 100.0 + rng.normal(0.0, 3.0, n)
    cy = 80.0 + rng.normal(0.0, 2.5, n)
    return pd.DataFrame(
        {
            "eye_frame": np.arange(n),
            "OE_timestamp": np.arange(n, dtype=np.int64) * 50,
            "ms_axis": np.arange(n, dtype=float),
            "center_x": cx,
            "center_y": cy,
            "width": np.full(n, 12.0),
            "height": np.full(n, 10.0),
            "phi": np.zeros(n),
        }
    )


def test_preview_kerr_angles_returns_finite_samples():
    preview = preview_kerr_angles(_preview_eye_df(), ref_x=100.0, ref_y=80.0)
    assert isinstance(preview, KerrAnglePreview)
    assert preview.n_input == 40
    assert preview.n_finite > 0
    assert np.isfinite(preview.f_z)
    assert preview.phi.shape == (40,)
    assert preview.theta.shape == (40,)
    assert np.count_nonzero(np.isfinite(preview.phi) & np.isfinite(preview.theta)) == (
        preview.n_finite
    )


def test_preview_kerr_angles_does_not_mutate_input():
    df = _preview_eye_df()
    before = df.copy()
    preview_kerr_angles(df, 100.0, 80.0)
    pd.testing.assert_frame_equal(df, before)


def test_preview_kerr_angles_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        preview_kerr_angles(pd.DataFrame(), 0.0, 0.0)


def test_append_adds_canonical_columns():
    out = append_angle_data(_eye(), _angles())
    assert {"k_phi", "k_theta"}.issubset(out.columns)
    assert "k_phi_x" not in out.columns
    assert out["k_phi"].notna().sum() == 5


def test_append_idempotent_when_angles_already_present():
    """Re-append onto hydrated frames must not create pandas merge suffixes."""
    once = append_angle_data(_eye(), _angles())
    twice = append_angle_data(once, _angles())
    assert list(c for c in twice.columns if c.startswith("k_phi")) == ["k_phi"]
    assert list(c for c in twice.columns if c.startswith("k_theta")) == ["k_theta"]
    check = check_appended_angles(twice, _angles(), side="left")
    assert check.columns_ok
    assert check.ok
    assert check.n_k_phi == 5


def test_append_strips_merge_suffix_leftovers():
    eye = _eye(with_angles=True)
    eye = eye.rename(columns={"k_phi": "k_phi_x", "k_theta": "k_theta_x"})
    eye["k_phi_y"] = 99.0
    eye["k_theta_y"] = 99.0
    out = append_angle_data(eye, _angles())
    assert "k_phi" in out.columns and "k_theta" in out.columns
    assert not any(c.endswith(("_x", "_y")) and c.startswith("k_") for c in out.columns)
    # New values from angle CSV, not the leftover 99s.
    assert out["k_phi"].iloc[0] == pytest.approx(10.0)


def test_append_preserves_caller_frame():
    """In-memory filtered frames must not be mutated in place."""
    eye = _eye()
    original_cols = list(eye.columns)
    _ = append_angle_data(eye, _angles())
    assert list(eye.columns) == original_cols
    assert "k_phi" not in eye.columns


def test_check_flags_missing_columns():
    eye = _eye()
    check = check_appended_angles(eye, _angles(), side="left")
    assert not check.columns_ok
    assert not check.ok
    assert any("missing" in m for m in check.messages)


def test_check_flags_timestamp_misalignment():
    eye = _eye()
    eye["OE_timestamp"] = eye["OE_timestamp"] + 999_999
    out = append_angle_data(eye, _angles())
    check = check_appended_angles(out, _angles(), side="right")
    assert check.columns_ok
    assert not check.ok
    assert check.n_k_phi == 0
    assert any("all-NaN" in m or "misalignment" in m for m in check.messages)
