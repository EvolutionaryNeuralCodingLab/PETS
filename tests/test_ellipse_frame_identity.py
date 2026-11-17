"""Tests for DLC frame identity and sync-free alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eye_tracking_system_tools.preprocessing.sync_free_eye_io import (
    align_ellipse_df_to_video_frames,
)


def test_align_ellipse_df_maps_first_fit_to_frame_zero():
    """After header=1 + iloc[1:], fit index k corresponds to video frame k."""
    n_data = 10  # body rows after trim
    n_ell = n_data - 2
    ellipse_df = pd.DataFrame(
        {
            "center_x": np.arange(n_ell, dtype=float),
            "center_y": np.arange(n_ell, dtype=float) + 0.5,
            "width": np.full(n_ell, 40.0),
            "height": np.full(n_ell, 20.0),
            "phi": np.linspace(0.1, 0.5, n_ell),
            "eye_frame": np.arange(n_ell, dtype=int),
        }
    )
    full = align_ellipse_df_to_video_frames(
        ellipse_df, n_video_frames=n_data, n_data_rows=n_data
    )
    assert len(full) == n_data
    assert full.loc[0, "center_x"] == 0.0
    assert full.loc[1, "center_x"] == 1.0
    assert np.isnan(full.loc[n_data - 2, "center_x"])
    assert np.isnan(full.loc[n_data - 1, "center_x"])
    assert list(full["eye_frame"]) == list(range(n_data))
