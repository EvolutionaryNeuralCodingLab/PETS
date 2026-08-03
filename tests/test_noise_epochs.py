"""Tests for noise-epoch catalogs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eye_tracking_system_tools.preprocessing.noise_epochs import (
    CATEGORY_LED_BLINK,
    CATEGORY_PUPIL_PERIMETER,
    append_epochs,
    apply_noise_epochs_to_block_csvs,
    epochs_to_frame_set,
    frames_to_epochs,
    list_categories,
    mask_eye_df_by_epochs,
    read_noise_epochs,
)


def test_frames_to_epochs_compresses_runs() -> None:
    ep = frames_to_epochs([1, 2, 3, 10, 11, 20])
    assert list(ep["start_frame"]) == [1, 10, 20]
    assert list(ep["end_frame"]) == [3, 11, 20]


def test_epochs_to_frame_set_filters_category() -> None:
    ep = pd.DataFrame(
        {
            "start_frame": [1, 10],
            "end_frame": [2, 10],
            "category": [CATEGORY_LED_BLINK, CATEGORY_PUPIL_PERIMETER],
        }
    )
    assert epochs_to_frame_set(ep, categories=[CATEGORY_LED_BLINK]) == {1, 2}
    assert epochs_to_frame_set(ep) == {1, 2, 10}


def test_append_and_read_roundtrip(tmp_path: Path) -> None:
    block = tmp_path / "block"
    (block / "analysis").mkdir(parents=True)
    path, n_frames, n_epochs = append_epochs(
        block,
        "left",
        category=CATEGORY_LED_BLINK,
        frames=[5, 6, 7, 20],
        replace_category=True,
    )
    assert path.name == "noise_epochs_left.csv"
    assert n_frames == 4
    assert n_epochs == 2
    df = read_noise_epochs(block, "left")
    assert list(df["category"]) == [CATEGORY_LED_BLINK, CATEGORY_LED_BLINK]
    # Replace category
    append_epochs(
        block,
        "left",
        category=CATEGORY_LED_BLINK,
        frames=[1],
        replace_category=True,
    )
    df2 = read_noise_epochs(block, "left")
    assert len(df2) == 1
    assert int(df2.iloc[0]["start_frame"]) == 1
    # Keep other category
    append_epochs(
        block,
        "left",
        category=CATEGORY_PUPIL_PERIMETER,
        frames=[100],
        replace_category=True,
    )
    assert set(list_categories(block)) == {
        CATEGORY_LED_BLINK,
        CATEGORY_PUPIL_PERIMETER,
    }


def test_mask_eye_df_by_epochs() -> None:
    df = pd.DataFrame(
        {
            "eye_frame": [0, 1, 2, 3],
            "center_x": [1.0, 2.0, 3.0, 4.0],
            "center_y": [1.0, 2.0, 3.0, 4.0],
            "width": [10.0, 10.0, 10.0, 10.0],
        }
    )
    epochs = frames_to_epochs([1, 2])
    epochs["category"] = CATEGORY_LED_BLINK
    masked, n_hit = mask_eye_df_by_epochs(
        df, epochs, frame_col="eye_frame", categories=[CATEGORY_LED_BLINK]
    )
    assert n_hit == 2
    assert np.isnan(masked.loc[1, "center_x"])
    assert masked.loc[0, "center_x"] == 1.0
    # Original untouched
    assert df.loc[1, "center_x"] == 2.0


def test_apply_noise_epochs_to_block_csvs(tmp_path: Path) -> None:
    block = tmp_path / "block"
    analysis = block / "analysis"
    analysis.mkdir(parents=True)
    append_epochs(
        block,
        "left",
        category=CATEGORY_PUPIL_PERIMETER,
        frames=[2],
        replace_category=True,
    )
    le = pd.DataFrame(
        {
            "L_eye_frame": [1, 2, 3],
            "center_x": [1.0, 2.0, 3.0],
            "center_y": [1.0, 2.0, 3.0],
            "width": [1.0, 1.0, 1.0],
            "height": [1.0, 1.0, 1.0],
            "phi": [0.0, 0.0, 0.0],
            "ellipse_size": [1.0, 1.0, 1.0],
            "center_x_corrected": [1.0, 2.0, 3.0],
            "center_y_corrected": [1.0, 2.0, 3.0],
        }
    )
    le.to_csv(analysis / "le_df.csv")
    report = apply_noise_epochs_to_block_csvs(
        block, categories=[CATEGORY_PUPIL_PERIMETER], eyes=["left"]
    )
    assert report["eyes"]["left"]["files"]["le_df.csv"] == 1
    loaded = pd.read_csv(analysis / "le_df.csv", index_col=0)
    assert np.isnan(loaded.loc[1, "center_x"])  # row for frame 2
