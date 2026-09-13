"""Rayleigh quiet-segment noise-core export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eye_tracking_system_tools.analysis.plot_bundle import is_plot_bundle
from eye_tracking_system_tools.analysis.rayleigh_noise_core_export import (
    PDF_NAME,
    PICKLE_NAME,
    SUMMARY_NAME,
    export_rayleigh_noise_core_window,
    figure_rayleigh_noise_core_window,
    pick_quiet_segments_from_d,
    resolve_noise_core_xmax,
)


def _rayleigh_sample(rng: np.random.Generator, n: int, B: float) -> np.ndarray:
    return np.hypot(rng.normal(0, B, n), rng.normal(0, B, n))


def test_figure_rayleigh_noise_core_window_renders():
    rng = np.random.default_rng(0)
    d = _rayleigh_sample(rng, 2000, 0.05)
    fig = figure_rayleigh_noise_core_window(d, B=0.05, threshold=0.8, title="test")
    assert fig.axes
    assert tuple(fig.get_size_inches()) == (3.0, 2.5)
    patches = fig.axes[0].patches
    assert patches
    assert patches[0].get_facecolor()[3] == 1.0
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.close(fig)


def test_xmax_shows_tail_and_threshold():
    rng = np.random.default_rng(4)
    d = _rayleigh_sample(rng, 4000, 0.1)
    xmax = resolve_noise_core_xmax(d, 0.1, 0.8)
    assert xmax >= 0.8 * 1.08 - 1e-9
    assert xmax >= 6.0 * 0.1
    fig = figure_rayleigh_noise_core_window(d, B=0.1, threshold=0.8, title="xlim")
    assert fig.axes[0].get_xlim()[1] >= 0.8
    import matplotlib.pyplot as plt

    plt.close(fig)
    # No threshold: still far enough to see the Rayleigh tail.
    xmax_nt = resolve_noise_core_xmax(d, 0.1, None)
    assert xmax_nt >= 6.0 * 0.1
    # Far threshold must not collapse the body into a few bins.
    from eye_tracking_system_tools.analysis.rayleigh_noise_core_export import (
        resolve_noise_core_n_bins,
    )

    xmax_far = resolve_noise_core_xmax(d, 0.1, 3.23)
    n_far = resolve_noise_core_n_bins(d, 0.1, xmax_far, 40)
    assert n_far > 40


def test_export_rayleigh_noise_core_window_writes_bundle(tmp_path: Path):
    rng = np.random.default_rng(1)
    d = _rayleigh_sample(rng, 1500, 0.04)
    written = export_rayleigh_noise_core_window(
        tmp_path,
        d=d,
        quiet_segments=[(1000.0, 2000.0)],
        block_label="PV_test block_001",
        block_path=tmp_path / "block_001",
        animal="PV_test",
        eye="both",
        mount_type="modular",
        threshold=0.8,
        B=0.04,
        show=False,
    )
    bundle = tmp_path / "rayleigh_noise_core"
    assert is_plot_bundle(bundle)
    assert written[PDF_NAME].is_file()
    assert written[PICKLE_NAME].is_file()
    assert written[SUMMARY_NAME].is_file()
    assert (bundle / "replot.py").is_file()

    import pickle

    with open(written[PICKLE_NAME], "rb") as f:
        payload = pickle.load(f)
    assert payload["n"] == 1500
    assert payload["snr_thr_over_2B"] == pytest.approx(0.8 / (2.0 * 0.04), rel=1e-9)


def test_figure_median_marker_has_no_2b_label():
    rng = np.random.default_rng(2)
    d = _rayleigh_sample(rng, 1500, 0.05)
    fig = figure_rayleigh_noise_core_window(
        d, B=0.05, threshold=0.8, noise_marker="median", title="median"
    )
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any(s.startswith("median=") for s in labels)
    assert not any("2B" in s for s in labels)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_write_s10_media_style(tmp_path: Path):
    rng = np.random.default_rng(5)
    meta = tmp_path / "metadata"
    meta.mkdir()
    for species, pkl_name, thr in (
        ("lizard", "S10a_lizard_rayleigh_noise_core.pkl", 0.8),
        ("mouse", "S10b_mouse_rayleigh_noise_core.pkl", 3.23),
        ("turtle", "S10c_turtle_rayleigh_noise_core.pkl", None),
    ):
        d = _rayleigh_sample(rng, 400, 0.05 if species == "lizard" else 0.2)
        written = export_rayleigh_noise_core_window(
            tmp_path / species,
            d=d,
            quiet_segments=[(0.0, 1000.0)],
            block_label=f"{species}_block",
            block_path=tmp_path / species / "block",
            animal=species,
            eye="both",
            mount_type=species,
            threshold=thr,
            show=False,
        )
        import shutil

        shutil.copy2(written[PICKLE_NAME], meta / pkl_name)
    from eye_tracking_system_tools.analysis.rayleigh_noise_core_export import (
        write_s10_media_style,
    )

    out = write_s10_media_style(tmp_path)
    assert (tmp_path / "media_style" / "caption.md").is_file()
    assert out["S10a_lizard.pdf"].is_file()
    text = (tmp_path / "media_style" / "caption.md").read_text()
    assert "SNR = threshold / median(D)" in text
    assert "2B" not in text

    rng = np.random.default_rng(3)
    d = _rayleigh_sample(rng, 800, 0.05)
    written = export_rayleigh_noise_core_window(
        tmp_path / "liz",
        d=d,
        quiet_segments=[(0.0, 1000.0)],
        block_label="PV_test",
        block_path=tmp_path / "block",
        animal="PV_test",
        eye="both",
        mount_type="rigid",
        threshold=0.8,
        show=False,
    )
    from eye_tracking_system_tools.analysis.rayleigh_noise_core_export import write_s10_ks_gof_csv

    csv_path = write_s10_ks_gof_csv(
        {"lizard": written[PICKLE_NAME]},
        tmp_path / "ks_rayleigh_gof.csv",
        n_sim=40,
        seed=0,
    )
    import pandas as pd

    table = pd.read_csv(csv_path)
    assert list(table["species"]) == ["lizard"]
    assert int(table.loc[0, "n"]) == 800
    assert np.isfinite(table.loc[0, "ks_statistic"])


def test_pick_quiet_segments_from_d_prefers_lowest_median_windows():
    t = np.arange(0.0, 10000.0, 10.0)
    d = np.full(t.shape, 1.0)
    d[(t >= 2000) & (t < 4000)] = 0.1
    d[(t >= 6000) & (t < 8000)] = 0.2
    segs = pick_quiet_segments_from_d(d, t, n_segments=2, win_ms=2000.0)
    assert len(segs) == 2
    starts = [s for s, _ in segs]
    assert min(starts) < 4000
    assert max(starts) >= 5000
