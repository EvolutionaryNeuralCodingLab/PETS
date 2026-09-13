"""Robust Rayleigh-core estimators."""

from __future__ import annotations

import numpy as np

from eye_tracking_system_tools.analysis.rayleigh_core import (
    RAYLEIGH_EXCESS_KURTOSIS,
    RAYLEIGH_SKEW,
    argmin_gof_hi_pct,
    argmin_shape_hi_pct,
    engbert_mad_sigma,
    gof_from_sweep,
    gof_onset_hi_pct,
    ks_truncated_rayleigh,
    mean_excess_d2,
    naive_rayleigh_mle,
    plateau_onset_hi_pct,
    rayleigh_qq,
    rayleigh_scale_from_median,
    sweep_hi_pct,
    truncated_rayleigh_mle,
)


def _rayleigh_sample(rng: np.random.Generator, n: int, B: float) -> np.ndarray:
    return np.hypot(rng.normal(0, B, n), rng.normal(0, B, n))


def test_median_and_mle_recover_B():
    rng = np.random.default_rng(0)
    d = _rayleigh_sample(rng, 80_000, 1.0)
    assert abs(rayleigh_scale_from_median(d) - 1.0) < 0.03
    assert abs(naive_rayleigh_mle(d) - 1.0) < 0.03


def test_truncated_mle_corrects_naive_low_bias():
    rng = np.random.default_rng(1)
    d = _rayleigh_sample(rng, 80_000, 1.0)
    c = 1.2
    clipped = d[d <= c]
    naive = naive_rayleigh_mle(clipped)
    trunc = truncated_rayleigh_mle(d, c)
    assert naive < 0.95
    assert abs(trunc - 1.0) < 0.05
    assert trunc > naive


def test_mean_excess_d2_flat_at_2B2():
    rng = np.random.default_rng(2)
    B = 1.0
    d = _rayleigh_sample(rng, 80_000, B)
    grid = np.linspace(0.05, 1.5, 12)
    mex = mean_excess_d2(d, grid)
    # stay inside the core (well below the far tail of a unit Rayleigh)
    core = mex.loc[mex["d_u"] <= 1.0, "mean_excess_d2"].to_numpy()
    assert np.all(np.isfinite(core))
    assert abs(float(np.median(core)) - 2.0 * B * B) < 0.15


def test_qq_hugs_diagonal_on_pure_rayleigh():
    rng = np.random.default_rng(3)
    d = _rayleigh_sample(rng, 40_000, 0.4)
    theo, samp = rayleigh_qq(d, 0.4, n_plot=200)
    # lower 80% of the QQ should match
    k = int(0.8 * theo.size)
    rel = np.median(np.abs(samp[:k] - theo[:k]) / np.maximum(theo[:k], 1e-9))
    assert rel < 0.05


def test_plateau_onset_on_pure_rayleigh_is_near_untrimmed():
    rng = np.random.default_rng(4)
    d = _rayleigh_sample(rng, 30_000, 1.0)
    table = sweep_hi_pct(d, np.arange(100.0, 89.5, -0.5))
    onset = plateau_onset_hi_pct(table, err_max=1.5)
    best = argmin_shape_hi_pct(table)
    assert onset >= 95.0
    assert best >= 90.0
    # untrimmed shape is already near the Rayleigh constants
    assert abs(float(table["skew"].iloc[0]) - RAYLEIGH_SKEW) < 0.15
    assert abs(float(table["kurtosis"].iloc[0]) - RAYLEIGH_EXCESS_KURTOSIS) < 0.2


def test_argmin_shape_prefers_core_on_heavy_tail():
    rng = np.random.default_rng(9)
    core = _rayleigh_sample(rng, 18_000, 0.2)
    tail = _rayleigh_sample(rng, 2_000, 4.0)
    d = np.concatenate([core, tail])
    table = sweep_hi_pct(d, np.arange(100.0, 69.5, -0.5))
    best = argmin_shape_hi_pct(table)
    assert np.isfinite(best)
    assert best < 100.0
    i = int((table["hi_pct"] - best).abs().idxmin())
    assert float(table.loc[i, "shape_err"]) == float(np.nanmin(table["shape_err"]))


def test_engbert_mad_matches_gaussian_sigma():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 0.25, 50_000)
    assert abs(engbert_mad_sigma(x) - 0.25) < 0.02


def test_ks_rayleigh_small_on_pure_sample():
    rng = np.random.default_rng(11)
    d = _rayleigh_sample(rng, 20_000, 0.4)
    from eye_tracking_system_tools.analysis.rayleigh_core import ks_rayleigh

    gof = ks_rayleigh(d, n_sim=80, rng=rng)
    assert gof["ks_statistic"] < 0.02
    assert gof["ks_p_lilliefors"] > 0.05


def test_ks_rayleigh_rejects_heavy_tail():
    rng = np.random.default_rng(12)
    core = _rayleigh_sample(rng, 8_000, 0.2)
    tail = _rayleigh_sample(rng, 2_000, 3.0)
    from eye_tracking_system_tools.analysis.rayleigh_core import ks_rayleigh

    gof = ks_rayleigh(np.concatenate([core, tail]), n_sim=0)
    assert gof["ks_statistic"] > 0.05
    assert gof["ks_p"] < 1e-6


def test_ks_small_on_truncated_pure_rayleigh():
    rng = np.random.default_rng(7)
    d = _rayleigh_sample(rng, 40_000, 1.0)
    c = 2.0
    B = truncated_rayleigh_mle(d, c)
    ks, _p = ks_truncated_rayleigh(d[d <= c], c, B, n_ks=8000, rng=rng)
    assert ks < 0.03


def test_gof_improves_when_heavy_tail_is_clipped():
    rng = np.random.default_rng(8)
    core = _rayleigh_sample(rng, 18_000, 0.2)
    tail = _rayleigh_sample(rng, 2_000, 4.0)
    d = np.concatenate([core, tail])
    table = sweep_hi_pct(d, np.array([100.0, 90.0]))
    gof = gof_from_sweep(d, table, n_ks=8000)
    ks_all = float(gof.loc[gof["hi_pct"] == 100.0, "ks"].iloc[0])
    ks_clip = float(gof.loc[gof["hi_pct"] == 90.0, "ks"].iloc[0])
    assert ks_clip < ks_all
    onset = gof_onset_hi_pct(gof)
    assert onset <= 100.0
    best = argmin_gof_hi_pct(gof)
    assert best == 90.0


def test_contaminated_median_stays_near_core():
    rng = np.random.default_rng(6)
    core = _rayleigh_sample(rng, 19_000, 0.2)
    tail = _rayleigh_sample(rng, 1_000, 3.0)
    d = np.concatenate([core, tail])
    assert abs(rayleigh_scale_from_median(d) - 0.2) < 0.04
    assert naive_rayleigh_mle(d) > 0.5
