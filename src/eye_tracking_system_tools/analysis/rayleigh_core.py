"""Robust Rayleigh-core estimators for inter-frame Δangle.

If ``Δφ, Δθ ~ N(0, B²)`` i.i.d., then ``D = hypot(Δφ, Δθ)`` is Rayleigh with
scale ``B``. Saccades live in the tail, so the noise number is a robust (or
truncated) estimate of ``B``, not the untrimmed MLE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Independent of B. skew = 2√π (π-3) / (4-π)^{3/2}
# excess kurtosis = -(6π²-24π+16)/(4-π)²
RAYLEIGH_SKEW = float(
    2.0 * np.sqrt(np.pi) * (np.pi - 3.0) / ((4.0 - np.pi) ** 1.5)
)
RAYLEIGH_EXCESS_KURTOSIS = float(
    -((6.0 * np.pi**2) - (24.0 * np.pi) + 16.0) / ((4.0 - np.pi) ** 2)
)
# median(D) = B * sqrt(2 ln 2)
RAYLEIGH_MEDIAN_FACTOR = float(np.sqrt(2.0 * np.log(2.0)))


def _finite_d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a) & (a >= 0)]


def naive_rayleigh_mle(arr: np.ndarray) -> float:
    """Untruncated MLE: B = sqrt(mean(D²) / 2). Tail-inflated if saccades remain."""
    a = _finite_d(arr)
    if a.size < 8:
        return float("nan")
    return float(np.sqrt(np.mean(a * a) / 2.0))


def rayleigh_pdf(x: np.ndarray, B: float) -> np.ndarray:
    """Rayleigh PDF, loc=0, scale B."""
    b = max(float(B), 1e-15)
    x = np.asarray(x, dtype=float)
    return (x / (b * b)) * np.exp(-0.5 * (x / b) ** 2)


def truncated_rayleigh_pdf(x: np.ndarray, B: float, c: float) -> np.ndarray:
    """Rayleigh PDF conditioned on ``D ≤ c``."""
    mass = 1.0 - np.exp(-0.5 * (float(c) / max(float(B), 1e-15)) ** 2)
    return rayleigh_pdf(x, B) / max(float(mass), 1e-15)


def rayleigh_scale_from_median(arr: np.ndarray) -> float:
    """High-breakdown scale: B = median(D) / sqrt(2 ln 2). No cutoff."""
    a = _finite_d(arr)
    if a.size < 8:
        return float("nan")
    med = float(np.median(a))
    if med <= 0:
        return float("nan")
    return med / RAYLEIGH_MEDIAN_FACTOR


def engbert_mad_sigma(arr: np.ndarray) -> float:
    """Engbert / Holmqvist MAD-σ: median(|x - median|) / 0.6745."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 8:
        return float("nan")
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad <= 0:
        return float("nan")
    return mad / 0.6745


def truncated_rayleigh_mle(
    arr: np.ndarray,
    c: float,
    *,
    max_iter: int = 60,
    tol: float = 1e-12,
) -> float:
    """MLE of B given ``D | D ≤ c`` (conditional / truncated Rayleigh).

    The naive MLE on the clipped sample is biased low. Fixed-point:

    ``B² = mean(D²)/2 + (c²/2) λ/(1-λ)``,  ``λ = exp(-c² / (2 B²))``.
    """
    c = float(c)
    if not np.isfinite(c) or c <= 0:
        return float("nan")
    a = _finite_d(arr)
    a = a[a <= c]
    if a.size < 8:
        return float("nan")
    mean_d2 = float(np.mean(a * a))
    b2 = mean_d2 / 2.0
    c2 = c * c
    for _ in range(int(max_iter)):
        if not np.isfinite(b2) or b2 <= 0:
            return float("nan")
        lam = float(np.exp(-c2 / (2.0 * b2)))
        if lam >= 1.0 - 1e-15:
            return float("nan")
        b2_new = mean_d2 / 2.0 + (c2 / 2.0) * lam / (1.0 - lam)
        if abs(b2_new - b2) <= tol * max(b2, 1e-18):
            b2 = b2_new
            break
        b2 = b2_new
    return float(np.sqrt(b2))


def shape_moments(arr: np.ndarray) -> tuple[float, float]:
    """Sample skew and excess (Fisher) kurtosis."""
    a = _finite_d(arr)
    if a.size < 8:
        return float("nan"), float("nan")
    return (
        float(stats.skew(a, bias=False)),
        float(stats.kurtosis(a, fisher=True, bias=False)),
    )


def shape_error(skew: np.ndarray | float, kurt: np.ndarray | float) -> np.ndarray:
    """Normalized distance to untruncated Rayleigh (skew, kurtosis)."""
    sk = np.asarray(skew, dtype=float)
    ku = np.asarray(kurt, dtype=float)
    return np.hypot((sk - RAYLEIGH_SKEW) / RAYLEIGH_SKEW, (ku - RAYLEIGH_EXCESS_KURTOSIS) / RAYLEIGH_EXCESS_KURTOSIS)


def sweep_hi_pct(
    arr: np.ndarray,
    hi_pcts: np.ndarray,
) -> pd.DataFrame:
    """Prefix of the sorted sample at each ``HI_PCT``: moments + naive and truncated B."""
    sorted_a = np.sort(_finite_d(arr))
    n = int(sorted_a.size)
    rows: list[dict[str, float | int]] = []
    for hi in np.asarray(hi_pcts, dtype=float):
        n_keep = max(8, int(np.floor(float(hi) / 100.0 * n)))
        n_keep = min(n_keep, n)
        sub = sorted_a[:n_keep]
        c = float(sub[-1]) if sub.size else float("nan")
        sk, ku = shape_moments(sub)
        rows.append(
            {
                "hi_pct": float(hi),
                "n": int(n_keep),
                "d_cut": c,
                "B_naive": naive_rayleigh_mle(sub),
                "B_trunc": truncated_rayleigh_mle(sub, c),
                "skew": sk,
                "kurtosis": ku,
            }
        )
    table = pd.DataFrame(rows)
    table["shape_err"] = shape_error(table["skew"].to_numpy(), table["kurtosis"].to_numpy())
    return table


def plateau_onset_hi_pct(
    table: pd.DataFrame,
    *,
    err_max: float = 1.5,
) -> float:
    """Least trimming that enters the Rayleigh-shape band.

    Walk ``HI_PCT`` from 100 downward; return the first (largest) value whose
    ``shape_err <= err_max``. ``nan`` if the sweep never enters the band.
    """
    if table.empty or "shape_err" not in table.columns:
        return float("nan")
    sub = table.sort_values("hi_pct", ascending=False)
    ok = sub["shape_err"].to_numpy() <= float(err_max)
    if not np.any(ok):
        return float("nan")
    return float(sub["hi_pct"].to_numpy()[np.argmax(ok)])


def argmin_shape_hi_pct(table: pd.DataFrame) -> float:
    """HI_PCT that minimizes shape_err (often past the plateau; comparison only)."""
    if table.empty or "shape_err" not in table.columns:
        return float("nan")
    err = table["shape_err"].to_numpy()
    if not np.any(np.isfinite(err)):
        return float("nan")
    return float(table["hi_pct"].iloc[int(np.nanargmin(err))])


def mean_excess_d2(
    arr: np.ndarray,
    thresholds_d: np.ndarray,
) -> pd.DataFrame:
    """Mean excess of ``D²`` vs a threshold on ``D``.

    If ``D²`` is exponential (true Rayleigh), mean excess is flat at ``2 B²``.
    """
    a = _finite_d(arr)
    z = np.sort(a * a)
    rows: list[dict[str, float | int]] = []
    for d_u in np.asarray(thresholds_d, dtype=float):
        if not np.isfinite(d_u) or d_u < 0:
            continue
        u = float(d_u * d_u)
        i = int(np.searchsorted(z, u, side="right"))
        tail = z[i:]
        n_tail = int(tail.size)
        if n_tail < 8:
            mex = float("nan")
        else:
            mex = float(np.mean(tail - u))
        rows.append(
            {
                "d_u": float(d_u),
                "u": u,
                "n_tail": n_tail,
                "mean_excess_d2": mex,
            }
        )
    return pd.DataFrame(rows)


def truncated_rayleigh_cdf(x: np.ndarray, B: float, c: float) -> np.ndarray:
    """CDF of ``D | D ≤ c`` under Rayleigh(B)."""
    x = np.asarray(x, dtype=float)
    b = max(float(B), 1e-15)
    c = max(float(c), 1e-15)
    num = 1.0 - np.exp(-0.5 * (np.clip(x, 0.0, None) / b) ** 2)
    den = 1.0 - np.exp(-0.5 * (c / b) ** 2)
    return np.clip(num / max(float(den), 1e-15), 0.0, 1.0)


def truncated_mean_loglik(arr: np.ndarray, c: float, B: float) -> float:
    """Mean log-density of the truncated Rayleigh on ``D ≤ c``."""
    a = _finite_d(arr)
    a = a[(a > 0) & (a <= float(c))]
    b = float(B)
    if a.size < 8 or not np.isfinite(b) or b <= 0 or not np.isfinite(c) or c <= 0:
        return float("nan")
    mass = 1.0 - np.exp(-0.5 * (c / b) ** 2)
    if mass <= 0:
        return float("nan")
    return float(np.mean(np.log(a) - 2.0 * np.log(b) - (a * a) / (2.0 * b * b) - np.log(mass)))


def ks_truncated_rayleigh(
    arr: np.ndarray,
    c: float,
    B: float,
    *,
    n_ks: int = 15000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """KS statistic of ``D ≤ c`` vs truncated Rayleigh(B). Lower is a better fit."""
    a = _finite_d(arr)
    a = a[a <= float(c)]
    if a.size < 8 or not np.isfinite(B) or B <= 0:
        return float("nan"), float("nan")
    if a.size > int(n_ks):
        rng = np.random.default_rng(0) if rng is None else rng
        a = rng.choice(a, size=int(n_ks), replace=False)
    cdf = lambda x, _B=float(B), _c=float(c): truncated_rayleigh_cdf(x, _B, _c)
    stat, p = stats.kstest(a, cdf)
    return float(stat), float(p)


def gof_from_sweep(
    arr: np.ndarray,
    table: pd.DataFrame,
    *,
    n_ks: int = 15000,
) -> pd.DataFrame:
    """KS and mean log-likelihood at each row of a ``sweep_hi_pct`` table."""
    sorted_a = np.sort(_finite_d(arr))
    n = int(sorted_a.size)
    rng = np.random.default_rng(0)
    rows: list[dict[str, float | int]] = []
    for rec in table.itertuples(index=False):
        hi = float(rec.hi_pct)
        n_keep = max(8, int(np.floor(hi / 100.0 * n)))
        n_keep = min(n_keep, n)
        sub = sorted_a[:n_keep]
        c = float(rec.d_cut)
        B = float(rec.B_trunc)
        ks, ks_p = ks_truncated_rayleigh(sub, c, B, n_ks=n_ks, rng=rng)
        rows.append(
            {
                "hi_pct": hi,
                "d_cut": c,
                "B_trunc": B,
                "ks": ks,
                "ks_p": ks_p,
                "mean_loglik": truncated_mean_loglik(sub, c, B),
            }
        )
    out = pd.DataFrame(rows)
    return out


def gof_onset_hi_pct(gof: pd.DataFrame, *, ks_tol: float = 1.15) -> float:
    """Largest HI_PCT whose KS is within ``ks_tol`` of the best (smallest) KS."""
    if gof.empty or "ks" not in gof.columns:
        return float("nan")
    ks = gof["ks"].to_numpy(dtype=float)
    if not np.any(np.isfinite(ks)):
        return float("nan")
    ks_min = float(np.nanmin(ks))
    if not np.isfinite(ks_min) or ks_min < 0:
        return float("nan")
    sub = gof.sort_values("hi_pct", ascending=False)
    ok = sub["ks"].to_numpy(dtype=float) <= ks_min * float(ks_tol)
    if not np.any(ok):
        return float("nan")
    return float(sub["hi_pct"].to_numpy()[np.argmax(ok)])


def rayleigh_qq(
    arr: np.ndarray,
    B: float,
    *,
    n_plot: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Theoretical Rayleigh(B) quantiles vs sample quantiles (even probability grid)."""
    a = np.sort(_finite_d(arr))
    B = float(B)
    if a.size < 8 or not np.isfinite(B) or B <= 0:
        empty = np.array([], dtype=float)
        return empty, empty
    n = min(int(n_plot), a.size)
    p = (np.arange(n) + 0.5) / n
    # skip p very close to 1 (tail quantile is noisy)
    p = p[p < 0.999]
    theo = B * np.sqrt(-2.0 * np.log(1.0 - p))
    idx = np.clip((p * a.size).astype(int), 0, a.size - 1)
    samp = a[idx]
    return theo, samp


def robust_scale_row(
    arr: np.ndarray,
    *,
    dphi: np.ndarray | None = None,
    dtheta: np.ndarray | None = None,
) -> dict[str, float | int]:
    """One-mount summary for step 1: median B, untrimmed MLE, optional axis MAD."""
    a = _finite_d(arr)
    out: dict[str, float | int] = {
        "n": int(a.size),
        "median_D": float(np.median(a)) if a.size else float("nan"),
        "B_med": rayleigh_scale_from_median(a),
        "B_mle_untrimmed": naive_rayleigh_mle(a),
        "sigma_phi": float("nan"),
        "sigma_theta": float("nan"),
    }
    if dphi is not None:
        out["sigma_phi"] = engbert_mad_sigma(dphi)
    if dtheta is not None:
        out["sigma_theta"] = engbert_mad_sigma(dtheta)
    return out
