"""Media mix model: geometric adstock + saturation, fit from first principles.

Conditional on each channel's nonlinear parameters (adstock decay, half-
saturation point) the model is linear, so fitting alternates between a
per-channel grid search over the nonlinear pair and an exact OLS solve for
the linear part (base, day-of-week, channel betas). Uncertainty comes from a
moving-block residual bootstrap that respects the serial structure.

Every claim is scored against simulator truth in tests/test_marketing_recovery.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from growth_lab.causal._regression import ols

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DECAY_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(19))  # 0.00..0.90
HALF_SAT_MULTIPLIERS: tuple[float, ...] = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)


def adstock(spend: FloatArray, decay: float) -> FloatArray:
    """Geometric carryover: a_t = spend_t + decay * a_{t-1}."""
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay out of [0, 1): {decay}")
    out = np.empty_like(spend)
    carry = 0.0
    for t in range(len(spend)):
        carry = spend[t] + decay * carry
        out[t] = carry
    return out


def saturate(adstocked: FloatArray, half_sat: float) -> FloatArray:
    """Michaelis-Menten saturation, 0..1 with value 0.5 at `half_sat`."""
    if half_sat <= 0:
        raise ValueError(f"half_sat must be positive: {half_sat}")
    result: FloatArray = adstocked / (adstocked + half_sat)
    return result


@dataclass(frozen=True)
class MmmFit:
    channels: tuple[str, ...]
    beta: FloatArray
    decay: FloatArray
    half_sat: FloatArray
    base: float
    dow_effect: FloatArray  # 7 entries, Monday first, Monday pinned to 0
    fitted: FloatArray
    r_squared: float
    contribution: FloatArray  # (n_days, n_channels)

    def roas(self, spend: FloatArray) -> FloatArray:
        """Incremental revenue per unit spend over the fitted window."""
        result: FloatArray = self.contribution.sum(axis=0) / spend.sum(axis=0)
        return result


def _design(
    spend: FloatArray, dow: IntArray, decay: FloatArray, half_sat: FloatArray
) -> FloatArray:
    n_days, n_ch = spend.shape
    sat = np.empty((n_days, n_ch))
    for c in range(n_ch):
        sat[:, c] = saturate(adstock(spend[:, c], float(decay[c])), float(half_sat[c]))
    dow_dummies = np.zeros((n_days, 6))
    for d in range(1, 7):
        dow_dummies[:, d - 1] = dow == d
    return np.column_stack([np.ones(n_days), dow_dummies, sat])


def fit_mmm(
    spend: FloatArray,
    revenue: FloatArray,
    day_of_week: IntArray,
    channels: tuple[str, ...],
    n_rounds: int = 3,
) -> MmmFit:
    """Coordinate descent: grid-search each channel's (decay, half_sat) with
    all others fixed, solving the linear part exactly at every candidate."""
    n_days, n_ch = spend.shape
    if len(revenue) != n_days or len(day_of_week) != n_days or len(channels) != n_ch:
        raise ValueError("inconsistent shapes")

    decay = np.zeros(n_ch)
    half_sat = np.array([max(float(np.median(spend[:, c])), 1e-9) for c in range(n_ch)])

    def sse(dec: FloatArray, hs: FloatArray) -> float:
        fit = ols(_design(spend, day_of_week, dec, hs), revenue)
        return float(fit.residuals @ fit.residuals)

    best = sse(decay, half_sat)
    for _ in range(n_rounds):
        improved = False
        for c in range(n_ch):
            median_adstocked = {
                d: max(float(np.median(adstock(spend[:, c], d))), 1e-9) for d in DECAY_GRID
            }
            for d in DECAY_GRID:
                for mult in HALF_SAT_MULTIPLIERS:
                    cand_decay = decay.copy()
                    cand_hs = half_sat.copy()
                    cand_decay[c] = d
                    cand_hs[c] = mult * median_adstocked[d]
                    candidate = sse(cand_decay, cand_hs)
                    if candidate < best - 1e-9:
                        best, decay, half_sat = candidate, cand_decay, cand_hs
                        improved = True
        if not improved:
            break

    final = ols(_design(spend, day_of_week, decay, half_sat), revenue)
    beta = final.beta[7:].astype(np.float64)
    if np.any(beta < 0):
        raise RuntimeError(
            f"MMM produced negative channel effects {beta.round(1).tolist()}: "
            "the model is misspecified for this data, refusing to report"
        )
    contribution = np.empty((n_days, n_ch))
    for c in range(n_ch):
        contribution[:, c] = beta[c] * saturate(
            adstock(spend[:, c], float(decay[c])), float(half_sat[c])
        )
    fitted = revenue - final.residuals
    total_ss = float(((revenue - revenue.mean()) ** 2).sum())
    r_squared = 1.0 - float(final.residuals @ final.residuals) / total_ss
    return MmmFit(
        channels=channels,
        beta=beta,
        decay=decay,
        half_sat=half_sat,
        base=float(final.beta[0]),
        dow_effect=np.concatenate(([0.0], final.beta[1:7])),
        fitted=fitted,
        r_squared=r_squared,
        contribution=contribution,
    )


def bootstrap_roas_ci(
    fit: MmmFit,
    spend: FloatArray,
    revenue: FloatArray,
    day_of_week: IntArray,
    n_boot: int = 200,
    block_days: int = 14,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Moving-block residual bootstrap CI for per-channel ROAS.

    Nonlinear parameters are held at their fitted values (standard practice
    for MMM intervals; documented limitation — intervals are conditional on
    the selected adstock/saturation shape).
    """
    rng = np.random.default_rng(seed)
    n_days = len(revenue)
    residuals = revenue - fit.fitted
    design = _design(spend, day_of_week, fit.decay, fit.half_sat)
    sat = design[:, 7:]
    n_blocks = math.ceil(n_days / block_days)

    roas_draws = np.empty((n_boot, spend.shape[1]))
    for b in range(n_boot):
        starts = rng.integers(0, n_days - block_days + 1, size=n_blocks)
        resampled = np.concatenate([residuals[s : s + block_days] for s in starts])[:n_days]
        y_star = fit.fitted + resampled
        beta_star = ols(design, y_star).beta[7:]
        contrib_star = sat * beta_star
        roas_draws[b] = contrib_star.sum(axis=0) / spend.sum(axis=0)

    lo = (1.0 - level) / 2.0
    lower: FloatArray = np.quantile(roas_draws, lo, axis=0)
    upper: FloatArray = np.quantile(roas_draws, 1.0 - lo, axis=0)
    return lower, upper
