"""Power analysis for two-arm conversion experiments.

Standard two-proportion z-test sample sizing. Every promise made here is
verified by Monte Carlo in tests/test_experiment_calibration.py: designs
sized by `required_n_per_arm` must actually achieve their stated power.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_NORMAL = NormalDist()


def _validate(p_baseline: float, alpha: float, power: float) -> None:
    if not 0.0 < p_baseline < 1.0:
        raise ValueError(f"baseline rate out of (0, 1): {p_baseline}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha out of (0, 1): {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power out of (0, 1): {power}")


def required_n_per_arm(
    p_baseline: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-arm sample size to detect a relative lift with a two-sided test."""
    _validate(p_baseline, alpha, power)
    if mde_relative <= 0.0:
        raise ValueError(f"mde_relative must be positive: {mde_relative}")
    p2 = p_baseline * (1.0 + mde_relative)
    if p2 >= 1.0:
        raise ValueError(f"treatment rate implied by MDE >= 1: {p2}")

    z_alpha = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    z_beta = _NORMAL.inv_cdf(power)
    p_bar = (p_baseline + p2) / 2.0
    delta = p2 - p_baseline

    numerator = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * math.sqrt(p_baseline * (1.0 - p_baseline) + p2 * (1.0 - p2))
    ) ** 2
    return math.ceil(numerator / delta**2)


def minimum_detectable_effect(
    n_per_arm: int,
    p_baseline: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest relative lift detectable at the given n (bisection inverse)."""
    _validate(p_baseline, alpha, power)
    if n_per_arm <= 1:
        raise ValueError(f"n_per_arm must exceed 1: {n_per_arm}")

    lo, hi = 1e-6, (1.0 - p_baseline) / p_baseline - 1e-9
    if required_n_per_arm(p_baseline, hi, alpha, power) > n_per_arm:
        raise ValueError(f"n_per_arm={n_per_arm} cannot detect any lift at this power")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if required_n_per_arm(p_baseline, mid, alpha, power) > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi
