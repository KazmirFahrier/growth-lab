"""Sharp regression discontinuity with a density-continuity gate.

If units can push themselves across the threshold, the design is broken; a
McCrary-style count comparison in narrow bins flanking the cutoff must pass
before any effect is estimated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from growth_lab.causal._regression import FloatArray, add_intercept, normal_two_sided_p, ols
from growth_lab.causal.exceptions import AssumptionViolation


@dataclass(frozen=True)
class RddResult:
    effect: float
    se: float
    bandwidth: float
    n_left: int
    n_right: int
    density_p_value: float


def sharp_rdd(
    running: FloatArray,
    outcome: FloatArray,
    cutoff: float,
    bandwidth: float | None = None,
    density_alpha: float = 0.001,
) -> RddResult:
    """Local-linear sharp RDD: separate fits each side, effect at the cutoff."""
    if len(running) != len(outcome):
        raise ValueError("running and outcome must be the same length")
    h = 0.5 * float(np.std(running)) if bandwidth is None else bandwidth
    if h <= 0:
        raise ValueError(f"bandwidth must be positive: {h}")

    centered = running - cutoff

    # Density gate: counts in narrow flanking bins should be comparable.
    bin_width = h / 5.0
    n_below = int(((centered >= -bin_width) & (centered < 0)).sum())
    n_above = int(((centered >= 0) & (centered < bin_width)).sum())
    if n_below + n_above < 20:
        raise ValueError("too few observations near the cutoff for a density check")
    z = (n_above - n_below) / math.sqrt(n_above + n_below)
    density_p = normal_two_sided_p(z)
    if density_p < density_alpha:
        raise AssumptionViolation(
            f"density discontinuity at the cutoff ({n_below} just below vs "
            f"{n_above} just above, p={density_p:.2e}): units appear to sort "
            "across the threshold; RDD is not identified"
        )

    left = (centered < 0) & (centered >= -h)
    right = (centered >= 0) & (centered <= h)
    if left.sum() < 10 or right.sum() < 10:
        raise ValueError("too few observations within the bandwidth")

    fit_left = ols(add_intercept(centered[left]), outcome[left])
    fit_right = ols(add_intercept(centered[right]), outcome[right])
    effect = float(fit_right.beta[0] - fit_left.beta[0])
    se = math.sqrt(float(fit_right.se[0]) ** 2 + float(fit_left.se[0]) ** 2)
    return RddResult(
        effect=effect,
        se=se,
        bandwidth=h,
        n_left=int(left.sum()),
        n_right=int(right.sum()),
        density_p_value=density_p,
    )
