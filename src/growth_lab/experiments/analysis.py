"""Fixed-horizon analysis: two-proportion z-test, Welch test, CUPED.

Large-sample normal inference throughout (documented, and honest: the
calibration suite verifies realized false-positive rates at the sample sizes
this platform actually recommends).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

_NORMAL = NormalDist()

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TestResult:
    """A two-sample comparison: difference, CI, and p-value."""

    estimate: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float
    n_control: int
    n_treatment: int
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha


def two_proportion_ztest(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided z-test for a difference in conversion rates.

    Pooled SE for the test statistic, unpooled SE for the CI (standard).
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("both arms need units")
    if not 0 <= conversions_control <= n_control or not 0 <= conversions_treatment <= n_treatment:
        raise ValueError("conversions out of range")

    p_c = conversions_control / n_control
    p_t = conversions_treatment / n_treatment
    diff = p_t - p_c

    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_control + 1.0 / n_treatment))
    if se_pool == 0.0:
        raise ValueError("degenerate data: pooled rate is 0 or 1")
    z = diff / se_pool
    p_value = 2.0 * (1.0 - _NORMAL.cdf(abs(z)))

    se_unpooled = math.sqrt(
        p_c * (1.0 - p_c) / n_control + p_t * (1.0 - p_t) / n_treatment
    )
    z_crit = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    return TestResult(
        estimate=diff,
        ci_low=diff - z_crit * se_unpooled,
        ci_high=diff + z_crit * se_unpooled,
        z=z,
        p_value=p_value,
        n_control=n_control,
        n_treatment=n_treatment,
        alpha=alpha,
    )


def welch_test(
    y_control: FloatArray,
    y_treatment: FloatArray,
    alpha: float = 0.05,
) -> TestResult:
    """Welch two-sample test on means (normal approximation, large n)."""
    n_c, n_t = len(y_control), len(y_treatment)
    if n_c < 2 or n_t < 2:
        raise ValueError("both arms need at least 2 observations")

    diff = float(y_treatment.mean() - y_control.mean())
    se = math.sqrt(float(y_control.var(ddof=1)) / n_c + float(y_treatment.var(ddof=1)) / n_t)
    if se == 0.0:
        raise ValueError("degenerate data: zero variance in both arms")
    z = diff / se
    p_value = 2.0 * (1.0 - _NORMAL.cdf(abs(z)))
    z_crit = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    return TestResult(
        estimate=diff,
        ci_low=diff - z_crit * se,
        ci_high=diff + z_crit * se,
        z=z,
        p_value=p_value,
        n_control=n_c,
        n_treatment=n_t,
        alpha=alpha,
    )


def cuped_theta(y: FloatArray, covariate: FloatArray) -> float:
    """OLS theta for CUPED adjustment: cov(y, x) / var(x)."""
    if len(y) != len(covariate):
        raise ValueError("y and covariate must be the same length")
    var = float(covariate.var(ddof=1))
    if var == 0.0:
        raise ValueError("covariate has zero variance; CUPED is undefined")
    cov = float(np.cov(y, covariate, ddof=1)[0, 1])
    return cov / var


def cuped_adjust(y: FloatArray, covariate: FloatArray, theta: float) -> FloatArray:
    """Variance-reduced outcome: y - theta * (x - mean(x)).

    `theta` must be estimated on pooled (or pre-experiment) data and passed
    in explicitly — recomputing it per-arm would bias the estimate.
    """
    adjusted: FloatArray = y - theta * (covariate - covariate.mean())
    return adjusted
