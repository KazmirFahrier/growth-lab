"""Inverse-propensity weighting with overlap and balance gates.

Two ways to fail, both fatal: propensities piling up at 0/1 (no overlap —
some units effectively never/always treated, so no counterfactual exists),
and covariates still imbalanced *after* weighting (the propensity model
didn't do its one job).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from growth_lab.causal._regression import FloatArray, add_intercept, fit_logistic, sigmoid
from growth_lab.causal.exceptions import AssumptionViolation


@dataclass(frozen=True)
class IpwResult:
    ate: float
    se: float
    propensity_min: float
    propensity_max: float
    max_weighted_smd: float


def _weighted_mean_and_se(y: FloatArray, w: FloatArray) -> tuple[float, float]:
    total = float(w.sum())
    mean = float((w * y).sum()) / total
    var = float((w**2 * (y - mean) ** 2).sum()) / total**2
    return mean, math.sqrt(var)


def ipw_ate(
    covariates: FloatArray,
    treated: npt.NDArray[np.bool_] | npt.NDArray[np.float64],
    outcome: FloatArray,
    overlap_eps: float = 0.02,
    max_extreme_share: float = 0.02,
    balance_threshold: float = 0.10,
) -> IpwResult:
    """Hajek IPW estimate of the ATE with mandatory diagnostics."""
    d = np.asarray(treated, dtype=bool)
    if not 0 < d.sum() < len(d):
        raise ValueError("need both treated and control units")

    design = add_intercept(covariates)
    beta = fit_logistic(design, d.astype(np.float64))
    e = sigmoid(design @ beta)

    extreme_share = float(((e < overlap_eps) | (e > 1.0 - overlap_eps)).mean())
    if extreme_share > max_extreme_share:
        raise AssumptionViolation(
            f"overlap failure: {extreme_share:.1%} of units have propensity "
            f"outside [{overlap_eps}, {1 - overlap_eps}]; for these units no "
            "counterfactual exists and IPW estimates would be dominated by "
            "a handful of extreme weights"
        )

    w_treated = d / e
    w_control = (~d) / (1.0 - e)

    # Balance gate: weighted standardized mean differences must be small.
    smds = []
    for j in range(covariates.shape[1]):
        x = covariates[:, j]
        m_t = float((w_treated * x).sum() / w_treated.sum())
        m_c = float((w_control * x).sum() / w_control.sum())
        pooled_sd = float(np.std(x))
        smds.append(abs(m_t - m_c) / pooled_sd if pooled_sd > 0 else 0.0)
    max_smd = max(smds)
    if max_smd > balance_threshold:
        raise AssumptionViolation(
            f"covariate balance failure after weighting (max SMD "
            f"{max_smd:.3f} > {balance_threshold}): the propensity model does "
            "not remove observed confounding"
        )

    mean_t, se_t = _weighted_mean_and_se(outcome, w_treated)
    mean_c, se_c = _weighted_mean_and_se(outcome, w_control)
    return IpwResult(
        ate=mean_t - mean_c,
        se=math.sqrt(se_t**2 + se_c**2),
        propensity_min=float(e.min()),
        propensity_max=float(e.max()),
        max_weighted_smd=max_smd,
    )
