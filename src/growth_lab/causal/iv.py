"""Two-stage least squares with a first-stage strength gate.

A weak instrument doesn't degrade 2SLS gracefully — it produces confident
garbage. First-stage F below the conventional threshold refuses to estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from growth_lab.causal._regression import FloatArray, add_intercept, ols
from growth_lab.causal.exceptions import AssumptionViolation

WEAK_INSTRUMENT_F = 10.0


@dataclass(frozen=True)
class IvResult:
    estimate: float
    se: float
    first_stage_f: float
    n: int


def two_stage_least_squares(
    instrument: FloatArray,
    exposure: FloatArray,
    outcome: FloatArray,
    f_threshold: float = WEAK_INSTRUMENT_F,
) -> IvResult:
    """2SLS for a single endogenous exposure and a single instrument."""
    n = len(instrument)
    if not (len(exposure) == len(outcome) == n):
        raise ValueError("instrument, exposure, and outcome must be the same length")

    first_stage = ols(add_intercept(instrument), exposure)
    f_stat = (float(first_stage.beta[1]) / float(first_stage.se[1])) ** 2
    if f_stat < f_threshold:
        raise AssumptionViolation(
            f"weak instrument: first-stage F={f_stat:.2f} < {f_threshold}; "
            "2SLS with a weak instrument yields confident garbage, refusing"
        )

    z_c = instrument - instrument.mean()
    cov_zx = float((z_c * (exposure - exposure.mean())).mean())
    cov_zy = float((z_c * (outcome - outcome.mean())).mean())
    estimate = cov_zy / cov_zx

    intercept = float(outcome.mean()) - estimate * float(exposure.mean())
    residuals = outcome - intercept - estimate * exposure
    var_u = float(residuals @ residuals) / (n - 2)
    var_z = float(z_c @ z_c) / n
    se = math.sqrt(var_u * var_z / (n * cov_zx**2))
    return IvResult(estimate=estimate, se=se, first_stage_f=f_stat, n=n)
