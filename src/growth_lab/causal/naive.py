"""The foil: naive estimators that ignore confounding.

These exist so the recovery table can show *how wrong* the obvious answer is
on each scenario — the bias is the demonstration, not a bug.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from growth_lab.causal._regression import FloatArray, add_intercept, ols


def naive_difference(
    outcome: FloatArray, treated: npt.NDArray[np.bool_] | npt.NDArray[np.float64]
) -> float:
    """Raw treated-minus-control mean difference."""
    d = np.asarray(treated, dtype=bool)
    return float(outcome[d].mean() - outcome[~d].mean())


def naive_ols_slope(exposure: FloatArray, outcome: FloatArray) -> float:
    """OLS slope of outcome on exposure, ignoring endogeneity."""
    return float(ols(add_intercept(exposure), outcome).beta[1])
