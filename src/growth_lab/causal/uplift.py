"""Uplift modeling: T-learner for heterogeneous treatment effects.

Answers "who responds", not just "does it work" — the difference between
sending a promo to everyone and sending it to the users it actually moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from growth_lab.causal._regression import FloatArray, add_intercept, fit_logistic, sigmoid


@dataclass(frozen=True)
class TLearnerResult:
    cate: FloatArray  # per-unit conditional average treatment effect

    def segment_effect(self, mask: npt.NDArray[np.bool_]) -> float:
        """Average predicted effect within a segment."""
        return float(self.cate[np.asarray(mask, dtype=bool)].mean())


def t_learner(
    covariates: FloatArray,
    treated: npt.NDArray[np.bool_] | npt.NDArray[np.float64],
    outcome: FloatArray,
) -> TLearnerResult:
    """Fit separate outcome models per arm; CATE is their prediction gap."""
    d = np.asarray(treated, dtype=bool)
    if d.sum() < 30 or (~d).sum() < 30:
        raise ValueError("need at least 30 units in each arm to fit per-arm models")

    design = add_intercept(covariates)
    beta_treated = fit_logistic(design[d], outcome[d])
    beta_control = fit_logistic(design[~d], outcome[~d])
    cate = sigmoid(design @ beta_treated) - sigmoid(design @ beta_control)
    return TLearnerResult(cate=cate)
