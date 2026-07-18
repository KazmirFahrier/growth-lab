"""Observational causal inference with mandatory assumption gates.

This package is sealed: it never imports the simulator or reads ground
truth. Its estimates are scored externally in tests/test_causal_recovery.py,
and its diagnostics are required to *refuse* datasets whose identifying
assumptions fail — a confident wrong answer is worse than no answer.
"""

from growth_lab.causal.did import DidResult, diff_in_diff
from growth_lab.causal.exceptions import AssumptionViolation
from growth_lab.causal.iv import IvResult, two_stage_least_squares
from growth_lab.causal.naive import naive_difference, naive_ols_slope
from growth_lab.causal.propensity import IpwResult, ipw_ate
from growth_lab.causal.rdd import RddResult, sharp_rdd
from growth_lab.causal.uplift import TLearnerResult, t_learner

__all__ = [
    "AssumptionViolation",
    "DidResult",
    "IpwResult",
    "IvResult",
    "RddResult",
    "TLearnerResult",
    "diff_in_diff",
    "ipw_ate",
    "naive_difference",
    "naive_ols_slope",
    "sharp_rdd",
    "t_learner",
    "two_stage_least_squares",
]
