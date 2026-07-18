"""Guardrail metrics: one-sided non-inferiority tests.

A guardrail passes only when the data affirmatively rules out a degradation
larger than the margin — absence of significance is not a pass. Inconclusive
guardrails block launches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()


@dataclass(frozen=True)
class GuardrailSpec:
    """A conversion-style guardrail with an absolute non-inferiority margin."""

    metric: str
    margin: float  # largest acceptable absolute drop in the rate
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.margin <= 0.0:
            raise ValueError(f"margin must be positive: {self.margin}")


@dataclass(frozen=True)
class GuardrailResult:
    metric: str
    rate_control: float
    rate_treatment: float
    margin: float
    z: float
    p_value: float
    passed: bool


def non_inferiority_test(
    spec: GuardrailSpec,
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
) -> GuardrailResult:
    """H0: p_t <= p_c - margin  vs  H1: p_t > p_c - margin.

    Rejecting H0 (p < alpha) means the metric is demonstrably not degraded
    by more than the margin — only then does the guardrail pass.
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("both arms need units")
    p_c = conversions_control / n_control
    p_t = conversions_treatment / n_treatment

    se = math.sqrt(p_c * (1.0 - p_c) / n_control + p_t * (1.0 - p_t) / n_treatment)
    if se == 0.0:
        raise ValueError(f"guardrail {spec.metric!r}: degenerate data, zero variance")
    z = (p_t - p_c + spec.margin) / se
    p_value = 1.0 - _NORMAL.cdf(z)
    return GuardrailResult(
        metric=spec.metric,
        rate_control=p_c,
        rate_treatment=p_t,
        margin=spec.margin,
        z=z,
        p_value=p_value,
        passed=p_value < spec.alpha,
    )
