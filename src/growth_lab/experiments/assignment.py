"""Deterministic experiment assignment and sample-ratio-mismatch detection.

Assignment is a pure function of (experiment, unit) — stable across sessions,
no state, no coordination. SRM failure means the data pipeline is broken;
readouts treat it as invalidating, never as a warning.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()

SRM_ALPHA_DEFAULT = 0.001  # conventional: SRM tests run at a strict threshold


def assign_arm(experiment: str, unit_id: int | str, n_arms: int = 2) -> int:
    """Deterministically assign a unit to an arm in [0, n_arms)."""
    if n_arms < 2:
        raise ValueError(f"n_arms must be >= 2: {n_arms}")
    digest = hashlib.sha256(f"{experiment}:{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_arms


@dataclass(frozen=True)
class SrmResult:
    """Outcome of a two-arm sample-ratio-mismatch check."""

    n_control: int
    n_treatment: int
    expected_treatment_share: float
    z: float
    p_value: float
    passed: bool


def srm_check(
    n_control: int,
    n_treatment: int,
    expected_treatment_share: float = 0.5,
    alpha: float = SRM_ALPHA_DEFAULT,
) -> SrmResult:
    """Two-sided exact-mean binomial z-test for assignment imbalance."""
    n = n_control + n_treatment
    if n == 0:
        raise ValueError("no units assigned")
    if not 0.0 < expected_treatment_share < 1.0:
        raise ValueError(f"expected share out of (0, 1): {expected_treatment_share}")

    p = expected_treatment_share
    se = math.sqrt(n * p * (1.0 - p))
    z = (n_treatment - n * p) / se
    p_value = 2.0 * (1.0 - _NORMAL.cdf(abs(z)))
    return SrmResult(
        n_control=n_control,
        n_treatment=n_treatment,
        expected_treatment_share=p,
        z=z,
        p_value=p_value,
        passed=p_value >= alpha,
    )
