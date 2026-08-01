"""Group-sequential testing with Lan-DeMets O'Brien-Fleming alpha spending.

Peeking at an experiment K times with a fixed z=1.96 threshold inflates the
false-positive rate to ~14% at K=5. This module computes exact spending
boundaries by numerically propagating the density of the partial-sum process,
so the *overall* type-I error across all looks equals alpha.

Both facts — the inflation and the fix — are verified by Monte Carlo in the
calibration suite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

_NORMAL = NormalDist()

FloatArray = npt.NDArray[np.float64]


def obf_spending(alpha: float, information_fraction: float) -> float:
    """Two-sided Lan-DeMets O'Brien-Fleming spending function.

    Symmetric two-sided form: each side spends the one-sided OBF function at
    alpha/2, i.e. 4 - 4*Phi(z_{alpha/4} / sqrt(t)). Spends exactly alpha at
    t=1 and matches published boundary tables (gsDesign/ldbounds).
    """
    if not 0.0 < information_fraction <= 1.0:
        raise ValueError(f"information fraction out of (0, 1]: {information_fraction}")
    z = _NORMAL.inv_cdf(1.0 - alpha / 4.0)
    return 4.0 * (1.0 - _NORMAL.cdf(z / math.sqrt(information_fraction)))


def _normal_pdf(x: FloatArray) -> FloatArray:
    result: FloatArray = np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return result


def obf_boundaries(
    n_looks: int,
    alpha: float = 0.05,
    grid_points: int = 2001,
    z_max: float = 8.0,
) -> tuple[float, ...]:
    """Symmetric z-scale rejection boundaries for equally spaced looks.

    Propagates the continuing density of S_k = sum of N(0,1) increments on a
    grid; at each look, bisects the boundary so the newly absorbed mass equals
    the spending increment. Total absorbed mass across looks is alpha by
    construction (asserted, loudly).
    """
    if n_looks < 1:
        raise ValueError(f"n_looks must be >= 1: {n_looks}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha out of (0, 1): {alpha}")

    spend = [obf_spending(alpha, (k + 1) / n_looks) for k in range(n_looks)]
    increments = np.diff(np.concatenate(([0.0], np.asarray(spend))))

    half_width = z_max * math.sqrt(n_looks)
    s: FloatArray = np.linspace(-half_width, half_width, grid_points, dtype=np.float64)
    ds = float(s[1] - s[0])
    density = _normal_pdf(s)  # S_1 ~ N(0, 1)

    def outside_mass(f: FloatArray, threshold: float) -> float:
        weight = np.clip((np.abs(s) - threshold) / ds + 0.5, 0.0, 1.0)
        return float(np.sum(f * weight) * ds)

    boundaries: list[float] = []
    total_absorbed = 0.0
    for k in range(1, n_looks + 1):
        if k > 1:
            density = (_normal_pdf(s[:, None] - s[None, :]) @ density) * ds
        target = float(increments[k - 1])
        lo, hi = 0.0, half_width
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if outside_mass(density, mid) > target:
                lo = mid
            else:
                hi = mid
        threshold = (lo + hi) / 2.0
        boundaries.append(threshold / math.sqrt(k))
        total_absorbed += outside_mass(density, threshold)
        weight = np.clip((np.abs(s) - threshold) / ds + 0.5, 0.0, 1.0)
        density = density * (1.0 - weight)

    if abs(total_absorbed - alpha) > 0.001:
        raise RuntimeError(
            f"boundary computation lost calibration: absorbed {total_absorbed:.5f}, "
            f"expected {alpha:.5f} — refusing to return miscalibrated boundaries"
        )
    return tuple(boundaries)


@dataclass(frozen=True)
class SequentialDesign:
    """A K-look group-sequential design with OBF spending."""

    n_looks: int
    alpha: float
    boundaries: tuple[float, ...]

    @classmethod
    def create(cls, n_looks: int, alpha: float = 0.05) -> SequentialDesign:
        return cls(n_looks=n_looks, alpha=alpha, boundaries=obf_boundaries(n_looks, alpha))

    def first_crossing(self, z_at_looks: FloatArray) -> int | None:
        """Index of the first look whose |z| crosses its boundary, else None."""
        if len(z_at_looks) != self.n_looks:
            raise ValueError(f"expected {self.n_looks} z-values, got {len(z_at_looks)}")
        crossed = np.abs(z_at_looks) > np.asarray(self.boundaries)
        hits = np.flatnonzero(crossed)
        return int(hits[0]) if len(hits) else None
