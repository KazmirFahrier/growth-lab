"""Budget allocation over fitted MMM response curves.

At steady state a channel's adstocked spend is s / (1 - decay), so expected
daily revenue is beta * sat(s / (1 - decay)). The curves are concave, so the
optimum equalizes marginal revenue across channels — found by bisecting the
shared marginal value (water-filling). No solver dependency, and optimality
is asserted, not assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ChannelResponse:
    name: str
    beta: float
    decay: float
    half_sat: float

    def __post_init__(self) -> None:
        if self.beta < 0 or not 0 <= self.decay < 1 or self.half_sat <= 0:
            raise ValueError(f"invalid response parameters for {self.name!r}")

    def revenue(self, daily_spend: float) -> float:
        a = daily_spend / (1.0 - self.decay)
        return self.beta * a / (a + self.half_sat)

    def marginal(self, daily_spend: float) -> float:
        g = 1.0 / (1.0 - self.decay)
        a = daily_spend * g
        return self.beta * g * self.half_sat / (a + self.half_sat) ** 2

    def spend_at_marginal(self, lam: float) -> float:
        """Spend at which marginal revenue equals `lam` (0 if never)."""
        g = 1.0 / (1.0 - self.decay)
        peak = self.beta * g / self.half_sat  # marginal at zero spend
        if lam >= peak:
            return 0.0
        a = math.sqrt(self.beta * g * self.half_sat / lam) - self.half_sat
        return a / g


@dataclass(frozen=True)
class Allocation:
    channels: tuple[str, ...]
    daily_spend: FloatArray
    expected_daily_revenue: float


def optimal_allocation(
    responses: tuple[ChannelResponse, ...],
    total_daily_budget: float,
    tol: float = 1e-9,
) -> Allocation:
    """Maximize steady-state revenue subject to the budget constraint."""
    if total_daily_budget <= 0:
        raise ValueError(f"budget must be positive: {total_daily_budget}")

    peak = max(r.marginal(0.0) for r in responses)
    lo, hi = 1e-12, peak  # lower lambda -> more total spend

    def spend_at(lam: float) -> FloatArray:
        return np.array([r.spend_at_marginal(lam) for r in responses])

    for _ in range(200):
        lam = math.sqrt(lo * hi)  # geometric bisection: lambda spans decades
        total = float(spend_at(lam).sum())
        if abs(total - total_daily_budget) < tol:
            break
        if total > total_daily_budget:
            lo = lam
        else:
            hi = lam
    spends = spend_at(lam)
    # exact budget: distribute any residual rounding proportionally
    if spends.sum() > 0:
        spends = spends * (total_daily_budget / spends.sum())
    revenue = float(sum(r.revenue(float(s)) for r, s in zip(responses, spends, strict=True)))
    return Allocation(
        channels=tuple(r.name for r in responses),
        daily_spend=spends,
        expected_daily_revenue=revenue,
    )
