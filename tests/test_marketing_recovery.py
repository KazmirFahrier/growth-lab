"""Phase 3 gate: marketing measurement scored against sealed truth.

MMM must recover carryover and ROAS; attribution's over-crediting of the
retargeting channel must be exposed and quantified; LTV hazards must match
the churn parameters; the budget optimizer must provably beat naive
allocations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from growth_lab.marketing import (
    ChannelResponse,
    bootstrap_roas_ci,
    fit_geometric_ltv,
    fit_mmm,
    last_touch,
    linear_touch,
    markov_removal,
    optimal_allocation,
)
from growth_lab.simulator import SimOutput, Truth
from growth_lab.simulator.scenarios import journey_paths, mmm_market

# --- MMM --------------------------------------------------------------------


@pytest.fixture(scope="module")
def mmm_scenario():  # type: ignore[no-untyped-def]
    return mmm_market()


@pytest.fixture(scope="module")
def mmm_fit(mmm_scenario):  # type: ignore[no-untyped-def]
    return fit_mmm(
        mmm_scenario.spend,
        mmm_scenario.revenue,
        mmm_scenario.day_of_week,
        mmm_scenario.channels,
    )


def test_mmm_fits_to_the_noise_floor(mmm_scenario, mmm_fit) -> None:  # type: ignore[no-untyped-def]
    """The honest fit-quality bar: residual sd must approach the true noise
    sd, not an arbitrary R2 number (R2's ceiling is set by the DGP)."""
    residual_sd = float((mmm_scenario.revenue - mmm_fit.fitted).std())
    assert residual_sd < 1.15 * mmm_scenario.true_noise_sd
    assert mmm_fit.r_squared > 0.8


def test_mmm_recovers_adstock_decay(mmm_scenario, mmm_fit) -> None:  # type: ignore[no-untyped-def]
    # decay is on a 0.05 grid; small channels identify it weakly, so the
    # bar is "right carryover regime", not the exact grid point
    for c, name in enumerate(mmm_scenario.channels):
        assert abs(mmm_fit.decay[c] - mmm_scenario.true_decay[c]) <= 0.155, name


def test_mmm_recovers_roas(mmm_scenario, mmm_fit) -> None:  # type: ignore[no-untyped-def]
    estimated = mmm_fit.roas(mmm_scenario.spend)
    for c, name in enumerate(mmm_scenario.channels):
        true_val = mmm_scenario.true_roas[c]
        rel_err = abs(estimated[c] - true_val) / true_val
        assert rel_err < 0.20, f"{name}: est {estimated[c]:.3f} vs true {true_val:.3f}"


def test_mmm_contribution_decomposition_adds_up(mmm_scenario, mmm_fit) -> None:  # type: ignore[no-untyped-def]
    reconstructed = (
        mmm_fit.base
        + mmm_fit.dow_effect[mmm_scenario.day_of_week]
        + mmm_fit.contribution.sum(axis=1)
    )
    assert np.allclose(reconstructed, mmm_fit.fitted, atol=1e-6)


def test_mmm_bootstrap_ci_is_informative_and_honest(mmm_scenario, mmm_fit) -> None:  # type: ignore[no-untyped-def]
    """Conditional-on-shape bootstrap intervals are known to undercover
    (they ignore adstock/saturation selection uncertainty — documented in
    bootstrap_roas_ci). The bar: intervals must be tight enough to be useful,
    truth must sit within CI plus a 10% model-selection margin for every
    channel, and at least half must be strictly covered."""
    lower, upper = bootstrap_roas_ci(
        mmm_fit,
        mmm_scenario.spend,
        mmm_scenario.revenue,
        mmm_scenario.day_of_week,
        n_boot=200,
    )
    estimated = mmm_fit.roas(mmm_scenario.spend)
    strictly_covered = 0
    for c, name in enumerate(mmm_scenario.channels):
        width = upper[c] - lower[c]
        assert 0 < width < 0.5 * estimated[c], f"{name}: uninformative CI"
        margin = 0.10 * estimated[c]
        assert lower[c] - margin <= mmm_scenario.true_roas[c] <= upper[c] + margin, name
        strictly_covered += bool(lower[c] <= mmm_scenario.true_roas[c] <= upper[c])
    assert strictly_covered >= 2


# --- attribution ------------------------------------------------------------


@pytest.fixture(scope="module")
def journeys():  # type: ignore[no-untyped-def]
    return journey_paths()


def test_last_touch_over_credits_retargeting(journeys) -> None:  # type: ignore[no-untyped-def]
    credit = last_touch(journeys.paths, journeys.converted)
    truth = journeys.true_incremental["display"]
    assert credit["display"] > 3.0 * truth, (
        f"last-touch gives display {credit['display']:.0f} vs true {truth:.0f} — "
        "the harvesting trap should be dramatic"
    )


def test_markov_less_wrong_than_last_touch_on_display(journeys) -> None:  # type: ignore[no-untyped-def]
    lt = last_touch(journeys.paths, journeys.converted)
    mk = markov_removal(journeys.paths, journeys.converted, journeys.channels)
    truth = journeys.true_incremental["display"]
    assert abs(mk["display"] - truth) < abs(lt["display"] - truth)


def test_attribution_credits_sum_to_conversions(journeys) -> None:  # type: ignore[no-untyped-def]
    total = float(journeys.converted.sum())
    for credits in (
        last_touch(journeys.paths, journeys.converted),
        linear_touch(journeys.paths, journeys.converted),
        markov_removal(journeys.paths, journeys.converted, journeys.channels),
    ):
        assert sum(credits.values()) == pytest.approx(total, rel=1e-9)


def test_no_attribution_model_measures_incrementality(journeys) -> None:  # type: ignore[no-untyped-def]
    """The honest headline: even the best attribution model here is not an
    incrementality estimate. Every model over-credits display."""
    truth = journeys.true_incremental["display"]
    for model in (last_touch, linear_touch):
        assert model(journeys.paths, journeys.converted)["display"] > 1.5 * truth
    mk = markov_removal(journeys.paths, journeys.converted, journeys.channels)
    assert mk["display"] > 1.5 * truth


# --- LTV --------------------------------------------------------------------


def test_ltv_recovers_churn_hazards(truth: Truth, sim: SimOutput) -> None:
    observation_end = pd.Timestamp(truth.start_date) + pd.Timedelta(days=truth.horizon_days - 1)
    estimate = fit_geometric_ltv(sim.signups, sim.transactions, observation_end)
    basic = estimate.plan("basic")
    pro = estimate.plan("pro")
    assert abs(basic.monthly_churn - truth.monthly_hazard_basic) < 0.015
    assert abs(pro.monthly_churn - truth.monthly_hazard_pro) < 0.010
    assert pro.ltv > basic.ltv  # higher price AND lower churn
    assert basic.ltv == pytest.approx(basic.monthly_price / basic.monthly_churn, rel=1e-9)


def test_ltv_by_channel_is_complete(truth: Truth, sim: SimOutput) -> None:
    observation_end = pd.Timestamp(truth.start_date) + pd.Timedelta(days=truth.horizon_days - 1)
    estimate = fit_geometric_ltv(sim.signups, sim.transactions, observation_end)
    expected = {"search", "social", "display", "video", "organic"}
    assert set(estimate.by_channel["channel"]) == expected
    assert (estimate.by_channel["avg_ltv"] > 0).all()


# --- budget optimizer -------------------------------------------------------


RESPONSES = (
    ChannelResponse("search", beta=4000.0, decay=0.2, half_sat=900.0),
    ChannelResponse("social", beta=2500.0, decay=0.5, half_sat=600.0),
    ChannelResponse("display", beta=1200.0, decay=0.7, half_sat=400.0),
    ChannelResponse("video", beta=800.0, decay=0.4, half_sat=300.0),
)


def test_optimizer_exhausts_budget() -> None:
    allocation = optimal_allocation(RESPONSES, total_daily_budget=1800.0)
    assert float(allocation.daily_spend.sum()) == pytest.approx(1800.0, rel=1e-9)
    assert (allocation.daily_spend >= 0).all()


def test_optimizer_equalizes_marginal_returns() -> None:
    allocation = optimal_allocation(RESPONSES, total_daily_budget=1800.0)
    marginals = [
        r.marginal(float(s))
        for r, s in zip(RESPONSES, allocation.daily_spend, strict=True)
        if s > 1e-6
    ]
    assert max(marginals) - min(marginals) < 0.01 * max(marginals)


def test_optimizer_beats_proportional_and_uniform() -> None:
    budget = 1800.0
    allocation = optimal_allocation(RESPONSES, total_daily_budget=budget)

    def revenue_of(spends: list[float]) -> float:
        return sum(r.revenue(s) for r, s in zip(RESPONSES, spends, strict=True))

    uniform = [budget / 4] * 4
    assert allocation.expected_daily_revenue > revenue_of(uniform)
    historical = [800.0, 500.0, 300.0, 200.0]  # the sim's static plan
    assert allocation.expected_daily_revenue > revenue_of(historical)


def test_optimizer_matches_brute_force_on_two_channels() -> None:
    two = RESPONSES[:2]
    budget = 1000.0
    allocation = optimal_allocation(two, total_daily_budget=budget)
    grid = np.linspace(0.0, budget, 2001)
    best = max(
        two[0].revenue(float(s)) + two[1].revenue(float(budget - s)) for s in grid
    )
    assert allocation.expected_daily_revenue == pytest.approx(best, rel=1e-4)
