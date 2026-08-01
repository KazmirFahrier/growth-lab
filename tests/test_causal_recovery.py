"""Phase 2 gate: naive is biased, causal recovers truth, broken data blocks.

Each scenario is scored three ways:
  1. the naive estimator must be *demonstrably wrong* (the trap is real),
  2. the causal estimator must land within tolerance of ground truth,
  3. the assumption-violating variant must raise, not return a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from growth_lab.causal import (
    AssumptionViolation,
    diff_in_diff,
    ipw_ate,
    naive_difference,
    naive_ols_slope,
    sharp_rdd,
    t_learner,
    two_stage_least_squares,
)
from growth_lab.causal._regression import add_intercept, ols
from growth_lab.simulator.scenarios import (
    geo_rollout,
    price_instrument,
    promo_email,
    spend_threshold,
)


def test_ols_rejects_nonfinite_inputs() -> None:
    design = add_intercept(np.array([1.0, 2.0, np.inf]))
    with pytest.raises(ValueError, match="must be finite"):
        ols(design, np.array([2.0, 4.0, 6.0]))


# --- DiD --------------------------------------------------------------------


def test_did_naive_cross_section_is_biased() -> None:
    s = geo_rollout()
    post = s.panel[s.panel["post"]]
    naive = naive_difference(post["y"].to_numpy(dtype=float), post["treated"].to_numpy(dtype=bool))
    assert abs(naive - s.true_att) > 0.01, "selection bias vanished; scenario is broken"


def test_did_recovers_truth() -> None:
    s = geo_rollout()
    result = diff_in_diff(s.panel)
    assert abs(result.att - s.true_att) < 0.002
    assert result.ci_low < s.true_att < result.ci_high


def test_did_blocks_on_differential_trends() -> None:
    s = geo_rollout(differential_trends=True)
    with pytest.raises(AssumptionViolation, match="parallel-trends"):
        diff_in_diff(s.panel)


# --- RDD --------------------------------------------------------------------


def test_rdd_naive_above_below_is_biased() -> None:
    s = spend_threshold()
    naive = naive_difference(s.outcome, s.running >= s.cutoff)
    assert abs(naive - s.true_effect) > 0.10


def test_rdd_recovers_truth() -> None:
    s = spend_threshold()
    result = sharp_rdd(s.running, s.outcome, s.cutoff)
    assert abs(result.effect - s.true_effect) < 0.04
    assert result.n_left > 100 and result.n_right > 100


def test_rdd_blocks_on_manipulation() -> None:
    s = spend_threshold(manipulated=True)
    with pytest.raises(AssumptionViolation, match="density"):
        sharp_rdd(s.running, s.outcome, s.cutoff)


# --- IPW --------------------------------------------------------------------


def test_ipw_naive_difference_is_biased() -> None:
    s = promo_email()
    naive = naive_difference(s.outcome, s.treated)
    assert naive - s.true_ate > 0.03, "self-selection bias vanished; scenario is broken"


def test_ipw_recovers_truth() -> None:
    s = promo_email()
    result = ipw_ate(s.covariates, s.treated, s.outcome)
    assert abs(result.ate - s.true_ate) < 0.02
    assert result.max_weighted_smd < 0.10


def test_ipw_blocks_without_overlap() -> None:
    s = promo_email(no_overlap=True)
    with pytest.raises(AssumptionViolation, match="overlap"):
        ipw_ate(s.covariates, s.treated, s.outcome)


# --- IV ---------------------------------------------------------------------


def test_iv_naive_ols_is_sign_flipped() -> None:
    s = price_instrument()
    naive = naive_ols_slope(s.price, s.demand)
    assert naive > 0 > s.true_price_coefficient, "endogeneity bias vanished"


def test_iv_recovers_truth() -> None:
    s = price_instrument()
    result = two_stage_least_squares(s.instrument, s.price, s.demand)
    assert abs(result.estimate - s.true_price_coefficient) < 0.08
    assert result.first_stage_f > 100


def test_iv_blocks_on_weak_instrument() -> None:
    s = price_instrument(weak=True)
    with pytest.raises(AssumptionViolation, match="weak instrument"):
        two_stage_least_squares(s.instrument, s.price, s.demand)


# --- uplift -----------------------------------------------------------------


def test_uplift_finds_heterogeneity() -> None:
    s = promo_email(heterogeneous=True, n=12_000)
    result = t_learner(s.covariates, s.treated, s.outcome)

    corr = float(np.corrcoef(result.cate, s.true_cate)[0, 1])
    assert corr > 0.5, f"CATE barely tracks truth (corr={corr:.2f})"

    responsive = s.covariates[:, 0] < 0  # low engagement: true effect 0.10
    est_responsive = result.segment_effect(responsive)
    est_unresponsive = result.segment_effect(~responsive)
    true_responsive = float(s.true_cate[responsive].mean())
    true_unresponsive = float(s.true_cate[~responsive].mean())
    assert abs(est_responsive - true_responsive) < 0.04
    assert abs(est_unresponsive - true_unresponsive) < 0.04
    assert est_responsive > est_unresponsive + 0.03  # ordering, with margin


def test_uplift_average_matches_ate() -> None:
    s = promo_email(n=10_000)
    result = t_learner(s.covariates, s.treated, s.outcome)
    assert abs(float(result.cate.mean()) - s.true_ate) < 0.02
