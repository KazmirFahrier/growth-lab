"""Unit tests: assignment determinism, SRM, guardrails, readout decisions."""

from __future__ import annotations

import numpy as np
import pytest

from growth_lab.experiments import (
    ArmCounts,
    Decision,
    ExperimentDesign,
    GuardrailSpec,
    Readout,
    assign_arm,
    evaluate,
    non_inferiority_test,
    srm_check,
    to_markdown,
)
from growth_lab.simulator import SimOutput

# --- assignment -------------------------------------------------------------


def test_assignment_deterministic() -> None:
    assert all(
        assign_arm("exp_pricing", uid) == assign_arm("exp_pricing", uid) for uid in range(200)
    )


def test_assignment_balanced() -> None:
    arms = [assign_arm("exp_pricing", uid) for uid in range(40_000)]
    share = sum(arms) / len(arms)
    assert abs(share - 0.5) < 0.01


def test_assignment_independent_across_experiments() -> None:
    a = np.array([assign_arm("exp_one", uid) for uid in range(20_000)])
    b = np.array([assign_arm("exp_two", uid) for uid in range(20_000)])
    assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.02


def test_assignment_rejects_single_arm() -> None:
    with pytest.raises(ValueError):
        assign_arm("exp", 1, n_arms=1)


# --- SRM --------------------------------------------------------------------


def test_srm_passes_when_balanced() -> None:
    assert srm_check(10_050, 9_950).passed


def test_srm_detects_imbalance() -> None:
    result = srm_check(52_500, 47_500)  # 5% skew at n=100k: unmistakable
    assert not result.passed
    assert result.p_value < 1e-6


# --- guardrails -------------------------------------------------------------


def test_guardrail_passes_when_clearly_non_inferior() -> None:
    spec = GuardrailSpec(metric="retention", margin=0.01)
    result = non_inferiority_test(spec, 5000, 50_000, 5010, 50_000)
    assert result.passed


def test_guardrail_fails_on_real_degradation() -> None:
    spec = GuardrailSpec(metric="retention", margin=0.01)
    result = non_inferiority_test(spec, 5000, 50_000, 4200, 50_000)
    assert not result.passed


def test_guardrail_inconclusive_is_not_a_pass() -> None:
    """Tiny sample: no evidence either way -> must NOT pass."""
    spec = GuardrailSpec(metric="retention", margin=0.01)
    result = non_inferiority_test(spec, 10, 100, 10, 100)
    assert not result.passed


# --- readout decisions ------------------------------------------------------

DESIGN = ExperimentDesign(
    name="onboarding_v2",
    primary_metric="trial_to_paid",
    baseline_rate=0.60,
    mde_relative=0.05,
    guardrails=(GuardrailSpec(metric="d7_retention", margin=0.02),),
)


def _arm(n: int, primary: int, guardrail: int) -> ArmCounts:
    return ArmCounts(
        n=n, primary_conversions=primary, guardrail_conversions={"d7_retention": guardrail}
    )


def _readout(control: ArmCounts, treatment: ArmCounts) -> Readout:
    return evaluate(DESIGN, control, treatment)


def test_readout_launch() -> None:
    n = DESIGN.planned_n_per_arm
    readout = _readout(_arm(n, int(n * 0.60), int(n * 0.50)), _arm(n, int(n * 0.64), int(n * 0.50)))
    assert readout.decision is Decision.LAUNCH


def test_readout_no_launch_when_negative() -> None:
    n = DESIGN.planned_n_per_arm
    readout = _readout(_arm(n, int(n * 0.60), int(n * 0.50)), _arm(n, int(n * 0.56), int(n * 0.50)))
    assert readout.decision is Decision.NO_LAUNCH


def test_readout_guardrail_blocks_launch() -> None:
    n = DESIGN.planned_n_per_arm
    readout = _readout(_arm(n, int(n * 0.60), int(n * 0.50)), _arm(n, int(n * 0.64), int(n * 0.44)))
    assert readout.decision is Decision.NO_LAUNCH
    assert "guardrail" in readout.reason


def test_readout_srm_invalidates() -> None:
    n = DESIGN.planned_n_per_arm
    skewed = _arm(int(n * 0.8), int(n * 0.5), int(n * 0.4))
    readout = _readout(_arm(n, int(n * 0.60), int(n * 0.50)), skewed)
    assert readout.decision is Decision.INVALID
    assert readout.primary is None  # no metric is even reported


def test_readout_continue_when_underpowered() -> None:
    n = DESIGN.planned_n_per_arm // 10
    readout = _readout(_arm(n, int(n * 0.60), int(n * 0.55)), _arm(n, int(n * 0.61), int(n * 0.55)))
    assert readout.decision is Decision.CONTINUE


def test_readout_missing_guardrail_counts_raises() -> None:
    n = DESIGN.planned_n_per_arm
    broken = ArmCounts(n=n, primary_conversions=int(n * 0.6), guardrail_conversions={})
    with pytest.raises(KeyError):
        evaluate(DESIGN, broken, broken)


def test_readout_markdown_renders() -> None:
    n = DESIGN.planned_n_per_arm
    readout = _readout(_arm(n, int(n * 0.60), int(n * 0.50)), _arm(n, int(n * 0.64), int(n * 0.50)))
    text = to_markdown(readout)
    assert "Decision: LAUNCH" in text
    assert "d7_retention" in text
    assert "95% CI" in text


# --- end to end on simulated users -----------------------------------------


def test_end_to_end_on_meridian_users(sim: SimOutput) -> None:
    """Assign real simulated users, inject a known lift, expect LAUNCH."""
    rng = np.random.default_rng(43)
    users = sim.signups.head(30_000).copy()
    arms = np.array([assign_arm("meridian_onboarding", uid) for uid in users["user_id"]])
    outcome = users["subscribed"].to_numpy(dtype=bool).copy()
    # inject a clear lift: 10% of treatment non-converters flip to converted
    treat_nonconv = (arms == 1) & ~outcome
    flip = rng.random(len(users)) < 0.10
    outcome = outcome | (treat_nonconv & flip)

    design = ExperimentDesign(
        name="meridian_onboarding",
        primary_metric="trial_to_paid",
        baseline_rate=float(outcome[arms == 0].mean()),
        mde_relative=0.05,
    )
    control = ArmCounts(int((arms == 0).sum()), int(outcome[arms == 0].sum()), {})
    treatment = ArmCounts(int((arms == 1).sum()), int(outcome[arms == 1].sum()), {})
    readout = evaluate(design, control, treatment)
    assert readout.srm.passed
    assert readout.decision in (Decision.LAUNCH, Decision.CONTINUE)
    assert readout.primary is not None and readout.primary.estimate > 0
