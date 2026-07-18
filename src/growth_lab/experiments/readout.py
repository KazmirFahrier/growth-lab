"""Automated experiment readout: from arm counts to a launch decision.

Decision rules are explicit, ordered, and each has a dedicated test:

1. SRM fails                                        -> INVALID
2. primary significant and negative                 -> NO_LAUNCH
3. primary significant positive, guardrails cleared -> LAUNCH
4. underpowered (n < planned)                       -> CONTINUE
5. any guardrail not cleared                        -> NO_LAUNCH (inconclusive blocks)
6. otherwise                                        -> NO_LAUNCH (powered, no effect)

Note rule 4 before rule 5: an underpowered experiment with an inconclusive
guardrail should collect more data, not be killed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from growth_lab.experiments.analysis import TestResult, two_proportion_ztest
from growth_lab.experiments.assignment import SrmResult, srm_check
from growth_lab.experiments.guardrails import GuardrailResult, GuardrailSpec, non_inferiority_test
from growth_lab.experiments.power import required_n_per_arm


class Decision(Enum):
    LAUNCH = "launch"
    NO_LAUNCH = "no_launch"
    CONTINUE = "continue"
    INVALID = "invalid"


@dataclass(frozen=True)
class ExperimentDesign:
    """Pre-registered design: metric, effect size, and guardrails."""

    name: str
    primary_metric: str
    baseline_rate: float
    mde_relative: float
    alpha: float = 0.05
    power: float = 0.8
    guardrails: tuple[GuardrailSpec, ...] = ()

    @property
    def planned_n_per_arm(self) -> int:
        return required_n_per_arm(self.baseline_rate, self.mde_relative, self.alpha, self.power)


@dataclass(frozen=True)
class ArmCounts:
    """Observed counts for one arm."""

    n: int
    primary_conversions: int
    guardrail_conversions: dict[str, int]


@dataclass(frozen=True)
class Readout:
    design: ExperimentDesign
    srm: SrmResult
    primary: TestResult | None
    guardrails: tuple[GuardrailResult, ...]
    decision: Decision
    reason: str


def evaluate(design: ExperimentDesign, control: ArmCounts, treatment: ArmCounts) -> Readout:
    """Apply the decision rules to observed counts."""
    srm = srm_check(control.n, treatment.n)
    if not srm.passed:
        return Readout(
            design=design,
            srm=srm,
            primary=None,
            guardrails=(),
            decision=Decision.INVALID,
            reason=(
                f"sample ratio mismatch (p={srm.p_value:.2e}): assignment or logging is "
                "broken; no metric from this experiment can be trusted"
            ),
        )

    primary = two_proportion_ztest(
        control.primary_conversions, control.n,
        treatment.primary_conversions, treatment.n,
        alpha=design.alpha,
    )

    guardrail_results = []
    for spec in design.guardrails:
        for arm, label in ((control, "control"), (treatment, "treatment")):
            if spec.metric not in arm.guardrail_conversions:
                raise KeyError(f"guardrail {spec.metric!r} missing from {label} counts")
        guardrail_results.append(
            non_inferiority_test(
                spec,
                control.guardrail_conversions[spec.metric], control.n,
                treatment.guardrail_conversions[spec.metric], treatment.n,
            )
        )
    guardrails = tuple(guardrail_results)

    guardrails_clear = all(g.passed for g in guardrails)
    underpowered = min(control.n, treatment.n) < design.planned_n_per_arm

    if primary.significant and primary.estimate < 0:
        decision, reason = Decision.NO_LAUNCH, "primary metric significantly negative"
    elif primary.significant and guardrails_clear:
        decision = Decision.LAUNCH
        reason = "primary metric significantly positive, guardrails clear"
    elif underpowered:
        decision, reason = (
            Decision.CONTINUE,
            f"underpowered: {min(control.n, treatment.n)} of "
            f"{design.planned_n_per_arm} planned units per arm",
        )
    elif not guardrails_clear:
        failed = [g.metric for g in guardrails if not g.passed]
        decision, reason = (
            Decision.NO_LAUNCH,
            f"guardrail(s) not cleared: {', '.join(failed)} (inconclusive blocks launch)",
        )
    else:
        decision, reason = (
            Decision.NO_LAUNCH,
            "fully powered and no significant effect on the primary metric",
        )
    return Readout(
        design=design,
        srm=srm,
        primary=primary,
        guardrails=guardrails,
        decision=decision,
        reason=reason,
    )


def to_markdown(readout: Readout) -> str:
    """Render a readout as a stakeholder-facing markdown summary."""
    d = readout.design
    lines = [
        f"# Experiment readout: {d.name}",
        "",
        f"**Decision: {readout.decision.value.upper()}** — {readout.reason}",
        "",
        f"SRM check: {'pass' if readout.srm.passed else 'FAIL'} "
        f"(control n={readout.srm.n_control:,}, treatment n={readout.srm.n_treatment:,}, "
        f"p={readout.srm.p_value:.3f})",
        "",
    ]
    if readout.primary is not None:
        p = readout.primary
        lines += [
            f"## Primary: {d.primary_metric}",
            "",
            f"- lift: {p.estimate:+.4f} (95% CI [{p.ci_low:+.4f}, {p.ci_high:+.4f}])",
            f"- p-value: {p.p_value:.4f} at alpha={d.alpha}",
            f"- planned n/arm: {d.planned_n_per_arm:,} "
            f"(baseline {d.baseline_rate:.3f}, MDE {d.mde_relative:+.1%})",
            "",
        ]
    if readout.guardrails:
        lines += ["## Guardrails", ""]
        lines += [
            f"- {g.metric}: {'pass' if g.passed else 'NOT CLEARED'} "
            f"(control {g.rate_control:.4f} -> treatment {g.rate_treatment:.4f}, "
            f"margin {g.margin}, p={g.p_value:.4f})"
            for g in readout.guardrails
        ]
        lines.append("")
    return "\n".join(lines)
