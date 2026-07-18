"""Experimentation platform: sizing, assignment, analysis, sequential
testing, guardrails, and automated readouts.

This package is sealed: it never reads simulator ground truth. Its error-rate
promises are verified externally by tests/test_experiment_calibration.py.
"""

from growth_lab.experiments.analysis import (
    TestResult,
    cuped_adjust,
    cuped_theta,
    two_proportion_ztest,
    welch_test,
)
from growth_lab.experiments.assignment import SrmResult, assign_arm, srm_check
from growth_lab.experiments.guardrails import GuardrailResult, GuardrailSpec, non_inferiority_test
from growth_lab.experiments.power import minimum_detectable_effect, required_n_per_arm
from growth_lab.experiments.readout import (
    ArmCounts,
    Decision,
    ExperimentDesign,
    Readout,
    evaluate,
    to_markdown,
)
from growth_lab.experiments.sequential import SequentialDesign, obf_boundaries, obf_spending

__all__ = [
    "ArmCounts",
    "Decision",
    "ExperimentDesign",
    "GuardrailResult",
    "GuardrailSpec",
    "Readout",
    "SequentialDesign",
    "SrmResult",
    "TestResult",
    "assign_arm",
    "cuped_adjust",
    "cuped_theta",
    "evaluate",
    "minimum_detectable_effect",
    "non_inferiority_test",
    "obf_boundaries",
    "obf_spending",
    "required_n_per_arm",
    "srm_check",
    "to_markdown",
    "two_proportion_ztest",
    "welch_test",
]
