"""Difference-in-differences with a mandatory parallel-trends placebo.

The placebo runs first: a DiD on the two halves of the *pre* period, where
the true effect is zero by construction. If it "finds" an effect, the trends
are not parallel and the estimator refuses to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from growth_lab.causal.exceptions import AssumptionViolation
from growth_lab.experiments.analysis import TestResult, welch_test


@dataclass(frozen=True)
class DidResult:
    att: float
    ci_low: float
    ci_high: float
    p_value: float
    placebo_p_value: float
    n_treated_units: int
    n_control_units: int


def _unit_delta_test(
    before: pd.DataFrame,
    after: pd.DataFrame,
    treated_by_unit: pd.Series,
    unit_col: str,
    outcome_col: str,
) -> TestResult:
    """Welch test on per-unit (after - before) changes, treated vs control.

    Aggregating to unit-level deltas clusters correctly by unit.
    """
    mean_before = before.groupby(unit_col)[outcome_col].mean()
    mean_after = after.groupby(unit_col)[outcome_col].mean()
    delta = (mean_after - mean_before).dropna()
    if len(delta) < 4:
        raise ValueError("too few units with both periods observed")
    is_treated = treated_by_unit.reindex(delta.index).astype(bool)
    return welch_test(
        delta[~is_treated].to_numpy(dtype=float),
        delta[is_treated].to_numpy(dtype=float),
    )


def diff_in_diff(
    panel: pd.DataFrame,
    unit_col: str = "geo",
    time_col: str = "day",
    outcome_col: str = "y",
    treated_col: str = "treated",
    post_col: str = "post",
    pretrend_alpha: float = 0.05,
) -> DidResult:
    """Two-group DiD on a unit x time panel."""
    missing = {unit_col, time_col, outcome_col, treated_col, post_col} - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")

    treated_by_unit = panel.groupby(unit_col)[treated_col].first()
    pre = panel[~panel[post_col]]
    post = panel[panel[post_col]]
    if pre.empty or post.empty:
        raise ValueError("panel needs both pre and post periods")

    # Placebo: DiD across the two halves of the pre period (truth is zero).
    median_pre_time = pre[time_col].median()
    placebo = _unit_delta_test(
        pre[pre[time_col] <= median_pre_time],
        pre[pre[time_col] > median_pre_time],
        treated_by_unit,
        unit_col,
        outcome_col,
    )
    if placebo.p_value < pretrend_alpha:
        raise AssumptionViolation(
            f"parallel-trends placebo failed (placebo ATT "
            f"{placebo.estimate:+.5f}, p={placebo.p_value:.2e}): treated and "
            "control units were already diverging before treatment; DiD is not "
            "identified on this panel"
        )

    main = _unit_delta_test(pre, post, treated_by_unit, unit_col, outcome_col)
    return DidResult(
        att=main.estimate,
        ci_low=main.ci_low,
        ci_high=main.ci_high,
        p_value=main.p_value,
        placebo_p_value=placebo.p_value,
        n_treated_units=int(treated_by_unit.sum()),
        n_control_units=int((~treated_by_unit.astype(bool)).sum()),
    )
