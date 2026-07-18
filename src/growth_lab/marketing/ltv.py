"""Subscription LTV from censored billing data.

Meridian is contractual, so the right model is churn-hazard survival, not
BG/NBD. Monthly churn is estimated as a censored-geometric MLE over renewal
trials: every observed billing after the first is a successful renewal, and a
subscriber whose next billing date fell inside the observation window but
never arrived is one observed failure. Users still active at the window edge
are censored — they contribute renewals, not failures. Ignoring censoring
overstates churn; the recovery test checks both hazards against truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PlanLtv:
    plan: str
    n_subscribers: int
    monthly_churn: float
    expected_lifetime_months: float
    monthly_price: float
    ltv: float


@dataclass(frozen=True)
class LtvEstimate:
    plans: tuple[PlanLtv, ...]
    by_channel: pd.DataFrame  # channel, n_paid, avg_ltv

    def plan(self, name: str) -> PlanLtv:
        for p in self.plans:
            if p.plan == name:
                return p
        raise KeyError(f"unknown plan: {name!r}")


def fit_geometric_ltv(
    signups: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_end: pd.Timestamp,
    billing_period_days: int = 30,
) -> LtvEstimate:
    """Censored-geometric churn MLE and LTV per plan and per channel."""
    paying = signups[signups["subscribed"]].copy()
    if paying.empty:
        raise ValueError("no subscribers in the data")

    per_user = (
        transactions.groupby("user_id")
        .agg(n_txns=("txn_id", "count"), avg_amount=("amount", "mean"))
        .reset_index()
    )
    users = paying.merge(per_user, on="user_id", how="left")
    if users["n_txns"].isna().any():
        raise ValueError("subscribed users with no transactions: data is inconsistent")

    # Next billing date after the last observed one; if it was due inside the
    # window and never happened, the user churned. Otherwise censored.
    next_due = users["signup_date"] + pd.to_timedelta(
        users["n_txns"] * billing_period_days, unit="D"
    )
    users["churned"] = next_due <= observation_end

    plan_estimates: list[PlanLtv] = []
    for plan_name, group in users.groupby("plan"):
        renewals = float((group["n_txns"] - 1).sum())
        failures = float(group["churned"].sum())
        if renewals + failures == 0:
            raise ValueError(f"plan {plan_name!r} has no renewal trials to learn from")
        hazard = failures / (failures + renewals)
        if hazard <= 0:
            raise ValueError(f"plan {plan_name!r}: no churn observed, LTV is unbounded")
        lifetime = 1.0 / hazard
        price = float(group["avg_amount"].mean())
        plan_estimates.append(
            PlanLtv(
                plan=str(plan_name),
                n_subscribers=len(group),
                monthly_churn=hazard,
                expected_lifetime_months=lifetime,
                monthly_price=price,
                ltv=price * lifetime,
            )
        )

    ltv_by_plan = {p.plan: p.ltv for p in plan_estimates}
    users["ltv"] = users["plan"].map(ltv_by_plan)
    by_channel = (
        users.groupby("channel")
        .agg(n_paid=("user_id", "count"), avg_ltv=("ltv", "mean"))
        .reset_index()
        .sort_values("avg_ltv", ascending=False)
        .reset_index(drop=True)
    )
    return LtvEstimate(plans=tuple(plan_estimates), by_channel=by_channel)
