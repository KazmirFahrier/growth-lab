"""Vectorized event generation for the Meridian subscription business.

Produces four tables:
  * ad_spend_daily  — day x channel delivery (what an ad platform would export)
  * signups         — row-level signups (paid channels + organic)
  * transactions    — row-level subscription billing events
  * users_latent    — HIDDEN: latent intent + uncensored lifetimes, for scoring only

The latent table never reaches the warehouse marts; it exists so the
calibration/scoring harness can compare estimates against truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from growth_lab.simulator.params import Truth

BILLING_PERIOD_DAYS = 30
ORGANIC = "organic"


@dataclass(frozen=True)
class SimOutput:
    """All tables produced by one simulation run."""

    ad_spend_daily: pd.DataFrame
    signups: pd.DataFrame
    transactions: pd.DataFrame
    users_latent: pd.DataFrame


def _daily_signup_rows(
    rng: np.random.Generator,
    truth: Truth,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate ad delivery and channel/organic signups.

    Returns (ad_spend_daily, signup_rows) where signup_rows has one row per
    signup with its (hidden) intent flag.
    """
    horizon = truth.horizon_days
    day_idx = np.arange(horizon)
    dow = np.array([(truth.start_date + timedelta(days=int(d))).weekday() for d in day_idx])
    dow_mult = np.asarray(truth.dow_multipliers, dtype=np.float64)[dow]

    ad_rows: list[pd.DataFrame] = []
    signup_parts: list[pd.DataFrame] = []

    for ch in truth.channels:
        base_impressions = ch.daily_spend / ch.cpm * 1000.0
        impressions = rng.poisson(base_impressions * dow_mult)
        share_high = ch.high_impression_share(truth.p_high_intent)
        impr_high = rng.binomial(impressions, share_high)
        impr_low = impressions - impr_high

        clicks_high = rng.binomial(impr_high, ch.ctr_high)
        clicks_low = rng.binomial(impr_low, ch.ctr_low)
        signups_high = rng.binomial(clicks_high, ch.cvr_high)
        signups_low = rng.binomial(clicks_low, ch.cvr_low)

        ad_rows.append(
            pd.DataFrame(
                {
                    "day_idx": day_idx,
                    "channel": ch.name,
                    "spend": np.full(horizon, ch.daily_spend),
                    "impressions": impressions,
                    "clicks": clicks_high + clicks_low,
                }
            )
        )
        for counts, intent in ((signups_high, True), (signups_low, False)):
            n = int(counts.sum())
            if n == 0:
                continue
            signup_parts.append(
                pd.DataFrame(
                    {
                        "day_idx": np.repeat(day_idx, counts),
                        "channel": ch.name,
                        "latent_high_intent": intent,
                    }
                )
            )

    # Organic signups: Poisson arrivals per intent group, no spend behind them.
    for mu, intent in (
        (truth.organic_daily_signups_high, True),
        (truth.organic_daily_signups_low, False),
    ):
        counts = rng.poisson(np.full(horizon, mu))
        signup_parts.append(
            pd.DataFrame(
                {
                    "day_idx": np.repeat(day_idx, counts),
                    "channel": ORGANIC,
                    "latent_high_intent": intent,
                }
            )
        )

    ad_spend_daily = pd.concat(ad_rows, ignore_index=True)
    signup_rows = pd.concat(signup_parts, ignore_index=True)
    # Shuffle then re-sort by day so user_id ordering carries no intent signal.
    signup_rows = (
        signup_rows.sample(frac=1.0, random_state=rng.integers(0, 2**31 - 1))
        .sort_values("day_idx", kind="stable")
        .reset_index(drop=True)
    )
    return ad_spend_daily, signup_rows


def _subscriptions(
    rng: np.random.Generator, truth: Truth, signup_rows: pd.DataFrame
) -> pd.DataFrame:
    """Draw trial->paid conversion, plan choice, and uncensored lifetime."""
    n = len(signup_rows)
    high = signup_rows["latent_high_intent"].to_numpy(dtype=bool)

    p_paid = np.where(high, truth.trial_to_paid_high, truth.trial_to_paid_low)
    subscribed = rng.random(n) < p_paid

    p_pro = np.where(high, truth.plan_pro_share_high, truth.plan_pro_share_low)
    pro = rng.random(n) < p_pro
    plan = np.where(pro, "pro", "basic").astype(object)
    plan[~subscribed] = None

    hazard = np.where(pro, truth.monthly_hazard_pro, truth.monthly_hazard_basic)
    lifetime = rng.geometric(hazard)  # months, >= 1
    lifetime_months = np.where(subscribed, lifetime, 0)

    out = signup_rows.copy()
    out["user_id"] = np.arange(n)
    out["subscribed"] = subscribed
    out["plan"] = plan
    out["lifetime_months"] = lifetime_months
    return out


def _transactions(rng: np.random.Generator, truth: Truth, users: pd.DataFrame) -> pd.DataFrame:
    """Monthly billing events, censored at the horizon, with fraud flags."""
    subs = users[users["subscribed"]].reset_index(drop=True)
    signup_day = subs["day_idx"].to_numpy(dtype=np.int64)
    lifetime = subs["lifetime_months"].to_numpy(dtype=np.int64)

    # Billing k = 0..n_txns-1 at signup_day + 30k, while inside the horizon.
    runway = (truth.horizon_days - 1 - signup_day) // BILLING_PERIOD_DAYS + 1
    n_txns = np.minimum(lifetime, runway)

    total = int(n_txns.sum())
    user_rep = np.repeat(subs["user_id"].to_numpy(), n_txns)
    plan_rep = np.repeat(subs["plan"].to_numpy(), n_txns)
    day_rep = np.repeat(signup_day, n_txns)
    # offset within each user's run: 0, 1, ..., n_txns[i]-1
    offsets = np.arange(total) - np.repeat(np.cumsum(n_txns) - n_txns, n_txns)
    txn_day = day_rep + offsets * BILLING_PERIOD_DAYS

    amount = np.where(plan_rep == "pro", truth.price_pro, truth.price_basic)
    is_fraud = rng.random(total) < truth.txn_fraud_rate

    return pd.DataFrame(
        {
            "txn_id": np.arange(total),
            "user_id": user_rep,
            "day_idx": txn_day,
            "amount": amount,
            "is_fraud": is_fraud,
        }
    )


def simulate(truth: Truth, seed: int | None = None) -> SimOutput:
    """Run one full simulation. Deterministic for a given (truth, seed)."""
    rng = np.random.default_rng(truth.seed_default if seed is None else seed)

    ad_spend_daily, signup_rows = _daily_signup_rows(rng, truth)
    users = _subscriptions(rng, truth, signup_rows)
    transactions = _transactions(rng, truth, users)

    def to_date(frame: pd.DataFrame, col: str) -> pd.Series:
        base = pd.Timestamp(truth.start_date)
        return base + pd.to_timedelta(frame[col], unit="D")

    ad_spend_daily = ad_spend_daily.assign(date=to_date(ad_spend_daily, "day_idx")).drop(
        columns="day_idx"
    )
    signups = users.assign(signup_date=to_date(users, "day_idx"))[
        ["user_id", "signup_date", "channel", "subscribed", "plan"]
    ]
    txns = transactions.assign(txn_date=to_date(transactions, "day_idx"))[
        ["txn_id", "user_id", "txn_date", "amount", "is_fraud"]
    ]
    users_latent = users[["user_id", "latent_high_intent", "subscribed", "plan", "lifetime_months"]]

    return SimOutput(
        ad_spend_daily=ad_spend_daily,
        signups=signups,
        transactions=txns,
        users_latent=users_latent,
    )
