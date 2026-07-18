"""Calibration gate: the simulator's outputs must match its own parameters.

If any of these fail, every downstream estimate in growth-lab is meaningless —
so they fail CI, loudly. Expectations are recomputed from truth.yaml, never
hard-coded, so parameter changes stay covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from growth_lab.simulator import SimOutput, Truth

REL_TOL = 0.05  # deterministic seed; tolerance covers one realized draw


def _realized_ad(sim: SimOutput) -> pd.DataFrame:
    return sim.ad_spend_daily.groupby("channel", as_index=True)[
        ["spend", "impressions", "clicks"]
    ].sum()


def test_ctr_matches_truth(truth: Truth, sim: SimOutput) -> None:
    agg = _realized_ad(sim)
    for ch in truth.channels:
        realized = agg.loc[ch.name, "clicks"] / agg.loc[ch.name, "impressions"]
        expected = ch.expected_ctr(truth.p_high_intent)
        assert abs(realized - expected) / expected < REL_TOL, ch.name


def test_cvr_matches_truth(truth: Truth, sim: SimOutput) -> None:
    agg = _realized_ad(sim)
    signups = sim.signups.groupby("channel")["user_id"].count()
    for ch in truth.channels:
        realized = signups[ch.name] / agg.loc[ch.name, "clicks"]
        expected = ch.expected_cvr(truth.p_high_intent)
        assert abs(realized - expected) / expected < REL_TOL, ch.name


def test_organic_volume_matches_truth(truth: Truth, sim: SimOutput) -> None:
    n_organic = int((sim.signups["channel"] == "organic").sum())
    expected = truth.horizon_days * (
        truth.organic_daily_signups_low + truth.organic_daily_signups_high
    )
    assert abs(n_organic - expected) / expected < REL_TOL


def test_fraud_rate_matches_truth(truth: Truth, sim: SimOutput) -> None:
    realized = float(sim.transactions["is_fraud"].mean())
    assert abs(realized - truth.txn_fraud_rate) / truth.txn_fraud_rate < 0.10


def test_churn_hazard_matches_truth(truth: Truth, sim: SimOutput) -> None:
    # Uncensored geometric lifetimes: mean must be 1/hazard per plan.
    subs = sim.users_latent[sim.users_latent["subscribed"]]
    for plan, hazard in (
        ("basic", truth.monthly_hazard_basic),
        ("pro", truth.monthly_hazard_pro),
    ):
        mean_lifetime = float(subs.loc[subs["plan"] == plan, "lifetime_months"].mean())
        expected = 1.0 / hazard
        assert abs(mean_lifetime - expected) / expected < REL_TOL, plan


def test_trial_to_paid_matches_truth(truth: Truth, sim: SimOutput) -> None:
    latent = sim.users_latent
    for high, expected in (
        (True, truth.trial_to_paid_high),
        (False, truth.trial_to_paid_low),
    ):
        realized = float(latent.loc[latent["latent_high_intent"] == high, "subscribed"].mean())
        assert abs(realized - expected) / expected < REL_TOL, f"high_intent={high}"


def test_dow_seasonality_present(truth: Truth, sim: SimOutput) -> None:
    ad = sim.ad_spend_daily.copy()
    ad["dow"] = pd.to_datetime(ad["date"]).dt.dayofweek
    by_dow = ad.groupby("dow")["impressions"].mean()
    mult = np.asarray(truth.dow_multipliers)
    # Friday (index 4) vs Sunday (index 6) delivery ratio must match config.
    realized_ratio = by_dow[4] / by_dow[6]
    expected_ratio = mult[4] / mult[6]
    assert abs(realized_ratio - expected_ratio) / expected_ratio < REL_TOL


def test_confounder_trap_exists(truth: Truth, sim: SimOutput) -> None:
    """Display signups must over-represent high intent — the attribution trap
    later phases are built to expose. If this fails, Phase 2/3 have nothing
    to demonstrate."""
    merged = sim.signups.merge(sim.users_latent[["user_id", "latent_high_intent"]], on="user_id")
    display_share = float(merged.loc[merged["channel"] == "display", "latent_high_intent"].mean())
    assert display_share > 2.0 * truth.p_high_intent

    # And the trap must make display's naive CVR *look* better than social's,
    # even though its per-intent conversion rates are not uniformly better.
    agg = _realized_ad(sim)
    signups = sim.signups.groupby("channel")["user_id"].count()
    cvr = {ch: signups[ch] / agg.loc[ch, "clicks"] for ch in ("display", "social")}
    assert cvr["display"] > cvr["social"]
