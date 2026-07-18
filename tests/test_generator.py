from __future__ import annotations

import pandas as pd

from growth_lab.simulator import SimOutput, Truth, simulate


def test_deterministic_for_seed(truth: Truth, sim: SimOutput) -> None:
    again = simulate(truth)
    for name in ("ad_spend_daily", "signups", "transactions", "users_latent"):
        pd.testing.assert_frame_equal(getattr(sim, name), getattr(again, name))


def test_different_seed_differs(truth: Truth, sim: SimOutput) -> None:
    other = simulate(truth, seed=truth.seed_default + 1)
    assert not sim.signups.equals(other.signups)


def test_no_transactions_outside_horizon(truth: Truth, sim: SimOutput) -> None:
    start = pd.Timestamp(truth.start_date)
    end = start + pd.Timedelta(days=truth.horizon_days - 1)
    assert (sim.transactions["txn_date"] >= start).all()
    assert (sim.transactions["txn_date"] <= end).all()


def test_no_transaction_before_signup(sim: SimOutput) -> None:
    merged = sim.transactions.merge(sim.signups[["user_id", "signup_date"]], on="user_id")
    assert (merged["txn_date"] >= merged["signup_date"]).all()


def test_every_subscriber_billed_at_least_once(sim: SimOutput) -> None:
    subscribers = set(sim.signups.loc[sim.signups["subscribed"], "user_id"])
    billed = set(sim.transactions["user_id"])
    assert subscribers == billed


def test_non_subscribers_never_billed(sim: SimOutput) -> None:
    non_subs = sim.signups.loc[~sim.signups["subscribed"], "user_id"]
    assert not set(non_subs) & set(sim.transactions["user_id"])


def test_latent_intent_not_in_visible_tables(sim: SimOutput) -> None:
    """The confounder is latent: it must never appear outside users_latent."""
    for frame in (sim.ad_spend_daily, sim.signups, sim.transactions):
        assert "latent_high_intent" not in frame.columns


def test_user_ids_unique_and_dense(sim: SimOutput) -> None:
    ids = sim.signups["user_id"]
    assert ids.is_unique
    assert ids.min() == 0 and ids.max() == len(ids) - 1
