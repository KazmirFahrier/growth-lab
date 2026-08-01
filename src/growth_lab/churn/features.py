"""Temporal feature engineering for churn prediction.

All features are derived exclusively from the warehouse (raw.* tables). The
hidden truth (sim_hidden.users_latent) is never accessed — the no-truth seal
is respected by construction.

Temporal split strategy:
  * Observation window: days [0, cutoff) from each user's signup
  * Prediction window: days [cutoff, cutoff + horizon) from signup
  * Label = 1 if zero transactions in the prediction window (= churned)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Default temporal split: 120 days observation, 30-day churn horizon.
DEFAULT_CUTOFF_DAYS = 120
DEFAULT_HORIZON_DAYS = 30


@dataclass(frozen=True)
class TrainingSet:
    """Leakage-safe training data with temporal split metadata."""

    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    cutoff_days: int
    horizon_days: int
    n_users: int
    churn_rate: float


def build_training_set(
    db_path: Path,
    cutoff_days: int = DEFAULT_CUTOFF_DAYS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> TrainingSet:
    """Build a leakage-safe training set from the warehouse.

    For every subscribed user, computes features from the observation window
    [signup, signup + cutoff_days) and labels from [signup + cutoff_days,
    signup + cutoff_days + horizon_days). Users with < cutoff_days of history
    are excluded.
    """
    con = duckdb.connect(str(db_path), read_only=True)

    try:
        query = f"""
        WITH
        user_base AS (
          SELECT
            s.user_id,
            s.signup_date,
            s.channel,
            s.plan,
            DATE_DIFF('day', s.signup_date, DATE '2025-07-01') AS max_days
          FROM raw.signups s
          WHERE s.subscribed = TRUE
            AND s.plan IS NOT NULL
        ),
        eligible AS (
          SELECT *
          FROM user_base
          WHERE max_days >= {cutoff_days} + {horizon_days}
        ),
        txn_window AS (
          SELECT
            e.user_id,
            t.txn_date,
            t.amount,
            t.is_fraud,
            DATE_DIFF('day', e.signup_date, t.txn_date) AS days_from_signup
          FROM eligible e
          LEFT JOIN raw.transactions t
            ON e.user_id = t.user_id
        ),
        obs_features AS (
          SELECT
            user_id,
            COUNT(CASE WHEN days_from_signup >= 0
                       AND days_from_signup < {cutoff_days}
                  THEN 1 END) AS txn_count_obs,
            COALESCE(SUM(CASE WHEN days_from_signup >= 0
                               AND days_from_signup < {cutoff_days}
                          THEN amount END), 0.0) AS total_spend_obs,
            COALESCE(AVG(CASE WHEN days_from_signup >= 0
                               AND days_from_signup < {cutoff_days}
                          THEN amount END), 0.0) AS avg_txn_amount_obs,
            MAX(CASE WHEN days_from_signup >= 0
                      AND days_from_signup < {cutoff_days}
                 THEN days_from_signup END) AS last_txn_day_obs,
            CAST(COALESCE(SUM(CASE WHEN days_from_signup >= 0
                                    AND days_from_signup < {cutoff_days}
                                    AND is_fraud
                               THEN 1 END), 0) > 0 AS INTEGER) AS had_fraud_obs
          FROM txn_window
          GROUP BY user_id
        ),
        labels AS (
          SELECT
            user_id,
            CASE WHEN COUNT(CASE WHEN days_from_signup >= {cutoff_days}
                                  AND days_from_signup < {cutoff_days} + {horizon_days}
                             THEN 1 END) = 0
                 THEN 1 ELSE 0 END AS churned
          FROM txn_window
          GROUP BY user_id
        )
        SELECT
          e.user_id,
          e.channel,
          e.plan,
          e.signup_date,
          DATE_DIFF('day', e.signup_date, DATE '2025-07-01') AS tenure_days,
          COALESCE(o.txn_count_obs, 0) AS txn_count_obs,
          o.total_spend_obs,
          o.avg_txn_amount_obs,
          COALESCE(o.last_txn_day_obs, 0) AS last_txn_day_obs,
          o.had_fraud_obs,
          l.churned
        FROM eligible e
        JOIN obs_features o ON e.user_id = o.user_id
        JOIN labels l ON e.user_id = l.user_id
        ORDER BY e.user_id
        """
        raw = con.execute(query).fetchdf()
    finally:
        con.close()

    if len(raw) == 0:
        raise RuntimeError("build_training_set: zero eligible users (check DB path / horizon)")

    # ── feature engineering (Python-side, post-SQL) ──────────────────────
    df = raw.copy()

    # Recency: days since last transaction (capped at cutoff)
    df["days_since_last_txn"] = cutoff_days - df["last_txn_day_obs"].clip(0)

    # Transaction frequency (txns per 30-day month in observation window)
    months_obs = cutoff_days / 30.0
    df["txn_freq_monthly"] = df["txn_count_obs"] / months_obs

    # Signup day-of-week (0=Mon..6=Sun) and month
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_dow"] = df["signup_date"].dt.dayofweek
    df["signup_month"] = df["signup_date"].dt.month

    # ── categorical encoding ─────────────────────────────────────────────
    categorical_cols: list[str] = []
    for col in ["channel", "plan"]:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        for c in dummies.columns:
            df[c] = dummies[c].astype(int)
            categorical_cols.append(c)

    # ── final feature matrix ─────────────────────────────────────────────
    numeric_features = [
        "tenure_days",
        "txn_count_obs",
        "total_spend_obs",
        "avg_txn_amount_obs",
        "days_since_last_txn",
        "txn_freq_monthly",
        "had_fraud_obs",
        "signup_dow",
        "signup_month",
    ]
    feature_names = numeric_features + categorical_cols

    X = df[feature_names].copy()
    y = df["churned"].to_numpy(dtype=np.int64)

    churn_rate = float(y.mean())

    return TrainingSet(
        X=X,
        y=y,
        feature_names=feature_names,
        cutoff_days=cutoff_days,
        horizon_days=horizon_days,
        n_users=len(df),
        churn_rate=churn_rate,
    )
