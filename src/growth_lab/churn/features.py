"""Temporal feature engineering for churn prediction.

All features are derived exclusively from observable warehouse tables. The
sealed simulator state is inaccessible by construction.

Temporal split strategy:
  * Observation window: days [0, cutoff) from each user's signup
  * Prediction window: days [cutoff, cutoff + horizon) from signup
  * Label = 1 if zero transactions in the prediction window (= churned)
  * Rows ordered by signup_date so a simple row split = temporal split.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import numpy.typing as npt
import pandas as pd

# Default temporal split: 120 days observation, 30-day churn horizon.
DEFAULT_CUTOFF_DAYS = 120
DEFAULT_HORIZON_DAYS = 30

# Where the canonical feature list is saved for the serving bundle.
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("GROWTH_LAB_MODEL_DIR", str(REPO_ROOT / "models")))
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


# ── ordered categorical levels (must match serving in serve/app.py) ──────
CHANNEL_LEVELS = ["display", "organic", "search", "social", "video"]
PLAN_LEVELS = ["basic", "pro"]

NUMERIC_FEATURES = [
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


def _categorical_dummy_names() -> list[str]:
    """Return the full set of one-hot column names (no drop_first).

    Must exactly match _build_feature_vector in serve/app.py."""
    names: list[str] = []
    for level in CHANNEL_LEVELS:
        names.append(f"channel_{level}")
    for level in PLAN_LEVELS:
        names.append(f"plan_{level}")
    return names


ALL_FEATURE_NAMES: list[str] = NUMERIC_FEATURES + _categorical_dummy_names()


@dataclass(frozen=True)
class TrainingSet:
    """Leakage-safe training data with temporal split metadata."""

    X: pd.DataFrame
    y: npt.NDArray[np.int64]
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
    signup + cutoff_days + horizon_days). Users with insufficient history
    are excluded. Rows are ordered by signup_date for temporal splitting.
    """
    if cutoff_days <= 0 or horizon_days <= 0:
        raise ValueError("cutoff_days and horizon_days must be positive")
    if not db_path.is_file():
        raise FileNotFoundError(f"warehouse does not exist: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        query = f"""
        WITH
        dataset_bounds AS (
          SELECT GREATEST(
            (SELECT MAX(date) FROM raw.ad_spend_daily),
            (SELECT MAX(signup_date) FROM raw.signups),
            (SELECT MAX(txn_date) FROM raw.transactions)
          ) AS dataset_end
        ),
        eligible AS (
          SELECT s.user_id, s.signup_date, s.channel, s.plan
          FROM raw.signups s
          CROSS JOIN dataset_bounds b
          WHERE s.subscribed = TRUE
            AND s.plan IS NOT NULL
            AND s.signup_date + INTERVAL '{cutoff_days + horizon_days} days' <= b.dataset_end
        ),
        txn_window AS (
          SELECT
            e.user_id,
            e.signup_date,
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
          COALESCE(o.txn_count_obs, 0) AS txn_count_obs,
          o.total_spend_obs,
          o.avg_txn_amount_obs,
          COALESCE(o.last_txn_day_obs, 0) AS last_txn_day_obs,
          o.had_fraud_obs,
          l.churned
        FROM eligible e
        JOIN obs_features o ON e.user_id = o.user_id
        JOIN labels l ON e.user_id = l.user_id
        ORDER BY e.signup_date, e.user_id
        """
        raw = con.execute(query).fetchdf()
    finally:
        con.close()

    if len(raw) == 0:
        raise RuntimeError("build_training_set: zero eligible users (check DB path / horizon)")

    # ── feature engineering (Python-side, post-SQL) ──────────────────────
    df = raw.copy()

    # Tenure at the observation cutoff (not a hardcoded endpoint date).
    df["tenure_days"] = cutoff_days

    # Recency: days since last transaction (capped at cutoff)
    df["days_since_last_txn"] = cutoff_days - df["last_txn_day_obs"].clip(0)

    # Transaction frequency (txns per 30-day month in observation window)
    months_obs = cutoff_days / 30.0
    df["txn_freq_monthly"] = df["txn_count_obs"] / months_obs

    # Signup day-of-week (0=Mon..6=Sun) and month
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["signup_dow"] = df["signup_date"].dt.dayofweek
    df["signup_month"] = df["signup_date"].dt.month

    # ── categorical encoding (FULL one-hot, no drop_first) ───────────────
    ch_dummies = pd.get_dummies(
        pd.Categorical(df["channel"], categories=CHANNEL_LEVELS),
        prefix="channel",
        dtype=int,
    )
    plan_dummies = pd.get_dummies(
        pd.Categorical(df["plan"], categories=PLAN_LEVELS),
        prefix="plan",
        dtype=int,
    )
    df = pd.concat([df, ch_dummies, plan_dummies], axis=1)

    # ── final feature matrix ─────────────────────────────────────────────
    X = df[ALL_FEATURE_NAMES].copy()
    y = df["churned"].to_numpy(dtype=np.int64)

    churn_rate = float(y.mean())

    # ── persist canonical feature list for serving ───────────────────────
    FEATURE_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_NAMES_PATH.write_text(json.dumps(ALL_FEATURE_NAMES, indent=2))

    return TrainingSet(
        X=X,
        y=y,
        feature_names=ALL_FEATURE_NAMES,
        cutoff_days=cutoff_days,
        horizon_days=horizon_days,
        n_users=len(df),
        churn_rate=churn_rate,
    )
