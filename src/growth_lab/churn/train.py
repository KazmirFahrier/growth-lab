"""Temporal churn training pipeline with MLflow tracking and model registry.

Trains XGBoost + baseline models, logs metrics/params/artifacts to MLflow,
registers the best model, and writes a versioned model artifact for serving.

Usage:
  growth-lab-train                          # full pipeline
  growth-lab-train --db data/growth_lab.duckdb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from growth_lab import __version__
from growth_lab.churn.features import build_training_set

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = Path(os.environ.get("GROWTH_LAB_DB", str(REPO_ROOT / "data/growth_lab.duckdb")))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
MODEL_DIR = Path(os.environ.get("GROWTH_LAB_MODEL_DIR", str(REPO_ROOT / "models")))
IntArray = npt.NDArray[np.int64]


def _eval_model(model: Any, X_test: pd.DataFrame, y_test: IntArray, name: str) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return {
        f"{name}_roc_auc": roc_auc_score(y_test, y_prob),
        f"{name}_avg_precision": average_precision_score(y_test, y_prob),
        f"{name}_log_loss": log_loss(y_test, y_prob, labels=[0, 1]),
        f"{name}_brier": brier_score_loss(y_test, y_prob),
        f"{name}_accuracy": float((y_pred == y_test).mean()),
    }


def _calibration_summary(
    model: Any, X_test: pd.DataFrame, y_test: IntArray, name: str, n_bins: int = 10
) -> dict[str, object]:
    """Binned calibration curve — fraction of positives per predicted-probability bin."""
    y_prob = model.predict_proba(X_test)[:, 1]
    df = pd.DataFrame({"prob": y_prob, "label": y_test})
    df["bin"] = pd.cut(df["prob"], bins=n_bins, include_lowest=True)
    summary = df.groupby("bin", observed=False).agg(
        mean_pred=("prob", "mean"), actual_rate=("label", "mean"), count=("label", "size")
    )
    summary = summary.dropna()
    return {
        f"{name}_calibration_bins": summary.reset_index().to_dict(orient="records"),
        f"{name}_ece": float(
            ((summary["mean_pred"] - summary["actual_rate"]).abs() * summary["count"]).sum()
            / summary["count"].sum()
        ),
    }


def train_pipeline(
    db_path: Path = DEFAULT_DB,
    cutoff_days: int = 120,
    horizon_days: int = 30,
    register_model: bool = True,
) -> dict[str, object]:
    """Run the full churn training pipeline with MLflow tracking."""

    # ── 1. Build leakage-safe training set ───────────────────────────────
    ts = build_training_set(db_path, cutoff_days=cutoff_days, horizon_days=horizon_days)
    X, y = ts.X, ts.y
    if len(X) < 100:
        raise ValueError("at least 100 eligible users are required for training")

    # ── 2. Temporal train/test split (last 20% of users by signup date) ──
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("temporal train and test partitions must each contain both classes")

    # ── 3. Temporal CV for hyperparameter selection ──────────────────────
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores: list[float] = []
    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train = X_train.iloc[train_idx]
        X_cv_val = X_train.iloc[val_idx]
        y_cv_train = y_train[train_idx]
        y_cv_val = y_train[val_idx]
        if len(np.unique(y_cv_train)) != 2 or len(np.unique(y_cv_val)) != 2:
            continue
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        clf.fit(X_cv_train, y_cv_train)
        cv_scores.append(roc_auc_score(y_cv_val, clf.predict_proba(X_cv_val)[:, 1]))
    if not cv_scores:
        raise ValueError("temporal cross validation produced no two class fold")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name=f"churn-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"):
        # ── log dataset metadata ─────────────────────────────────────────
        mlflow.log_params(
            {
                "cutoff_days": cutoff_days,
                "horizon_days": horizon_days,
                "n_users": ts.n_users,
                "churn_rate": ts.churn_rate,
                "n_features": len(ts.feature_names),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "cv_mean_auc": float(np.mean(cv_scores)),
                "cv_std_auc": float(np.std(cv_scores)),
            }
        )
        mlflow.log_dict(ts.feature_names, "feature_names.json")

        # ── 4. Train models ──────────────────────────────────────────────
        models: dict[str, Any] = {}

        # Dummy baseline
        dummy = DummyClassifier(strategy="stratified", random_state=42)
        dummy.fit(X_train, y_train)
        models["dummy"] = dummy

        # Logistic regression baseline
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        models["logistic"] = lr

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)
        models["random_forest"] = rf

        # XGBoost (primary)
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb

        # ── 5. Evaluate all models ───────────────────────────────────────
        results: dict[str, object] = {}
        best_auc = float("-inf")
        best_name = ""

        for name, model in models.items():
            metrics = _eval_model(model, X_test, y_test, name)
            cal = _calibration_summary(model, X_test, y_test, name)
            results.update(metrics)
            results.update(cal)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(cal, f"{name}_calibration.json")

            if metrics[f"{name}_roc_auc"] > best_auc:
                best_auc = metrics[f"{name}_roc_auc"]
                best_name = name

        # ── 6. Feature importance (XGBoost) ──────────────────────────────
        importance = pd.DataFrame(
            {
                "feature": ts.feature_names,
                "importance": xgb.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        mlflow.log_dict(importance.to_dict(orient="records"), "feature_importance.json")

        # ── 7. Log and register best model ────────────────────────────────
        best_model = models[best_name]
        if register_model:
            mlflow.sklearn.log_model(
                best_model,
                name="model",
                registered_model_name="churn-predictor",
            )
        else:
            mlflow.sklearn.log_model(best_model, name="model")

        # Save model artifact for serving
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib

            model_path = Path(tmpdir) / "churn_model.joblib"
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(str(model_path), "serving_artifact")

            # Also save a versioned copy in models/
            dest = MODEL_DIR / "churn_model.joblib"
            joblib.dump(best_model, dest)
        model_sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()

        # ── 8. Model card ─────────────────────────────────────────────────
        model_card = {
            "model_version": __version__,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "best_model": best_name,
            "best_test_auc": best_auc,
            "model_sha256": model_sha256,
            "n_users": ts.n_users,
            "churn_rate": ts.churn_rate,
            "feature_names": ts.feature_names,
            "metrics": dict(results),
        }
        mlflow.log_dict(model_card, "model_card.json")
        (MODEL_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2, default=str))
        results["model_card"] = model_card

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to DuckDB file")
    parser.add_argument("--cutoff-days", type=int, default=120)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--no-register", action="store_true", help="Skip MLflow model registration")
    args = parser.parse_args()

    results = train_pipeline(
        db_path=args.db,
        cutoff_days=args.cutoff_days,
        horizon_days=args.horizon_days,
        register_model=not args.no_register,
    )
    print(json.dumps(results.get("model_card", {}), indent=2, default=str))


if __name__ == "__main__":
    main()
