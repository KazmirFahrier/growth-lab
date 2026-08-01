"""Temporal churn training pipeline with MLflow tracking and model registry.

Selects a champion through temporal cross validation, evaluates it once on an
untouched final holdout, and writes a versioned model artifact for serving.

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


def _candidate_models() -> dict[str, Any]:
    """Create fresh candidate estimators for one fit cycle."""
    return {
        "dummy": DummyClassifier(strategy="stratified", random_state=42),
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42, solver="liblinear"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        ),
    }


def _select_champion(
    X_train: pd.DataFrame,
    y_train: IntArray,
    n_splits: int = 3,
) -> tuple[str, dict[str, list[float]]]:
    """Select a candidate using temporal validation data only."""
    scores: dict[str, list[float]] = {name: [] for name in _candidate_models()}
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X_train):
        X_cv_train = X_train.iloc[train_idx]
        X_cv_val = X_train.iloc[val_idx]
        y_cv_train = y_train[train_idx]
        y_cv_val = y_train[val_idx]
        if len(np.unique(y_cv_train)) != 2 or len(np.unique(y_cv_val)) != 2:
            continue
        for name, model in _candidate_models().items():
            model.fit(X_cv_train, y_cv_train)
            probability = model.predict_proba(X_cv_val)[:, 1]
            scores[name].append(float(roc_auc_score(y_cv_val, probability)))

    valid_folds = len(next(iter(scores.values())))
    if valid_folds < 2:
        raise ValueError("temporal cross validation requires at least two two class folds")
    mean_scores = {name: float(np.mean(values)) for name, values in scores.items()}
    selected_name = max(mean_scores.items(), key=lambda item: item[1])[0]
    return selected_name, scores


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

    # ── 2. Final temporal holdout (last 20% by signup date) ──────────────
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("temporal train and test partitions must each contain both classes")

    # ── 3. Candidate selection on development data only ──────────────────
    selected_name, candidate_cv_scores = _select_champion(X_train, y_train)
    candidate_cv_mean = {
        name: float(np.mean(scores)) for name, scores in candidate_cv_scores.items()
    }
    candidate_cv_std = {
        name: float(np.std(scores)) for name, scores in candidate_cv_scores.items()
    }
    valid_cv_folds = len(candidate_cv_scores[selected_name])

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
                "selection_method": "temporal_cross_validation",
                "selection_metric": "roc_auc",
                "selection_folds": valid_cv_folds,
                "selection_cv_mean_auc": candidate_cv_mean[selected_name],
                "selection_cv_std_auc": candidate_cv_std[selected_name],
            }
        )
        mlflow.log_metrics(
            {
                f"{name}_cv_roc_auc_mean": mean_auc
                for name, mean_auc in candidate_cv_mean.items()
            }
            | {
                f"{name}_cv_roc_auc_std": candidate_cv_std[name]
                for name in candidate_cv_mean
            }
        )
        mlflow.log_dict(ts.feature_names, "feature_names.json")

        # ── 4. Fit the selected candidate on all development data ────────
        selected_model = _candidate_models()[selected_name]
        selected_model.fit(X_train, y_train)

        # ── 5. Evaluate the selected model once on the final holdout ──────
        results: dict[str, object] = {}
        test_metrics = _eval_model(selected_model, X_test, y_test, selected_name)
        calibration = _calibration_summary(selected_model, X_test, y_test, selected_name)
        results.update(test_metrics)
        results.update(calibration)
        mlflow.log_metrics(test_metrics)
        mlflow.log_dict(calibration, f"{selected_name}_calibration.json")
        final_test_auc = test_metrics[f"{selected_name}_roc_auc"]

        # ── 6. Feature importance for the selected model, when available ─
        importance_values: npt.NDArray[np.float64] | None = None
        if hasattr(selected_model, "feature_importances_"):
            importance_values = np.asarray(selected_model.feature_importances_, dtype=np.float64)
        elif hasattr(selected_model, "coef_"):
            importance_values = np.abs(np.asarray(selected_model.coef_[0], dtype=np.float64))
        if importance_values is not None:
            importance = pd.DataFrame(
                {
                    "feature": ts.feature_names,
                    "importance": importance_values,
                }
            ).sort_values("importance", ascending=False)
            mlflow.log_dict(importance.to_dict(orient="records"), "feature_importance.json")

        # ── 7. Log and register the selected model ────────────────────────
        if register_model:
            mlflow.sklearn.log_model(
                selected_model,
                name="model",
                registered_model_name="churn-predictor",
            )
        else:
            mlflow.sklearn.log_model(selected_model, name="model")

        # Save model artifact for serving
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            import joblib

            model_path = Path(tmpdir) / "churn_model.joblib"
            joblib.dump(selected_model, model_path)
            mlflow.log_artifact(str(model_path), "serving_artifact")

            # Also save a versioned copy in models/
            dest = MODEL_DIR / "churn_model.joblib"
            joblib.dump(selected_model, dest)
        model_sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()

        # ── 8. Model card ─────────────────────────────────────────────────
        model_card = {
            "model_version": __version__,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "best_model": selected_name,
            "best_test_auc": final_test_auc,
            "final_test_auc": final_test_auc,
            "selection_method": "temporal_cross_validation",
            "selection_metric": "roc_auc",
            "selection_folds": valid_cv_folds,
            "selection_cv_mean_auc": candidate_cv_mean[selected_name],
            "selection_cv_std_auc": candidate_cv_std[selected_name],
            "candidate_cv_auc": {
                name: {
                    "mean": candidate_cv_mean[name],
                    "std": candidate_cv_std[name],
                    "fold_scores": scores,
                }
                for name, scores in candidate_cv_scores.items()
            },
            "model_sha256": model_sha256,
            "n_users": ts.n_users,
            "churn_rate": ts.churn_rate,
            "feature_names": ts.feature_names,
            "test_metrics": dict(results),
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
