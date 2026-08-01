# growth-lab

Decision science with a checkable ground truth.

Every dataset here is produced by a simulator of **Meridian**, a fictional
subscription business, whose data-generating process is fully specified in
[`truth.yaml`](truth.yaml). That makes every claim in this repo falsifiable:
causal estimates, marketing measurement, and forecasts (Phases 1–5) are scored
against parameters the estimators are structurally barred from reading —
`tests/test_no_truth_leak.py` enforces the seal.

Companion project: [campaign-copilot](../campaign-copilot) — the AI-agent side
of the same domain. campaign-copilot answers *"can an agent answer marketing
questions safely?"*; growth-lab answers *"can we measure what actually works?"*

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 0 | Simulator + DuckDB/dbt warehouse + semantic layer | ✅ |
| 1 | Experimentation platform (power, SRM, sequential, CUPED) | ✅ |
| 2 | Observational causal inference (DiD, RDD, PSM/IPW, IV, uplift) | planned |
| 3 | Marketing measurement (Bayesian MMM, attribution vs. incrementality, LTV) | planned |
| 4 | Forecasting, anomaly detection, risk & calibration | planned |
| 5 | Decision delivery (dashboard + auto-generated growth review) | planned |
| 6 | Integration with campaign-copilot + audit | planned |

## Design

**The simulator is the oracle.** Latent user intent confounds everything: high-intent
users click more, convert more, and are over-served by the retargeting-like
`display` channel. Naive channel CVR therefore *lies* — the calibration suite
asserts the trap exists (`test_confounder_trap_exists`), and later phases exist
to defuse it.

**Calibration is a CI gate.** `tests/test_calibration.py` recomputes analytic
expectations from `truth.yaml` and fails the build if the simulator's realized
CTR, CVR, churn, fraud rate, or seasonality drift from them. If the simulator
is wrong, everything downstream is meaningless — so it crashes, loudly.

**Metrics are ratio-of-sums by construction.** The semantic layer
(`warehouse/semantic.py`) stores aggregate SQL expressions only; averaging
daily ratios is unrepresentable. Unknown metrics raise, they don't default.

**The experimentation platform's error rates are themselves under test.**
`experiments/` provides power analysis, deterministic hash assignment with SRM
detection, two-proportion and Welch tests, CUPED variance reduction,
Lan-DeMets O'Brien-Fleming sequential boundaries (computed by numerical
density propagation, validated against published tables), non-inferiority
guardrails, and an automated readout with explicit launch/no-launch rules.
`tests/test_experiment_calibration.py` is the Monte Carlo gate: A/A
false-positive rates must sit within sampling error of alpha, designs sized
for 80% power must deliver it, naive peeking must reproduce the known FPR
inflation, and the sequential boundaries must cure it — or CI fails.

## Layout

```
truth.yaml                sealed DGP (simulator + tests only)
src/growth_lab/
  simulator/              params loader + vectorized event generation
  warehouse/              DuckDB landing, dbt orchestration, semantic layer
  experiments/            power, assignment/SRM, CUPED, sequential, readouts
dbt/                      staging views + star-schema marts
tests/                    calibration gate, invariants, seal enforcement
```

Warehouse schemas: `raw` (as-landed) → `staging` (dbt views) → `marts`
(star schema: `dim_users`, `fct_transactions`, `mart_daily_channel`).
`sim_hidden.users_latent` holds latent truth for the scoring harness; dbt
never reads it.

## Quickstart

```bash
pip install -e ".[dev]"
python -m growth_lab build          # simulate → DuckDB → dbt → metric summary
pytest                              # calibration gate + invariants
ruff check . && mypy               # style + strict types
```

## Production ML — Churn Risk System

The repository includes a production-grade churn prediction vertical slice:
temporal feature engineering, MLflow-tracked training, a FastAPI prediction
service, Docker packaging, monitoring, and a coverage-gated CI pipeline.

### Architecture

```
┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│  truth.yaml  │    │  DuckDB       │    │  dbt          │
│  (sealed)    │───▶│  warehouse    │───▶│  star schema  │
└──────────────┘    └──────┬───────┘    └───────┬───────┘
                           │                    │
                    ┌──────▼───────┐    ┌───────▼───────┐
                    │  features.py │    │  train.py     │
                    │  temporal    │───▶│  XGBoost +    │
                    │  split, no   │    │  baselines    │
                    │  leakage     │    └───────┬───────┘
                    └──────────────┘            │
                                        ┌───────▼───────┐
                                        │  MLflow       │
                                        │  tracking +   │
                                        │  registry     │
                                        └───────┬───────┘
                                                │
                                        ┌───────▼───────┐
                                        │  models/      │
                                        │  churn_model  │
                                        │  .joblib      │
                                        └───────┬───────┘
                                                │
┌──────────────┐    ┌──────────────┐    ┌───────▼───────┐
│  Prometheus  │◀───│  FastAPI     │◀───│  Docker /     │
│  /metrics    │    │  serve/app   │    │  Cloud Run    │
└──────────────┘    └──────┬───────┘    └───────────────┘
                           │
                    ┌──────▼───────┐
                    │  monitoring/ │
                    │  drift       │
                    │  calibration │
                    │  health      │
                    └──────────────┘
```

### Components

| Module | Purpose | Key files |
|--------|---------|-----------|
| `churn/features.py` | Temporal feature engineering from warehouse (no truth.yaml access) | SQL + pandas pipeline, leakage-safe split |
| `churn/train.py` | MLflow-tracked training: XGBoost vs Logistic Regression vs Random Forest vs Dummy | TimeSeriesSplit CV, calibration, model card |
| `serve/app.py` | FastAPI prediction service with Prometheus metrics | `/predict`, `/predict/batch`, `/health`, `/metrics` |
| `serve/schemas.py` | Pydantic request/response validation | `PredictionRequest`, `PredictionResponse`, `HealthResponse` |
| `monitor/drift.py` | KS-statistic feature drift detection | Reference distribution comparison |
| `monitor/calibration.py` | Expected Calibration Error (ECE) tracking | Binned probability calibration |
| `monitor/health.py` | Latency (p50/p95/p99), throughput, data freshness | Warehouse timestamp checks |

### Service Contract

**`POST /predict`** — Single churn prediction

```json
// Request
{
  "user_id": 42,
  "channel": "search",
  "plan": "pro",
  "tenure_days": 120,
  "txn_count_obs": 4,
  "total_spend_obs": 79.96,
  "avg_txn_amount_obs": 19.99,
  "days_since_last_txn": 15,
  "txn_freq_monthly": 1.0,
  "had_fraud_obs": 0,
  "signup_dow": 2,
  "signup_month": 3
}

// Response
{
  "user_id": 42,
  "churn_probability": 0.0823,
  "churn_prediction": false,
  "model_version": "0.1.0",
  "timestamp": "2025-07-01T12:00:00Z"
}
```

**`POST /predict/batch`** — Up to 1,000 predictions in one request.

**`GET /health`** — Liveness check with model version and uptime.
**`GET /ready`** — Readiness probe (503 if model not loaded).
**`GET /metrics`** — Prometheus scrape endpoint.
**`GET /model`** — Model metadata (feature names, training date).

### Deployment

**Local (Docker Compose):**
```bash
docker-compose up --build       # serve (port 8000) + MLflow (port 5000)
```

**Cloud Run:**
```bash
gcloud builds submit --config=cloudbuild.yaml
```

**Manual:**
```bash
growth-lab-train                          # train + register model
growth-lab-serve                          # start FastAPI (port 8000)
```

### Monitoring

All monitoring is Prometheus-native, exposed at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `churn_predictions_total{outcome}` | Counter | Predictions by churn/retain |
| `churn_prediction_latency_seconds` | Histogram | Per-request latency |
| `churn_model_loaded` | Gauge | 1 if model loaded, 0 otherwise |
| `churn_prediction_errors_total` | Counter | Prediction errors |
| `churn_request_size` | Histogram | Batch request sizes |

Additional scheduled checks via `monitor/`:
- **Data drift**: KS-statistic per feature against training reference
- **Calibration**: ECE on binned predictions (threshold ≤ 0.05)
- **Data freshness**: Latest warehouse transaction timestamp

### Retraining (Champion/Challenger)

```bash
# Full retraining pipeline
python -m growth_lab build                        # refresh simulated data
growth-lab-train                                  # train new model → MLflow
python -c "
from growth_lab.churn.train import train_pipeline
results = train_pipeline(register_model=True)
# Compare new vs. registered champion in MLflow UI (port 5000)
"
```

If the new model beats the champion on test-set ROC-AUC, promote it via the
MLflow registry or by copying `models/churn_model.joblib` into the serving
volume.

### Measured Latency (reference)

| Percentile | Latency |
|-----------|---------|
| p50 | < 2 ms |
| p95 | < 5 ms |
| p99 | < 10 ms |

*Measured on a single CPU (M1 Pro) with XGBoost, batch size 1, excluding network.*

### CI Pipeline

| Gate | Tool | Threshold |
|------|------|-----------|
| Lint | ruff | zero violations |
| Types | mypy (strict) | zero errors |
| Unit + Integration tests | pytest | 65% coverage minimum |
| Seal enforcement | test_no_truth_leak.py | zero forbidden imports |
| Docker build | docker | image builds |
| Smoke test | curl /health | HTTP 200 |
| Integration (main only) | simulate → train | end-to-end pipeline |
