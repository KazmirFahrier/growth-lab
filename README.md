# growth-lab

Decision science with a checkable ground truth.

Every dataset here is produced by a simulator of **Meridian**, a fictional
subscription business, whose data-generating process is fully specified in
[`truth.yaml`](truth.yaml). That makes every claim in this repo falsifiable:
causal estimates, marketing measurement, and forecasts (Phases 1–5) are scored
against parameters the estimators are structurally barred from reading —
`tests/test_no_truth_leak.py` enforces the seal.

Companion project: [campaign-copilot](https://github.com/KazmirFahrier/campaign-copilot) — the AI-agent side
of the same domain. campaign-copilot answers *"can an agent answer marketing
questions safely?"*; growth-lab answers *"can we measure what actually works?"*

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 0 | Simulator + DuckDB/dbt warehouse + semantic layer | ✅ |
| 1 | Experimentation platform (power, SRM, sequential, CUPED) | ✅ |
| 2 | Observational causal inference (DiD, RDD, IPW, IV, uplift) | ✅ |
| 3 | Marketing measurement (MMM, attribution vs. incrementality, LTV, budget) | ✅ |
| 4 | Forecasting, anomaly detection, risk & calibration | ✅ |
| 5 | Decision delivery (dashboard + provenance-tracked growth review) | ✅ |
| 6 | campaign-copilot bridge + audit-as-code | ✅ |

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

**Causal estimators must refuse broken data.** `causal/` implements DiD,
sharp RDD, IPW, 2SLS, and a T-learner from first principles (the only deps
are numpy and pandas), each behind a mandatory diagnostic: parallel-trends
placebo, density continuity at the cutoff, overlap and post-weighting
balance, first-stage F. When an assumption fails, the estimator raises
`AssumptionViolation` instead of returning a number. The recovery gate in
`tests/test_causal_recovery.py` requires all three legs per scenario: the
naive answer is provably biased, the causal answer lands on the sealed
truth, and the assumption-violating variant is refused. See it yourself:

```
$ python -m growth_lab causal-report
scenario         method         naive    causal     truth
---------------------------------------------------------
geo rollout      DiD          +0.0256   +0.0081   +0.0080
spend threshold  sharp RDD    +0.6856   +0.1689   +0.1500
promo email      IPW          +0.1894   +0.0440   +0.0600
price change     2SLS         +1.3184   -0.4022   -0.4000
```

**Marketing measurement is scored against truth, including its blind spots.**
`marketing/` implements MMM (geometric adstock + saturation, fit by
coordinate descent with exact OLS conditioning, moving-block bootstrap
intervals), three attribution models (last-touch, linear, Markov removal
effect), censored-geometric subscription LTV, and a water-filling budget
optimizer whose optimality is verified against brute force. The recovery
gate demands MMM ROAS within 20% of truth on a DGP with go-dark windows,
and — the honest headline — proves that *no* attribution model measures
incrementality: all three over-credit the retargeting channel that harvests
users already about to convert, with last-touch off by more than 3x.

**Forecasts must beat "same day last week" or fail CI.** `forecasting/`
implements Holt-Winters and a from-scratch gradient-boosted-stumps
forecaster behind a rolling-origin backtest harness that owns the train/test
boundary (a spy-model test proves nothing ever sees past its fold's origin).
MASE is computed against seasonal-naive on identical folds; MASE >= 1 fails
the build — models must earn their complexity. Quantile paths (P10/P50/P90)
are scored with pinball loss, and bottom-up hierarchical forecasts are
coherence-checked.

**Risk models are deployment-gated on calibration, not just AUC.** `risk/`
has a robust MAD-residual anomaly detector and a from-scratch isolation
forest, both scored against injected shocks (precision and recall >= 0.8);
a fraud model gated on ECE < 0.02 with reliability curves and Brier score;
a cost-optimal threshold that must land on the analytic c_fp/(c_fp+c_fn)
optimum; and a PSI drift monitor verified in both directions — silent on a
fresh sample of the same population, alarming on a shifted regime.

**growth-lab is an agent toolkit.** `integrations/` mirrors
campaign-copilot's tool contract (`ToolSpec`, `ToolResult`, failures as
results with machine-readable codes, `numeric_facts()` for grounding) and
exposes the semantic layer, LTV, forecasting, and the MMM budget planner as
mountable tools. The flagship demo answers "should we shift budget?" from
fitted response curves, and the bridge gate verifies that every number a
tool states in its content is licensed by its own data — so
campaign-copilot's grounding checker passes these answers by construction.
The audit is code too: `tests/test_readme_claims.py` fails CI if this
README documents a CLI command that doesn't exist, marks a phase done
without its packages, or quotes a causal-report table that no longer
matches actual output.

**Every reported number carries its lineage.** `reporting/` renders a
weekly growth review (markdown + .pptx via one shared Figure layer) where
each KPI embeds the exact semantic-layer SQL that produced it and each
model output names its estimator. The memo template contains no digits; the
provenance gate extracts every numeric token from the rendered memo and
fails if any lacks a backing figure. `python -m growth_lab weekly-review`
produces both files; `streamlit run dashboard/app.py` serves the
interactive version (KPIs, channel views, MMM response curves with a budget
slider, forecast + anomaly review, and a live experiment-readout
calculator).

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
  causal/                 DiD, RDD, IPW, 2SLS, uplift — with assumption gates
  marketing/              MMM, attribution, LTV, budget optimizer
  forecasting/            Holt-Winters, boosted stumps, backtest, hierarchy
  risk/                   anomaly detection, calibration, drift (PSI)
  reporting/              Figure provenance, growth review (md + pptx)
  integrations/           agent tool bridge (campaign-copilot contract)
  service/                authenticated API, health, readiness, telemetry
dashboard/                Streamlit app (optional extra: pip install -e ".[dashboard]")
dbt/                      staging views + star-schema marts
tests/                    calibration gate, invariants, seal enforcement
docs/                     operations and incident runbook
```

Warehouse schemas: `raw` (as-landed) → `staging` (dbt views) → `marts`
(star schema: `dim_users`, `fct_transactions`, `mart_daily_channel`).
`sim_hidden.users_latent` holds latent truth for the scoring harness; dbt
never reads it.

## Quickstart

```bash
pip install -e ".[dev]"
python -m growth_lab build          # simulate → DuckDB → dbt → metric summary
python -m growth_lab causal-report  # naive vs causal vs truth table
python -m growth_lab weekly-review  # provenance-tracked memo (md + pptx)
python -m growth_lab export-mmm     # MMM params artifact for the agent bridge
pytest                              # all gates: calibration, recovery, provenance
ruff check . && mypy               # style + strict types
```

## Production service

The production boundary exposes governed metrics, forecasting, and budget
planning through an authenticated FastAPI service. Raw SQL is never accepted
from clients. Metric filters are typed and bound as DuckDB parameters. The
runtime also provides correlation IDs, structured JSON logs, health and
readiness probes, bounded request bodies, security headers, and Prometheus
text metrics.

Create a secret with at least thirty two characters, then start the hardened
container profile:

```bash
export GROWTH_LAB_API_KEY="$(openssl rand -hex 32)"
docker compose up --build
curl http://127.0.0.1:8000/readyz
curl -H "X-API-Key: $GROWTH_LAB_API_KEY" http://127.0.0.1:8000/metrics
```

The image runs as an unprivileged user with a read only filesystem, all Linux
capabilities removed, a process limit, and explicit CPU and memory limits.
See [`docs/operations.md`](docs/operations.md) for configuration, deployment,
monitoring, rollback, and incident procedures.

## Why this project exists

NYC data science postings cluster into four families: product/
experimentation, marketing/ads measurement, fintech forecasting & risk, and
generalist ML. growth-lab covers all four in one coherent repo whose every
claim is falsifiable against a sealed ground truth — and pairs with
[campaign-copilot](https://github.com/KazmirFahrier/campaign-copilot) to make a single two-repo story:
*building AI systems, and building the measurement science that keeps them
honest.*
