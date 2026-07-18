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
