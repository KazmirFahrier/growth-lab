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
| 2 | Observational causal inference (DiD, RDD, IPW, IV, uplift) | ✅ |
| 3 | Marketing measurement (MMM, attribution vs. incrementality, LTV, budget) | ✅ |
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
