"""CLI: `python -m growth_lab build | causal-report`."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from growth_lab.simulator import load_truth, simulate
from growth_lab.warehouse.load import build_all
from growth_lab.warehouse.semantic import compute_metrics


def _build(db_path: Path, seed: int | None) -> None:
    truth = load_truth()
    sim = simulate(truth, seed=seed)
    build_all(sim, db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        summary = compute_metrics(
            con, ["spend", "signups", "paid_signups", "ctr", "cvr", "cac", "revenue"],
            by=["channel"],
        )
    finally:
        con.close()
    print(f"warehouse built at {db_path}")
    print(summary.to_string(index=False))


def _causal_report() -> None:
    """Recovery table: what the naive answer says, what the causal estimator
    says, and what the truth is. The gap in column one is the point."""
    from growth_lab.causal import (
        diff_in_diff,
        ipw_ate,
        naive_difference,
        naive_ols_slope,
        sharp_rdd,
        two_stage_least_squares,
    )
    from growth_lab.simulator.scenarios import (
        geo_rollout,
        price_instrument,
        promo_email,
        spend_threshold,
    )

    rows: list[tuple[str, str, float, float, float]] = []

    did_s = geo_rollout()
    post = did_s.panel[did_s.panel["post"]]
    rows.append(
        (
            "geo rollout",
            "DiD",
            naive_difference(post["y"].to_numpy(dtype=float), post["treated"].to_numpy()),
            diff_in_diff(did_s.panel).att,
            did_s.true_att,
        )
    )

    rdd_s = spend_threshold()
    rows.append(
        (
            "spend threshold",
            "sharp RDD",
            naive_difference(rdd_s.outcome, rdd_s.running >= rdd_s.cutoff),
            sharp_rdd(rdd_s.running, rdd_s.outcome, rdd_s.cutoff).effect,
            rdd_s.true_effect,
        )
    )

    ipw_s = promo_email()
    rows.append(
        (
            "promo email",
            "IPW",
            naive_difference(ipw_s.outcome, ipw_s.treated),
            ipw_ate(ipw_s.covariates, ipw_s.treated, ipw_s.outcome).ate,
            ipw_s.true_ate,
        )
    )

    iv_s = price_instrument()
    rows.append(
        (
            "price change",
            "2SLS",
            naive_ols_slope(iv_s.price, iv_s.demand),
            two_stage_least_squares(iv_s.instrument, iv_s.price, iv_s.demand).estimate,
            iv_s.true_price_coefficient,
        )
    )

    header = f"{'scenario':<16} {'method':<10} {'naive':>9} {'causal':>9} {'truth':>9}"
    print(header)
    print("-" * len(header))
    for scenario, method, naive, causal, truth_val in rows:
        print(f"{scenario:<16} {method:<10} {naive:>+9.4f} {causal:>+9.4f} {truth_val:>+9.4f}")


def _weekly_review(db_path: Path, out_dir: Path) -> None:
    from growth_lab.reporting import build_weekly_review, export_pptx

    if not db_path.exists():
        raise SystemExit(f"no warehouse at {db_path}; run `python -m growth_lab build` first")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        review = build_weekly_review(con)
    finally:
        con.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "weekly_review.md"
    md_path.write_text(review.markdown)
    pptx_path = export_pptx(review, out_dir / "weekly_review.pptx")
    print(f"wrote {md_path}")
    print(f"wrote {pptx_path}")


def _export_mmm(out_path: Path) -> None:
    """Fit the MMM on the media-market scenario and write a truth-free
    parameter artifact for the agent bridge's budget planner."""
    import json

    import numpy as np

    from growth_lab.marketing import fit_mmm
    from growth_lab.simulator.scenarios import mmm_market

    scenario = mmm_market()
    fit = fit_mmm(scenario.spend, scenario.revenue, scenario.day_of_week, scenario.channels)
    payload = {
        "source": "fit_mmm on the media-market scenario (see growth_lab.marketing.mmm)",
        "r_squared": round(fit.r_squared, 4),
        "channels": [
            {
                "name": name,
                "beta": round(float(fit.beta[c]), 2),
                "decay": round(float(fit.decay[c]), 3),
                "half_sat": round(float(fit.half_sat[c]), 2),
                "current_daily_spend": round(float(np.mean(scenario.spend[:, c])), 2),
            }
            for c, name in enumerate(scenario.channels)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="growth_lab")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="simulate + load DuckDB + run dbt")
    build.add_argument("--db", type=Path, default=Path("data/growth_lab.duckdb"))
    build.add_argument("--seed", type=int, default=None)
    sub.add_parser("causal-report", help="naive vs causal vs truth recovery table")
    review = sub.add_parser("weekly-review", help="provenance-tracked growth review (md + pptx)")
    review.add_argument("--db", type=Path, default=Path("data/growth_lab.duckdb"))
    review.add_argument("--out", type=Path, default=Path("reports"))
    export = sub.add_parser("export-mmm", help="fitted MMM params for the agent bridge")
    export.add_argument("--out", type=Path, default=Path("models/mmm.json"))
    args = parser.parse_args()

    if args.command == "build":
        _build(args.db, args.seed)
    elif args.command == "weekly-review":
        _weekly_review(args.db, args.out)
    elif args.command == "export-mmm":
        _export_mmm(args.out)
    else:
        _causal_report()


if __name__ == "__main__":
    main()
