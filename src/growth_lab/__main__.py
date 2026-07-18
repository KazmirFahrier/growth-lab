"""CLI: `python -m growth_lab build` — simulate, land, and model the warehouse."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from growth_lab.simulator import load_truth, simulate
from growth_lab.warehouse.load import build_all
from growth_lab.warehouse.semantic import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(prog="growth_lab")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="simulate + load DuckDB + run dbt")
    build.add_argument("--db", type=Path, default=Path("data/growth_lab.duckdb"))
    build.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    truth = load_truth()
    sim = simulate(truth, seed=args.seed)
    build_all(sim, args.db)

    con = duckdb.connect(str(args.db), read_only=True)
    try:
        summary = compute_metrics(
            con, ["spend", "signups", "paid_signups", "ctr", "cvr", "cac", "revenue"],
            by=["channel"],
        )
    finally:
        con.close()
    print(f"warehouse built at {args.db}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
