"""Load simulator output into DuckDB and build the dbt star schema.

Layout:
  raw.*     — as-landed tables (ad platform export + app events)
  staging.* — dbt views (renames/casts)
  marts.*   — dbt tables (star schema + daily channel mart)

This package is sealed: it never sees latent truth. The scoring harness lands
its hidden schema separately via the simulator package.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import duckdb

from growth_lab.simulator.generate import SimOutput

REPO_ROOT = Path(__file__).resolve().parents[3]
DBT_DIR = REPO_ROOT / "dbt"
DB_ENV_VAR = "GROWTH_LAB_DB"


def build_warehouse(sim: SimOutput, db_path: Path) -> None:
    """Create the DuckDB file and land raw + hidden tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        frames = {
            "raw.ad_spend_daily": sim.ad_spend_daily,
            "raw.signups": sim.signups,
            "raw.transactions": sim.transactions,
        }
        for table, frame in frames.items():
            con.register("_incoming", frame)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _incoming")
            con.unregister("_incoming")
            n_db = con.execute(f"SELECT count(*) FROM {table}").fetchone()
            if n_db is None or int(n_db[0]) != len(frame):
                raise RuntimeError(f"row count mismatch landing {table}")
    finally:
        con.close()


def run_dbt(db_path: Path) -> None:
    """Run the dbt project against `db_path`. Raises loudly on any failure."""
    env = dict(os.environ)
    env[DB_ENV_VAR] = str(db_path)
    result = subprocess.run(
        ["dbt", "run", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dbt run failed (exit {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def build_all(sim: SimOutput, db_path: Path) -> None:
    """Land raw tables and build the full star schema."""
    build_warehouse(sim, db_path)
    run_dbt(db_path)
