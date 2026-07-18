"""Land the latent-truth table for the scoring harness.

Lives in the simulator package deliberately: the sealed warehouse code must
never know this table exists. A production-like build (`python -m growth_lab
build`) does not call this — the shipped warehouse contains no truth at all.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from growth_lab.simulator.generate import SimOutput


def land_hidden(sim: SimOutput, db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS sim_hidden")
        con.register("_hidden", sim.users_latent)
        con.execute("CREATE OR REPLACE TABLE sim_hidden.users_latent AS SELECT * FROM _hidden")
        con.unregister("_hidden")
    finally:
        con.close()
