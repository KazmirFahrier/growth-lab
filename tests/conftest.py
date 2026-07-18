from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from growth_lab.simulator import SimOutput, Truth, land_hidden, load_truth, simulate
from growth_lab.warehouse.load import build_all


@pytest.fixture(scope="session")
def truth() -> Truth:
    return load_truth()


@pytest.fixture(scope="session")
def sim(truth: Truth) -> SimOutput:
    return simulate(truth)


@pytest.fixture(scope="session")
def warehouse(
    truth: Truth, sim: SimOutput, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[duckdb.DuckDBPyConnection]:
    db_path: Path = tmp_path_factory.mktemp("wh") / "growth_lab.duckdb"
    build_all(sim, db_path)
    land_hidden(sim, db_path)  # scoring harness only; dbt has already run without it
    con = duckdb.connect(str(db_path), read_only=True)
    yield con
    con.close()
