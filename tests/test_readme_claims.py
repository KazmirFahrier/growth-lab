"""Phase 6 gate (audit-as-code): the README cannot drift from the code.

The campaign-copilot audit found claimed-but-unimplemented modules; here
that class of defect fails CI directly: every CLI command the README
mentions must exist, every layout entry must be on disk, every completed
phase must have its packages, and the causal-report table printed in the
README must byte-match what the code actually prints today.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
MAIN_SOURCE = (ROOT / "src" / "growth_lab" / "__main__.py").read_text()


def test_every_documented_cli_command_exists() -> None:
    documented = set(re.findall(r"python -m growth_lab ([\w-]+)", README))
    implemented = set(re.findall(r'add_parser\(\s*"([\w-]+)"', MAIN_SOURCE))
    assert documented, "README documents no CLI commands?"
    missing = documented - implemented
    assert not missing, f"README documents commands that do not exist: {sorted(missing)}"


def test_every_layout_entry_exists() -> None:
    layout = README.split("## Layout")[1].split("```")[1]
    for line in layout.strip().splitlines():
        entry = line.strip().split()[0].rstrip("/")
        if not entry or entry.endswith((".yaml", ".md")):
            path = ROOT / entry if entry else None
        else:
            candidates = (ROOT / entry, ROOT / "src" / "growth_lab" / entry)
            path = next((c for c in candidates if c.exists()), candidates[0])
        if path is not None:
            assert path.exists(), f"README layout mentions missing path: {entry}"


def test_completed_phases_have_their_packages() -> None:
    packages_by_phase = {
        "0": ["simulator", "warehouse"],
        "1": ["experiments"],
        "2": ["causal"],
        "3": ["marketing"],
        "4": ["forecasting", "risk"],
        "5": ["reporting"],
        "6": ["integrations"],
    }
    for row in re.findall(r"\|\s*(\d)\s*\|[^|]+\|\s*✅\s*\|", README):
        for package in packages_by_phase[row]:
            assert (ROOT / "src" / "growth_lab" / package / "__init__.py").exists(), (
                f"phase {row} is marked done but package {package!r} is missing"
            )


def test_readme_causal_table_matches_actual_output() -> None:
    """The recovery table quoted in the README must be exactly what the
    code prints — stale docs fail the build."""
    from growth_lab.__main__ import _causal_report

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _causal_report()
    for line in buffer.getvalue().strip().splitlines():
        assert line.rstrip() in README, f"README causal-report table is stale: {line!r}"


def test_constraints_file_pins_core_dependencies() -> None:
    constraints = (ROOT / "constraints.txt").read_text()
    for package in ("numpy==", "pandas==", "duckdb==", "dbt-duckdb==", "python-pptx=="):
        assert package in constraints, f"constraints.txt missing pin for {package}"


def test_ci_installs_with_constraints() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "-c constraints.txt" in ci
