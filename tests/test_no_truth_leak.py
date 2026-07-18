"""The seal on truth.yaml: estimator- and warehouse-facing code must never
read the ground truth. Only the simulator (it IS the DGP) and tests may.

This is enforced structurally, not by convention.
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "growth_lab"
FORBIDDEN_COMMON = ("load_truth", "truth.yaml", "simulator.params", "users_latent", "sim_hidden")
# Estimator packages must not import the simulator at all; the warehouse may
# land simulator *output* (ingestion) but never its parameters or latents.
SEALED_PACKAGES: dict[str, tuple[str, ...]] = {
    "warehouse": FORBIDDEN_COMMON,
    "experiments": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "causal": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
}


def test_sealed_packages_never_touch_truth() -> None:
    violations: list[str] = []
    for package, forbidden in SEALED_PACKAGES.items():
        for path in (SRC / package).rglob("*.py"):
            text = path.read_text()
            violations.extend(
                f"{path.relative_to(SRC)}: contains {token!r}"
                for token in forbidden
                if token in text
            )
    assert not violations, "\n".join(violations)


def test_dbt_models_never_touch_hidden_schema() -> None:
    dbt_models = Path(__file__).resolve().parents[1] / "dbt" / "models"
    violations = [
        str(path)
        for path in dbt_models.rglob("*.sql")
        if "sim_hidden" in path.read_text() or "users_latent" in path.read_text()
    ]
    assert not violations, "\n".join(violations)
