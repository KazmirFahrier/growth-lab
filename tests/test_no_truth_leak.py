"""Structural seal preventing production code from reading simulator truth."""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "growth_lab"
FORBIDDEN_COMMON = ("load_truth", "truth.yaml", "simulator.params", "users_latent", "sim_hidden")
SEALED_PACKAGES: dict[str, tuple[str, ...]] = {
    "warehouse": FORBIDDEN_COMMON,
    "experiments": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "causal": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "marketing": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "forecasting": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "risk": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "reporting": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "integrations": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "service": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "churn": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "serve": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
    "monitor": (*FORBIDDEN_COMMON, "growth_lab.simulator"),
}


def test_sealed_package_list_is_current() -> None:
    """A new production package cannot silently opt out of the audit."""
    on_disk = {
        path.name
        for path in SRC.iterdir()
        if path.is_dir() and (path / "__init__.py").exists() and path.name != "simulator"
    }
    assert on_disk == set(SEALED_PACKAGES), (
        f"packages missing from seal audit: {on_disk ^ set(SEALED_PACKAGES)}"
    )


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
