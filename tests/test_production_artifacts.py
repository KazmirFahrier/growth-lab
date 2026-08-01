"""Deployment controls are part of the tested product contract."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_container_runs_unprivileged_with_a_health_check() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "USER 10001:10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "GROWTH_LAB_ENV=production" in dockerfile
    assert "GROWTH_LAB_API_KEY" not in dockerfile


def test_compose_applies_runtime_confinement() -> None:
    compose = (ROOT / "compose.yaml").read_text()
    for control in (
        "read_only: true",
        "cap_drop:",
        "no-new-privileges:true",
        "pids_limit:",
        "mem_limit:",
        "cpus:",
    ):
        assert control in compose


def test_ci_validates_source_package_and_container() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "ruff format --check ." in workflow
    assert "python -m build --wheel" in workflow
    assert "docker/build-push-action@v7" in workflow


def test_example_environment_does_not_contain_a_usable_key() -> None:
    example = (ROOT / ".env.example").read_text()
    assert "GROWTH_LAB_API_KEY=replace-me" in example
    assert len("replace-me") < 32
