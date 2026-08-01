"""Figure provenance: every number in a report traces to its computation.

A `Figure` is a value plus its lineage — the SQL text or estimator that
produced it. Reports are assembled exclusively from `Figure.render()`
output, so an unsourced number cannot appear in a memo; the provenance gate
in tests/test_reporting.py verifies this by exhaustive extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def fmt_scalar(value: float, unit: str) -> str:
    if unit == "$":
        return f"${value:,.2f}"
    if unit == "%":
        return f"{value * 100.0:.1f}%"
    if unit == "int":
        return f"{round(value):,}"
    return f"{value:,.2f}"


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Minimal markdown table with the same numeric formatting rules as
    scalar figures (no tabulate dependency)."""
    columns = list(frame.columns)

    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class Figure:
    slug: str
    title: str
    value: float | pd.DataFrame
    source: str
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.slug:
            raise ValueError("figure slug must be non-empty")
        if not self.source.strip():
            raise ValueError(f"figure {self.slug!r} has no provenance source")

    def render(self) -> str:
        if isinstance(self.value, pd.DataFrame):
            return frame_to_markdown(self.value)
        return fmt_scalar(float(self.value), self.unit)


@dataclass
class FigureRegistry:
    _figures: dict[str, Figure] = field(default_factory=dict)

    def add(self, figure: Figure) -> Figure:
        if figure.slug in self._figures:
            raise ValueError(f"duplicate figure slug: {figure.slug!r}")
        self._figures[figure.slug] = figure
        return figure

    def get(self, slug: str) -> Figure:
        if slug not in self._figures:
            raise KeyError(f"unknown figure: {slug!r}; known: {sorted(self._figures)}")
        return self._figures[slug]

    def all(self) -> tuple[Figure, ...]:
        return tuple(self._figures.values())
