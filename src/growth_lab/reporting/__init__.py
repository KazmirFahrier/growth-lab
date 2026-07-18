"""Decision delivery: provenance-tracked reports from the warehouse.

This package is sealed: it never imports the simulator or reads ground
truth. Every number it publishes carries its lineage; the provenance gate in
tests/test_reporting.py extracts every numeric token from the rendered memo
and demands a backing figure.
"""

from growth_lab.reporting.growth_review import GrowthReview, build_weekly_review
from growth_lab.reporting.pptx_export import export_pptx
from growth_lab.reporting.provenance import Figure, FigureRegistry, fmt_scalar, frame_to_markdown

__all__ = [
    "Figure",
    "FigureRegistry",
    "GrowthReview",
    "build_weekly_review",
    "export_pptx",
    "fmt_scalar",
    "frame_to_markdown",
]
