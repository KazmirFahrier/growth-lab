"""Assumption failures are hard stops, not footnotes."""

from __future__ import annotations


class AssumptionViolation(RuntimeError):
    """Raised when an identifying assumption fails its diagnostic.

    Estimators in this package refuse to return a number when the assumption
    that gives the number meaning is contradicted by the data. A confident
    wrong answer is worse than no answer.
    """
