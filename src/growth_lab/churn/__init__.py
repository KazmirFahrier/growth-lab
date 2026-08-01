"""Churn risk prediction — production ML vertical slice.

This package is sealed: it never reads simulator ground truth. All features
are derived from the warehouse (raw.* → staging.* → marts.*) and temporal
splits prevent leakage by construction.

Lives alongside the existing `experiments` and `warehouse` packages under the
same no-truth seal enforced by tests/test_no_truth_leak.py.
"""

from growth_lab.churn.features import build_training_set
from growth_lab.churn.train import train_pipeline

__all__ = ["build_training_set", "train_pipeline"]
