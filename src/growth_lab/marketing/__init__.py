"""Marketing measurement: MMM, attribution, LTV, and budget optimization.

This package is sealed: it never imports the simulator or reads ground
truth. Its estimates are scored externally in tests/test_marketing_recovery.py.
"""

from growth_lab.marketing.attribution import last_touch, linear_touch, markov_removal
from growth_lab.marketing.budget import Allocation, ChannelResponse, optimal_allocation
from growth_lab.marketing.ltv import LtvEstimate, PlanLtv, fit_geometric_ltv
from growth_lab.marketing.mmm import MmmFit, adstock, bootstrap_roas_ci, fit_mmm, saturate

__all__ = [
    "Allocation",
    "ChannelResponse",
    "LtvEstimate",
    "MmmFit",
    "PlanLtv",
    "adstock",
    "bootstrap_roas_ci",
    "fit_geometric_ltv",
    "fit_mmm",
    "last_touch",
    "linear_touch",
    "markov_removal",
    "optimal_allocation",
    "saturate",
]
