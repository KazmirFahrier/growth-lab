"""The campaign-copilot bridge: growth-lab as an agent toolkit.

This package is sealed: it never imports the simulator or reads ground
truth. MMM parameters arrive as a truth-free JSON artifact produced by
`python -m growth_lab export-mmm`.
"""

from growth_lab.integrations.contract import Tool, ToolResult, ToolSpec
from growth_lab.integrations.tools import (
    BudgetPlannerTool,
    ForecastRevenueTool,
    LtvSummaryTool,
    QueryGrowthMetricsTool,
    default_toolkit,
)

__all__ = [
    "BudgetPlannerTool",
    "ForecastRevenueTool",
    "LtvSummaryTool",
    "QueryGrowthMetricsTool",
    "Tool",
    "ToolResult",
    "ToolSpec",
    "default_toolkit",
]
