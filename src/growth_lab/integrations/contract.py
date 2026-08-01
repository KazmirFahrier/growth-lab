"""The agent tool contract, mirrored from campaign-copilot.

growth-lab does not depend on campaign-copilot (or vice versa); instead this
module reproduces the same structural contract — `ToolSpec`, `ToolResult`
with `success`/`failure`/`reference`, and the rule that **a tool failure is
a result, never an exception**. campaign-copilot's `Tool` protocol is
runtime-checkable and duck-typed, so tools built here satisfy it as-is, and
its grounding checker can call `numeric_facts()` on our results to verify
that every number the agent states was licensed by a tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Tool", "ToolResult", "ToolSpec"]


@dataclass(frozen=True)
class ToolSpec:
    """What the model is told about a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def as_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def as_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call. `content` is what the model reads;
    `data` is what the grounding checker reads."""

    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    grounds_numbers: bool = True

    @classmethod
    def failure(cls, code: str, message: str) -> ToolResult:
        return cls(ok=False, content=f"{code}: {message}", error_code=code)

    @classmethod
    def success(cls, content: str, **data: Any) -> ToolResult:
        return cls(ok=True, content=content, data=data)

    @classmethod
    def reference(cls, content: str, **data: Any) -> ToolResult:
        """A success that licenses no numbers (documents, not measurements)."""
        return cls(ok=True, content=content, data=data, grounds_numbers=False)

    def numeric_facts(self) -> list[float]:
        """Every number this result licenses the agent to state."""
        if not self.grounds_numbers:
            return []
        found: list[float] = []

        def walk(node: Any) -> None:
            if isinstance(node, bool):
                return
            if isinstance(node, int | float):
                found.append(float(node))
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list | tuple):
                for value in node:
                    walk(value)

        walk(self.data)
        return found

    def __str__(self) -> str:
        status = "ok" if self.ok else f"error[{self.error_code}]"
        return f"<ToolResult {status} {json.dumps(self.content[:80])}>"


@runtime_checkable
class Tool(Protocol):
    """Anything the agent may call."""

    @property
    def spec(self) -> ToolSpec: ...

    def run(self, **kwargs: Any) -> ToolResult: ...
