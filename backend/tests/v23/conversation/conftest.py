"""Ring 3 fixtures — scripted MockOpus driver.

The :class:`ScriptedMockOpus` collects an expected ordered list of
``(tool_name, args_predicate)`` steps and replays them against a
callable (Ring 1 direct call or Ring 2 stdio call). Each ``step()``
returns the tool result; assertions live in the test body.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class ScriptedMockOpus:
    """Replays a fixed sequence of tool calls against a dispatcher.

    Used by Ring 3 placeholder tests to assert the canonical Pattern-A
    cold-start flow (analyze → propose → inspect → export) survives any
    refactor of the underlying tools.
    """

    dispatcher: Callable[[str, dict[str, Any]], Any]
    transcript: list[tuple[str, dict[str, Any], Any]] = field(default_factory=list)

    def step(
        self,
        tool: str,
        args: dict[str, Any],
        expect: Callable[[Any], bool] | None = None,
    ) -> Any:
        result = self.dispatcher(tool, args)
        self.transcript.append((tool, args, result))
        if expect is not None:
            assert expect(result), f"{tool} response failed: {result!r}"
        return result


def _direct_tool_dispatcher(tool: str, args: dict[str, Any]) -> Any:
    from backend.mcp.registry import call_mcp_tool

    return call_mcp_tool(tool, args)


@pytest.fixture
def mock_opus() -> ScriptedMockOpus:
    """Yield a :class:`ScriptedMockOpus` wired to the direct MCP registry."""
    return ScriptedMockOpus(dispatcher=_direct_tool_dispatcher)
