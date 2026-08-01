"""D2.6 — ``WoodblockError`` 4-tier severity + ``ToolResult`` envelope.

Authority: ``/tmp/research-v23-mcp-edges.md`` (38-entry error catalog,
4-tier taxonomy: refusal / warn / degraded / fail).

Opus reads ``WoodblockError`` returns. ``retry_with`` carries concrete
parameter overrides Opus can pass into the next tool call. ``recoverable``
controls whether Opus offers a retry or relays terminally.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class ErrorTier(StrEnum):
    """4-tier error taxonomy (addendum-v2 §error-surface)."""

    REFUSAL = "refusal"  # input invalid; Opus asks for a different input
    WARN = "warn"  # non-blocking annotation
    DEGRADED = "degraded"  # partial result shipped; Opus may retry/escalate
    FAIL = "fail"  # hard failure; Opus relays diagnostic terminally


ErrorTierLiteral = Literal["refusal", "warn", "degraded", "fail"]


class WoodblockError(BaseModel):
    """A single structured error emitted by any v23-MCP tool.

    Multiple errors may fire on one tool invocation (e.g. ``OKLAB_DRIFT``
    warn + ``WALL_TIME_EXHAUSTED`` degraded) — they ride together inside a
    ``ToolResult.errors`` list. Opus reads each tier separately.
    """

    model_config = ConfigDict(frozen=True)

    tier: ErrorTierLiteral
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1)
    hint: str | None = None
    recoverable: bool = True
    retry_with: dict[str, Any] | None = None
    context: dict[str, Any] = Field(default_factory=dict)


T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    """Generic envelope returned by every v23-MCP tool.

    ``ok=True`` + ``data`` populated == happy path.
    ``ok=False`` + ``data=None`` + at least one ``fail``-tier error == hard fail.
    Mixed (``ok=True`` + ``data`` populated + non-empty ``errors``) carries
    warnings or degradation notes alongside a useable result.
    """

    model_config = ConfigDict(frozen=False)

    ok: bool = True
    data: T | None = None
    errors: list[WoodblockError] = Field(default_factory=list)
