"""D2.6 RED — ToolResult envelope + WoodblockError 4-tier severity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_woodblock_error_has_four_tier_severity() -> None:
    from backend.mcp.errors import ErrorTier, WoodblockError

    assert {t.value for t in ErrorTier} == {"refusal", "warn", "degraded", "fail"}
    err = WoodblockError(
        tier="warn",
        code="INPUT_LOW_DPI",
        message="Low resolution.",
        recoverable=True,
    )
    assert err.tier == "warn"
    assert err.recoverable is True


def test_tool_result_envelope_carries_errors() -> None:
    from backend.mcp.errors import ToolResult, WoodblockError

    tr = ToolResult[dict](
        ok=True,
        data={"plan_id": "01HABC"},
        errors=[
            WoodblockError(
                tier="warn",
                code="OKLAB_DRIFT",
                message="dark-saturated region",
                recoverable=True,
            ),
            WoodblockError(
                tier="degraded",
                code="WALL_TIME_EXHAUSTED",
                message="solver stopped at level 2x",
                recoverable=True,
                retry_with={"solve_profile": "thorough"},
            ),
        ],
    )
    assert len(tr.errors) == 2
    assert tr.errors[1].retry_with == {"solve_profile": "thorough"}


def test_woodblock_error_rejects_unknown_tier() -> None:
    from backend.mcp.errors import WoodblockError

    with pytest.raises(ValidationError):
        WoodblockError(
            tier="catastrophic",  # type: ignore[arg-type]
            code="X",
            message="x",
            recoverable=False,
        )
