"""D9B — Tier 4 session tools (4 tools, all REAL — backing module shipped at D3.3)."""

from __future__ import annotations

import json
import shutil
from typing import Any

from backend.mcp import paths
from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23 import session as _sess


def list_sessions(limit: int = 20) -> ToolResult[dict[str, Any]]:
    if limit < 1 or limit > 200:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_LIMIT",
                    message=f"limit must be in [1, 200], got {limit}",
                    recoverable=True,
                ),
            ],
        )
    sessions_dir = paths.WB_DATA_DIR / "sessions"
    if not sessions_dir.is_dir():
        return ToolResult(ok=True, data={"sessions": [], "active": None})
    entries: list[dict[str, Any]] = []
    for sd in sorted(sessions_dir.iterdir(), reverse=True)[:limit]:
        sf = sd / "session.json"
        if not sf.is_file():
            continue
        try:
            payload = json.loads(sf.read_text())
            payload["plan_count"] = (
                sum(1 for _ in (sd / "plans").glob("*")) if (sd / "plans").is_dir() else 0
            )
            entries.append(payload)
        except json.JSONDecodeError:
            continue
    return ToolResult(ok=True, data={"sessions": entries, "active": _sess.current_session()})


def purge_session(session_id: str) -> ToolResult[dict[str, Any]]:
    try:
        sdir = paths.session_dir(session_id)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal", code="INVALID_SESSION_ID", message=str(exc), recoverable=True
                ),
            ],
        )
    if not sdir.is_dir():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="SESSION_NOT_FOUND",
                    message=f"session {session_id!r} not found",
                    recoverable=True,
                ),
            ],
        )
    shutil.rmtree(sdir)
    return ToolResult(ok=True, data={"session_id": session_id, "purged": True})


def set_session(session_id: str) -> ToolResult[dict[str, Any]]:
    try:
        _sess.set_current_session(session_id)
    except ValueError as exc:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal", code="SESSION_NOT_FOUND", message=str(exc), recoverable=True
                ),
            ],
        )
    return ToolResult(ok=True, data={"session_id": session_id, "active": True})


def current_session() -> ToolResult[dict[str, Any]]:
    sid = _sess.current_session()
    return ToolResult(ok=True, data={"session_id": sid, "active": sid is not None})


__all__ = ["list_sessions", "purge_session", "set_session", "current_session"]
