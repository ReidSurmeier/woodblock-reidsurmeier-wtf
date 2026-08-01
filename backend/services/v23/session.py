"""D3.3 — session lifecycle for v23-MCP woodblock_stack.

Each session is a directory under ``~/.woodblock/v23/sessions/<ulid>/``
containing ``session.json`` (created_at, pid, heartbeat) + plans under
``plans/``. The ``current_session`` pointer at ``~/.woodblock/v23/current_session``
holds the active session ULID so multi-tool flows skip passing session_id
explicitly.

All writes are atomic via ``tempfile + os.replace`` and guarded by
``filelock.FileLock`` so concurrent claude-p ticks don't corrupt state.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Final

from filelock import FileLock

from backend.mcp import paths
from backend.mcp.paths import new_ulid

_SESSION_FILENAME: Final[str] = "session.json"
_LOCK_SUFFIX: Final[str] = ".lock"
_CURRENT_POINTER: Final[str] = "current_session"


@dataclass(frozen=True)
class Session:
    """A live session — the directory + the ULID."""

    session_id: str
    dir: Path


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def _atomic_write_json(target: Path, payload: dict) -> None:
    """Write JSON to ``target`` atomically via tempfile + rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _session_lock(sid: str) -> FileLock:
    return FileLock(str(paths.session_dir(sid) / (_SESSION_FILENAME + _LOCK_SUFFIX)))


def new_session() -> Session:
    """Create a fresh session directory + write session.json under lock."""
    sid = new_ulid()
    sdir = paths.session_dir(sid)
    sdir.mkdir(parents=True, exist_ok=True)
    now = _now_iso()
    payload = {
        "session_id": sid,
        "created_at": now,
        "pid": os.getpid(),
        "heartbeat": now,
        "schema_version": "v23.0",
    }
    with _session_lock(sid):
        _atomic_write_json(sdir / _SESSION_FILENAME, payload)
    return Session(session_id=sid, dir=sdir)


def touch_session(session_id: str) -> None:
    """Update the heartbeat timestamp on an existing session, under lock."""
    sdir = paths.session_dir(session_id)
    sj = sdir / _SESSION_FILENAME
    if not sj.is_file():
        raise ValueError(f"session not found: {session_id}")
    with _session_lock(session_id):
        payload = json.loads(sj.read_text())
        payload["heartbeat"] = _now_iso()
        payload["pid"] = os.getpid()
        _atomic_write_json(sj, payload)


def _pointer_path() -> Path:
    return paths.WB_DATA_DIR / _CURRENT_POINTER


def set_current_session(session_id: str) -> None:
    """Set the current-session pointer. Rejects unknown session IDs."""
    sdir = paths.session_dir(session_id)
    if not (sdir / _SESSION_FILENAME).is_file():
        raise ValueError(f"unknown session_id: {session_id}")
    pointer = _pointer_path()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=pointer.name + ".", suffix=".tmp", dir=str(pointer.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(session_id)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, pointer)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def current_session() -> str | None:
    """Return the current session ULID, or None if no pointer set."""
    pointer = _pointer_path()
    if not pointer.is_file():
        return None
    sid = pointer.read_text().strip()
    return sid or None
