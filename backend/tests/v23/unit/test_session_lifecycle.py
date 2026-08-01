"""D3.3 RED — session lifecycle (atomic write + filelock + PID heartbeat)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    # Force re-resolution of WB_DATA_DIR
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)


def test_new_session_writes_session_json(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import new_session

    s = new_session()
    assert s.session_id
    assert len(s.session_id) == 26  # ULID
    assert s.dir.is_dir()
    sj = s.dir / "session.json"
    assert sj.is_file()
    payload = json.loads(sj.read_text())
    assert payload["session_id"] == s.session_id
    assert payload["created_at"]
    assert payload["pid"] == os.getpid()
    assert payload["heartbeat"]


def test_session_dir_under_data_dir(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp import paths
    from backend.services.v23.session import new_session

    s = new_session()
    assert paths.WB_DATA_DIR in s.dir.parents
    assert s.dir.parent.name == "sessions"


def test_current_session_pointer_round_trips(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import current_session, new_session, set_current_session

    a = new_session()
    set_current_session(a.session_id)
    assert current_session() == a.session_id

    b = new_session()
    set_current_session(b.session_id)
    assert current_session() == b.session_id


def test_current_session_returns_none_when_unset(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import current_session

    assert current_session() is None


def test_set_current_session_rejects_unknown_id(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import set_current_session

    with pytest.raises(ValueError):
        set_current_session("01HABC0000000000000000000A")  # never created


def test_heartbeat_refreshes_on_touch(tmp_path: Path, monkeypatch) -> None:
    import time

    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import new_session, touch_session

    s = new_session()
    first = json.loads((s.dir / "session.json").read_text())["heartbeat"]
    time.sleep(0.01)
    touch_session(s.session_id)
    second = json.loads((s.dir / "session.json").read_text())["heartbeat"]
    assert second > first


def test_filelock_prevents_concurrent_write(tmp_path: Path, monkeypatch) -> None:
    """Two attempts to write the same session.json under lock — one waits."""
    import threading
    import time

    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import new_session, touch_session

    s = new_session()
    held = threading.Event()
    released = threading.Event()

    def holder() -> None:
        # Touch while holding the lock by sleeping in the writer path
        from filelock import FileLock

        lock = FileLock(str(s.dir / "session.json.lock"))
        with lock:
            held.set()
            time.sleep(0.05)
            released.set()

    t = threading.Thread(target=holder)
    t.start()
    held.wait(timeout=1.0)
    start = time.time()
    touch_session(s.session_id)
    elapsed = time.time() - start
    t.join()
    assert released.is_set()
    # touch_session had to wait ≥ part of the holder's sleep
    assert elapsed > 0.02
