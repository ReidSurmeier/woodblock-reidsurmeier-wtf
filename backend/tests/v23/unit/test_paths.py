"""D3.2 RED — WB_DATA_DIR + session_dir / plan_dir path resolver."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_data_dir_defaults_under_home(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("WOODBLOCK_HOME", raising=False)
    # Reload to pick up env state
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)

    assert paths.WB_DATA_DIR.name == "v23"
    assert paths.WB_DATA_DIR.parent.name == ".woodblock"


def test_data_dir_respects_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path / "wb"))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)
    assert paths.WB_DATA_DIR == (tmp_path / "wb" / "v23").resolve()


def test_session_dir_resolves_under_data_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)
    sid = "01HABC0000000000000000000A"
    sd = paths.session_dir(sid)
    assert sd.parent.name == "sessions"
    assert sd.name == sid
    assert paths.WB_DATA_DIR in sd.parents


def test_plan_dir_under_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)
    sid = "01HABC0000000000000000000A"
    pid = "plan_01HABC0000000000000000000B"
    pd = paths.plan_dir(sid, pid)
    assert pd.name == pid
    assert pd.parent.name == "plans"
    assert pd.parents[1] == paths.session_dir(sid)


def test_session_dir_rejects_path_traversal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)
    with pytest.raises(ValueError):
        paths.session_dir("../escape")
    with pytest.raises(ValueError):
        paths.session_dir("01HA/../../etc")


def test_plan_dir_rejects_path_traversal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)
    with pytest.raises(ValueError):
        paths.plan_dir("01HABC0000000000000000000A", "plan_../escape")
