"""D1.2 RED — pyproject declares woodblock_stack v23-MCP package."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PYPROJECT = REPO / "pyproject.toml"


def test_pyproject_exists() -> None:
    assert PYPROJECT.is_file(), f"missing {PYPROJECT}"


def test_pyproject_declares_woodblock_stack() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    assert data["project"]["name"] == "woodblock_stack"
    assert data["project"]["version"]
    assert (
        "python" in data["project"]["requires-python"].lower()
        or ">=" in data["project"]["requires-python"]
    )


def test_pyproject_declares_woodblock_mcp_entrypoint() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    scripts = data["project"].get("scripts", {})
    assert "woodblock-mcp" in scripts, f"missing console script woodblock-mcp; got {scripts}"
