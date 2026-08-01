"""D1.1 RED — v23-MCP scaffold shape test.

Asserts the minimum directory + file tree exists so D1.2 (pyproject) +
D1.3 (mock server) + D1.4 (banned-terms lint) can layer on top.

Failing this means the v23 build hasn't been scaffolded yet.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

REQUIRED_FILES = [
    "backend/mcp/__init__.py",
    "backend/services/v23/__init__.py",
    "backend/tests/v23/conftest.py",
    ".github/workflows/v23.yml",
]

REQUIRED_DIRS = [
    "backend/mcp",
    "backend/services/v23",
    "backend/tests/v23",
    "backend/tests/v23/scaffold",
    ".github/workflows",
]


def test_v23_dirs_exist() -> None:
    missing = [d for d in REQUIRED_DIRS if not (REPO / d).is_dir()]
    assert not missing, f"missing dirs: {missing}"


def test_v23_files_exist() -> None:
    missing = [f for f in REQUIRED_FILES if not (REPO / f).is_file()]
    assert not missing, f"missing files: {missing}"
