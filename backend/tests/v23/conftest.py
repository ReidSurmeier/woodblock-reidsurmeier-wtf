"""pytest fixtures for the v23-MCP test rings.

Ring 1: direct python (`tests/v23/direct/` and `tests/v23/scaffold/`).
Ring 2: transport (`tests/v23/transport/`).
Ring 3: mock-Opus conversation (`tests/v23/conversation/`).
Ring 4: per-stage (`tests/v23/stages/`).
Ring 5: corpus regression (`tests/v23/corpus/`).

Autouse fixture ``disable_sam_by_default`` sets ``WOODBLOCK_DISABLE_SAM=1``
for every test so the orchestrator + propose_stack skip the v20 SAM HTTP
gateway (which is unreachable from the test environment). Tests that
specifically exercise SAM behaviour patch the env or the sam_client
module directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def disable_sam_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")


@pytest.fixture(autouse=True)
def disable_solver_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip S4+S5 by default. Solver-exercising tests delenv to opt in."""
    monkeypatch.setenv("WOODBLOCK_DISABLE_SOLVER", "1")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def corpus_dir(repo_root: Path) -> Path:
    return repo_root / "corpus"
