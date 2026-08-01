"""Ring 1 fixtures — direct-invocation tool tests.

Scope: lightweight per-test scratch dir + a mock image handle + a mock
plan id. No FastMCP, no transport, no GPU. These keep Ring 1 dev-loop
under a second per test.

Real implementations land alongside the D9 tool decorators; until then,
the fixtures stay structural so the placeholder smoke test xfails
cleanly and goes green the moment D9 wires the tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


@pytest.fixture
def tmp_session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test session root, isolated via ``WB_DATA_DIR``.

    Mirrors D3.2 path-resolution: every session/plan dir lives under
    ``$WB_DATA_DIR/sessions/<ulid>``. Setting it to a tmp dir prevents
    Ring 1 tests from touching ``~/.woodblock/v23``.
    """
    root = tmp_path / "wb-data"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WB_DATA_DIR", str(root))
    return root


@dataclass(frozen=True)
class MockImageHandle:
    """Stand-in for the D4 ``IngestResult`` returned by S1.

    Holds the minimum fields Ring 1 contract tests need: a session-bound
    path + the deterministic sha256 + image dims. Real type lands in D4.
    """

    path: Path
    sha256: str
    width: int
    height: int


@pytest.fixture
def mock_image_handle(tmp_session_dir: Path) -> MockImageHandle:
    """Return a structural ``ImageHandle`` without running S1 ingest.

    The on-disk file is a 1-byte stub; tests that need real pixels should
    use the synthetic helper in ``_helpers.synthetic_fixtures``.
    """
    p = tmp_session_dir / "ingest_mock.png"
    p.write_bytes(b"\x00")
    return MockImageHandle(
        path=p,
        sha256="0" * 64,
        width=256,
        height=256,
    )


@pytest.fixture
def mock_plan_id() -> str:
    """A deterministic ULID-shaped plan id for envelope-shape assertions."""
    return "plan_01HZK6V0MOCK0000000000000"
