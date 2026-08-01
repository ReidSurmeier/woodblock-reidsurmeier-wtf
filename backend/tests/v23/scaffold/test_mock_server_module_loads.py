"""D1.3 RED — v23 mock server module imports cleanly.

In D1, the server is a stub: importable, exposes ``app`` (or callable) +
``main()`` entry point, but does NOT need FastMCP wired (that lands in
D19 per the build sequence).
"""

from __future__ import annotations


def test_v23_server_import_returns_fastmcp_app() -> None:
    from backend.mcp import v23_server

    assert hasattr(v23_server, "app"), "expected v23_server.app"
    assert hasattr(v23_server, "main"), "expected v23_server.main()"
    assert callable(v23_server.main)


def test_v23_server_app_name_is_woodblock_stack() -> None:
    from backend.mcp import v23_server

    # ``app`` may be a FastMCP instance later; for D1, any object exposing a
    # ``name`` attribute equal to ``woodblock_stack`` satisfies the contract.
    name = getattr(v23_server.app, "name", None)
    assert name == "woodblock_stack", f"expected name='woodblock_stack', got {name}"
