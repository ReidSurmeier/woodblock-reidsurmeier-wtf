"""D3.1 RED — ULID minter."""

from __future__ import annotations


def test_ulid_is_26_chars() -> None:
    from backend.mcp.paths import new_ulid

    u = new_ulid()
    assert isinstance(u, str)
    assert len(u) == 26


def test_ulid_is_lex_sortable_over_time() -> None:
    import time

    from backend.mcp.paths import new_ulid

    a = new_ulid()
    time.sleep(0.005)
    b = new_ulid()
    assert a < b, f"ULIDs must sort chronologically: {a} >= {b}"


def test_ulid_is_unique_across_rapid_calls() -> None:
    from backend.mcp.paths import new_ulid

    ulids = {new_ulid() for _ in range(200)}
    assert len(ulids) == 200
