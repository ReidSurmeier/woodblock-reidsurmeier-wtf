"""v23 filesystem layout: ULID minter + WB_DATA_DIR + session/plan paths."""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode_crockford(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


_last_ts_ms = 0
_last_rand = 0


def new_ulid() -> str:
    """Mint a 26-char Crockford-base32 ULID. Lex-sortable by time."""
    global _last_ts_ms, _last_rand
    ts_ms = int(time.time() * 1000)
    if ts_ms == _last_ts_ms:
        # Monotonic increment of the random tail to preserve sort order within the same ms.
        _last_rand += 1
        rand = _last_rand
    else:
        _last_ts_ms = ts_ms
        rand = secrets.randbits(80)
        _last_rand = rand
    ts_part = _encode_crockford(ts_ms, 10)
    rand_part = _encode_crockford(rand, 16)
    return ts_part + rand_part


def _resolve_data_dir() -> Path:
    home = os.environ.get("WOODBLOCK_HOME")
    if home:
        base = Path(home).expanduser().resolve()
    else:
        base = (Path.home() / ".woodblock").resolve()
    return (base / "v23").resolve()


WB_DATA_DIR: Path = _resolve_data_dir()


def _validate_segment(name: str, label: str) -> None:
    if not name:
        raise ValueError(f"{label} must be non-empty")
    if "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise ValueError(f"{label} contains illegal characters: {name!r}")


def session_dir(session_id: str) -> Path:
    _validate_segment(session_id, "session_id")
    p = (WB_DATA_DIR / "sessions" / session_id).resolve()
    if WB_DATA_DIR not in p.parents:
        raise ValueError(f"session_dir escaped WB_DATA_DIR: {p}")
    return p


def plan_dir(session_id: str, plan_id: str) -> Path:
    _validate_segment(plan_id, "plan_id")
    sd = session_dir(session_id)
    p = (sd / "plans" / plan_id).resolve()
    if sd not in p.parents:
        raise ValueError(f"plan_dir escaped session_dir: {p}")
    return p
