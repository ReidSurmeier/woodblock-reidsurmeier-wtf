"""D1.4 RED — banned-terms lint over v23 source.

Posture lock from CONTEXT.md + addendum-v2: never claim "recovered true
underlayers", "actual block", "ground-truth stack", or "detected
underprint". This grep is the source-of-truth enforcement.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SCAN_DIRS = ["backend/mcp", "backend/services/v23"]

BANNED = re.compile(
    r"recovered.{0,20}underlayer"
    r"|true underlayer"
    r"|actual block"
    r"|ground.truth.*stack"
    r"|detect.{0,20}underprint",
    re.IGNORECASE,
)


def test_no_banned_terms_in_v23_code() -> None:
    hits: list[tuple[str, int, str]] = []
    for d in SCAN_DIRS:
        path = REPO / d
        if not path.exists():
            continue
        for p in path.rglob("*.py"):
            for lineno, line in enumerate(p.read_text().splitlines(), start=1):
                if BANNED.search(line):
                    hits.append((str(p.relative_to(REPO)), lineno, line.strip()))
    assert not hits, "WB-LANG-01 violations:\n" + "\n".join(
        f"  {file_name}:{line_number}: {line}" for file_name, line_number, line in hits
    )
