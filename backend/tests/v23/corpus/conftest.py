"""Ring 5 fixtures — corpus tier loader + per-fixture parametrize.

Reads ``corpus_tiers.yaml`` once per session, exposes:

- ``corpus_tiers``: parsed YAML dict
- ``corpus_fixture``: a parametrized fixture yielding one corpus dir at
  a time with its tier label + ΔE gate. Tests iterate via
  ``@pytest.mark.parametrize`` over the same fixture ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

try:  # PyYAML is in dev extras; fall back to a minimal parser if missing.
    import yaml as _yaml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover — only on bare envs
    _yaml = None

TIERS_PATH = Path(__file__).with_name("corpus_tiers.yaml")
REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_ROOT = REPO_ROOT / "corpus"


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Tiny fallback parser — only handles the fixed shape of corpus_tiers.yaml.

    Used only when PyYAML isn't installed (e.g. ultra-bare CI). Strict:
    raises if it sees a structure it doesn't understand so silent drift
    against the canonical YAML is impossible.
    """
    result: dict[str, Any] = {}
    current_tier: dict[str, Any] | None = None
    current_key: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if indent == 0 and body.endswith(":"):
            current_tier = {}
            result[body[:-1]] = current_tier
            current_key = None
        elif indent == 2 and ":" in body and current_tier is not None:
            k, v = body.split(":", 1)
            v = v.strip()
            if not v:
                current_tier[k] = {} if k != "fixtures" else []
                current_key = k
            else:
                current_tier[k] = v.strip('"')
                current_key = None
        elif indent == 4 and ":" in body and current_tier is not None and current_key == "gate":
            k, v = body.split(":", 1)
            current_tier["gate"][k] = float(v.strip())
        elif body.startswith("- ") and current_tier is not None and current_key == "fixtures":
            current_tier["fixtures"].append(body[2:].strip())
        else:
            raise ValueError(f"unrecognised line in corpus_tiers.yaml: {raw!r}")
    return result


@dataclass(frozen=True)
class CorpusFixture:
    """One corpus directory paired with its tier label + ΔE gate."""

    id: str
    tier: str
    path: Path
    de_mean_gate: float
    de_p95_gate: float


def _load_tiers() -> dict[str, Any]:
    text = TIERS_PATH.read_text()
    if _yaml is not None:
        return _yaml.safe_load(text)
    return _minimal_yaml_load(text)


@pytest.fixture(scope="session")
def corpus_tiers() -> dict[str, Any]:
    return _load_tiers()


def _all_fixtures() -> list[CorpusFixture]:
    out: list[CorpusFixture] = []
    tiers = _load_tiers()
    for tier_name, body in tiers.items():
        gate = body["gate"]
        for fid in body["fixtures"]:
            out.append(
                CorpusFixture(
                    id=fid,
                    tier=tier_name,
                    path=CORPUS_ROOT / fid,
                    de_mean_gate=float(gate["de_mean"]),
                    de_p95_gate=float(gate["de_p95"]),
                )
            )
    return out


@pytest.fixture(params=_all_fixtures(), ids=lambda f: f"{f.tier}::{f.id}")
def corpus_fixture(request: pytest.FixtureRequest) -> CorpusFixture:
    return request.param  # type: ignore[no-any-return]
