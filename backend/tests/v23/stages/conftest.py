"""Ring 4 fixtures — per-stage isolation harness.

The ``synthetic_3imp_stack`` fixture is the workhorse: a deterministic
256×256 3-impression ground truth from
:mod:`backend.tests.v23._helpers.synthetic_fixtures`. S5 solver tests
use it for the 5-step smoke recovery (D7.1) without paying SAM cost.
"""

from __future__ import annotations

import pytest

from backend.tests.v23._helpers.synthetic_fixtures import (
    SyntheticStack,
    make_3imp_synthetic,
)


@pytest.fixture
def synthetic_3imp_stack() -> SyntheticStack:
    """Return a deterministic 256×256 3-impression ground truth.

    Identity hash is stable across runs (seed=0). Use ``.rgb`` as the
    image input the solver sees and ``.alpha`` as the answer key.
    """
    return make_3imp_synthetic(seed=0)
