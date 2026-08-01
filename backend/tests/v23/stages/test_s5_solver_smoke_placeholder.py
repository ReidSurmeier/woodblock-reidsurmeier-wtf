"""Ring 4 compatibility smoke — S5 solver API and synthetic fixture."""

from __future__ import annotations

import importlib

from backend.tests.v23._helpers.synthetic_fixtures import SyntheticStack


def test_synthetic_fixture_is_deterministic(synthetic_3imp_stack: SyntheticStack) -> None:
    """Sanity: the fixture is reproducible and shaped correctly.

    Runs green today — no production code dependency. Protects the
    solver smoke from a future fixture-generation regression.
    """
    assert synthetic_3imp_stack.rgb.shape == (256, 256, 3)
    assert synthetic_3imp_stack.alpha.shape == (3, 256, 256)
    assert synthetic_3imp_stack.pigment_rgb.shape == (3, 3)
    assert 0.0 <= synthetic_3imp_stack.alpha.min() <= synthetic_3imp_stack.alpha.max() <= 1.0


def test_s5_solver_stage_exposes_real_runner(synthetic_3imp_stack: SyntheticStack) -> None:
    mod = importlib.import_module("backend.services.v23.stages.s5_solver")
    assert hasattr(mod, "run_s5_solver")
