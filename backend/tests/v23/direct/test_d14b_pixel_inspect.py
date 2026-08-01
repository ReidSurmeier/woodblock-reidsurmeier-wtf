"""D14.b — dE_at + pigment_at against a real persisted plan.

Exercises the full path:
1. Run propose_stack with solver enabled on a tiny synthetic image
2. Load plan -> assert alpha_stack.npy + target.npy persist
3. Call dE_at(plan_id, x, y) -> assert real ΔE + RGB triples come back
4. Call pigment_at(plan_id, x, y) -> assert per-impression rows sorted by order_step
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    from backend.mcp import paths

    importlib.reload(paths)
    from backend.services.v23 import session as _sess

    importlib.reload(_sess)
    from backend.services.v23 import orchestrator as _orch

    importlib.reload(_orch)


def _write_test_image(p: Path, h: int = 16, w: int = 16) -> Path:
    """Tiny gradient PNG — enough for solver to converge fast."""
    rng = np.random.default_rng(42)
    arr = rng.integers(80, 200, (h, w, 3), dtype=np.uint8)
    img_path = p / "tiny.png"
    Image.fromarray(arr, "RGB").save(img_path)
    return img_path


@pytest.fixture
def real_plan(tmp_path: Path, monkeypatch):
    """Run full S1→S7 pipeline on a tiny image with solver enabled."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    img_path = _write_test_image(tmp_path)
    from backend.services.v23 import orchestrator as _orch

    plan = _orch.run_pipeline_partial(str(img_path), solve_profile="fast")
    return plan


def test_real_plan_persists_alpha_stack_and_target(real_plan) -> None:
    assert real_plan.alpha_stack_path is not None
    assert Path(real_plan.alpha_stack_path).is_file()
    target_path = Path(real_plan.alpha_stack_path).parent / "target.npy"
    assert target_path.is_file()
    # Shape sanity
    alpha = np.load(real_plan.alpha_stack_path)
    target = np.load(target_path)
    assert alpha.ndim == 3  # (M, H, W)
    assert target.shape == (real_plan.height, real_plan.width, 3)


def test_dE_at_returns_real_values(real_plan) -> None:
    from backend.mcp.tools import introspection

    r = introspection.dE_at(real_plan.plan_id, 5, 5)
    assert r.ok is True, r.errors
    assert r.data["dE"] is not None
    assert isinstance(r.data["dE"], float)
    assert r.data["dE"] >= 0.0
    assert r.data["target_rgb"] is not None
    assert r.data["rendered_rgb"] is not None
    assert len(r.data["target_rgb"]) == 3
    assert len(r.data["rendered_rgb"]) == 3
    assert r.data["metric"] == "deltaE76"


def test_dE_at_rejects_out_of_bounds(real_plan) -> None:
    from backend.mcp.tools import introspection

    r = introspection.dE_at(real_plan.plan_id, 999, 999)
    assert r.ok is False
    assert r.errors[0].code == "OUT_OF_BOUNDS"


def test_pigment_at_returns_real_stack(real_plan) -> None:
    from backend.mcp.tools import introspection

    r = introspection.pigment_at(real_plan.plan_id, 5, 5)
    assert r.ok is True, r.errors
    impressions = r.data["impressions"]
    # Should have at least one impression with alpha > threshold
    assert isinstance(impressions, list)
    for row in impressions:
        assert "order_step" in row
        assert "pigment_id" in row
        assert "pigment_name" in row
        assert "alpha" in row
        assert row["alpha"] >= 0.01
    # Sorted by order_step ascending
    order_steps = [r["order_step"] for r in impressions]
    assert order_steps == sorted(order_steps)


def test_pigment_at_rejects_out_of_bounds(real_plan) -> None:
    from backend.mcp.tools import introspection

    r = introspection.pigment_at(real_plan.plan_id, 999, 999)
    assert r.ok is False
    assert r.errors[0].code == "OUT_OF_BOUNDS"
