"""D14.k — inspect_plan focus=heatmap and focus=quad real artifacts."""

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
    rng = np.random.default_rng(42)
    arr = rng.integers(80, 200, (h, w, 3), dtype=np.uint8)
    img_path = p / "tiny.png"
    Image.fromarray(arr, "RGB").save(img_path)
    return img_path


@pytest.fixture
def real_plan(tmp_path: Path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    img_path = _write_test_image(tmp_path)
    from backend.services.v23 import orchestrator as _orch

    return _orch.run_pipeline_partial(str(img_path), solve_profile="fast")


def test_inspect_heatmap_real(real_plan) -> None:
    from backend.mcp.tools import core

    r = core.inspect_plan(real_plan.plan_id, focus="heatmap")
    assert r.ok is True, r.errors
    assert r.data["artifact_path"] is not None
    assert Path(r.data["artifact_path"]).is_file()
    # Open + verify dimensions match plan
    img = Image.open(r.data["artifact_path"])
    assert img.size == (real_plan.width, real_plan.height)
    assert r.data["metric"] == "deltaE76"


def test_inspect_quad_real(real_plan) -> None:
    from backend.mcp.tools import core

    r = core.inspect_plan(real_plan.plan_id, focus="quad")
    assert r.ok is True, r.errors
    assert r.data["artifact_path"] is not None
    assert Path(r.data["artifact_path"]).is_file()
    # Quad grid is 2x2 the plan size
    img = Image.open(r.data["artifact_path"])
    assert img.size == (real_plan.width * 2, real_plan.height * 2)


def test_inspect_propose_stack_no_impl_pending(real_plan) -> None:
    """propose_stack should emit zero errors now — solver IS real."""
    from PIL import Image as _Image

    from backend.mcp.tools import core

    tmp_image = Path(real_plan.alpha_stack_path).parent / "for_repropose.png"
    arr = np.full((16, 16, 3), 128, dtype=np.uint8)
    _Image.fromarray(arr, "RGB").save(tmp_image)
    r = core.propose_stack(str(tmp_image), solve_profile="fast")
    assert r.ok is True
    # No more stale IMPL_PENDING_SOLVER banner
    assert all(e.code != "IMPL_PENDING_SOLVER" for e in r.errors)
