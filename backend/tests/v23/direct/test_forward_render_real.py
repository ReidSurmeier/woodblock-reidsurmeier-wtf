"""D13.a RED — forward_render real path renders a persisted plan."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)


def _png_path(tmp_path: Path) -> Path:
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:, :8] = (200, 150, 100)
    arr[:, 8:] = (60, 100, 110)
    p = tmp_path / "test.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


def test_forward_render_real_returns_composite_path(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.mcp.tools import core
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    res = core.forward_render(plan.plan_id)
    assert res.ok is True
    assert res.data["composite_path"] is not None
    assert Path(res.data["composite_path"]).is_file()
    assert res.data["render_tier"] == "t1_mixbox"


def test_forward_render_unknown_plan_falls_back(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.forward_render("plan_does_not_exist")
    assert res.ok is True
    assert res.data["composite_path"] is None
    codes = {e.code for e in res.errors}
    assert "PLAN_NOT_FOUND" in codes or any(c.startswith("IMPL_PENDING") for c in codes)


def test_simulate_candidate_stack_alias_returns_same_data(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.mcp.tools import core
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    a = core.forward_render(plan.plan_id)
    b = core.simulate_candidate_stack(plan.plan_id)
    assert set(a.data.keys()) == set(b.data.keys())
    assert a.data["render_tier"] == b.data["render_tier"]
