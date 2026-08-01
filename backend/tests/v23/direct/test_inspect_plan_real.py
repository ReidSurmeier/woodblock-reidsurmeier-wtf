"""D12.b RED — inspect_plan real artifact generation."""

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


def _png_path(tmp_path: Path, h: int = 16, w: int = 16) -> Path:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (210, 170, 140)
    arr[:, w // 2 :] = (50, 100, 120)
    p = tmp_path / "test.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


def test_inspect_plan_composite_returns_real_png_path(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.mcp.tools import core
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    res = core.inspect_plan(plan.plan_id, focus="composite")
    assert res.ok is True
    path = res.data["artifact_path"]
    assert path is not None
    assert Path(path).is_file()
    assert path.endswith(".png")
    # Decode + verify dims
    img = Image.open(path)
    assert img.size == (plan.width, plan.height)


def test_inspect_plan_recipe_returns_markdown_path(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.mcp.tools import core
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    res = core.inspect_plan(plan.plan_id, focus="recipe")
    assert res.ok is True
    path = res.data["artifact_path"]
    assert path.endswith(".md")
    text = Path(path).read_text()
    assert "Impression" in text
    assert "pre-mixed" in text.lower()


def test_inspect_plan_per_impression_returns_list(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.mcp.tools import core
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    res = core.inspect_plan(plan.plan_id, focus="per_impression")
    assert res.ok is True
    paths = res.data["impression_paths"]
    assert isinstance(paths, list)
    assert len(paths) == len(plan.impressions)
    for p in paths:
        assert Path(p).is_file()
        assert p.endswith(".png")


def test_inspect_plan_confidence_returns_summary(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.mcp.tools import core
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    res = core.inspect_plan(plan.plan_id, focus="confidence")
    assert res.ok is True
    assert "state_summary" in res.data
    assert len(res.data["state_summary"]) == len(plan.impressions)


def test_inspect_plan_unknown_plan_returns_neutral_mock(tmp_path: Path, monkeypatch) -> None:
    """Unknown plan_id falls back to neutral mock (backwards-compat)."""
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.inspect_plan("plan_does_not_exist", focus="composite")
    # Tool still returns ok=True with degraded errors so Opus doesn't error out
    assert res.ok is True
    codes = {e.code for e in res.errors}
    assert any(c.startswith("IMPL_PENDING") or c == "PLAN_NOT_FOUND" for c in codes)


def test_inspect_plan_rejects_invalid_focus(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.inspect_plan("plan_x", focus="quantum")  # type: ignore[arg-type]
    assert res.ok is False
    assert any(e.code == "INVALID_FOCUS_MODE" for e in res.errors)
