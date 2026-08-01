"""D14.n — export_svg + export_block_svgs + generate_carve_order real."""

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


@pytest.fixture
def real_plan(tmp_path: Path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    rng = np.random.default_rng(42)
    arr = rng.integers(80, 200, (16, 16, 3), dtype=np.uint8)
    img_path = tmp_path / "tiny.png"
    Image.fromarray(arr, "RGB").save(img_path)
    from backend.services.v23 import orchestrator as _orch

    return _orch.run_pipeline_partial(str(img_path), solve_profile="fast")


def test_export_svg_all_impressions_real(real_plan) -> None:
    if not real_plan.impressions:
        pytest.skip("solver produced 0 impressions")
    from backend.mcp.tools import carve

    r = carve.export_svg(real_plan.plan_id)
    assert r.ok is True, r.errors
    assert r.data["vectoriser"] == "skimage_marching_squares_v1"
    for entry in r.data["svg_paths"]:
        assert Path(entry["svg_path"]).is_file()
        content = Path(entry["svg_path"]).read_text()
        assert "<svg" in content
        assert 'viewBox="0 0' in content


def test_export_svg_specific_impression_real(real_plan) -> None:
    if not real_plan.impressions:
        pytest.skip("solver produced 0 impressions")
    iid = real_plan.impressions[0]["id"]
    from backend.mcp.tools import carve

    r = carve.export_svg(real_plan.plan_id, impression_ids=[iid])
    assert r.ok is True, r.errors
    assert len(r.data["svg_paths"]) == 1
    assert r.data["svg_paths"][0]["impression_id"] == iid


def test_export_svg_unknown_id_refuses(real_plan) -> None:
    from backend.mcp.tools import carve

    r = carve.export_svg(real_plan.plan_id, impression_ids=["imp_does_not_exist"])
    assert r.ok is False
    assert r.errors[0].code == "UNKNOWN_IMPRESSION_ID"


def test_export_block_svgs_real(real_plan) -> None:
    if real_plan.block_count == 0:
        pytest.skip("S7 did not pack blocks")
    from backend.mcp.tools import carve

    r = carve.export_block_svgs(real_plan.plan_id)
    assert r.ok is True, r.errors
    assert r.data["block_count"] == real_plan.block_count
    for block_id, path in r.data["block_svg_paths"].items():
        assert Path(path).is_file()
        assert block_id.startswith("blk_")
    assert len(r.data["kento_marks"]) >= 3


def test_generate_carve_order_real(real_plan) -> None:
    if not real_plan.impressions:
        pytest.skip("solver produced 0 impressions")
    from backend.mcp.tools import carve

    r = carve.generate_carve_order(real_plan.plan_id)
    assert r.ok is True, r.errors
    assert r.data["step_count"] == len(real_plan.impressions)
    for step in r.data["steps"]:
        assert "tool_recommendation" in step
        assert step["block_id"].startswith("blk_")
        assert step["impression_id"] in [imp["id"] for imp in real_plan.impressions]
    assert isinstance(r.data["estimated_hours"], float)
