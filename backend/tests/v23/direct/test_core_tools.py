"""D9A RED — Tier 0 core flow tool decorators (10 tools, mock-real hybrid).

Per addendum-v5 ship-state: every tool wired with stable signature +
valid Pydantic + ToolResult[T]. Tools with backing implementation
return real data; tools awaiting full solver return ``degraded`` tier
``IMPL_PENDING`` with hint pointing at the substep that lands real logic.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def _png_path(tmp_path: Path, h: int = 32, w: int = 32) -> Path:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = 200
    arr[:, :, 1] = 150
    arr[:, :, 2] = 100
    p = tmp_path / "test.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)


def test_all_10_tier_0_tools_importable() -> None:
    from backend.mcp.tools import core

    expected = {
        "ingest_reference_image",
        "analyze_image",
        "build_hue_family_map",
        "propose_stack",
        "inspect_plan",
        "forward_render",
        "score_stack_delta_e",
        "score_candidate_stack",
        "export_print_plan",
        "generate_print_recipe_report",
    }
    for name in expected:
        assert hasattr(core, name), f"missing tool: {name}"
        assert callable(getattr(core, name)), f"not callable: {name}"


def test_ingest_reference_image_returns_tool_result(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.ingest_reference_image(str(_png_path(tmp_path)))
    assert res.ok is True
    assert res.data is not None
    assert res.data["image_sha256"]
    assert res.data["width"] == 32
    assert res.data["height"] == 32
    assert res.data["session_id"]


def test_analyze_image_returns_measurables_only(tmp_path: Path, monkeypatch) -> None:
    """Addendum-v3 fix 5: NO subject_label, NO intent classification."""
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.analyze_image(str(_png_path(tmp_path)))
    assert res.ok is True
    data = res.data
    assert "subject_label" not in data
    assert "intent" not in data
    assert "width" in data
    assert "height" in data
    assert "mpx" in data
    assert "dominant_family" in data
    assert "family_areas" in data


def test_build_hue_family_map_returns_per_family_areas(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.build_hue_family_map(str(_png_path(tmp_path)))
    assert res.ok is True
    fam = res.data["family_areas"]
    assert set(fam.keys()) >= {"cream", "cool", "flesh", "warm", "shadow", "detail", "accent"}
    total = sum(fam.values())
    assert abs(total - 1.0) < 0.05  # areas should sum to ≈ 1


def test_propose_stack_is_real_solver_post_d14h(tmp_path: Path, monkeypatch) -> None:
    """Post-D14.h: solver IS real. No IMPL_PENDING_SOLVER banner expected."""
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.propose_stack(str(_png_path(tmp_path)), solve_profile="default")
    assert res.ok is True
    assert res.data is not None
    assert res.data["plan_id"]
    # Solver is real now — no stale impl-pending banner
    codes = {e.code for e in res.errors}
    assert "IMPL_PENDING_SOLVER" not in codes


def test_propose_stack_rejects_invalid_solve_profile(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.propose_stack(str(_png_path(tmp_path)), solve_profile="ultra")
    assert res.ok is False
    codes = {e.code for e in res.errors}
    assert "INVALID_SOLVE_PROFILE" in codes
    assert any(e.tier == "refusal" for e in res.errors)


def test_inspect_plan_supports_six_focus_modes(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    for focus in ("composite", "heatmap", "per_impression", "confidence", "quad", "recipe"):
        res = core.inspect_plan("plan_stub", focus=focus)
        # In mock mode any focus must succeed (returning a placeholder reference)
        assert res.ok is True, f"focus={focus} unexpectedly failed: {res.errors}"
        assert res.data is not None


def test_forward_render_alias_simulate_candidate_stack(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    a = core.forward_render("plan_stub")
    b = core.simulate_candidate_stack("plan_stub")
    assert a.ok == b.ok
    assert set(a.data.keys()) == set(b.data.keys())


def test_score_stack_delta_e_returns_mean_p95(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.score_stack_delta_e("plan_stub")
    assert res.ok is True
    assert "dE_mean" in res.data
    assert "dE_p95" in res.data


def test_score_candidate_stack_returns_5_component_breakdown(tmp_path: Path, monkeypatch) -> None:
    """Return the five scoring components required by addendum-v3 fix 4."""
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.score_candidate_stack("plan_stub")
    assert res.ok is True
    data = res.data
    for k in (
        "overall",
        "visual_match",
        "carveability",
        "simplicity",
        "underprint_utility",
        "template_fit",
    ):
        assert k in data, f"missing component: {k}"
        assert 0.0 <= data[k] <= 1.0, f"{k} out of [0,1]: {data[k]}"
    assert "component_weights" in data
    assert "notes" in data


def test_export_print_plan_returns_zip_and_recipe_paths(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.export_print_plan("plan_stub", out_dir=str(tmp_path))
    assert res.ok is True
    assert "zip_path" in res.data
    assert "recipe_path" in res.data


def test_generate_print_recipe_report_returns_numbered_impressions(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.generate_print_recipe_report("plan_stub", format="markdown")
    assert res.ok is True
    md = res.data["markdown"]
    assert "Impression 01" in md
    assert "Impression 02" in md
    # Plain-language pigment-family description must appear
    lines = [ln for ln in md.splitlines() if ln.startswith("Impression ")]
    assert len(lines) >= 1


def test_export_print_plan_recipe_carries_mixing_qualifier(tmp_path: Path, monkeypatch) -> None:
    """Addendum-v4 WB-LANG-02: t1_mixbox must declare 'as if pre-mixed' qualifier."""
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import core

    res = core.generate_print_recipe_report("plan_stub", format="markdown")
    md = res.data["markdown"].lower()
    assert "pre-mixed" in md or "as if mixed" in md, (
        "WB-LANG-02 violation: t1 recipe must qualify Mixbox output as mixing-not-overprint"
    )
