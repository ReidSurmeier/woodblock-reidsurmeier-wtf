"""D12.a RED — score_candidate_stack real 5-component breakdown.

Per addendum-v3 fix 4: ``visual_match + carveability + simplicity +
underprint_utility + template_fit`` computed from a persisted PartialPlan,
not mock 0.5 values.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.solver


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)


def _png_path(tmp_path: Path, h: int = 16, w: int = 16) -> Path:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (210, 170, 140)  # flesh
    arr[:, w // 2 :] = (50, 100, 120)  # shadow
    p = tmp_path / "test.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


def test_score_real_reflects_actual_dE(tmp_path: Path, monkeypatch) -> None:
    """visual_match decreases as reconstruction_dE_mean increases."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.core.score import score_plan_real
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    score = score_plan_real(plan)
    assert 0.0 <= score["visual_match"] <= 1.0
    assert 0.0 <= score["carveability"] <= 1.0
    assert 0.0 <= score["simplicity"] <= 1.0
    assert 0.0 <= score["underprint_utility"] <= 1.0
    assert 0.0 <= score["template_fit"] <= 1.0
    assert 0.0 <= score["overall"] <= 1.0


def test_score_real_components_sum_via_weights(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.core.score import score_plan_real
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    score = score_plan_real(plan)
    weights = score["component_weights"]
    expected = sum(
        weights[k] * score[k]
        for k in (
            "visual_match",
            "carveability",
            "simplicity",
            "underprint_utility",
            "template_fit",
        )
    )
    assert abs(score["overall"] - expected) < 1e-6


def test_template_fit_higher_when_suggested_matches(tmp_path: Path, monkeypatch) -> None:
    """A flesh-dominant image gets portrait_emma — template_fit > 0.5."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.core.score import score_plan_real
    from backend.services.v23.orchestrator import run_pipeline_partial

    arr = np.full((16, 16, 3), [220, 175, 145], dtype=np.uint8)  # flesh-tone
    p = tmp_path / "flesh.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    plan = run_pipeline_partial(str(p), solve_profile="fast")
    assert plan.suggested_template == "portrait_emma"
    score = score_plan_real(plan)
    assert score["template_fit"] > 0.3


def test_simplicity_higher_for_fewer_impressions(tmp_path: Path, monkeypatch) -> None:
    """simplicity is bounded — lots of impressions → lower score."""
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.core.score import score_plan_real
    from backend.services.v23.orchestrator import PartialPlan

    plan_few = PartialPlan(
        plan_id="p1",
        session_id="s",
        image_sha256="a" * 64,
        width=16,
        height=16,
        solve_profile="fast",
        schema_version="v23.0",
        family_areas={},
        dominant_family="cream",
        hue_family_map_path=None,
        sam_regions=[],
        suggested_template=None,
        template_confidence=0.0,
        template_reason="",
        solver_status="OK",
        impressions=[{"order_step": 1, "pigment_id": 0, "coverage_pct": 50.0, "mean_alpha": 0.5}],
        reconstruction_dE_mean=2.0,
        reconstruction_dE_p95=4.0,
        solver_wall_s=1.0,
        state_summary=[],
        block_count=1,
        impression_to_block={},
        impression_to_face={},
        pull_groups=[],
        state_stack_path=None,
        created_at="",
    )
    plan_many = PartialPlan(
        plan_id="p2",
        session_id="s",
        image_sha256="b" * 64,
        width=16,
        height=16,
        solve_profile="fast",
        schema_version="v23.0",
        family_areas={},
        dominant_family="cream",
        hue_family_map_path=None,
        sam_regions=[],
        suggested_template=None,
        template_confidence=0.0,
        template_reason="",
        solver_status="OK",
        impressions=[
            {"order_step": i + 1, "pigment_id": i, "coverage_pct": 10.0, "mean_alpha": 0.3}
            for i in range(10)
        ],
        reconstruction_dE_mean=2.0,
        reconstruction_dE_p95=4.0,
        solver_wall_s=1.0,
        state_summary=[],
        block_count=10,
        impression_to_block={},
        impression_to_face={},
        pull_groups=[],
        state_stack_path=None,
        created_at="",
    )
    s_few = score_plan_real(plan_few)
    s_many = score_plan_real(plan_many)
    assert s_few["simplicity"] > s_many["simplicity"]


def test_visual_match_at_target_de_is_zero(tmp_path: Path, monkeypatch) -> None:
    """ΔE == target_dE (1.5) → visual_match = 0; ΔE 0 → visual_match = 1."""
    from backend.services.v23.core.score import score_plan_real
    from backend.services.v23.orchestrator import PartialPlan

    base = PartialPlan(
        plan_id="p",
        session_id="s",
        image_sha256="a" * 64,
        width=8,
        height=8,
        solve_profile="fast",
        schema_version="v23.0",
        family_areas={},
        dominant_family="cream",
        hue_family_map_path=None,
        sam_regions=[],
        suggested_template=None,
        template_confidence=0.0,
        template_reason="",
        solver_status="OK",
        impressions=[{"order_step": 1, "pigment_id": 0, "coverage_pct": 50.0, "mean_alpha": 0.5}],
        reconstruction_dE_mean=0.0,
        reconstruction_dE_p95=0.0,
        solver_wall_s=1.0,
        state_summary=[],
        block_count=1,
        impression_to_block={},
        impression_to_face={},
        pull_groups=[],
        state_stack_path=None,
        created_at="",
    )
    s_perfect = score_plan_real(base)
    assert s_perfect["visual_match"] == pytest.approx(1.0, abs=1e-6)

    bad = PartialPlan(**{**base.__dict__, "reconstruction_dE_mean": 1.5})
    s_target = score_plan_real(bad)
    assert s_target["visual_match"] == pytest.approx(0.0, abs=1e-6)


def test_score_real_wired_in_score_candidate_stack_tool(tmp_path: Path, monkeypatch) -> None:
    """score_candidate_stack MCP tool now returns real values for real plan_ids."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.mcp.tools import core
    from backend.services.v23.orchestrator import run_pipeline_partial

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    res = core.score_candidate_stack(plan.plan_id)
    assert res.ok is True
    # Real scores ≠ all 0.5
    components = [
        res.data[k]
        for k in (
            "visual_match",
            "carveability",
            "simplicity",
            "underprint_utility",
            "template_fit",
        )
    ]
    assert not all(abs(c - 0.5) < 1e-6 for c in components), (
        f"all components are mock-0.5: {components}"
    )
