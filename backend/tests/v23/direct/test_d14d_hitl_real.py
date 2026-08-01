"""D14.d — compare_plans + merge_impressions real wiring."""

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


def _write_test_image(p: Path, h: int = 16, w: int = 16, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.integers(80, 200, (h, w, 3), dtype=np.uint8)
    img_path = p / f"tiny_{seed}.png"
    Image.fromarray(arr, "RGB").save(img_path)
    return img_path


@pytest.fixture(scope="module")
def generated_two_plans(tmp_path_factory: pytest.TempPathFactory):
    """Generate the expensive solver-backed pair once for this test module."""
    plan_root = tmp_path_factory.mktemp("hitl-plans")
    module_patch = pytest.MonkeyPatch()
    _isolate(module_patch, plan_root)
    module_patch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    module_patch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    from backend.services.v23 import orchestrator as _orch

    p1 = _orch.run_pipeline_partial(
        str(_write_test_image(plan_root, seed=1)), solve_profile="fast"
    )
    p2 = _orch.run_pipeline_partial(
        str(_write_test_image(plan_root, seed=2)), solve_profile="fast"
    )
    yield plan_root, p1, p2
    module_patch.undo()


@pytest.fixture
def two_plans(generated_two_plans, monkeypatch):
    """Restore the shared plan root before each test uses the cached pair."""
    plan_root, p1, p2 = generated_two_plans
    _isolate(monkeypatch, plan_root)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    monkeypatch.setenv("WOODBLOCK_DISABLE_SAM", "1")
    return p1, p2


def test_compare_plans_returns_real_dE_delta(two_plans) -> None:
    p1, p2 = two_plans
    from backend.mcp.tools import hitl

    r = hitl.compare_plans(p1.plan_id, p2.plan_id)
    assert r.ok is True, r.errors
    assert "dE_delta_mean" in r.data
    assert "dE_delta_p95" in r.data
    # Real values (not the all-zero mock)
    assert r.data["dE_delta_mean"] is not None
    assert r.data["impression_count_a"] == len(p1.impressions)
    assert r.data["impression_count_b"] == len(p2.impressions)


def test_compare_plans_unknown_refuses(tmp_path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import hitl

    r = hitl.compare_plans("nope_a", "nope_b")
    assert r.ok is False


def test_pin_region_forbid_real(two_plans) -> None:
    p1, _ = two_plans
    if not p1.impressions:
        pytest.skip("solver produced 0 impressions on this seed")
    pid = p1.impressions[0]["pigment_id"]
    pigment_names = [
        "cadmium_yellow",
        "hansa_yellow",
        "cadmium_orange",
        "cadmium_red",
        "quinacridone_magenta",
        "cobalt_violet",
        "ultramarine_blue",
        "cobalt_blue",
        "viridian_green",
        "forest_green",
        "burnt_sienna",
        "raw_umber",
        "ivory_black",
    ]
    pigment_name = pigment_names[pid]
    from backend.mcp.tools import hitl

    r = hitl.pin_region(p1.plan_id, {"bbox": [0, 0, 8, 8]}, "forbid", pigment_id=pigment_name)
    assert r.ok is True, r.errors
    assert r.data["new_plan_id"] != p1.plan_id
    # New plan must have ZERO alpha in that region for that pigment
    import numpy as np

    from backend.services.v23 import orchestrator as _orch

    new_plan = _orch.load_plan(r.data["new_plan_id"])
    new_alpha = np.load(new_plan.alpha_stack_path)
    # Find impressions with that pigment
    for i, p_idx in enumerate(new_plan.pigment_idx):
        if p_idx == pid:
            assert (new_alpha[i, 0:8, 0:8] == 0).all(), f"forbid failed for impression {i}"


def test_pin_region_force_real(two_plans) -> None:
    p1, _ = two_plans
    if not p1.impressions:
        pytest.skip("solver produced 0 impressions on this seed")
    pid = p1.impressions[0]["pigment_id"]
    pigment_names = [
        "cadmium_yellow",
        "hansa_yellow",
        "cadmium_orange",
        "cadmium_red",
        "quinacridone_magenta",
        "cobalt_violet",
        "ultramarine_blue",
        "cobalt_blue",
        "viridian_green",
        "forest_green",
        "burnt_sienna",
        "raw_umber",
        "ivory_black",
    ]
    pigment_name = pigment_names[pid]
    from backend.mcp.tools import hitl

    r = hitl.pin_region(p1.plan_id, {"bbox": [0, 0, 8, 8]}, "force", pigment_id=pigment_name)
    assert r.ok is True, r.errors


def test_pin_region_unknown_pigment_refuses(two_plans) -> None:
    p1, _ = two_plans
    from backend.mcp.tools import hitl

    r = hitl.pin_region(p1.plan_id, {"bbox": [0, 0, 8, 8]}, "forbid", pigment_id="unobtanium")
    assert r.ok is False
    assert r.errors[0].code == "UNKNOWN_PIGMENT"


def test_adjust_pull_groups_real_merge(two_plans) -> None:
    p1, _ = two_plans
    if p1.block_count < 2:
        pytest.skip("plan has <2 blocks")
    from backend.mcp.tools import hitl

    r = hitl.adjust_pull_groups(p1.plan_id, {"merge_pull_groups": [0, 1]})
    assert r.ok is True, r.errors
    assert r.data["new_plan_id"] != p1.plan_id


def test_alternative_stacks_real(two_plans) -> None:
    p1, _ = two_plans
    if not p1.impressions:
        pytest.skip("solver produced 0 impressions on this seed")
    from backend.mcp.tools import hitl

    r = hitl.alternative_stacks(p1.plan_id, n=2)
    assert r.ok is True, r.errors
    assert r.data["n_requested"] == 2
    assert r.data["n_succeeded"] >= 1
    alts = [a for a in r.data["alternatives"] if a.get("status") == "OK"]
    assert len(alts) >= 1
    for a in alts:
        assert a["dE_mean"] is not None
        assert a["plan_id"] != p1.plan_id
    # Ranked ascending
    assert alts == sorted(alts, key=lambda a: a["dE_mean"])


def test_alternative_stacks_invalid_n_refuses(two_plans) -> None:
    p1, _ = two_plans
    from backend.mcp.tools import hitl

    r = hitl.alternative_stacks(p1.plan_id, n=0)
    assert r.ok is False
    r2 = hitl.alternative_stacks(p1.plan_id, n=11)
    assert r2.ok is False


def test_merge_by_hue_family_unknown_family_refuses(two_plans) -> None:
    p1, _ = two_plans
    from backend.mcp.tools import hitl

    r = hitl.merge_impressions_by_hue_family(p1.plan_id, "unobtanium")
    assert r.ok is False
    assert r.errors[0].code == "UNKNOWN_HUE_FAMILY"


def test_merge_by_hue_family_real_or_insufficient(two_plans) -> None:
    """At least one family path either merges or refuses INSUFFICIENT — never IMPL_PENDING."""
    p1, _ = two_plans
    from backend.mcp.tools import hitl

    for fam in ("cream", "cool", "warm", "shadow", "detail"):
        r = hitl.merge_impressions_by_hue_family(p1.plan_id, fam)
        # Either succeeds with a new plan_id OR refuses with the real code
        if r.ok:
            assert r.data["new_plan_id"] != p1.plan_id
        else:
            assert r.errors[0].code in (
                "INSUFFICIENT_FAMILY_MATCH",
                "INSUFFICIENT_IMPRESSIONS",
                "UNKNOWN_IMPRESSION_ID",
            )
    # No IMPL_PENDING anywhere = real path always taken
    assert True  # smoke for above invariants


def test_split_impression_by_mask_island_real(two_plans) -> None:
    p1, _ = two_plans
    if not p1.impressions:
        pytest.skip("solver produced 0 impressions on this seed")
    target_id = p1.impressions[0]["id"]
    from backend.mcp.tools import hitl

    r = hitl.split_impression(p1.plan_id, target_id, by="mask_island")
    assert r.ok is True, r.errors
    assert r.data["new_plan_id"] != p1.plan_id
    assert r.data["child_count"] >= 1
    # New plan must load + have >=parent impression count
    from backend.services.v23 import orchestrator as _orch

    split = _orch.load_plan(r.data["new_plan_id"])
    assert len(split.impressions) >= len(p1.impressions)


def test_split_by_hue_subcluster_real(two_plans) -> None:
    p1, _ = two_plans
    if not p1.impressions:
        pytest.skip("solver produced 0 impressions on this seed")
    target_id = p1.impressions[0]["id"]
    from backend.mcp.tools import hitl

    r = hitl.split_impression(p1.plan_id, target_id, by="hue_subcluster")
    assert r.ok is True, r.errors
    # Should either split successfully or note single-cluster degenerate
    if r.data["child_count"] >= 2:
        assert r.data["new_plan_id"] != p1.plan_id
        assert len(r.data["cluster_centroids_rgb"]) >= 2
        from backend.services.v23 import orchestrator as _orch

        new = _orch.load_plan(r.data["new_plan_id"])
        assert len(new.impressions) >= len(p1.impressions)


def test_split_impression_unknown_id_refuses(two_plans) -> None:
    p1, _ = two_plans
    from backend.mcp.tools import hitl

    r = hitl.split_impression(p1.plan_id, "imp_does_not_exist", by="mask_island")
    assert r.ok is False
    assert r.errors[0].code == "UNKNOWN_IMPRESSION_ID"


def test_merge_impressions_real_creates_new_plan(two_plans) -> None:
    p1, _ = two_plans
    if len(p1.impressions) < 2:
        pytest.skip("solver produced <2 impressions on this seed")
    ids = [imp["id"] for imp in p1.impressions[:2]]
    from backend.mcp.tools import hitl

    r = hitl.merge_impressions(p1.plan_id, ids)
    assert r.ok is True, r.errors
    assert r.data["new_plan_id"] != p1.plan_id
    assert r.data["merged_count"] == 2
    # New plan must load
    from backend.services.v23 import orchestrator as _orch

    merged = _orch.load_plan(r.data["new_plan_id"])
    assert len(merged.impressions) == len(p1.impressions) - 1
