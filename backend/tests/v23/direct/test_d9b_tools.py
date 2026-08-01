"""D9B RED — Tier 1+2+3+4+5+6 tools (30 tools). One smoke per tool: signature + ToolResult shape."""

from __future__ import annotations

from pathlib import Path


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)


# Tier 1 — HITL (8 tools)


def test_pin_region_unknown_plan_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import hitl

    r = hitl.pin_region("plan_a", {"bbox": [0, 0, 10, 10]}, "force", pigment_id="cadmium_yellow")
    assert r.ok is False  # plan unknown -> refusal


def test_pin_region_rejects_unknown_action() -> None:
    from backend.mcp.tools import hitl

    r = hitl.pin_region("p", {}, "annihilate")  # type: ignore[arg-type]
    assert r.ok is False
    assert any(e.code == "INVALID_PIN_ACTION" for e in r.errors)


def test_alternative_stacks_unknown_plan_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import hitl

    r = hitl.alternative_stacks("plan_a", n=3)
    assert r.ok is False  # plan unknown -> refusal


def test_alternative_stacks_rejects_invalid_n() -> None:
    from backend.mcp.tools import hitl

    r = hitl.alternative_stacks("p", n=0)
    assert r.ok is False
    r2 = hitl.alternative_stacks("p", n=11)
    assert r2.ok is False


def test_generate_stack_candidates_is_alias(tmp_path: Path, monkeypatch) -> None:
    """Both must take the same path through alternative_stacks — same refusal."""
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import hitl

    a = hitl.alternative_stacks("p", n=2)
    b = hitl.generate_stack_candidates("p", n=2)
    assert a.ok is b.ok
    assert [e.code for e in a.errors] == [e.code for e in b.errors]


def test_compare_plans_unknown_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import hitl

    r = hitl.compare_plans("plan_a", "plan_b")
    assert r.ok is False  # both plans unknown -> refusal


def test_merge_impressions_requires_two() -> None:
    from backend.mcp.tools import hitl

    r = hitl.merge_impressions("plan_a", ["imp_001"])
    assert r.ok is False


def test_merge_impressions_unknown_plan_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import hitl

    r = hitl.merge_impressions("plan_a", ["imp_001", "imp_002"])
    assert r.ok is False  # plan unknown -> refusal


def test_split_impression_rejects_bad_mode() -> None:
    from backend.mcp.tools import hitl

    r = hitl.split_impression("p", "imp_001", by="quantum")  # type: ignore[arg-type]
    assert r.ok is False


def test_simplify_masks_returns_new_plan_id() -> None:
    from backend.mcp.tools import hitl

    r = hitl.simplify_masks_for_carving("plan_a")
    assert r.ok is True
    assert r.data["new_plan_id"] != "plan_a"


# Tier 2 — Calibration (5 tools)


def test_capture_swatch_rejects_missing_file(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import calibration

    r = calibration.capture_swatch("/nope/missing.jpg", {}, {})
    assert r.ok is False
    assert any(e.code == "SWATCH_FILE_MISSING" for e in r.errors)


def test_capture_swatch_invalid_layout_refuses(tmp_path: Path, monkeypatch) -> None:
    """Mock-shape layout {rows, cols} alone is invalid post-D14.l (needs origin + cell_px)."""
    _isolate(monkeypatch, tmp_path)
    import numpy as np
    from PIL import Image as _Image

    f = tmp_path / "swatch.png"
    _Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8), "RGB").save(f)
    from backend.mcp.tools import calibration

    r = calibration.capture_swatch(str(f), {"rows": 5, "cols": 13}, {"variant": "passport"})
    assert r.ok is False
    assert any(e.code == "INVALID_LAYOUT" for e in r.errors)


def test_apply_calibration_unknown_refuses(tmp_path: Path, monkeypatch) -> None:
    """Post-D14.l: apply_calibration validates the id exists (not generic_mixbox_13)."""
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import calibration

    r = calibration.apply_calibration("cal_fitted_test")
    assert r.ok is False
    assert any(e.code == "CALIBRATION_NOT_FOUND" for e in r.errors)


def test_apply_calibration_builtin_passthrough(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import calibration

    r = calibration.apply_calibration("generic_mixbox_13")
    assert r.ok is True
    assert r.data["applied"] is True


def test_list_calibrations_empty(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import calibration

    r = calibration.list_calibrations()
    assert r.ok is True
    assert r.data["calibrations"] == []


def test_inspect_unknown_calibration(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import calibration

    r = calibration.inspect_calibration("cal_does_not_exist")
    assert r.ok is False
    assert any(e.code == "CALIBRATION_NOT_FOUND" for e in r.errors)


# Tier 3 — Introspection (6 tools, all real read-only)


def test_get_pigments_returns_13() -> None:
    from backend.mcp.tools import introspection

    r = introspection.get_pigments()
    assert r.ok is True
    assert r.data["count"] == 13


def test_get_emma_priors_has_six_families() -> None:
    from backend.mcp.tools import introspection

    r = introspection.get_emma_priors()
    assert r.ok is True
    assert len(r.data["hue_families"]) == 7  # 6 + accent


def test_get_defaults_has_solve_profile_table() -> None:
    from backend.mcp.tools import introspection

    r = introspection.get_defaults()
    assert r.ok is True
    assert "solve_profile_walltime_s" in r.data


def test_solver_telemetry_returns_mock() -> None:
    from backend.mcp.tools import introspection

    r = introspection.solver_telemetry("plan_x")
    assert r.ok is True
    assert "pyramid_levels_completed" in r.data


def test_dE_at_rejects_negative_coords() -> None:
    from backend.mcp.tools import introspection

    r = introspection.dE_at("plan_x", -1, 0)
    assert r.ok is False


def test_pigment_at_unknown_plan_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import introspection

    r = introspection.pigment_at("plan_x", 5, 5)
    assert r.ok is False
    assert r.errors[0].code in ("PLAN_NOT_FOUND", "NO_ACTIVE_SESSION")


def test_dE_at_unknown_plan_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import introspection

    r = introspection.dE_at("plan_x", 5, 5)
    assert r.ok is False
    assert r.errors[0].code in ("PLAN_NOT_FOUND", "NO_ACTIVE_SESSION")


# Tier 4 — Session (4 tools, real)


def test_list_sessions_empty(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import session

    r = session.list_sessions()
    assert r.ok is True
    assert r.data["sessions"] == []


def test_purge_unknown_session(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import session

    r = session.purge_session("01HABC0000000000000000000A")
    assert r.ok is False


def test_set_unknown_session(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import session

    r = session.set_session("01HABC0000000000000000000A")
    assert r.ok is False


def test_current_session_returns_none_initially(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import session

    r = session.current_session()
    assert r.ok is True
    assert r.data["active"] is False


def test_list_sessions_rejects_huge_limit(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import session

    r = session.list_sessions(limit=1000)
    assert r.ok is False


# Tier 5 — Carve handoff (3 tools)


def test_export_svg_unknown_plan_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import carve

    r = carve.export_svg("plan_x")
    assert r.ok is False
    assert r.errors[0].code in ("PLAN_NOT_FOUND", "NO_ACTIVE_SESSION")


def test_export_block_svgs_unknown_plan_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import carve

    r = carve.export_block_svgs("plan_x")
    assert r.ok is False
    assert r.errors[0].code in ("PLAN_NOT_FOUND", "NO_ACTIVE_SESSION")


def test_generate_carve_order_unknown_plan_refuses(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import carve

    r = carve.generate_carve_order("plan_x")
    assert r.ok is False
    assert r.errors[0].code in ("PLAN_NOT_FOUND", "NO_ACTIVE_SESSION")


# Tier 6 — Overlay (4 tools)


def test_get_render_tier_defaults_to_t1(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import overlay

    r = overlay.get_render_tier()
    assert r.ok is True
    assert r.data["tier"] == "t1_mixbox"


def test_simulate_overprint_runs_t1(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import overlay

    r = overlay.simulate_overprint("plan_x")
    assert r.ok is True
    assert r.data["tier"] == "t1_mixbox"


def test_upload_swatch_matrix_missing_csv(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import overlay

    r = overlay.upload_swatch_overprint_matrix("/nope/missing.csv")
    assert r.ok is False


def test_upload_swatch_matrix_real_csv(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    csv = tmp_path / "matrix.csv"
    csv.write_text("base,top,dilution,r,g,b\n0,7,0.5,128,120,100\n")
    from backend.mcp.tools import overlay

    r = overlay.upload_swatch_overprint_matrix(str(csv))
    assert r.ok is True
    assert r.data["tier_after_ingest"] == "t2_empirical"


def test_compare_render_tiers_returns_dE_deltas() -> None:
    from backend.mcp.tools import overlay

    r = overlay.compare_render_tiers("plan_x")
    assert r.ok is True
    assert "tier_renders" in r.data


# Full surface count — addendum-v5 lock


def test_tier_modules_export_expected_tool_counts() -> None:
    from backend.mcp.tools import calibration, carve, hitl, introspection, overlay, session

    expected = {
        hitl: 10,  # 8 listed + 2 aliases
        calibration: 5,
        introspection: 6,
        session: 4,
        carve: 3,
        overlay: 4,
    }
    for mod, count in expected.items():
        assert len(mod.__all__) == count, (
            f"{mod.__name__} __all__ len {len(mod.__all__)} != {count}"
        )
