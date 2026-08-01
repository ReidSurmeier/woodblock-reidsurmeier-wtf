"""D14.o — empirical overlay tier and tier comparison real wiring."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from PIL import Image


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    from backend.mcp import paths

    importlib.reload(paths)
    from backend.services.v23 import session as _sess

    importlib.reload(_sess)
    from backend.services.v23 import orchestrator as _orch

    importlib.reload(_orch)
    from backend.mcp.tools import overlay

    importlib.reload(overlay)


def _tiny_image(tmp_path: Path) -> Path:
    arr = np.zeros((12, 12, 3), dtype=np.uint8)
    arr[:, :6] = (220, 170, 120)
    arr[:, 6:] = (30, 80, 130)
    p = tmp_path / "target.png"
    Image.fromarray(arr, "RGB").save(p)
    return p


def _swatch_csv(tmp_path: Path) -> Path:
    p = tmp_path / "overprint_matrix.csv"
    p.write_text(
        "base,top,dilution,r,g,b\n"
        "cadmium_yellow,cobalt_blue,0.50,110,120,80\n"
        "cadmium_red,ivory_black,0.25,120,30,40\n"
    )
    return p


def test_upload_swatch_overprint_matrix_builds_t2_lut(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import overlay

    r = overlay.upload_swatch_overprint_matrix(str(_swatch_csv(tmp_path)))
    assert r.ok is True, r.errors
    assert r.data["rows_ingested"] == 2
    assert Path(r.data["lut_path"]).is_file()
    assert r.data["tier_after_ingest"] == "t2_empirical"

    tier = overlay.get_render_tier()
    assert tier.ok is True
    assert tier.data["empirical_lut_available"] is True
    assert tier.data["tier"] == "t2_empirical"


def test_simulate_overprint_uses_empirical_t2(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import overlay
    from backend.services.v23 import orchestrator as _orch

    overlay.upload_swatch_overprint_matrix(str(_swatch_csv(tmp_path)))
    plan = _orch.run_pipeline_partial(str(_tiny_image(tmp_path)), solve_profile="fast")
    r = overlay.simulate_overprint(plan.plan_id)
    assert r.ok is True, r.errors
    assert r.data["tier"] == "t2_empirical"
    assert Path(r.data["composite_path"]).is_file()
    assert r.data["empirical_lut"]["rows"] == 2


def test_compare_render_tiers_outputs_t1_t2_delta(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.tools import overlay
    from backend.services.v23 import orchestrator as _orch

    overlay.upload_swatch_overprint_matrix(str(_swatch_csv(tmp_path)))
    plan = _orch.run_pipeline_partial(str(_tiny_image(tmp_path)), solve_profile="fast")
    r = overlay.compare_render_tiers(plan.plan_id)
    assert r.ok is True, r.errors
    assert Path(r.data["tier_renders"]["t1_mixbox"]).is_file()
    assert Path(r.data["tier_renders"]["t2_empirical"]).is_file()
    assert r.data["tier_renders"]["t3_spectral"] is None
    assert "t2_empirical_vs_t1_mixbox" in r.data["dE_deltas"]
