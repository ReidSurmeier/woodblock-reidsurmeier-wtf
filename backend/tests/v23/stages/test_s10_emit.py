"""D11.e RED — S10 ZIP emitter (real export_print_plan).

Bundles the persisted PartialPlan + forward-rendered composite_preview +
per-impression PNGs + manifest.json (v23.0 schema) + recipe.md into a
single ZIP. Per addendum-v3 fix 6: NEVER claim "recovered underlayers";
the manifest carries the "plausible underprint candidates" posture verbatim.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)


def _png_path(tmp_path: Path, h: int = 16, w: int = 16) -> Path:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (200, 150, 100)
    arr[:, w // 2 :] = (60, 100, 110)
    p = tmp_path / "test.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


def test_emit_zip_creates_archive_with_required_entries(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.orchestrator import run_pipeline_partial
    from backend.services.v23.stages.s10_emit import emit_plan_zip

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    zip_path = emit_plan_zip(plan.plan_id, out_dir=tmp_path)
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "recipe.md" in names
        assert "composite_preview.png" in names
        assert "README.txt" in names
        # At least one per-impression entry under impressions/
        assert any(n.startswith("impressions/") and n.endswith(".png") for n in names)


def test_emit_zip_manifest_uses_v23_schema(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.orchestrator import run_pipeline_partial
    from backend.services.v23.stages.s10_emit import emit_plan_zip

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    zip_path = emit_plan_zip(plan.plan_id, out_dir=tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["schema_version"] == "v23.0"
    assert "plan_id" in manifest
    assert "impressions" in manifest
    assert "blocks" in manifest
    assert "calibration" in manifest
    assert manifest["calibration"]["source"] == "generic_mixbox_13"


def test_emit_zip_recipe_carries_pre_mixed_qualifier(tmp_path: Path, monkeypatch) -> None:
    """Addendum-v4 WB-LANG-02: t1_mixbox must declare 'as if pre-mixed' qualifier."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.orchestrator import run_pipeline_partial
    from backend.services.v23.stages.s10_emit import emit_plan_zip

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    zip_path = emit_plan_zip(plan.plan_id, out_dir=tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        recipe = zf.read("recipe.md").decode("utf-8").lower()
    assert "pre-mixed" in recipe or "as if mixed" in recipe


def test_emit_zip_banned_terms_lint(tmp_path: Path, monkeypatch) -> None:
    """WB-LANG-01: manifest + recipe must NOT claim recovered underlayers."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.orchestrator import run_pipeline_partial
    from backend.services.v23.stages.s10_emit import emit_plan_zip

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    zip_path = emit_plan_zip(plan.plan_id, out_dir=tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        manifest = zf.read("manifest.json").decode("utf-8").lower()
        recipe = zf.read("recipe.md").decode("utf-8").lower()
    for banned in (
        "recovered underlayer",
        "true underlayer",
        "ground-truth stack",
        "ground truth stack",
    ):
        assert banned not in manifest, f"manifest has banned term: {banned}"
        assert banned not in recipe, f"recipe has banned term: {banned}"


def test_emit_zip_rejects_unknown_plan(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.stages.s10_emit import EmitError, emit_plan_zip

    with pytest.raises(EmitError):
        emit_plan_zip("plan_does_not_exist", out_dir=tmp_path)


def test_emit_zip_composite_preview_is_valid_png(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.orchestrator import run_pipeline_partial
    from backend.services.v23.stages.s10_emit import emit_plan_zip

    plan = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    zip_path = emit_plan_zip(plan.plan_id, out_dir=tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        composite_bytes = zf.read("composite_preview.png")
    img = Image.open(io.BytesIO(composite_bytes))
    assert img.size == (plan.width, plan.height)
    assert img.mode == "RGB"
