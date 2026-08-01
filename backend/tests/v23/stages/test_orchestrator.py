"""D10.b RED — S1→S2→S3 orchestrator chains shipped stages.

The orchestrator at `backend.services.v23.orchestrator.run_pipeline_partial`
wires S1 (ingest) + S2 (SAM, mocked when no sidecar) + S3 (hue family map)
into a single call that returns a Plan-stub-with-real-artifacts.

S4-S10 still return IMPL_PENDING_* in the Plan but S1-S3 produce real
files under the session dir.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)


def _png_path(tmp_path: Path, h: int = 32, w: int = 32) -> Path:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, : w // 2] = (220, 175, 145)  # flesh-ish
    arr[: h // 2, w // 2 :] = (60, 120, 110)  # shadow teal
    arr[h // 2 :, : w // 2] = (245, 235, 210)  # cream
    arr[h // 2 :, w // 2 :] = (20, 20, 20)  # detail
    p = tmp_path / "test.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


def _mock_sam(monkeypatch) -> None:
    """Stub the SAM HTTP gateway so the orchestrator runs without v20 sidecar.

    Also un-sets the autouse ``WOODBLOCK_DISABLE_SAM=1`` from the v23
    conftest so the orchestrator actually CALLS the mocked sam_client.
    """
    import base64
    from io import BytesIO

    from backend.services.v23.io import sam_client

    monkeypatch.delenv("WOODBLOCK_DISABLE_SAM", raising=False)

    def fake(url, *, files, params, timeout):
        # Tiny single-region response
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        buf = BytesIO()
        Image.fromarray(mask, mode="L").save(buf, format="PNG")
        return {
            "image_sha256": params["image_sha256"],
            "regions": [
                {
                    "region_id": "rgn_000",
                    "bbox": [2, 2, 6, 6],
                    "area_px": 36,
                    "mask_png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "mean_oklab": [0.5, 0.0, 0.0],
                }
            ],
            "sam_wall_s": 0.01,
        }

    monkeypatch.setattr(sam_client, "_post_sam", fake)


def test_partial_pipeline_returns_plan_with_real_artifacts(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _mock_sam(monkeypatch)
    # Disable solver so the test stays fast and asserts S1-S3 artifacts only.
    monkeypatch.setenv("WOODBLOCK_DISABLE_SOLVER", "1")
    from backend.services.v23.orchestrator import run_pipeline_partial

    result = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="default")
    assert result.plan_id.startswith("plan_")
    assert result.image_sha256
    assert result.session_id
    # Real S1-S3 artifacts must be on disk
    assert result.hue_family_map_path is not None
    assert Path(result.hue_family_map_path).is_file()
    # SAM regions parsed
    assert len(result.sam_regions) >= 1
    # S5 disabled by env → status reflects skip
    assert result.solver_status == "IMPL_PENDING"
    assert result.impressions == []


def test_partial_pipeline_with_solver_returns_real_impressions(tmp_path: Path, monkeypatch) -> None:
    """When WOODBLOCK_DISABLE_SOLVER unset, S4+S5+S6+S7 run + populate plan."""
    _isolate(monkeypatch, tmp_path)
    _mock_sam(monkeypatch)
    monkeypatch.delenv("WOODBLOCK_DISABLE_SOLVER", raising=False)
    from backend.services.v23.orchestrator import run_pipeline_partial

    result = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    assert result.solver_status == "OK"
    assert len(result.impressions) >= 1
    assert result.reconstruction_dE_mean is not None
    assert result.solver_wall_s > 0.0
    for imp in result.impressions:
        assert "order_step" in imp
        assert "pigment_id" in imp
        assert "coverage_pct" in imp
    # S6 + S7 wired
    assert len(result.state_summary) == len(result.impressions)
    for s in result.state_summary:
        assert "visible_pct" in s
    assert result.block_count >= 1
    assert len(result.impression_to_block) == len(result.impressions)
    assert all("::face_" in v for v in result.impression_to_face.values())
    assert result.state_stack_path is not None
    assert Path(result.state_stack_path).is_file()


def test_partial_pipeline_persists_plan_json(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _mock_sam(monkeypatch)
    from backend.services.v23.orchestrator import load_plan, run_pipeline_partial

    result = run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast")
    loaded = load_plan(result.plan_id)
    assert loaded.plan_id == result.plan_id
    assert loaded.image_sha256 == result.image_sha256
    assert loaded.solve_profile == "fast"
    assert loaded.dominant_family == result.dominant_family


def test_partial_pipeline_strategy_template_suggestion(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _mock_sam(monkeypatch)
    # Synth a flesh-dominant image
    arr = np.full((32, 32, 3), [225, 175, 145], dtype=np.uint8)
    p = tmp_path / "flesh.png"
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    p.write_bytes(buf.getvalue())

    from backend.services.v23.orchestrator import run_pipeline_partial

    result = run_pipeline_partial(str(p), solve_profile="default")
    assert result.suggested_template == "portrait_emma"


def test_partial_pipeline_rejects_invalid_solve_profile(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _mock_sam(monkeypatch)
    from backend.services.v23.orchestrator import OrchestratorError, run_pipeline_partial

    with pytest.raises(OrchestratorError) as ei:
        run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="ultra")
    err = ei.value.error
    assert err.code == "INVALID_SOLVE_PROFILE"
    assert err.tier == "refusal"


def test_partial_pipeline_resolves_profile_m_prior_defaults() -> None:
    from backend.services.v23 import orchestrator

    assert orchestrator._warmstart_palette_size("fast", None) == 6
    assert orchestrator._warmstart_palette_size("default", None) == 8
    assert orchestrator._warmstart_palette_size("thorough", None) == 10
    assert orchestrator._warmstart_palette_size("fast", 12) == 12


def test_partial_pipeline_rejects_invalid_m_prior(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _mock_sam(monkeypatch)
    from backend.services.v23.orchestrator import OrchestratorError, run_pipeline_partial

    with pytest.raises(OrchestratorError) as ei:
        run_pipeline_partial(str(_png_path(tmp_path)), solve_profile="fast", m_prior=13)
    err = ei.value.error
    assert err.code == "INVALID_M_PRIOR"
    assert err.tier == "refusal"


def test_partial_pipeline_propagates_ingest_errors(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    _mock_sam(monkeypatch)
    from backend.services.v23.orchestrator import OrchestratorError, run_pipeline_partial

    with pytest.raises(OrchestratorError) as ei:
        run_pipeline_partial("/does/not/exist.png", solve_profile="default")
    assert ei.value.error.code == "INPUT_FILE_MISSING"


def test_load_unknown_plan_raises(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.orchestrator import OrchestratorError, load_plan

    with pytest.raises(OrchestratorError):
        load_plan("plan_does_not_exist")
