"""D5.1 RED — S2 SAM HTTP client.

Calls the v20-side ``POST /api/sam`` route (merged at fd74964) and parses
the response into ``SAMRegion`` Pydantic models for downstream stages.
HTTP transport is mocked at the client level so tests are fast + don't
depend on the live v20 sidecar.
"""

from __future__ import annotations

import base64
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


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _fake_sam_response(n_regions: int = 8, width: int = 64, height: int = 64) -> dict:
    """Build a plausible v20 /api/sam JSON response with N synthetic regions."""
    regions = []
    for i in range(n_regions):
        mask = np.zeros((height, width), dtype=np.uint8)
        # carve out a small square per region
        y0 = (i * 7) % (height - 8)
        x0 = (i * 11) % (width - 8)
        mask[y0 : y0 + 8, x0 : x0 + 8] = 255
        buf = io.BytesIO()
        Image.fromarray(mask, mode="L").save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        lightness = float(0.3 + i * 0.05)
        a_axis = float(-0.1 + i * 0.02)
        b_axis = float(0.1 - i * 0.02)
        regions.append(
            {
                "region_id": f"rgn_{i:03d}",
                "bbox": [x0, y0, 8, 8],
                "area_px": 64,
                "mask_png_b64": mask_b64,
                "mean_oklab": [lightness, a_axis, b_axis],
            }
        )
    return {
        "image_sha256": "f" * 64,
        "regions": regions,
        "sam_wall_s": 1.23,
    }


def test_mocked_sam_returns_8_regions(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.io import sam_client
    from backend.services.v23.stages import s2_sam

    fake = _fake_sam_response(n_regions=8)

    def fake_post(url: str, *, files, params, timeout: float) -> dict:
        return fake

    monkeypatch.setattr(sam_client, "_post_sam", fake_post)

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    img_path = tmp_path / "x.png"
    img_path.write_bytes(_png_bytes(rgb))

    result = s2_sam.run_s2_sam(img_path, image_sha256="f" * 64)
    assert result.image_sha256 == "f" * 64
    assert len(result.regions) == 8
    r0 = result.regions[0]
    assert r0.region_id == "rgn_000"
    assert r0.area_px == 64
    assert r0.bbox == (0, 0, 8, 8)
    assert isinstance(r0.mask_path, Path)
    assert r0.mask_path.is_file()
    assert len(r0.mean_oklab) == 3
    assert result.sam_wall_s > 0.0


def test_sam_client_serializes_min_region_area(monkeypatch) -> None:
    from backend.services.v23.io import sam_client

    captured: dict = {}

    def fake_post(url, *, files, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return {"image_sha256": "0" * 64, "regions": [], "sam_wall_s": 0.1}

    monkeypatch.setattr(sam_client, "_post_sam", fake_post)
    payload = b"\x89PNG-stub"
    sam_client.call_sam_endpoint(payload, image_sha256="0" * 64, min_region_area_px=250)
    assert captured["params"]["image_sha256"] == "0" * 64
    assert captured["params"]["min_region_area_px"] == 250
    assert captured["timeout"] >= 600.0


def test_s2_sam_caches_by_image_sha256(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.io import sam_client
    from backend.services.v23.stages import s2_sam

    calls = {"count": 0}

    def fake_post(url, *, files, params, timeout):
        calls["count"] += 1
        return _fake_sam_response(n_regions=3)

    monkeypatch.setattr(sam_client, "_post_sam", fake_post)

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    img_path = tmp_path / "x.png"
    img_path.write_bytes(_png_bytes(rgb))

    s2_sam.run_s2_sam(img_path, image_sha256="f" * 64)
    s2_sam.run_s2_sam(img_path, image_sha256="f" * 64)

    assert calls["count"] == 1  # second call must hit the cache


def test_sam_endpoint_url_honors_env(monkeypatch) -> None:
    monkeypatch.setenv("WOODBLOCK_SAM_URL", "http://sidecar.test:8001/api/sam")
    import importlib

    from backend.services.v23.io import sam_client

    importlib.reload(sam_client)
    assert sam_client.SAM_ENDPOINT_URL == "http://sidecar.test:8001/api/sam"


def test_sam_timeout_raises_structured_error(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.io import sam_client
    from backend.services.v23.stages import s2_sam

    class _FakeTimeout(Exception):
        pass

    def fake_post(url, *, files, params, timeout):
        raise _FakeTimeout("simulated read timeout")

    monkeypatch.setattr(sam_client, "_post_sam", fake_post)
    monkeypatch.setattr(sam_client, "_TIMEOUT_EXCEPTIONS", (_FakeTimeout,))

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    img_path = tmp_path / "x.png"
    img_path.write_bytes(_png_bytes(rgb))

    with pytest.raises(s2_sam.SamGatewayError) as ei:
        s2_sam.run_s2_sam(img_path, image_sha256="b" * 64)
    err = ei.value.error
    assert err.code == "SAM_TIMEOUT"
    assert err.tier == "degraded"
