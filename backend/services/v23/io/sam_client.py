"""HTTP client for the v20 ``POST /api/sam`` route (merged at fd74964).

The transport is intentionally tiny so tests can monkeypatch ``_post_sam``
without spinning the real httpx stack. ``SAM_ENDPOINT_URL`` honors the
``WOODBLOCK_SAM_URL`` env so deployment can rewire the v20 sidecar host.

The 600s read timeout matches the v20-side SAM_TIMEOUT cap; see the memory
note ``feedback_colorsep_large_image_timeout.md``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

SAM_ENDPOINT_URL: str = os.environ.get("WOODBLOCK_SAM_URL", "http://100.67.23.102:8001/api/sam")
DEFAULT_TIMEOUT_S: float = 600.0
_TIMEOUT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    TimeoutError,
)


def _post_sam(
    url: str,
    *,
    files: dict[str, Any],
    params: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Thin wrapper around httpx.post returning parsed JSON. Mockable."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, files=files, params=params)
    resp.raise_for_status()
    return resp.json()


def call_sam_endpoint(
    image_bytes: bytes,
    *,
    image_sha256: str,
    min_region_area_px: int = 500,
    max_regions: int = 64,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """POST the image bytes to the v20 SAM route + return the parsed JSON body."""
    timeout = float(timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S)
    return _post_sam(
        SAM_ENDPOINT_URL,
        files={"image": ("image.png", image_bytes, "image/png")},
        params={
            "image_sha256": image_sha256,
            "min_region_area_px": min_region_area_px,
            "max_regions": max_regions,
        },
        timeout=timeout,
    )
