"""D5 — S2 SAM region prior stage.

Calls the v20 ``/api/sam`` HTTP endpoint via
:mod:`backend.services.v23.io.sam_client`, parses regions into
``SAMRegion`` records, persists mask PNGs under
``~/.woodblock/v23/cache/sam/<sha>/``, and caches the full result so
repeat calls on the same ``image_sha256`` skip the HTTP round trip.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

from backend.mcp import paths
from backend.mcp.errors import WoodblockError
from backend.services.v23.io import sam_client


@dataclass(frozen=True)
class SAMRegion:
    """One SAM-proposed region. Mask PNG is on disk, not inlined."""

    region_id: str
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    area_px: int
    mask_path: Path
    mean_oklab: tuple[float, float, float]


@dataclass(frozen=True)
class S2Result:
    """S2 stage output. Downstream stages key off ``image_sha256``."""

    image_sha256: str
    regions: list[SAMRegion] = field(default_factory=list)
    sam_wall_s: float = 0.0


class SamGatewayError(Exception):
    """Raised when the SAM HTTP gateway fails or times out."""

    def __init__(self, error: WoodblockError) -> None:
        super().__init__(error.message)
        self.error = error


def _cache_dir(image_sha256: str) -> Path:
    base = paths.WB_DATA_DIR / "cache" / "sam" / image_sha256
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_manifest(image_sha256: str) -> Path:
    return _cache_dir(image_sha256) / "result.json"


def _load_cached(image_sha256: str) -> S2Result | None:
    manifest = _cache_manifest(image_sha256)
    if not manifest.is_file():
        return None
    raw = json.loads(manifest.read_text())
    regions = [
        SAMRegion(
            region_id=r["region_id"],
            bbox=tuple(r["bbox"]),  # type: ignore[arg-type]
            area_px=int(r["area_px"]),
            mask_path=Path(r["mask_path"]),
            mean_oklab=tuple(r["mean_oklab"]),  # type: ignore[arg-type]
        )
        for r in raw["regions"]
    ]
    return S2Result(
        image_sha256=raw["image_sha256"],
        regions=regions,
        sam_wall_s=float(raw.get("sam_wall_s", 0.0)),
    )


def _persist_mask_png(image_sha256: str, region_id: str, mask_b64: str) -> Path:
    mask_bytes = base64.b64decode(mask_b64.encode("ascii"))
    out = _cache_dir(image_sha256) / f"{region_id}.png"
    out.write_bytes(mask_bytes)
    return out


def _persist_manifest(result: S2Result) -> None:
    payload = {
        "image_sha256": result.image_sha256,
        "sam_wall_s": result.sam_wall_s,
        "regions": [
            {
                "region_id": r.region_id,
                "bbox": list(r.bbox),
                "area_px": r.area_px,
                "mask_path": str(r.mask_path),
                "mean_oklab": list(r.mean_oklab),
            }
            for r in result.regions
        ],
    }
    _cache_manifest(result.image_sha256).write_text(json.dumps(payload, indent=2))


def run_s2_sam(image_path: Path | str, *, image_sha256: str) -> S2Result:
    """Fetch SAM regions for ``image_path`` keyed on ``image_sha256``.

    Cache-first: if a prior call materialised the result under the
    session cache it is returned without an HTTP round-trip.
    """
    cached = _load_cached(image_sha256)
    if cached is not None:
        return cached

    image_bytes = Path(image_path).read_bytes()
    try:
        body = sam_client.call_sam_endpoint(image_bytes, image_sha256=image_sha256)
    except sam_client._TIMEOUT_EXCEPTIONS as exc:
        raise SamGatewayError(
            WoodblockError(
                tier="degraded",
                code="SAM_TIMEOUT",
                message=f"SAM HTTP gateway timed out: {exc!r}",
                hint="auto-retry the request with upscale disabled or reduce image size",
                recoverable=True,
                retry_with={"upscale": False},
            )
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise SamGatewayError(
            WoodblockError(
                tier="degraded",
                code="SAM_GATEWAY_ERROR",
                message=f"SAM HTTP gateway error: {exc!r}",
                hint="check that the v20 colorsep-backend is reachable",
                recoverable=True,
            )
        ) from exc

    regions: list[SAMRegion] = []
    for raw in body.get("regions", []):
        mask_path = _persist_mask_png(image_sha256, raw["region_id"], raw["mask_png_b64"])
        regions.append(
            SAMRegion(
                region_id=raw["region_id"],
                bbox=tuple(raw["bbox"]),  # type: ignore[arg-type]
                area_px=int(raw["area_px"]),
                mask_path=mask_path,
                mean_oklab=tuple(raw["mean_oklab"]),  # type: ignore[arg-type]
            )
        )
    result = S2Result(
        image_sha256=body["image_sha256"],
        regions=regions,
        sam_wall_s=float(body.get("sam_wall_s", 0.0)),
    )
    _persist_manifest(result)
    return result
