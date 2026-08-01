"""D4 — S1 ingest_reference_image stage.

Load a PNG/JPG path, strip EXIF user fields (no metadata leaks), bound at
12 Mpx, sha256 the canonical (re-encoded) bytes, register the handle in
the active session (auto-creating one if none exists).

Returns ImageHandle. Downstream stages key off ``handle.image_sha256``.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.mcp.errors import WoodblockError
from backend.services.v23 import session as _sess

_MAX_MPX: float = 12.0


@dataclass(frozen=True)
class ImageHandle:
    """Reference to an ingested image stored under a session."""

    image_sha256: str
    width: int
    height: int
    array: np.ndarray
    canonical_bytes: bytes
    session_id: str


class IngestError(Exception):
    """Raised when S1 ingest rejects the input."""

    def __init__(self, error: WoodblockError) -> None:
        super().__init__(error.message)
        self.error = error


def _strip_exif(im: Image.Image) -> Image.Image:
    """Return a fresh image with NO EXIF, NO XMP, NO ICC user fields.

    PIL holds metadata in ``im.info``; copying pixels into a new Image
    drops it. ICC profile + simple format hints survive only via explicit
    re-attachment, which we deliberately skip.
    """
    rgb = im.convert("RGB")
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    return clean


def _canonicalise_png(clean: Image.Image) -> bytes:
    """Re-encode as PNG with no metadata. Deterministic for fixed pixels."""
    buf = io.BytesIO()
    clean.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def ingest_reference_image(path: str | Path) -> ImageHandle:
    """Load + strip + hash + register one image. Raises IngestError on refusal."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(
            WoodblockError(
                tier="refusal",
                code="INPUT_FILE_MISSING",
                message=f"file not found: {p}",
                hint="confirm the path exists and is readable",
                recoverable=True,
            )
        )

    try:
        raw = Image.open(p)
        raw.load()
    except Exception as exc:  # noqa: BLE001
        raise IngestError(
            WoodblockError(
                tier="refusal",
                code="INPUT_DECODE_FAILED",
                message=f"image decode failed: {exc!r}",
                hint="re-export the image as PNG or JPEG and retry",
                recoverable=True,
            )
        ) from exc

    mpx = (raw.width * raw.height) / 1_000_000.0
    if mpx > _MAX_MPX:
        raise IngestError(
            WoodblockError(
                tier="refusal",
                code="INPUT_TOO_LARGE",
                message=(
                    "Image too large. Resize below 12 megapixels and try again. "
                    f"(got {mpx:.1f} Mpx)"
                ),
                hint="downscale the long edge so width × height ≤ 12 million pixels",
                recoverable=True,
                context={"mpx": round(mpx, 2), "width": raw.width, "height": raw.height},
            )
        )

    clean = _strip_exif(raw)
    canonical = _canonicalise_png(clean)
    sha = hashlib.sha256(canonical).hexdigest()
    arr = np.array(clean, dtype=np.uint8)

    sid = _sess.current_session()
    if sid is None:
        s = _sess.new_session()
        _sess.set_current_session(s.session_id)
        sid = s.session_id

    # Persist a small JSON manifest of the ingest under the session
    session_root = _sess.paths.session_dir(sid)
    ingests = session_root / "ingests"
    ingests.mkdir(parents=True, exist_ok=True)
    (ingests / f"{sha}.json").write_text(
        json.dumps(
            {
                "image_sha256": sha,
                "source_path": str(p),
                "width": clean.width,
                "height": clean.height,
                "mpx": round(mpx, 4),
                "canonical_bytes_len": len(canonical),
            },
            indent=2,
        )
    )

    return ImageHandle(
        image_sha256=sha,
        width=clean.width,
        height=clean.height,
        array=arr,
        canonical_bytes=canonical,
        session_id=sid,
    )
