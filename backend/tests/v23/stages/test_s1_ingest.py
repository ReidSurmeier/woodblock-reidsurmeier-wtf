"""D4.1 + D4.2 RED — S1 ingest_reference_image stage.

S1 owns: load PNG/JPG bytes → strip EXIF (no user metadata leaks) →
sha256 the cleaned bytes → register the handle in the active session.
Returns an ImageHandle that downstream stages key off.
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


def _png_bytes(rgb: np.ndarray, *, exif: dict[int, str] | None = None) -> bytes:
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    save_kwargs: dict = {"format": "PNG"}
    if exif:
        ex = Image.Exif()
        for k, v in exif.items():
            ex[k] = v
        save_kwargs["exif"] = ex.tobytes()
    img.save(buf, **save_kwargs)
    return buf.getvalue()


def _jpeg_with_exif(rgb: np.ndarray, exif: dict[int, str]) -> bytes:
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    ex = Image.Exif()
    for k, v in exif.items():
        ex[k] = v
    img.save(buf, format="JPEG", exif=ex.tobytes(), quality=90)
    return buf.getvalue()


def test_loads_png_returns_rgb_array(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.stages.s1_ingest import ingest_reference_image

    rgb = np.tile(np.array([[10, 20, 30]], dtype=np.uint8), (32, 16, 1))  # (32,16,3)
    src = tmp_path / "x.png"
    src.write_bytes(_png_bytes(rgb))

    handle = ingest_reference_image(src)
    assert handle.array.shape == (32, 16, 3)
    assert handle.array.dtype == np.uint8
    assert handle.width == 16 and handle.height == 32
    np.testing.assert_array_equal(handle.array, rgb)


def test_strips_exif_user_fields(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.stages.s1_ingest import ingest_reference_image

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    # 0x013B Artist, 0x9286 UserComment, 0x8825 GPSInfo dir pointer
    dirty = tmp_path / "dirty.jpg"
    dirty.write_bytes(_jpeg_with_exif(rgb, {0x013B: "Reid", 0x9286: "secret note"}))

    handle = ingest_reference_image(dirty)
    # Round-trip through PIL to confirm EXIF gone in canonical bytes
    canonical = Image.open(io.BytesIO(handle.canonical_bytes))
    leftover = canonical.getexif()
    assert 0x013B not in leftover, f"Artist leaked: {leftover.get(0x013B)!r}"
    assert 0x9286 not in leftover, f"UserComment leaked: {leftover.get(0x9286)!r}"


def test_sha256_deterministic_after_strip(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.stages.s1_ingest import ingest_reference_image

    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    # Same pixels, different EXIF — canonical bytes (after strip) must hash identically
    a.write_bytes(_png_bytes(rgb, exif={0x013B: "Alice"}))
    b.write_bytes(_png_bytes(rgb, exif={0x013B: "Bob"}))

    ha = ingest_reference_image(a)
    hb = ingest_reference_image(b)
    assert ha.image_sha256 == hb.image_sha256
    assert len(ha.image_sha256) == 64


def test_rejects_over_12_mpx(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.mcp.errors import WoodblockError
    from backend.services.v23.stages.s1_ingest import (
        IngestError,
        ingest_reference_image,
    )

    # 13 Mpx = 4000 × 3250 — over the cap
    huge = np.zeros((3250, 4000, 3), dtype=np.uint8)
    src = tmp_path / "huge.png"
    src.write_bytes(_png_bytes(huge))

    with pytest.raises(IngestError) as ei:
        ingest_reference_image(src)
    err: WoodblockError = ei.value.error
    assert err.code == "INPUT_TOO_LARGE"
    assert err.tier == "refusal"
    assert err.recoverable is True


def test_registers_handle_in_session(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import current_session, new_session, set_current_session
    from backend.services.v23.stages.s1_ingest import ingest_reference_image

    s = new_session()
    set_current_session(s.session_id)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    src = tmp_path / "t.png"
    src.write_bytes(_png_bytes(rgb))

    handle = ingest_reference_image(src)
    assert handle.session_id == s.session_id
    # Handle persisted under the session dir
    assert (s.dir / "ingests" / f"{handle.image_sha256}.json").is_file()
    assert current_session() == s.session_id


def test_falls_back_to_auto_session_when_none_active(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import current_session
    from backend.services.v23.stages.s1_ingest import ingest_reference_image

    assert current_session() is None
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    src = tmp_path / "t.png"
    src.write_bytes(_png_bytes(rgb))

    handle = ingest_reference_image(src)
    # ingest auto-creates a session when none is active
    assert handle.session_id
    assert current_session() == handle.session_id
