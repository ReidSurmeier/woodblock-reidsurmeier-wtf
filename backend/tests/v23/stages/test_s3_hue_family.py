"""D10.a RED — S3 hue-family classifier (real stage).

S3 turns an ingested image array into a per-pixel hue-family label map
+ per-family area stats. Used by:
- core.analyze_image (real measurables)
- core.build_hue_family_map (per-family map artifact)
- S4 Tan warm-start (initial pigment-region assignment)
- templates.suggest_template (picker hints)

7 families: cream / cool / flesh / warm / shadow / detail / accent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    import importlib

    from backend.mcp import paths

    importlib.reload(paths)


def test_classify_pure_flesh_image() -> None:
    """A solid flesh-tone image (RGB ~225, 175, 145) should classify as flesh."""
    from backend.services.v23.stages.s3_hue_family import classify_hue_families

    arr = np.full((32, 32, 3), [225, 175, 145], dtype=np.uint8)
    result = classify_hue_families(arr)
    assert result.dominant_family in ("flesh", "warm")
    assert result.family_areas["flesh"] + result.family_areas["warm"] > 0.6


def test_classify_pure_shadow_image() -> None:
    """A solid teal shadow (RGB ~60, 100, 100) should classify as shadow."""
    from backend.services.v23.stages.s3_hue_family import classify_hue_families

    arr = np.full((32, 32, 3), [60, 100, 100], dtype=np.uint8)
    result = classify_hue_families(arr)
    assert result.dominant_family in ("shadow", "cool")


def test_classify_pure_dark_image() -> None:
    """Black image should classify as detail."""
    from backend.services.v23.stages.s3_hue_family import classify_hue_families

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    result = classify_hue_families(arr)
    assert result.dominant_family == "detail"
    assert result.family_areas["detail"] > 0.9


def test_classify_pure_cream_image() -> None:
    """Cream image (RGB ~245, 235, 210) should classify as cream."""
    from backend.services.v23.stages.s3_hue_family import classify_hue_families

    arr = np.full((32, 32, 3), [245, 235, 210], dtype=np.uint8)
    result = classify_hue_families(arr)
    assert result.dominant_family == "cream"


def test_family_areas_sum_to_one() -> None:
    from backend.services.v23.stages.s3_hue_family import classify_hue_families

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    result = classify_hue_families(arr)
    total = sum(result.family_areas.values())
    assert abs(total - 1.0) < 1e-3, f"family areas sum to {total}, expected ≈ 1"


def test_label_map_shape_matches_input() -> None:
    from backend.services.v23.stages.s3_hue_family import classify_hue_families

    arr = np.zeros((48, 32, 3), dtype=np.uint8)
    result = classify_hue_families(arr)
    assert result.label_map.shape == (48, 32)
    assert result.label_map.dtype == np.uint8


def test_label_map_values_are_valid_family_indices() -> None:
    from backend.services.v23.stages.s3_hue_family import (
        FAMILY_LABEL_TO_INDEX,
        classify_hue_families,
    )

    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    result = classify_hue_families(arr)
    valid_indices = set(FAMILY_LABEL_TO_INDEX.values())
    unique_indices = set(np.unique(result.label_map).tolist())
    assert unique_indices.issubset(valid_indices)


def test_run_s3_persists_family_map_under_session(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import new_session, set_current_session
    from backend.services.v23.stages.s3_hue_family import run_s3_hue_family

    s = new_session()
    set_current_session(s.session_id)
    arr = np.full((16, 16, 3), [180, 80, 90], dtype=np.uint8)
    result = run_s3_hue_family(arr, image_sha256="a" * 64)

    expected_path = s.dir / "hue_family_maps" / f"{'a' * 64}.png"
    assert expected_path.is_file()
    assert result.label_map_path == expected_path


def test_run_s3_handles_no_active_session_auto_create(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    from backend.services.v23.session import current_session
    from backend.services.v23.stages.s3_hue_family import run_s3_hue_family

    assert current_session() is None
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    result = run_s3_hue_family(arr, image_sha256="b" * 64)
    assert current_session() is not None
    assert result.dominant_family == "detail"
