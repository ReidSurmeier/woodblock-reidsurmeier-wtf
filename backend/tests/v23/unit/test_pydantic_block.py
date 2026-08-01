"""D2.1 RED — Block + Pigment Pydantic round-trip."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError


def test_pigment_round_trips_through_json() -> None:
    from backend.services.v23.types import Pigment

    p = Pigment(
        id="cadmium_yellow",
        name="Cadmium Yellow",
        rgb=(254, 236, 0),
        hex="#feec00",
        family="warm",
        density=0.65,
        opacity_curve={"1x": 1.0, "0.5x": 0.6, "0.25x": 0.3},
    )
    j = p.model_dump_json()
    restored = Pigment.model_validate_json(j)
    assert restored == p
    assert restored.k_opacity(0.5) == 0.6
    assert restored.calibration_source == "generic_mixbox_13"


def test_pigment_rejects_invalid_family() -> None:
    from backend.services.v23.types import Pigment

    with pytest.raises(ValidationError):
        Pigment(
            id="x",
            name="x",
            rgb=(0, 0, 0),
            hex="#000000",
            family="not_a_family",  # type: ignore[arg-type]
            density=0.5,
        )


def test_pigment_rejects_invalid_calibration_source() -> None:
    from backend.services.v23.types import Pigment

    with pytest.raises(ValidationError):
        Pigment(
            id="x",
            name="x",
            rgb=(0, 0, 0),
            hex="#000000",
            family="cool",
            density=0.5,
            calibration_source="hand_picked",  # regex mismatch
        )


def test_pigment_is_frozen() -> None:
    from backend.services.v23.types import Pigment

    p = Pigment(id="x", name="x", rgb=(0, 0, 0), hex="#000000", family="cool", density=0.5)
    with pytest.raises(ValidationError):
        p.id = "y"  # type: ignore[misc]


def test_block_round_trips_through_json() -> None:
    from backend.services.v23.types import Block

    b = Block(
        id="blk_03",
        face_ids=("blk_03::face_a", "blk_03::face_b"),
        material="shina_plywood",
        sheet_w_mm=406.4,
        sheet_h_mm=558.8,
        impression_ids=("imp_001", "imp_004", "imp_011"),
        dsatur_color=2,
    )
    j = b.model_dump_json()
    restored = Block.model_validate_json(j)
    assert restored == b
    parsed = json.loads(j)
    assert parsed["id"] == "blk_03"
    assert parsed["face_ids"] == ["blk_03::face_a", "blk_03::face_b"]


def test_block_defaults_to_maple_plywood() -> None:
    from backend.services.v23.types import Block

    b = Block(
        id="blk_00",
        face_ids=("blk_00::face_a",),
        sheet_w_mm=300.0,
        sheet_h_mm=400.0,
        impression_ids=(),
        dsatur_color=0,
    )
    assert b.material == "maple_plywood"
