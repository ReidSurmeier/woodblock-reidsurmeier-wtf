"""D2.5 RED — Plan model rejects ``mode`` (addendum-v3 fix 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


def _stub_pigment():
    from backend.services.v23.types import Pigment

    return Pigment(
        id="cad_yellow",
        name="Cadmium Yellow",
        rgb=(254, 236, 0),
        hex="#feec00",
        family="warm",
        density=0.6,
    )


def _stub_plan(tmp_path: Path, **overrides):
    from backend.services.v23.types import (
        Block,
        Impression,
        Mask,
        Plan,
        PullGroup,
    )

    state_path = tmp_path / "s.png"
    state_path.write_bytes(b"\x89PNG")
    pig = _stub_pigment()
    mask = Mask(
        width=8, height=8, state_path=state_path, confidence="visible-in-final", confidence_dE=1.2
    )
    imp = Impression(
        id="imp_001",
        order_step=1,
        block_id="blk_00",
        block_face_id="blk_00::face_a",
        pull_group=0,
        pigment_id=pig.id,
        mask=mask,
        hidden_coverage_ref="hc_imp_001",
        luminance_okL=0.9,
        coverage_pct=22.5,
    )
    blk = Block(
        id="blk_00",
        face_ids=("blk_00::face_a",),
        sheet_w_mm=400,
        sheet_h_mm=560,
        impression_ids=("imp_001",),
        dsatur_color=0,
    )
    pg = PullGroup(
        block_face_id="blk_00::face_a",
        order_step=1,
        pull_group=0,
        impression_ids=("imp_001",),
    )
    kwargs = dict(
        plan_id="01HABC123",
        target_image_sha256="a" * 64,
        width=512,
        height=512,
        pigments=[pig],
        blocks=[blk],
        impressions=[imp],
        pull_groups=[pg],
        strategy_template="portrait_emma",
        solve_profile="default",
        reconstruction_dE_mean=1.30,
        reconstruction_dE_p95=2.80,
        solver_wall_s=85.4,
    )
    kwargs.update(overrides)
    return Plan(**kwargs)


def test_plan_round_trips_through_json(tmp_path: Path) -> None:
    p = _stub_plan(tmp_path)
    restored = type(p).model_validate_json(p.model_dump_json())
    assert restored.plan_id == p.plan_id
    assert restored.strategy_template == "portrait_emma"
    assert restored.solve_profile == "default"
    assert restored.schema_version == "v23.0"


def test_plan_rejects_mode_field(tmp_path: Path) -> None:
    """addendum-v3 fix 3 — ``mode`` must NOT appear on Plan."""
    with pytest.raises(ValidationError):
        _stub_plan(tmp_path, mode="default")


def test_plan_rejects_invalid_solve_profile(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _stub_plan(tmp_path, solve_profile="ultra")


def test_plan_rejects_invalid_strategy_template(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _stub_plan(tmp_path, strategy_template="medieval_woodcut")
