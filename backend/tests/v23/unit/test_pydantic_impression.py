"""D2.2/D2.3/D2.4 RED — Impression + Mask + PullGroup."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


def test_mask_path_validates_png_extension(tmp_path: Path) -> None:
    from backend.services.v23.types import Mask

    good = tmp_path / "state.png"
    good.write_bytes(b"\x89PNG\r\n\x1a\n")
    m = Mask(
        width=10,
        height=10,
        state_path=good,
        confidence="visible-in-final",
        confidence_dE=1.4,
    )
    assert m.confidence == "visible-in-final"
    # alpha is optional
    assert m.alpha_path is None
    # round-trip
    j = m.model_dump_json()
    restored = Mask.model_validate_json(j)
    assert restored.state_path == m.state_path

    with pytest.raises(ValidationError):
        Mask(
            width=10,
            height=10,
            state_path=tmp_path / "state.jpg",
            confidence="visible-in-final",
            confidence_dE=1.4,
        )


def test_impression_validates_threestate_mask_label(tmp_path: Path) -> None:
    from backend.services.v23.types import Impression, Mask

    state_path = tmp_path / "s.png"
    state_path.write_bytes(b"\x89PNG")
    mask = Mask(
        width=64,
        height=64,
        state_path=state_path,
        confidence="inferred-underprint",
        confidence_dE=2.1,
    )
    imp = Impression(
        id="imp_007",
        order_step=4,
        block_id="blk_03",
        block_face_id="blk_03::face_a",
        pull_group=1,
        pigment_id="cadmium_yellow",
        mask=mask,
        hidden_coverage_ref="hc_imp_007",
        luminance_okL=0.78,
        coverage_pct=12.4,
    )
    assert imp.order_step == 4
    # ambiguous confidence requires alt_ids
    j = imp.model_dump_json()
    Impression.model_validate_json(j)


def test_pull_group_derives_from_impressions() -> None:
    from backend.services.v23.types import PullGroup

    pg = PullGroup(
        block_face_id="blk_03::face_a",
        order_step=4,
        pull_group=1,
        impression_ids=("imp_007", "imp_008"),
        label="warm trio",
    )
    assert pg.impression_ids == ("imp_007", "imp_008")
    j = pg.model_dump_json()
    restored = PullGroup.model_validate_json(j)
    assert restored == pg


def test_pull_group_requires_at_least_one_impression() -> None:
    from backend.services.v23.types import PullGroup

    with pytest.raises(ValidationError):
        PullGroup(
            block_face_id="blk_03::face_a",
            order_step=4,
            pull_group=1,
            impression_ids=(),
        )
