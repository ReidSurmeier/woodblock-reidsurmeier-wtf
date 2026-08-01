"""Minimum-viable Pydantic factories for v23-MCP tests.

These wrap ``backend.services.v23.types`` so every ring can mint a valid
``Pigment`` / ``Block`` / ``Plan`` in one line without re-stating every
required field. Defaults intentionally match the smallest legal payload
that satisfies the v2 validators in D2.

Note: importing these from a test triggers ``backend.services.v23.types``
import. That module lands in D2 and is already green at scaffold time.
"""

from __future__ import annotations

from typing import Any

from backend.services.v23 import types as T


def make_pigment(**overrides: Any) -> T.Pigment:
    """Return a valid :class:`~backend.services.v23.types.Pigment`."""
    base: dict[str, Any] = dict(
        id="pig_test_01",
        name="Test Cool",
        rgb=(80, 110, 165),
        hex="#506ea5",
        family="cool",
        density=0.5,
        opacity_curve={"1x": 1.0, "0.5x": 0.55, "0.25x": 0.28},
    )
    base.update(overrides)
    return T.Pigment(**base)


def make_block(**overrides: Any) -> T.Block:
    """Return a valid :class:`~backend.services.v23.types.Block`."""
    base: dict[str, Any] = dict(
        id="blk_test_01",
        face_ids=("blk_test_01::face_a",),
        material="maple_plywood",
        sheet_w_mm=300.0,
        sheet_h_mm=400.0,
        impression_ids=(),
        dsatur_color=0,
    )
    base.update(overrides)
    return T.Block(**base)


def make_mask(**overrides: Any) -> T.Mask:
    """Return a valid :class:`~backend.services.v23.types.Mask`."""
    base: dict[str, Any] = dict(
        width=256,
        height=256,
        state_path="/tmp/v23-test/imp_001_state.png",
        alpha_path=None,
        confidence="visible-in-final",
        confidence_dE=0.0,
        confidence_alt_ids=[],
    )
    base.update(overrides)
    return T.Mask(**base)


def make_impression(**overrides: Any) -> T.Impression:
    """Return a valid :class:`~backend.services.v23.types.Impression`."""
    base: dict[str, Any] = dict(
        id="imp_001",
        order_step=1,
        block_id="blk_test_01",
        block_face_id="blk_test_01::face_a",
        pull_group=0,
        pigment_id="pig_test_01",
        mask=make_mask(),
        hidden_coverage_ref="coverage_v1",
        luminance_okL=0.5,
        coverage_pct=42.0,
        notes="",
    )
    base.update(overrides)
    return T.Impression(**base)


def make_plan(**overrides: Any) -> T.Plan:
    """Return a valid :class:`~backend.services.v23.types.Plan`.

    Default plan has 1 pigment / 1 block / 1 impression — the minimum the
    Pydantic validators accept. Override any field to specialise.
    """
    pig = make_pigment()
    blk = make_block()
    imp = make_impression()
    base: dict[str, Any] = dict(
        plan_id="plan_test_01HZK6V0000000000000000",
        schema_version="v23.0",
        target_image_sha256="e" * 64,
        width=256,
        height=256,
        pigments=[pig],
        blocks=[blk],
        impressions=[imp],
        pull_groups=[],
        strategy_template=None,
        solve_profile="default",
        reconstruction_dE_mean=1.2,
        reconstruction_dE_p95=2.8,
        solver_wall_s=12.5,
        tensors={},
        warnings=[],
    )
    base.update(overrides)
    return T.Plan(**base)
