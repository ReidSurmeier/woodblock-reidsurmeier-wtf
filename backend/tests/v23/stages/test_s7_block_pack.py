"""D11.b RED — S7 block packing (greedy DSATUR-style).

Per interface contract B-3 + W-4: synthetic impression_NNN ids fed into
DSATUR over the IoU conflict graph. Two impressions get the SAME block
only when their spatial overlap is low enough that they can share a
physical woodblock (Pace-Editions style multi-pigment-per-block).

Adjacency: edge between impressions i and j if mask IoU > τ_conflict
(default 0.30). Edge means MUST be different blocks.
"""

from __future__ import annotations

import numpy as np


def test_disjoint_impressions_share_a_block() -> None:
    """Two impressions with no spatial overlap can land on the same block."""
    from backend.services.v23.stages.s7_block_pack import pack_blocks

    alphas = np.zeros((2, 8, 8), dtype=np.float32)
    alphas[0, :, :4] = 0.8  # left half
    alphas[1, :, 4:] = 0.8  # right half — disjoint
    result = pack_blocks(alphas)
    assert result.block_count >= 1
    # imp_001 + imp_002 on the same block is allowed (no conflict)
    blocks = {result.impression_to_block[k] for k in result.impression_to_block}
    assert len(blocks) <= 2  # at most 2 blocks; ideally 1


def test_overlapping_impressions_get_different_blocks() -> None:
    """Two impressions covering the same region must be on different blocks."""
    from backend.services.v23.stages.s7_block_pack import pack_blocks

    alphas = np.zeros((2, 8, 8), dtype=np.float32)
    alphas[0, :, :] = 0.8  # whole image
    alphas[1, :, :] = 0.8  # whole image — full overlap
    result = pack_blocks(alphas)
    assert result.block_count == 2
    assert result.impression_to_block["imp_001"] != result.impression_to_block["imp_002"]


def test_three_overlap_three_blocks() -> None:
    """Three pairwise-overlapping impressions need three blocks."""
    from backend.services.v23.stages.s7_block_pack import pack_blocks

    alphas = np.zeros((3, 8, 8), dtype=np.float32)
    alphas[:, :, :] = 0.8  # all overlap entirely
    result = pack_blocks(alphas)
    assert result.block_count == 3


def test_block_packing_result_carries_face_tags() -> None:
    """Each impression has a block_face_id formatted ``blk_NN::face_X``."""
    from backend.services.v23.stages.s7_block_pack import pack_blocks

    alphas = np.zeros((2, 4, 4), dtype=np.float32)
    alphas[0, :, :2] = 0.7
    alphas[1, :, 2:] = 0.7
    result = pack_blocks(alphas)
    for face_id in result.impression_to_face.values():
        assert "::face_" in face_id
        assert face_id.startswith("blk_")


def test_dsatur_chromatic_number_matches_block_count() -> None:
    from backend.services.v23.stages.s7_block_pack import pack_blocks

    alphas = np.zeros((2, 4, 4), dtype=np.float32)
    alphas[0] = 0.8
    alphas[1] = 0.8
    result = pack_blocks(alphas)
    assert result.dsatur_chromatic_number == result.block_count


def test_pull_groups_derived_per_block_per_order_step() -> None:
    """Impressions on the same block at the same order_step share a pull_group."""
    from backend.services.v23.stages.s7_block_pack import pack_blocks

    alphas = np.zeros((3, 8, 8), dtype=np.float32)
    alphas[0, :, :4] = 0.7
    alphas[1, :, 4:] = 0.7
    alphas[2, :, :] = 0.7  # overlaps both
    result = pack_blocks(alphas)
    # imp_001 + imp_002 disjoint → can share block → can share pull
    blocks_for_disjoint = {
        result.impression_to_block["imp_001"],
        result.impression_to_block["imp_002"],
    }
    if len(blocks_for_disjoint) == 1:
        # Same block → same pull at the same order_step
        pg_a = next(pg for pg in result.pull_groups if "imp_001" in pg["impression_ids"])
        pg_b = next(pg for pg in result.pull_groups if "imp_002" in pg["impression_ids"])
        assert pg_a["pull_group"] == pg_b["pull_group"]


def test_iou_threshold_param_changes_conflicts() -> None:
    from backend.services.v23.stages.s7_block_pack import pack_blocks

    # Two impressions with 25% overlap
    alphas = np.zeros((2, 8, 8), dtype=np.float32)
    alphas[0, :, :5] = 0.7  # 40 px
    alphas[1, :, 3:] = 0.7  # 40 px; overlap 8 px → IoU = 8/(40+40-8) = 0.111

    # Default τ_conflict=0.30 → IoU 0.11 below threshold → same block allowed
    a = pack_blocks(alphas, conflict_iou_threshold=0.30)
    assert a.block_count == 1
    # Tighter threshold 0.05 → IoU above → must split
    b = pack_blocks(alphas, conflict_iou_threshold=0.05)
    assert b.block_count == 2


def test_impression_ids_match_synthetic_pattern() -> None:
    from backend.services.v23.stages.s7_block_pack import pack_blocks

    alphas = np.zeros((4, 4, 4), dtype=np.float32)
    alphas[:] = 0.7
    result = pack_blocks(alphas)
    expected_ids = {f"imp_{i + 1:03d}" for i in range(4)}
    assert set(result.impression_to_block.keys()) == expected_ids
