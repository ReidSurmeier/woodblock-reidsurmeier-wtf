"""D11.b — S7 block packing.

Greedy DSATUR-style graph coloring over the impression-IoU conflict
graph. Per interface contract B-3 + W-4: synthetic ``imp_NNN`` ids feed
the conflict graph; edge between i and j iff their visible-mask IoU
exceeds ``conflict_iou_threshold``. Two non-conflicting impressions can
share a physical block + (when at the same Order step) a Pull group —
this is the Pace-Editions multi-pigment-per-block move that Emma uses.

Wave A integration via :mod:`backend.algorithms.decomposition.dsatur_color_aware`
is deferred to v23.1 — for day-1 we ship a small greedy coloring that
deterministically passes the unit tests + handles the corpus-scale
adjacency. The Wave A module remains available for the higher-fidelity
v23.1 lift (it adds OKLab tiebreaking and chromatic-number minimisation
guarantees that the greedy doesn't).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

_VIS_THRESHOLD: float = 0.40  # α above this counts as "printed" for IoU


@dataclass(frozen=True)
class BlockPackingResult:
    """Output of :func:`pack_blocks`. Pure Python — no numpy fields."""

    block_count: int
    impression_to_block: dict[str, int]
    impression_to_face: dict[str, str]
    pull_groups: list[dict[str, Any]]
    dsatur_chromatic_number: int


def _binary_mask(alpha: NDArray[np.float32]) -> NDArray[np.bool_]:
    return alpha >= _VIS_THRESHOLD


def _iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return inter / union


def _greedy_color(
    n: int,
    conflict_neighbours: list[set[int]],
) -> tuple[list[int], int]:
    """Greedy chromatic coloring: assign lowest non-conflicting color per node.

    Order nodes by descending degree (DSATUR-lite tiebreak via insertion order).
    Returns (colors_per_node, chromatic_number).
    """
    order = sorted(range(n), key=lambda i: -len(conflict_neighbours[i]))
    colors = [-1] * n
    for node in order:
        used = {colors[nb] for nb in conflict_neighbours[node] if colors[nb] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[node] = c
    return colors, max(colors) + 1 if colors else 0


def pack_blocks(
    alpha_stack: NDArray[np.float32],
    *,
    conflict_iou_threshold: float = 0.30,
) -> BlockPackingResult:
    """Greedy DSATUR coloring over IoU conflict graph.

    Args:
        alpha_stack: ``(M, H, W)`` float in [0, 1]; impressions in print order.
        conflict_iou_threshold: IoU above which two impressions MUST land on
            different blocks (default 0.30).

    Returns:
        :class:`BlockPackingResult` with impression→block dict, face tags,
        derived pull groups, and the DSATUR chromatic number.
    """
    m = int(alpha_stack.shape[0])
    impression_ids = [f"imp_{i + 1:03d}" for i in range(m)]
    masks = [_binary_mask(alpha_stack[i]) for i in range(m)]

    # Build conflict adjacency
    conflict_neighbours: list[set[int]] = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            iou = _iou(masks[i], masks[j])
            if iou > conflict_iou_threshold:
                conflict_neighbours[i].add(j)
                conflict_neighbours[j].add(i)

    colors, chromatic = _greedy_color(m, conflict_neighbours)

    # Map color index → block id; assign face_a as default
    impression_to_block: dict[str, int] = {impression_ids[i]: colors[i] for i in range(m)}
    impression_to_face: dict[str, str] = {
        impression_ids[i]: f"blk_{colors[i]:02d}::face_a" for i in range(m)
    }

    # Derive pull groups: same block + same order_step → same pull
    # (Order step == impression index for day-1; multi-pull-per-block lands later.)
    pull_groups: list[dict[str, Any]] = []
    seen: dict[tuple[int, int], int] = {}
    for i, imp_id in enumerate(impression_ids):
        block = colors[i]
        order = i + 1
        key = (block, order)
        if key not in seen:
            seen[key] = len(pull_groups)
            pull_groups.append(
                {
                    "block_id": f"blk_{block:02d}",
                    "order_step": order,
                    "pull_group": len(pull_groups),
                    "impression_ids": [imp_id],
                }
            )
        else:
            pull_groups[seen[key]]["impression_ids"].append(imp_id)

    # Collapse impressions on same block (any order step) into shared pulls when
    # they have no conflict — Pace-Editions: ink multiple pigments on one block
    # face, pull together. We mark them as the same pull_group iff their block
    # matches and they are pairwise non-conflicting.
    block_to_impressions: dict[int, list[int]] = {}
    for i, c in enumerate(colors):
        block_to_impressions.setdefault(c, []).append(i)
    for block_idx, members in block_to_impressions.items():
        if len(members) <= 1:
            continue
        # Reassign all same-block impressions to the LOWEST pull_group of any
        # member so disjoint-on-same-block impressions share their pull tag.
        affected = [
            j for j, pg in enumerate(pull_groups) if pg["block_id"] == f"blk_{block_idx:02d}"
        ]
        if not affected:
            continue
        canonical_pg = min(
            pg["pull_group"] for pg in pull_groups if pg["block_id"] == f"blk_{block_idx:02d}"
        )
        merged_imp_ids: list[str] = []
        merged_order = None
        for idx in affected:
            merged_imp_ids.extend(pull_groups[idx]["impression_ids"])
            if merged_order is None:
                merged_order = pull_groups[idx]["order_step"]
        # Remove old entries, append one merged
        pull_groups = [pg for pg in pull_groups if pg["block_id"] != f"blk_{block_idx:02d}"]
        pull_groups.append(
            {
                "block_id": f"blk_{block_idx:02d}",
                "order_step": merged_order or 1,
                "pull_group": canonical_pg,
                "impression_ids": merged_imp_ids,
            }
        )

    return BlockPackingResult(
        block_count=chromatic,
        impression_to_block=impression_to_block,
        impression_to_face=impression_to_face,
        pull_groups=pull_groups,
        dsatur_chromatic_number=chromatic,
    )


__all__ = ["BlockPackingResult", "pack_blocks"]
