"""D10.c — S4 Tan warm-start bridge.

Wraps Wave A's :func:`backend.algorithms.decomposition.tan_rgb_geometry.decompose_image`
(convex-hull palette + Delaunay barycentric weights summing to 1) and
remaps each hull vertex's RGB anchor onto the v23 Mixbox 13-pigment
catalog. The output ``alpha_stack`` is what S5 inverse solver consumes
as a warm-start so L-BFGS doesn't begin from random noise.

Per interface contract W-2: vertices that snap to the same Mixbox
pigment have their barycentric weights aggregated, NOT duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from backend.algorithms.decomposition import tan_rgb_geometry as _tan
from backend.services.v23.core import forward_render_jax as _fr


@dataclass(frozen=True)
class WarmStartResult:
    """Output of :func:`tan_to_pigment_warmstart`. Pure numpy, no I/O."""

    alpha_stack: NDArray[np.float32]  # (M, H, W), barycentric weights per Mixbox pigment
    pigment_idx: tuple[int, ...]  # length M, each ∈ [0, 12]
    palette_rgb: NDArray[np.float32]  # (M, 3) snapped pigment RGB anchors


def _snap_palette_to_pigment(palette_rgb_01: NDArray[np.float32]) -> NDArray[np.int32]:
    """Snap each palette vertex to its nearest Mixbox pigment index by Euclidean RGB distance."""
    pigments = _fr.PIGMENT_TABLE  # (13, 3) in [0,1]
    # Broadcast: palette (K, 1, 3) vs pigments (1, 13, 3) → (K, 13) distances
    diffs = palette_rgb_01[:, None, :] - pigments[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    return np.argmin(d2, axis=-1).astype(np.int32)


def _aggregate_duplicate_pigments(
    alphas: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
) -> tuple[NDArray[np.float32], tuple[int, ...]]:
    """Collapse vertices that snap to the same pigment by summing alphas."""
    h, w = alphas.shape[:2]
    unique_pigments: list[int] = []
    pigment_to_slot: dict[int, int] = {}
    for pid in pigment_idx.tolist():
        if pid not in pigment_to_slot:
            pigment_to_slot[pid] = len(unique_pigments)
            unique_pigments.append(pid)

    m = len(unique_pigments)
    out = np.zeros((m, h, w), dtype=np.float32)
    for source_slot, pid in enumerate(pigment_idx.tolist()):
        target_slot = pigment_to_slot[pid]
        out[target_slot] += alphas[..., source_slot]
    return out, tuple(unique_pigments)


def tan_to_pigment_warmstart(
    rgb: NDArray[np.uint8],
    *,
    target_palette_size: int = 8,
) -> WarmStartResult:
    """Run Tan RGB-geometry decomposition + remap palette onto Mixbox catalog.

    Args:
        rgb: ``(H, W, 3)`` uint8 image.
        target_palette_size: hint to Tan's convex-hull extractor for palette size.

    Returns:
        ``WarmStartResult`` with ``alpha_stack`` shape ``(M, H, W)``,
        ``pigment_idx`` length M (unique Mixbox indices after de-duplication),
        and ``palette_rgb`` the snapped pigment RGB anchors in [0, 1].
    """
    palette, alphas_hwM = _tan.decompose_image(rgb, target_palette_size=target_palette_size)
    # palette shape (K, 3) float [0,1]; alphas_hwM shape (H, W, K)
    pigment_idx = _snap_palette_to_pigment(palette.astype(np.float32))
    # Convert to (K, H, W)
    alpha_first = np.transpose(alphas_hwM, (2, 0, 1)).astype(np.float32)
    collapsed, unique_pigments = _aggregate_duplicate_pigments(
        np.transpose(alpha_first, (1, 2, 0)), pigment_idx
    )
    snapped_palette = _fr.PIGMENT_TABLE[np.array(unique_pigments)]
    return WarmStartResult(
        alpha_stack=collapsed,
        pigment_idx=unique_pigments,
        palette_rgb=snapped_palette.astype(np.float32),
    )


__all__ = ["WarmStartResult", "tan_to_pigment_warmstart"]
