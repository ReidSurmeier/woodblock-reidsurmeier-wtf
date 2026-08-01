"""D11.b — Topology repair (addendum-v3 fix 1 home).

Per addendum-v3 fix 1: rules 6 + 7 (no-tiny-hidden-islands,
few-broad-over-many-tiny) are NOT in the differentiable optimizer loss.
They run POST-SOLVE as topology scoring + a single-pass morph_open +
morph_close repair, gated by a ΔE-regression guard.

Pipeline:
1. ``topology_score(alpha_stack)`` — diagnostic-only, returns per-impression
   tiny-island count + mean island area for the manifest.
2. ``morph_repair_stack(alpha_stack)`` — single-pass binary_opening (removes
   tiny islands < min_island_px) + binary_closing (fills small gaps,
   dilates by registration tolerance for kento). Returns alpha_stack
   with the original alpha values preserved INSIDE the repaired mask
   and zero OUTSIDE.
3. ``run_topology_repair(alpha_stack, pigment_idx, target_rgb)`` —
   wraps the repair, recomputes forward-render ΔE, and if the repair
   worsens ΔE by more than ``de_regression_guard`` it REJECTS the
   repair and returns the original alpha_stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from skimage import measure, morphology

from backend.services.v23.core import forward_render_jax

_VIS_THRESHOLD: float = 0.40


@dataclass(frozen=True)
class TopologyScore:
    """Per-impression diagnostic counts. No mutation of the stack."""

    tiny_island_counts: list[int]
    mean_island_areas_px: list[float]
    total_island_counts: list[int]


@dataclass(frozen=True)
class TopologyRepairResult:
    """Output of :func:`run_topology_repair`."""

    alpha_stack: NDArray[np.float32]
    repair_accepted: bool
    original_dE: float
    repaired_dE: float
    reason: str


def topology_score(
    alpha_stack: NDArray[np.float32],
    *,
    min_island_px: int = 60,
    vis_threshold: float = _VIS_THRESHOLD,
) -> TopologyScore:
    """Count tiny islands + mean island area per impression."""
    m = int(alpha_stack.shape[0])
    tiny: list[int] = []
    means: list[float] = []
    totals: list[int] = []
    for i in range(m):
        mask = alpha_stack[i] >= vis_threshold
        labels = measure.label(mask, connectivity=2)
        if labels.max() == 0:
            tiny.append(0)
            means.append(0.0)
            totals.append(0)
            continue
        regions = measure.regionprops(labels)
        areas = [int(r.area) for r in regions]
        tiny_count = int(sum(1 for a in areas if a < min_island_px))
        tiny.append(tiny_count)
        # Mean of NON-TINY islands so the manifest reflects "broad" island size
        non_tiny = [a for a in areas if a >= min_island_px]
        means.append(float(np.mean(non_tiny)) if non_tiny else 0.0)
        totals.append(int(len(areas)))
    return TopologyScore(
        tiny_island_counts=tiny,
        mean_island_areas_px=means,
        total_island_counts=totals,
    )


def morph_repair_stack(
    alpha_stack: NDArray[np.float32],
    *,
    min_island_px: int = 60,
    close_radius: int = 1,
    vis_threshold: float = _VIS_THRESHOLD,
) -> NDArray[np.float32]:
    """Single-pass morph_open(min_island) + morph_close(close_radius) per impression."""
    m = int(alpha_stack.shape[0])
    out = np.zeros_like(alpha_stack)
    for i in range(m):
        mask = alpha_stack[i] >= vis_threshold
        # Remove tiny islands
        # scikit-image's current ``max_size`` boundary removes components
        # smaller than or equal to the value. Subtract one to preserve the
        # existing contract: remove islands strictly smaller than
        # ``min_island_px``.
        max_tiny_size = max(int(min_island_px) - 1, 0)
        opened = morphology.remove_small_objects(mask, max_size=max_tiny_size)
        # Fill small holes + dilate for registration tolerance
        if close_radius > 0:
            footprint = morphology.disk(int(close_radius))
            closed = morphology.closing(opened, footprint=footprint)
        else:
            closed = opened
        # Preserve original alpha values inside the repaired mask. Pixels
        # newly-filled by closing (mask=True but original α < vis_threshold)
        # get the impression's median visible alpha so they actually print.
        visible_alphas = alpha_stack[i][alpha_stack[i] >= vis_threshold]
        fill = float(np.median(visible_alphas)) if visible_alphas.size > 0 else vis_threshold
        filled = np.where(
            closed & (alpha_stack[i] < vis_threshold),
            fill,
            alpha_stack[i],
        )
        out[i] = np.where(closed, filled, 0.0).astype(np.float32)
    return out


def _forward_dE_proxy(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    target_rgb: NDArray[np.float32],
) -> float:
    """Compute mean RGB-space L2 ΔE proxy via the JAX forward render."""
    alpha_hwm = np.transpose(alpha_stack, (1, 2, 0))
    rgb = np.asarray(
        forward_render_jax.forward_render(
            jnp.asarray(alpha_hwm, dtype=jnp.float32),
            jnp.asarray(pigment_idx, dtype=jnp.int32),
        )
    )
    diff = rgb - target_rgb
    return float(np.sqrt(np.mean(diff * diff)) * 255.0 * 0.1)  # rough scale


def run_topology_repair(
    alpha_stack: NDArray[np.float32],
    *,
    pigment_idx: NDArray[np.int32],
    target_rgb: NDArray[np.float32],
    min_island_px: int = 60,
    close_radius: int = 1,
    de_regression_guard: float = 1.0,
) -> TopologyRepairResult:
    """Run topology repair, recompute ΔE, REJECT if repair worsens by > guard.

    Per interface contract B-5: ΔE-regression-guarded single-pass repair.
    """
    original_de = _forward_dE_proxy(alpha_stack, pigment_idx, target_rgb)
    repaired = morph_repair_stack(
        alpha_stack,
        min_island_px=min_island_px,
        close_radius=close_radius,
    )
    repaired_de = _forward_dE_proxy(repaired, pigment_idx, target_rgb)

    if repaired_de - original_de > de_regression_guard:
        return TopologyRepairResult(
            alpha_stack=alpha_stack,
            repair_accepted=False,
            original_dE=original_de,
            repaired_dE=repaired_de,
            reason=(
                f"repair worsened ΔE by {repaired_de - original_de:.3f} "
                f"(> guard {de_regression_guard}) — kept original"
            ),
        )
    return TopologyRepairResult(
        alpha_stack=repaired,
        repair_accepted=True,
        original_dE=original_de,
        repaired_dE=repaired_de,
        reason=f"repair accepted (ΔE delta {repaired_de - original_de:+.3f})",
    )


__all__ = [
    "TopologyScore",
    "TopologyRepairResult",
    "topology_score",
    "morph_repair_stack",
    "run_topology_repair",
]
