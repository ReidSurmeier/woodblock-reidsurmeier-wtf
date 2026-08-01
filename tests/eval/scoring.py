"""Scoring primitives for the eval pipeline.

V2 scaffold: only the cheap, dependency-free functions (`summarize`, `iou_per_mask`)
are implemented. The colour-conversion + Hungarian-match math lives behind
NotImplementedError stubs until MVP-A wires `colour-science` and `scipy`.

Reference: validation-system-v1.md sections 2 + 10.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .eval_result import SummaryStats


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB to CIELAB conversion.

    Args:
        rgb: (H, W, 3) uint8 in 0..255 OR float in 0..1.

    Returns:
        (H, W, 3) float32 Lab — L in 0..100, a/b in roughly -128..127.

    MVP-A: replace with `colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))`.
    """
    raise NotImplementedError("Wire colour-science in MVP-A")


def delta_e2000_image(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Per-pixel ΔE2000 between two sRGB images.

    Args:
        img_a: (H, W, 3) sRGB.
        img_b: (H, W, 3) sRGB. Must match shape of img_a.

    Returns:
        (H, W) float32 ΔE2000 map. Lower = closer match.

    MVP-A: `colour.delta_E(lab_a, lab_b, method="CIE 2000")`.
    """
    raise NotImplementedError("Wire colour-science delta_E_CIE2000 in MVP-A")


def summarize(dE_map: np.ndarray) -> SummaryStats:
    """Reduce a ΔE (or any error) map to mean/p50/p95/p99/max.

    Casts to Python floats explicitly so the result is JSON-serializable.
    numpy scalars (np.float32) silently break `json.dumps`.
    """
    flat = np.asarray(dE_map).flatten()
    return SummaryStats(
        mean=float(np.mean(flat)),
        p50=float(np.percentile(flat, 50)),
        p95=float(np.percentile(flat, 95)),
        p99=float(np.percentile(flat, 99)),
        max=float(np.max(flat)),
    )


def iou_per_mask(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two binary masks.

    Accepts bool / uint8 / float masks — anything castable to bool. By convention
    iou(empty, empty) = 1.0 (no false positives, no misses).
    """
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    union = int((pred_b | gt_b).sum())
    if union == 0:
        return 1.0
    inter = int((pred_b & gt_b).sum())
    return float(inter / union)


def hungarian_match_blocks(pred_blocks: dict[str, Any], gt_blocks: dict[str, Any]) -> float:
    """Hungarian-matched block IoU.

    Builds a cost matrix C[i, j] = 1 - IoU(pred_blocks[i], gt_blocks[j]),
    solves the assignment, then returns the mean IoU of matched pairs (weighted
    by ground-truth block area? — TBD in V3 when the block schema lands).

    V2 stub: not implemented. V3 will wire `scipy.optimize.linear_sum_assignment`.
    """
    raise NotImplementedError("Implement in V3 once block schema lands")
