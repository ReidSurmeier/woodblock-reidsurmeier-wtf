"""Deterministic synthetic fixtures for v23 stage + solver tests.

Used by:
- Ring 4 ``stages/`` placeholder tests (S1/S2/S5)
- D7 ``solver_smoke/test_5step_recovery.py`` (lands in D7)

Produces a 256×256 RGB image composed of three known impressions:

1. cream base (full coverage, dilute)
2. cool mid-tone (centred disk, mid coverage)
3. sumi detail (cross-hatched rectangle, opaque)

The ground-truth α-stack is returned alongside the rendered RGB so a
solver can be scored against a known answer without any inverse step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H = W = 256
N_IMPRESSIONS = 3


@dataclass(frozen=True)
class SyntheticStack:
    """Container for a synthetic 3-impression ground truth.

    Attributes
    ----------
    rgb : ``(H, W, 3) uint8``
        Forward-rendered RGB image — what the solver sees as input.
    alpha : ``(N, H, W) float32`` in ``[0, 1]``
        Per-impression coverage. Layer order = print order (bottom→top).
    pigment_rgb : ``(N, 3) uint8``
        Per-impression flat pigment colour used in the forward render.
    """

    rgb: np.ndarray
    alpha: np.ndarray
    pigment_rgb: np.ndarray


def make_3imp_synthetic(seed: int = 0) -> SyntheticStack:
    """Return a fully reproducible 3-impression 256×256 ground truth.

    Determinism: any two calls with the same ``seed`` produce identical
    bytes (numpy default RNG is seeded; no global state is touched).
    """
    rng = np.random.default_rng(seed)

    pigment_rgb = np.array(
        [
            [240, 232, 210],  # cream base
            [80, 110, 165],  # cool mid
            [25, 25, 30],  # sumi detail (near-black)
        ],
        dtype=np.uint8,
    )

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # impression 0 — cream base, full coverage, slight noise
    a0 = np.full((H, W), 0.92, dtype=np.float32)
    a0 += rng.normal(0.0, 0.01, size=(H, W)).astype(np.float32)
    a0 = np.clip(a0, 0.0, 1.0)

    # impression 1 — cool centred disk, radius 70, soft edge
    r2 = (xx - W / 2) ** 2 + (yy - H / 2) ** 2
    a1 = np.clip(1.0 - r2 / (70**2), 0.0, 1.0).astype(np.float32) * 0.85

    # impression 2 — sumi rectangle 60..200 × 100..160 w/ a cross-hatch
    a2 = np.zeros((H, W), dtype=np.float32)
    a2[100:160, 60:200] = 1.0
    a2 *= ((xx.astype(int) % 8) < 4).astype(np.float32) * 0.95

    alpha = np.stack([a0, a1, a2], axis=0)

    # Naive over-compositing — solver fixture only needs *a* deterministic
    # forward render; the real km-forward lands in D6.
    rgb = np.full((H, W, 3), 255, dtype=np.float32)
    for k in range(N_IMPRESSIONS):
        a = alpha[k][..., None]
        rgb = (1 - a) * rgb + a * pigment_rgb[k].astype(np.float32)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    return SyntheticStack(rgb=rgb_u8, alpha=alpha, pigment_rgb=pigment_rgb)
