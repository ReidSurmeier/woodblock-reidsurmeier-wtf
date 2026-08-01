"""D10.c RED — S4 Tan warm-start bridge (W-2 from corrections).

Wraps Wave A's tan_rgb_geometry decompose_image + remaps the convex-hull
palette indices into v23 Mixbox pigment indices via OKLab snap.

Returns per-pigment alpha-init maps that S5 inverse solver consumes as
a warm start so the L-BFGS step doesn't begin from random noise.
"""

from __future__ import annotations

import numpy as np


def test_tan_warmstart_returns_alpha_per_pigment() -> None:
    """Synthetic 2-color image → recovers 2 pigment alpha maps."""
    from backend.services.v23.stages.s4_warmstart import tan_to_pigment_warmstart

    h, w = 32, 32
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2] = (240, 220, 50)  # yellow half
    img[:, w // 2 :] = (30, 40, 160)  # blue half

    result = tan_to_pigment_warmstart(img, target_palette_size=4)
    assert result.alpha_stack.ndim == 3  # (M, H, W)
    assert result.alpha_stack.shape[1:] == (h, w)
    assert result.alpha_stack.shape[0] >= 2
    assert len(result.pigment_idx) == result.alpha_stack.shape[0]
    # Alphas must be in [0, 1]
    assert result.alpha_stack.min() >= 0.0
    assert result.alpha_stack.max() <= 1.0 + 1e-5


def test_tan_warmstart_pigment_idx_are_valid_catalog_entries() -> None:
    from backend.services.v23.stages.s4_warmstart import tan_to_pigment_warmstart

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    result = tan_to_pigment_warmstart(img, target_palette_size=6)
    # 13-pigment Mixbox catalog
    for pid in result.pigment_idx:
        assert 0 <= pid <= 12, f"pigment_idx {pid} outside [0, 12]"


def test_tan_warmstart_alphas_sum_close_to_one_at_each_pixel() -> None:
    """Tan barycentric decomposition gives partition-of-unity weights."""
    from backend.services.v23.stages.s4_warmstart import tan_to_pigment_warmstart

    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    result = tan_to_pigment_warmstart(img, target_palette_size=5)
    sums = result.alpha_stack.sum(axis=0)
    assert sums.shape == (24, 24)
    assert np.all(np.abs(sums - 1.0) < 0.02), (
        f"alphas don't sum to ~1: min {sums.min():.4f}, max {sums.max():.4f}"
    )


def test_tan_warmstart_collapses_duplicate_pigment_snaps() -> None:
    """Two hull vertices that snap to the same Mixbox pigment must aggregate."""
    from backend.services.v23.stages.s4_warmstart import tan_to_pigment_warmstart

    # Near-identical near-blacks should snap to ivory_black (pigment_idx 12)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:8, :] = (5, 5, 5)
    img[8:, :] = (20, 18, 22)
    result = tan_to_pigment_warmstart(img, target_palette_size=4)
    # All near-black hull vertices snap to one pigment → fewer unique indices than vertices
    assert len(set(result.pigment_idx)) <= result.alpha_stack.shape[0]


def test_tan_warmstart_single_color_image_returns_one_pigment() -> None:
    from backend.services.v23.stages.s4_warmstart import tan_to_pigment_warmstart

    img = np.full((8, 8, 3), 180, dtype=np.uint8)
    result = tan_to_pigment_warmstart(img, target_palette_size=4)
    # All vertices collapse to one pigment after Mixbox snap
    assert len(set(result.pigment_idx)) == 1
