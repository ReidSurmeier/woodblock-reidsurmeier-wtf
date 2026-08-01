"""D11.b RED — topology repair (S8 carveability + addendum-v3 fix 1 home).

Per addendum-v3 fix 1: rules 6+7 (no-tiny-hidden-islands, few-broad-over-many-tiny)
are NOT in the optimizer loss. They run POST-SOLVE here:

1. Compute topology_score(plan) — tiny-island count, mean-island area.
2. Single-pass morph_open + morph_close per impression mask.
3. Re-compute forward render + ΔE; if repair worsens ΔE > 1.0, REJECT.

Returns the repaired alpha_stack + diagnostics.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("skimage")


def test_repair_uses_current_skimage_api() -> None:
    from backend.services.v23.core.topology_repair import morph_repair_stack

    alpha = np.zeros((1, 8, 8), dtype=np.float32)
    alpha[0, 2:6, 2:6] = 0.8
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        repaired = morph_repair_stack(alpha, min_island_px=4, close_radius=1)

    assert repaired[0, 3, 3] > 0.5
    assert not [warning for warning in warning_records if warning.category is FutureWarning]


def test_topology_score_counts_tiny_islands() -> None:
    from backend.services.v23.core.topology_repair import topology_score

    alpha = np.zeros((1, 16, 16), dtype=np.float32)
    # One big island
    alpha[0, 2:12, 2:12] = 0.8
    # Three tiny isolated specks
    alpha[0, 0, 0] = 0.8
    alpha[0, 0, 15] = 0.8
    alpha[0, 15, 0] = 0.8

    score = topology_score(alpha, min_island_px=8)
    assert score.tiny_island_counts == [3]
    assert score.mean_island_areas_px[0] >= 100  # big island is ~100 px


def test_repair_removes_tiny_islands() -> None:
    from backend.services.v23.core.topology_repair import morph_repair_stack

    alpha = np.zeros((1, 16, 16), dtype=np.float32)
    alpha[0, 4:12, 4:12] = 0.8
    alpha[0, 0, 0] = 0.8
    alpha[0, 1, 0] = 0.8  # tiny 2-pixel speck

    repaired = morph_repair_stack(alpha, min_island_px=8)
    assert repaired[0, 0, 0] == 0.0
    assert repaired[0, 1, 0] == 0.0
    # Big island survives
    assert repaired[0, 8, 8] > 0.5


def test_repair_fills_small_holes() -> None:
    from backend.services.v23.core.topology_repair import morph_repair_stack

    alpha = np.full((1, 16, 16), 0.0, dtype=np.float32)
    alpha[0, 4:12, 4:12] = 0.8  # big block
    alpha[0, 7, 7] = 0.0  # single-pixel hole
    repaired = morph_repair_stack(alpha, min_island_px=4, close_radius=1)
    assert repaired[0, 7, 7] > 0.5  # hole filled


def test_repair_rejected_when_de_regresses() -> None:
    """If morph repair worsens ΔE by > de_regression_guard, original returned."""
    import jax.numpy as jnp

    from backend.services.v23.core import forward_render_jax
    from backend.services.v23.core.topology_repair import run_topology_repair

    alpha = np.zeros((1, 16, 16), dtype=np.float32)
    alpha[0, 4:12, 4:12] = 0.8
    pigment_idx = np.array([0], dtype=np.int32)
    # Target = the forward-rendered composite so original_dE ≈ 0 and any
    # destructive repair triggers an obvious regression.
    alpha_hwm = np.transpose(alpha, (1, 2, 0))
    target = np.asarray(
        forward_render_jax.forward_render(
            jnp.asarray(alpha_hwm, dtype=jnp.float32),
            jnp.asarray(pigment_idx, dtype=jnp.int32),
        )
    ).astype(np.float32)

    # Aggressive min_island erases the only impression → ΔE jumps → REJECTED.
    result = run_topology_repair(
        alpha,
        pigment_idx=pigment_idx,
        target_rgb=target,
        min_island_px=10000,
        de_regression_guard=1.0,
    )
    assert result.repair_accepted is False
    np.testing.assert_array_equal(result.alpha_stack, alpha)


def test_repair_accepted_when_de_neutral() -> None:
    """Repair should be accepted when removing tiny islands doesn't hurt ΔE."""
    from backend.services.v23.core.topology_repair import run_topology_repair

    alpha = np.zeros((1, 16, 16), dtype=np.float32)
    alpha[0, 4:12, 4:12] = 0.8  # one big island
    alpha[0, 0, 0] = 0.8  # one tiny pixel
    pigment_idx = np.array([0], dtype=np.int32)
    # Target close to forward-rendered result so the tiny pixel doesn't matter
    target = np.full((16, 16, 3), 0.6, dtype=np.float32)

    result = run_topology_repair(
        alpha,
        pigment_idx=pigment_idx,
        target_rgb=target,
        min_island_px=4,
        de_regression_guard=10.0,
    )
    assert result.repair_accepted is True
    assert result.alpha_stack[0, 0, 0] == 0.0


def test_topology_score_per_impression_is_list_length_M() -> None:
    from backend.services.v23.core.topology_repair import topology_score

    alpha = np.zeros((3, 8, 8), dtype=np.float32)
    alpha[0, 2:6, 2:6] = 0.7
    alpha[1, 0:3, 0:3] = 0.7
    alpha[2, 5:8, 5:8] = 0.7
    score = topology_score(alpha, min_island_px=4)
    assert len(score.tiny_island_counts) == 3
    assert len(score.mean_island_areas_px) == 3


def test_repair_preserves_shape_and_dtype() -> None:
    from backend.services.v23.core.topology_repair import morph_repair_stack

    alpha = np.zeros((2, 8, 8), dtype=np.float32)
    alpha[0] = 0.5
    alpha[1] = 0.7
    repaired = morph_repair_stack(alpha, min_island_px=2)
    assert repaired.shape == (2, 8, 8)
    assert repaired.dtype == np.float32
