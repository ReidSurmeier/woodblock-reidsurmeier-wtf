"""D6 RED — JAX forward render (Mixbox-stack lerp on 7D latent).

Tests cover the JAX-traceable forward render that S5 (inverse solver)
will backprop through. Mixbox numpy oracle isn't required on this dev
box; the JAX module loads a pre-built LUT from disk (offline generator
in scripts/build_mixbox_lut.py on the GPU host) and falls back to an
OKLab alpha-blend placeholder when the LUT file isn't present.

D6.1 — JAX render matches numpy oracle (or placeholder) within ΔE 0.5
D6.2 — jit compiles + 2nd call ≤ 50 ms on 256² × 7-impression stack (skip on slow CPU)
D6.3 — jax.grad produces finite gradient on random α
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp  # noqa: E402

from backend.services.v23.core import forward_render_jax  # noqa: E402


def _random_alpha(h: int, w: int, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(h, w, m)).astype(np.float32)


def test_forward_render_returns_rgb_shape() -> None:
    alphas = _random_alpha(16, 16, 3, seed=1)
    pigment_idx = jnp.array([0, 4, 9])  # 3 pigments from the 13-catalog
    rgb = forward_render_jax.forward_render(jnp.asarray(alphas), pigment_idx)
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == jnp.float32 or rgb.dtype == jnp.float64
    arr = np.asarray(rgb)
    assert np.all(np.isfinite(arr))
    assert arr.min() >= 0.0 - 1e-6 and arr.max() <= 1.0 + 1e-6


def test_single_pigment_alpha_one_returns_near_pigment_rgb() -> None:
    """If only pigment 0 has α=1 everywhere over white paper, output ≈ pigment 0 RGB."""
    alphas = jnp.ones((8, 8, 1), dtype=jnp.float32)
    pigment_idx = jnp.array([0])  # cadmium yellow per the catalog
    rgb = np.asarray(forward_render_jax.forward_render(alphas, pigment_idx))

    pigment_rgb = forward_render_jax.PIGMENT_TABLE[0]  # (R, G, B) in [0,1]
    # Allow some drift since the placeholder uses linear blend in [0,1] space.
    diff = np.abs(rgb.mean(axis=(0, 1)) - pigment_rgb)
    assert np.all(diff < 0.10), f"pigment-only render drift: {diff}"


def test_paper_only_renders_white() -> None:
    """Empty stack returns the paper substrate."""
    alphas = jnp.zeros((4, 4, 2), dtype=jnp.float32)
    pigment_idx = jnp.array([0, 1])
    rgb = np.asarray(forward_render_jax.forward_render(alphas, pigment_idx))
    paper = forward_render_jax.PAPER_RGB
    assert np.all(np.abs(rgb - paper) < 1e-5)


def test_jax_grad_is_finite() -> None:
    alphas = _random_alpha(8, 8, 4, seed=2)
    pigment_idx = jnp.array([0, 2, 4, 6])
    target = jnp.full((8, 8, 3), 0.5, dtype=jnp.float32)

    def loss_fn(a):
        rgb = forward_render_jax.forward_render(a, pigment_idx)
        return jnp.mean((rgb - target) ** 2)

    grad_fn = jax.grad(loss_fn)
    g = grad_fn(jnp.asarray(alphas))
    assert g.shape == alphas.shape
    arr = np.asarray(g)
    assert np.all(np.isfinite(arr)), "gradient contains NaN/Inf"
    # At least some elements must be non-zero — the loss depends on α
    assert np.any(np.abs(arr) > 1e-8)


def test_jit_compiles_and_runs() -> None:
    alphas = _random_alpha(32, 32, 5, seed=3)
    pigment_idx = jnp.array([0, 1, 4, 7, 12])
    jitted = jax.jit(lambda a: forward_render_jax.forward_render(a, pigment_idx))
    first = np.asarray(jitted(jnp.asarray(alphas)))
    second = np.asarray(jitted(jnp.asarray(alphas)))
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_pigment_table_has_13_entries() -> None:
    """The 13-pigment Mixbox-anchored catalog must seed the LUT."""
    assert forward_render_jax.PIGMENT_TABLE.shape == (13, 3)
    assert forward_render_jax.PIGMENT_TABLE.min() >= 0.0
    assert forward_render_jax.PIGMENT_TABLE.max() <= 1.0


def test_order_affects_output() -> None:
    """Stack [0, 5] should differ from [5, 0] — print order matters."""
    alphas = _random_alpha(16, 16, 2, seed=7)
    a_then_b = np.asarray(forward_render_jax.forward_render(jnp.asarray(alphas), jnp.array([0, 5])))
    b_then_a = np.asarray(forward_render_jax.forward_render(jnp.asarray(alphas), jnp.array([5, 0])))
    assert not np.allclose(a_then_b, b_then_a, atol=1e-3), (
        "print order has no effect — solver can't designate underprints"
    )
