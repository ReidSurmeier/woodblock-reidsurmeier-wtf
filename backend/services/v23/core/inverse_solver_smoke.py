"""D7.1 — Synthetic inverse-solver smoke (5-step L-BFGS recovery).

Solves the toy inverse problem for the D7 smoke fixture:

    given target RGB + known pigment_idx, recover alphas such that
    forward_render(alphas, pigment_idx) ≈ target

This is NOT the production S5 solver — it's the minimum loop that
proves the JAX forward render is gradient-correct + L-BFGS converges.
The real S5 solver (D7.2+, D10) adds 8-term loss, coarse-to-fine
pyramid, sparsity + TV + the addendum-v3 family-aligned-support rules.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from numpy.typing import NDArray

from backend.services.v23.core import forward_render_jax


def _sigmoid_box(alpha_raw: jnp.ndarray) -> jnp.ndarray:
    """Map ``α_raw ∈ R`` → ``α ∈ (0, 1)`` so L-BFGS works unconstrained."""
    return jax.nn.sigmoid(alpha_raw)


def _solver_loss(
    alpha_raw: jnp.ndarray,
    target_rgb: jnp.ndarray,
    pigment_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Sum-squared error in RGB space.

    Uses ``jnp.sum`` (NOT ``jnp.mean``) so the gradient magnitude scales
    with the image, giving L-BFGS's backtracking line search a useful
    initial step. For convergence diagnostics divide by ``H*W*3`` to get
    per-pixel MSE.
    """
    alpha = _sigmoid_box(alpha_raw)
    rgb = forward_render_jax.forward_render(alpha, pigment_idx)
    return jnp.sum((rgb - target_rgb) ** 2)


def solve_3imp_smoke(
    target_rgb: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    *,
    n_iters: int = 80,
    seed: int = 42,
) -> NDArray[np.float32]:
    """Recover α via L-BFGS from a random init.

    Returns the recovered α in ``[0, 1]`` of shape ``(H, W, M)``.
    """
    h, w, _ = target_rgb.shape
    m = int(len(pigment_idx))
    rng = np.random.default_rng(seed)
    # Mild random init in logit space; sigmoid(0) = 0.5 so this lands around 0.5
    init_raw = rng.normal(0.0, 0.5, size=(h, w, m)).astype(np.float32)

    target_j = jnp.asarray(target_rgb, dtype=jnp.float32)
    pigments_j = jnp.asarray(pigment_idx, dtype=jnp.int32)

    def loss_fn(a_raw):
        return _solver_loss(a_raw, target_j, pigments_j)

    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=int(n_iters), tol=1e-7)
    result = solver.run(jnp.asarray(init_raw))
    alpha_raw_final = result.params
    alpha = jax.nn.sigmoid(alpha_raw_final)
    return np.asarray(alpha)


def solve_3imp_smoke_with_history(
    target_rgb: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    *,
    n_iters: int = 80,
    seed: int = 42,
    lr: float = 2.0,
) -> list[float]:
    """Same as :func:`solve_3imp_smoke` but returns per-iter loss for diagnostics."""
    h, w, _ = target_rgb.shape
    m = int(len(pigment_idx))
    rng = np.random.default_rng(seed)
    init_raw = rng.normal(0.0, 0.5, size=(h, w, m)).astype(np.float32)

    target_j = jnp.asarray(target_rgb, dtype=jnp.float32)
    pigments_j = jnp.asarray(pigment_idx, dtype=jnp.int32)

    def loss_fn(a_raw):
        return _solver_loss(a_raw, target_j, pigments_j)

    history: list[float] = []
    a_raw = jnp.asarray(init_raw)
    grad_fn = jax.grad(loss_fn)
    # Plain gradient descent stand-in just to assert downward trend — the
    # production smoke uses jaxopt.LBFGS via solve_3imp_smoke_with_loss.
    for _ in range(n_iters):
        loss = float(loss_fn(a_raw))
        history.append(loss)
        g = grad_fn(a_raw)
        a_raw = a_raw - lr * g
    return history


def solve_3imp_smoke_with_loss(
    target_rgb: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    *,
    n_iters: int = 200,
    seed: int = 42,
) -> tuple[NDArray[np.float32], float, float]:
    """Recover α + return (alphas, initial_loss, final_loss) for the smoke gate."""
    h, w, _ = target_rgb.shape
    m = int(len(pigment_idx))
    rng = np.random.default_rng(seed)
    init_raw = rng.normal(0.0, 0.5, size=(h, w, m)).astype(np.float32)

    target_j = jnp.asarray(target_rgb, dtype=jnp.float32)
    pigments_j = jnp.asarray(pigment_idx, dtype=jnp.int32)

    def loss_fn(a_raw):
        return _solver_loss(a_raw, target_j, pigments_j)

    init_loss = float(loss_fn(jnp.asarray(init_raw)))
    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=int(n_iters), tol=1e-7)
    result = solver.run(jnp.asarray(init_raw))
    alpha_raw_final = result.params
    alpha = jax.nn.sigmoid(alpha_raw_final)
    final_loss = float(loss_fn(alpha_raw_final))
    return np.asarray(alpha), init_loss, final_loss


__all__ = [
    "solve_3imp_smoke",
    "solve_3imp_smoke_with_history",
    "solve_3imp_smoke_with_loss",
]
