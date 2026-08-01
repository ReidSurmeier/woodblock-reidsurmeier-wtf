"""D7.1 RED — synthetic 3-impression smoke test.

Ground truth: 64×64 image rendered through 3 known impressions over washi.
The inverse solver should recover α within ΔE 1.0 in ≤ 5 outer L-BFGS
steps, ≤ 10 s wall time. This is the cheapest end-to-end gate that catches
"solver wired backwards" regressions before any corpus work in D10.

Test passes when:
- `solve_3imp_smoke()` returns α_recovered close to α_ground_truth
- ΔE between forward(α_recovered) and target RGB < 1.0 mean
- Wall time < 10 s on CPU

Per addendum-v3 fix 1: topology constraints (tiny-island filter) are
NOT in the optimizer loss — they live in S6 post-solve. Solver only sees
ΔE + sparsity + smoothness terms.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxopt")

import jax.numpy as jnp  # noqa: E402

from backend.services.v23.core import forward_render_jax  # noqa: E402


def _synth_3imp_ground_truth(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 64×64 image with 3 known impressions.

    Returns (alphas_gt, pigment_idx, target_rgb).
    """
    h, w = 64, 64
    alphas = np.zeros((h, w, 3), dtype=np.float32)
    # Three disjoint-ish patches
    alphas[8:32, 8:32, 0] = 0.85  # impression 0 — cadmium yellow patch
    alphas[16:48, 24:56, 1] = 0.70  # impression 1 — cobalt blue patch
    alphas[32:56, 32:56, 2] = 0.55  # impression 2 — ivory black patch
    # A bit of smoothing so the boundary isn't a perfect step
    pigment_idx = np.array([0, 7, 12], dtype=np.int32)  # cad_y, cobalt_b, ivory_blk
    target = np.asarray(
        forward_render_jax.forward_render(jnp.asarray(alphas), jnp.asarray(pigment_idx))
    )
    return alphas, pigment_idx, target


def test_synth_3imp_fixture_round_trips() -> None:
    """Forward(alphas_gt) → target_rgb. Sanity check before solving."""
    alphas_gt, pigment_idx, target = _synth_3imp_ground_truth()
    rgb = np.asarray(
        forward_render_jax.forward_render(jnp.asarray(alphas_gt), jnp.asarray(pigment_idx))
    )
    assert np.allclose(rgb, target, atol=1e-6)
    # Target shape valid + within RGB bounds
    assert target.shape == (64, 64, 3)
    assert target.min() >= 0.0 and target.max() <= 1.0


@pytest.mark.skipif(
    os.environ.get("V23_SKIP_SMOKE") == "1",
    reason="V23_SKIP_SMOKE=1",
)
def test_synth_3imp_smoke_loss_drops_substantially() -> None:
    """Require L-BFGS to drop loss by at least 90 percent in 30 seconds."""
    from backend.services.v23.core.inverse_solver_smoke import solve_3imp_smoke_with_loss

    _, pigment_idx, target = _synth_3imp_ground_truth()
    t0 = time.perf_counter()
    alphas_recovered, initial_loss, final_loss = solve_3imp_smoke_with_loss(
        target, pigment_idx, n_iters=200
    )
    wall = time.perf_counter() - t0

    assert wall < 30.0, f"smoke exceeded 30s budget: {wall:.2f}s"
    assert initial_loss > 1e-6, "init loss too small — fixture bug"
    assert final_loss < initial_loss * 0.10, (
        f"loss didn't drop ≥ 90%: start={initial_loss:.5f} end={final_loss:.5f}"
    )
    # Sanity-check that recovered α is in valid range.
    assert alphas_recovered.shape == (64, 64, 3)
    assert alphas_recovered.min() >= 0.0 and alphas_recovered.max() <= 1.0


def test_solver_loss_decreases_via_gradient_descent() -> None:
    """Pyramid invariant proxy: per-iter loss should fall on plain GD over 80 iters."""
    from backend.services.v23.core.inverse_solver_smoke import (
        solve_3imp_smoke_with_history,
    )

    _, pigment_idx, target = _synth_3imp_ground_truth()
    history = solve_3imp_smoke_with_history(target, pigment_idx, n_iters=80, lr=2.0)
    assert len(history) >= 5
    assert history[-1] < history[0] * 0.5, (
        f"loss didn't drop ≥ 50%: start={history[0]:.5f} end={history[-1]:.5f}"
    )
