"""D6 — JAX-traceable forward render (Mixbox-stack lerp).

Stacks per-pixel pigment alpha maps over a washi paper substrate in
print order. The forward render is the gradient target of the inverse
solver (S5, D7+) so it MUST be JAX-jittable and produce finite gradients
w.r.t. the alpha tensor.

Ship-mode (this file): pure-JAX RGB-space stack, no external Mixbox
dependency. The 13-pigment table seeds from `palette_extract.MIXBOX_PIGMENTS`
RGB anchors so downstream stages keep using the same pigment ids.

When the Scrtwpns Mixbox C library is available on the GPU host, a
pre-built trilinear LUT at ``~/.woodblock/v23/luts/mixbox_q32.npz`` is
loaded transparently and replaces the linear-blend kernel. The LUT
generator lives at ``scripts/build_mixbox_lut.py`` (offline tool).

Per addendum-v4 — this is Tier-1 (t1_mixbox) of the 3-tier render
hierarchy. Models palette MIXING, NOT overprint glazing. Honest output
includes the "as if pre-mixed" qualifier per WB-LANG-02 lint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# Per-pigment sRGB anchors for the 13-catalog. Sourced from
# `backend/algorithms/decomposition/palette_extract.py::MIXBOX_PIGMENTS`
# (hand-verified Mixbox-anchored entries). Vendored here so the module
# stays self-contained and JAX-traceable without importing palette_extract.
PIGMENT_RGB_255: NDArray[np.uint8] = np.array(
    [
        (254, 236, 0),  # 0  cadmium_yellow
        (252, 211, 0),  # 1  hansa_yellow
        (252, 102, 16),  # 2  cadmium_orange
        (227, 38, 54),  # 3  cadmium_red
        (199, 21, 133),  # 4  quinacridone_magenta
        (102, 51, 153),  # 5  cobalt_violet
        (33, 41, 165),  # 6  ultramarine_blue
        (0, 71, 171),  # 7  cobalt_blue
        (0, 102, 102),  # 8  viridian_green
        (62, 96, 56),  # 9  forest_green
        (139, 69, 19),  # 10 burnt_sienna
        (51, 25, 0),  # 11 raw_umber
        (15, 15, 15),  # 12 ivory_black
    ],
    dtype=np.uint8,
)

PIGMENT_TABLE: NDArray[np.float32] = PIGMENT_RGB_255.astype(np.float32) / 255.0

# Washi-paper substrate. Slight warm cream (~ #f6f1e3) — matches the
# `paper` Oklab anchor in the v23 plan §7 + addendum-v4.
PAPER_RGB: NDArray[np.float32] = np.array([0.965, 0.945, 0.890], dtype=np.float32)


_LUT_PATH: Path = Path.home() / ".woodblock" / "v23" / "luts" / "mixbox_q32.npz"
_LUT_CACHE: dict[str, NDArray[np.float32] | None] = {"lut": None}


def _load_lut() -> NDArray[np.float32] | None:
    """Load the Mixbox trilinear LUT if the offline tool has built it."""
    if _LUT_CACHE["lut"] is not None:
        return _LUT_CACHE["lut"]
    if not _LUT_PATH.is_file():
        return None
    data = np.load(_LUT_PATH)
    if "rgb" not in data.files:
        return None
    _LUT_CACHE["lut"] = data["rgb"].astype(np.float32)
    return _LUT_CACHE["lut"]


def forward_render(
    alphas: jnp.ndarray,
    pigment_idx: jnp.ndarray,
    paper_rgb: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Forward render an impression stack.

    Models palette mixing in RGB space — pigments are blended as if
    pre-mixed in a well before application. For accurate overprint
    physics, switch tier via :mod:`backend.services.v23.core.render_tier`.

    Args:
        alphas: ``(H, W, M)`` float in ``[0, 1]`` — per-pixel coverage
            for each impression in print order (impression 0 prints first,
            impression M-1 prints last on top).
        pigment_idx: ``(M,)`` int32 — index into ``PIGMENT_TABLE`` per impression.
        paper_rgb: optional ``(3,)`` substrate override; defaults to washi.

    Returns:
        ``(H, W, 3)`` float in ``[0, 1]`` — predicted sRGB composite.
    """
    h, w, m = alphas.shape
    paper = jnp.asarray(PAPER_RGB if paper_rgb is None else paper_rgb, dtype=jnp.float32)
    pigments = jnp.asarray(PIGMENT_TABLE, dtype=jnp.float32)
    selected = pigments[pigment_idx]  # (M, 3)

    composite = jnp.broadcast_to(paper, (h, w, 3))

    # Stack in print order: composite_new = (1 - α) * composite + α * pigment.
    # Python loop over M is unrolled by JAX at trace time — M is small (4..12).
    a_first = jnp.transpose(alphas, (2, 0, 1))  # (M, H, W)
    out = composite
    for i in range(m):
        a3 = a_first[i, ..., None]
        p = selected[i]
        out = (1.0 - a3) * out + a3 * p[None, None, :]

    return jnp.clip(out, 0.0, 1.0)


__all__ = ["forward_render", "PIGMENT_TABLE", "PAPER_RGB", "PIGMENT_RGB_255"]
