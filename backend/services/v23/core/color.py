"""ΔE color-distance utilities for v23-MCP.

Authority: research-v23-mcp-defaults.md §3 — perceptual ΔE in CIE Lab D65 is
the gate metric. ΔE76 is the cheap monotone variant (sqrt of summed Lab
component squared diffs). ΔE2000 lands when the corpus gate needs sub-1.0
precision; for v23 ship-day, ΔE76 with sRGB→Lab D65 is the canonical pipeline
metric and replaces the RGB-L2 proxy that the solver was previously emitting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _srgb_linearise(c: NDArray[np.floating]) -> NDArray[np.floating]:
    """sRGB gamma -> linear RGB. Branchless via np.where."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab_f(t: NDArray[np.floating]) -> NDArray[np.floating]:
    """The Lab nonlinearity. Cube root for large t, linear blend below ε."""
    eps = 0.008856
    return np.where(t > eps, np.cbrt(t), 7.787 * t + 16.0 / 116.0)


def srgb_to_lab(rgb: NDArray[np.floating]) -> NDArray[np.floating]:
    """sRGB in [0, 1] -> CIE Lab D65. Input shape ``(..., 3)``; output same."""
    if rgb.dtype not in (np.float32, np.float64):
        rgb = rgb.astype(np.float32)
    rgb_lin = _srgb_linearise(rgb)
    r, g, b = rgb_lin[..., 0], rgb_lin[..., 1], rgb_lin[..., 2]
    # sRGB linear -> XYZ (D65)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    # XYZ -> Lab (D65 ref white)
    xn, yn, zn = 0.95047, 1.0, 1.08883
    fx, fy, fz = _lab_f(x / xn), _lab_f(y / yn), _lab_f(z / zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return np.stack([L, a, b_lab], axis=-1)


def delta_e76(lab_a: NDArray[np.floating], lab_b: NDArray[np.floating]) -> NDArray[np.floating]:
    """ΔE*ab (1976). Input shape ``(..., 3)`` for both; output ``(...,)``."""
    diff = lab_a - lab_b
    return np.sqrt((diff * diff).sum(axis=-1))


def rgb_delta_e76(rgb_a: NDArray[np.floating], rgb_b: NDArray[np.floating]) -> NDArray[np.floating]:
    """sRGB-in-[0,1] -> ΔE76 in Lab. Convenience wrapper."""
    return delta_e76(srgb_to_lab(rgb_a), srgb_to_lab(rgb_b))


def delta_e_summary(
    rendered_rgb: NDArray[np.floating],
    target_rgb: NDArray[np.floating],
) -> dict[str, float]:
    """Compute mean + p95 ΔE76 between two (H, W, 3) sRGB images in [0, 1]."""
    dE = rgb_delta_e76(rendered_rgb, target_rgb)
    return {
        "dE_mean": float(dE.mean()),
        "dE_p95": float(np.percentile(dE, 95)),
        "dE_max": float(dE.max()),
    }


__all__ = [
    "srgb_to_lab",
    "delta_e76",
    "rgb_delta_e76",
    "delta_e_summary",
]
