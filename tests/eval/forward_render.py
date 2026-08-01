"""Forward-render master gate: pigment masks + palette -> reconstructed RGB.

This is the MOST IMPORTANT function in the eval pipeline. Every engine produces
(masks, palette, order); this function turns those back into an image. The eval
then compares that image to the original via ΔE2000 — if mean < 1.5 and p95 < 3.0,
the engine's decomposition is faithful.

Pure CPU. Deterministic. No GPU.

V2 scaffold: returns a substrate-filled canvas. MVP-A wires `pymixbox` for the
real K-M layered composition.

Reference: validation-system-v1.md sections 2 + 10.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

OrderMode = Literal["light_to_dark", "dark_to_light"]
RGB = tuple[int, int, int]


def forward_render_km(
    per_pigment_masks: list[np.ndarray],
    palette_rgb: list[RGB],
    order_mode: OrderMode = "light_to_dark",
    substrate_rgb: RGB = (255, 255, 255),
) -> np.ndarray:
    """Forward-render N translucent pigment layers via K-M stacking.

    Args:
        per_pigment_masks: list of (H, W) alpha maps, each in [0, 1]. The i-th
            mask is the spatial coverage of palette_rgb[i].
        palette_rgb: list of (R, G, B) sRGB tuples, 1:1 with masks.
        order_mode: print order. `light_to_dark` = lightest pigment first
            (matches woodblock convention). `dark_to_light` is for engines
            that emit reversed order.
        substrate_rgb: paper color. Defaults to white washi.

    Returns:
        (H, W, 3) uint8 sRGB image — the predicted print.

    MVP-A real implementation:
        1. Sort pigments by luminance (Y in XYZ) per order_mode.
        2. Initialize composite = substrate (broadcast to HxWx3).
        3. For each pigment in order:
             pigment_latent = pymixbox.rgb_to_latent(palette_rgb[i])
             composite_latent = pymixbox.rgb_to_latent(composite)
             alpha = per_pigment_masks[i][..., None]  # broadcast to 3 channels
             new_latent = (1 - alpha) * composite_latent + alpha * pigment_latent
             composite = pymixbox.latent_to_rgb(new_latent)
        4. Return composite as uint8.

    Performance budget: 12 Mpx <3s on CPU. No GPU needed — pymixbox is pure NumPy.

    Invariants the real impl MUST hold:
        - Output shape == input mask shape.
        - len(per_pigment_masks) == len(palette_rgb).
        - Empty mask list returns substrate-filled canvas (handled here).
    """
    if len(per_pigment_masks) != len(palette_rgb):
        raise ValueError(
            f"mask/palette mismatch: {len(per_pigment_masks)} masks vs {len(palette_rgb)} colors"
        )

    if not per_pigment_masks:
        # No layers -> bare substrate. Use a default canvas so callers don't crash.
        h, w = 100, 100
    else:
        h, w = per_pigment_masks[0].shape

    # V2 stub: return substrate-filled canvas. Real K-M lands in MVP-A.
    canvas = np.full((h, w, 3), substrate_rgb, dtype=np.uint8)
    return canvas
