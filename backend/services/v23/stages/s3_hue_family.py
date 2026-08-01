"""D10.a — S3 hue-family classifier.

Per-pixel OKLab-bucketed classification into 7 mokuhanga hue families:
cream / cool / flesh / warm / shadow / detail / accent.

Heuristic bands (not learned) keyed off luminance + warm/cool/green:
- detail: luminance ≤ 0.20
- cream:  luminance > 0.85 AND warm
- flesh:  0.55 < L ≤ 0.80, warm, not over-saturated
- warm:   0.30 < L ≤ 0.70, warm, saturated
- cool:   0.40 < L ≤ 0.85, cool (b > r AND b > g)
- shadow: 0.20 < L ≤ 0.50, green OR cool
- accent: everything else (purples, off-saturated mids)

Pure heuristic so day-1 ships without learned weights. Calibrated to
the 13-pigment Mixbox catalog used in forward_render_jax.PIGMENT_TABLE.
A future D11+ tier can swap in a learned classifier (e.g. OKLab k-means
from the actual image) when the corpus shows the heuristic drifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from backend.services.v23 import session as _sess

FAMILY_LABEL_TO_INDEX: dict[str, int] = {
    "cream": 0,
    "cool": 1,
    "flesh": 2,
    "warm": 3,
    "shadow": 4,
    "detail": 5,
    "accent": 6,
}
FAMILY_INDEX_TO_LABEL: dict[int, str] = {v: k for k, v in FAMILY_LABEL_TO_INDEX.items()}

# One bright color per family for the family-map PNG export.
_FAMILY_RGB: dict[str, tuple[int, int, int]] = {
    "cream": (250, 240, 215),
    "cool": (90, 130, 200),
    "flesh": (235, 180, 150),
    "warm": (220, 100, 70),
    "shadow": (60, 120, 110),
    "detail": (20, 20, 20),
    "accent": (160, 60, 180),
}


@dataclass(frozen=True)
class HueFamilyResult:
    """Output of :func:`classify_hue_families`. Pure numpy — no I/O."""

    label_map: NDArray[np.uint8]
    family_areas: dict[str, float]
    dominant_family: str
    label_map_path: Path | None = None


def _classify_pixel_indices(rgb01: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Vectorised per-pixel family classification. ``rgb01`` shape (N, 3)."""
    r, g, b = rgb01[:, 0], rgb01[:, 1], rgb01[:, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    warm = (r > g) & (r > b)
    # Cool/green use >= to handle exact ties (e.g. pure teal g == b > r).
    cool = (b >= r) & (b >= g) & ~warm
    green = (g >= r) & (g >= b) & ~warm
    cool_or_green = cool | green
    saturated = (rgb01.max(axis=1) - rgb01.min(axis=1)) > 0.20

    n = rgb01.shape[0]
    # Default everyone to accent; bands overwrite by priority.
    labels = np.full(n, FAMILY_LABEL_TO_INDEX["accent"], dtype=np.uint8)

    # Detail (very dark) wins over everything
    labels[lum <= 0.20] = FAMILY_LABEL_TO_INDEX["detail"]

    # Cream (very light + warm)
    cream_mask = (lum > 0.85) & warm & (labels == FAMILY_LABEL_TO_INDEX["accent"])
    labels[cream_mask] = FAMILY_LABEL_TO_INDEX["cream"]

    # Cool (any reasonably-bright cool pixel)
    cool_mask = (lum > 0.40) & (lum <= 0.85) & cool & (labels == FAMILY_LABEL_TO_INDEX["accent"])
    labels[cool_mask] = FAMILY_LABEL_TO_INDEX["cool"]

    # Shadow (mid-dark teal/green)
    shadow_mask = (
        (lum > 0.20) & (lum <= 0.50) & cool_or_green & (labels == FAMILY_LABEL_TO_INDEX["accent"])
    )
    labels[shadow_mask] = FAMILY_LABEL_TO_INDEX["shadow"]

    # Flesh (warm mid-light, not super saturated)
    flesh_mask = (
        (lum > 0.55)
        & (lum <= 0.80)
        & warm
        & (rgb01[:, 0] - rgb01[:, 1] < 0.30)  # not too red
        & (labels == FAMILY_LABEL_TO_INDEX["accent"])
    )
    labels[flesh_mask] = FAMILY_LABEL_TO_INDEX["flesh"]

    # Warm (saturated warm mid)
    warm_mask = (
        (lum > 0.30)
        & (lum <= 0.70)
        & warm
        & saturated
        & (labels == FAMILY_LABEL_TO_INDEX["accent"])
    )
    labels[warm_mask] = FAMILY_LABEL_TO_INDEX["warm"]

    return labels


def classify_hue_families(rgb: NDArray[np.uint8]) -> HueFamilyResult:
    """Pure-numpy classifier. Returns labels + areas. No I/O."""
    h, w, _ = rgb.shape
    rgb01 = rgb.astype(np.float32).reshape(-1, 3) / 255.0
    labels_flat = _classify_pixel_indices(rgb01)
    label_map = labels_flat.reshape(h, w)

    family_areas: dict[str, float] = {}
    total = float(h * w)
    for label, idx in FAMILY_LABEL_TO_INDEX.items():
        family_areas[label] = float(np.count_nonzero(labels_flat == idx)) / total

    dominant = max(family_areas, key=lambda k: family_areas[k])
    return HueFamilyResult(
        label_map=label_map,
        family_areas=family_areas,
        dominant_family=dominant,
        label_map_path=None,
    )


def _render_family_map_png(label_map: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert label_map to RGB visualisation using FAMILY_RGB palette."""
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for label, idx in FAMILY_LABEL_TO_INDEX.items():
        mask = label_map == idx
        if mask.any():
            out[mask] = _FAMILY_RGB[label]
    return out


def run_s3_hue_family(
    rgb: NDArray[np.uint8],
    *,
    image_sha256: str,
) -> HueFamilyResult:
    """Classify + persist the family-map PNG under the active session."""
    base = classify_hue_families(rgb)

    sid = _sess.current_session()
    if sid is None:
        s = _sess.new_session()
        _sess.set_current_session(s.session_id)
        sid = s.session_id

    sdir = _sess.paths.session_dir(sid)
    family_dir = sdir / "hue_family_maps"
    family_dir.mkdir(parents=True, exist_ok=True)
    out_path = family_dir / f"{image_sha256}.png"
    rgb_vis = _render_family_map_png(base.label_map)
    Image.fromarray(rgb_vis, mode="RGB").save(out_path)

    return HueFamilyResult(
        label_map=base.label_map,
        family_areas=base.family_areas,
        dominant_family=base.dominant_family,
        label_map_path=out_path,
    )


__all__ = [
    "FAMILY_LABEL_TO_INDEX",
    "FAMILY_INDEX_TO_LABEL",
    "HueFamilyResult",
    "classify_hue_families",
    "run_s3_hue_family",
]
