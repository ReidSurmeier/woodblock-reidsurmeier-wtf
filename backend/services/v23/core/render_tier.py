"""D6.5 — Render tier dispatch (addendum-v4 lock).

Three forward-render tiers chosen at solve time based on the artist's
available calibration + the current stack depth:

- ``t1_mixbox``    — Mixbox-stack lerp in RGB space (ships v23, models MIXING)
- ``t2_empirical`` — 2-layer LUT from artist swatch sheet (ships v23.1, OVERPRINT)
- ``t3_spectral``  — K-M two-flux recursion + 8λ (K,S) spectral fit (ships v24)

Selection logic (addendum-v4 §"3-tier forward-render architecture"):

    if spectral_ks_available AND stack_depth > 3:
        return "t3_spectral"
    if empirical_lut_available:
        return "t2_empirical"
    return "t1_mixbox"

Diagnostics surface the addendum-v4 LERP_OVERLAP_GT3 warning when a
deep stack runs on t1 — heads-up to the artist that Mixbox's mixing
approximation degrades past 3 stacked translucent impressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from backend.mcp.errors import WoodblockError

RenderTier = Literal["t1_mixbox", "t2_empirical", "t3_spectral"]

_VALID_TIERS: frozenset[str] = frozenset(("t1_mixbox", "t2_empirical", "t3_spectral"))
_LERP_OVERLAP_MAX: int = 3


@dataclass(frozen=True)
class RenderTierContext:
    """Inputs to :func:`choose_render_tier`."""

    calibration_id: str | None
    empirical_lut_available: bool
    spectral_ks_available: bool
    stack_depth: int


@dataclass(frozen=True)
class RenderTierDiagnosis:
    """Result of :func:`diagnose_render_tier`. Carries any structured warnings."""

    tier: RenderTier
    warnings: list[WoodblockError] = field(default_factory=list)


def choose_render_tier(ctx: RenderTierContext) -> RenderTier:
    """Pick the highest-fidelity tier whose prerequisites are satisfied."""
    if ctx.spectral_ks_available and ctx.stack_depth > _LERP_OVERLAP_MAX:
        return "t3_spectral"
    if ctx.empirical_lut_available:
        return "t2_empirical"
    return "t1_mixbox"


def diagnose_render_tier(*, tier: RenderTier, stack_depth: int) -> RenderTierDiagnosis:
    """Surface warnings for the chosen tier given the stack depth."""
    if tier not in _VALID_TIERS:
        raise ValueError(f"invalid render tier: {tier!r}")

    warnings: list[WoodblockError] = []
    if tier == "t1_mixbox" and stack_depth > _LERP_OVERLAP_MAX:
        warnings.append(
            WoodblockError(
                tier="warn",
                code="LERP_OVERLAP_GT3",
                message=(
                    f"Mixbox lerp degrades past 3 stacked translucents "
                    f"(stack depth {stack_depth}). Predicted composite "
                    "is directionally correct but absolute color may drift "
                    "ΔE 4-8 vs actual overprint."
                ),
                hint="upload a swatch overprint matrix to unlock t2_empirical",
                recoverable=True,
                context={"stack_depth": stack_depth, "tier": tier},
            )
        )
    return RenderTierDiagnosis(tier=tier, warnings=warnings)


__all__ = [
    "RenderTier",
    "RenderTierContext",
    "RenderTierDiagnosis",
    "choose_render_tier",
    "diagnose_render_tier",
]
