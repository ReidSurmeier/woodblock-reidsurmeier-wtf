"""D6.5 RED — render_tier dispatch contract (addendum-v4 lock).

Three tiers per `/tmp/research-v23-mcp-user-addendum-v4.md`:
- t1_mixbox    — ships v23, models palette mixing (Mixbox lerp)
- t2_empirical — ships v23.1, 2-layer LUT from artist swatch sheet
- t3_spectral  — ships v24, K-M two-flux recursion + 8λ (K,S) fit

Selection rules:
- t3_spectral when spectral_ks fit available AND stack_depth > 3
- t2_empirical when empirical LUT available
- t1_mixbox otherwise (day-1 default)
"""

from __future__ import annotations

import pytest


def test_default_tier_is_t1_mixbox_with_no_calibration() -> None:
    from backend.services.v23.core.render_tier import RenderTierContext, choose_render_tier

    ctx = RenderTierContext(
        calibration_id=None,
        empirical_lut_available=False,
        spectral_ks_available=False,
        stack_depth=4,
    )
    assert choose_render_tier(ctx) == "t1_mixbox"


def test_promotes_to_t2_empirical_when_swatch_lut_present() -> None:
    from backend.services.v23.core.render_tier import RenderTierContext, choose_render_tier

    ctx = RenderTierContext(
        calibration_id="cal_2026-05-12_winter-cherry",
        empirical_lut_available=True,
        spectral_ks_available=False,
        stack_depth=4,
    )
    assert choose_render_tier(ctx) == "t2_empirical"


def test_promotes_to_t3_spectral_when_spectral_and_deep_stack() -> None:
    from backend.services.v23.core.render_tier import RenderTierContext, choose_render_tier

    ctx = RenderTierContext(
        calibration_id="cal_2026-05-12_full-spectral",
        empirical_lut_available=True,
        spectral_ks_available=True,
        stack_depth=5,
    )
    assert choose_render_tier(ctx) == "t3_spectral"


def test_t3_blocked_by_shallow_stack_falls_back_to_t2() -> None:
    """Shallow stacks (≤3) don't need spectral physics; t2 is enough."""
    from backend.services.v23.core.render_tier import RenderTierContext, choose_render_tier

    ctx = RenderTierContext(
        calibration_id="cal_2026-05-12_full-spectral",
        empirical_lut_available=True,
        spectral_ks_available=True,
        stack_depth=3,
    )
    assert choose_render_tier(ctx) == "t2_empirical"


def test_lerp_overlap_warning_fires_on_deep_t1() -> None:
    """Per overlap-math §8: stack > 3 on t1 emits LERP_OVERLAP_GT3 warning."""
    from backend.services.v23.core.render_tier import diagnose_render_tier

    diag = diagnose_render_tier(
        tier="t1_mixbox",
        stack_depth=5,
    )
    codes = {w.code for w in diag.warnings}
    assert "LERP_OVERLAP_GT3" in codes
    fired = next(w for w in diag.warnings if w.code == "LERP_OVERLAP_GT3")
    assert fired.tier == "warn"


def test_t2_with_shallow_stack_no_warning() -> None:
    from backend.services.v23.core.render_tier import diagnose_render_tier

    diag = diagnose_render_tier(tier="t2_empirical", stack_depth=2)
    assert diag.warnings == []


def test_invalid_tier_raises() -> None:
    from backend.services.v23.core.render_tier import diagnose_render_tier

    with pytest.raises(ValueError):
        diagnose_render_tier(tier="t99_quantum", stack_depth=1)  # type: ignore[arg-type]


def test_render_tier_is_literal_typed() -> None:
    """The returned tier must be one of the three locked names."""
    from backend.services.v23.core.render_tier import (
        RenderTierContext,
        choose_render_tier,
    )

    ctx = RenderTierContext(
        calibration_id=None,
        empirical_lut_available=False,
        spectral_ks_available=False,
        stack_depth=1,
    )
    out = choose_render_tier(ctx)
    assert out in ("t1_mixbox", "t2_empirical", "t3_spectral")
    # mypy/type-check shape — also runtime-asserted via Literal at use sites
    assert isinstance(out, str)
