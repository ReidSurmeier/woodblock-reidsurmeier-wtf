"""D8 — Strategy templates + measurable picker hints.

Three hardcoded named templates per addendum-v2 §"Strategy template library":

- ``portrait_emma`` — cream / cool / flesh / warm / shadow / detail
  Emma-inspired 6-impression flesh-tone portrait stack.
- ``landscape`` — cream / sky_cool / foliage_green / warm_earth / shadow / detail
  Open-air scene stack.
- ``high_chroma_graphic`` — base / dominant / complement / accent / dark_key
  Flat-color graphic / poster stack.

Per addendum-v3 fix 5: backend computes measurable features only. Opus
makes the semantic subject-classification call. :func:`pick_template_hints`
returns numeric measurables (dominant family, flesh-area %, family count,
etc.) and never a ``subject_label``. :func:`suggest_template` ranks the
three templates by how well measurable evidence matches each, but Opus
is free to override the suggestion in ``propose_stack(strategy_template=…)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TemplateId = Literal["portrait_emma", "landscape", "high_chroma_graphic"]
HueFamily = Literal[
    "cream",
    "cool",
    "flesh",
    "warm",
    "shadow",
    "detail",
    "accent",
]


@dataclass(frozen=True)
class TemplateSlot:
    """One slot in a Strategy template."""

    family: HueFamily
    role: str  # human-readable: "base", "support", "accent", "key", ...
    expected_okL: float  # typical OKLab L for this slot (light→dark soft prior)
    expected_area_pct: float  # typical % of image area for this family


@dataclass(frozen=True)
class StrategyTemplate:
    """An ordered list of impression slots forming a print plan template."""

    template_id: TemplateId
    slots: tuple[TemplateSlot, ...]
    name: str  # human-readable display name


@dataclass(frozen=True)
class TemplateSuggestion:
    """Picker output. ``template_id=None`` means no clear match — Opus picks."""

    template_id: TemplateId | None
    confidence: float  # 0..1, how well measurables match this template's prior
    reason: str  # human-readable explanation


# Portrait Emma (Chuck Close pattern): pale base → cool shadow support →
# pink flesh midtones → warm cheek accents → teal shadow → black detail
_PORTRAIT_EMMA = StrategyTemplate(
    template_id="portrait_emma",
    name="Portrait (Emma-style)",
    slots=(
        TemplateSlot(
            family="cream", role="paper-warmed base", expected_okL=0.92, expected_area_pct=18.0
        ),
        TemplateSlot(
            family="cool",
            role="under-shadow + hair-edge support",
            expected_okL=0.78,
            expected_area_pct=15.0,
        ),
        TemplateSlot(
            family="flesh", role="face midtones field", expected_okL=0.72, expected_area_pct=24.0
        ),
        TemplateSlot(
            family="warm",
            role="cheeks / lips / eye-corner accents",
            expected_okL=0.58,
            expected_area_pct=12.0,
        ),
        TemplateSlot(
            family="shadow", role="shadow-side teal", expected_okL=0.42, expected_area_pct=10.0
        ),
        TemplateSlot(
            family="detail",
            role="hair / eyes / mouth / nose lines",
            expected_okL=0.18,
            expected_area_pct=8.0,
        ),
    ),
)

# Landscape: paper → sky → foliage → earth → shadow → detail
_LANDSCAPE = StrategyTemplate(
    template_id="landscape",
    name="Landscape",
    slots=(
        TemplateSlot(
            family="cream", role="paper / cream reserve", expected_okL=0.93, expected_area_pct=12.0
        ),
        TemplateSlot(family="cool", role="sky cool", expected_okL=0.78, expected_area_pct=22.0),
        TemplateSlot(
            family="shadow", role="foliage green", expected_okL=0.50, expected_area_pct=24.0
        ),
        TemplateSlot(family="warm", role="warm earth", expected_okL=0.55, expected_area_pct=18.0),
        TemplateSlot(
            family="shadow", role="ground shadow", expected_okL=0.35, expected_area_pct=12.0
        ),
        TemplateSlot(family="detail", role="key detail", expected_okL=0.18, expected_area_pct=6.0),
    ),
)

# High-chroma graphic: base support → dominant hue → complement shadow → accent → dark key
_HIGH_CHROMA = StrategyTemplate(
    template_id="high_chroma_graphic",
    name="High-chroma graphic",
    slots=(
        TemplateSlot(
            family="cream", role="base support", expected_okL=0.90, expected_area_pct=10.0
        ),
        TemplateSlot(family="warm", role="dominant hue", expected_okL=0.65, expected_area_pct=40.0),
        TemplateSlot(
            family="cool", role="complement shadow", expected_okL=0.55, expected_area_pct=15.0
        ),
        TemplateSlot(family="accent", role="accent pop", expected_okL=0.50, expected_area_pct=18.0),
        TemplateSlot(family="detail", role="dark key", expected_okL=0.15, expected_area_pct=8.0),
    ),
)


TEMPLATES: dict[TemplateId, StrategyTemplate] = {
    "portrait_emma": _PORTRAIT_EMMA,
    "landscape": _LANDSCAPE,
    "high_chroma_graphic": _HIGH_CHROMA,
}


# ---------------------------------------------------------------------------
# Picker hints (measurable features only — no subject classification here)
# ---------------------------------------------------------------------------


def pick_template_hints(*, family_areas: dict[str, float]) -> dict[str, float | str | int]:
    """Compute measurable picker hints from per-family area fractions.

    Args:
        family_areas: dict ``{family_name: fraction_of_image_in_[0,1]}``.

    Returns:
        dict of measurables only:
        - ``dominant_family``: family with max area
        - ``max_family_area_pct``: float in 0..100
        - ``flesh_area_pct``: float in 0..100
        - ``cool_area_pct``: float in 0..100
        - ``warm_area_pct``: float in 0..100
        - ``shadow_area_pct``: float in 0..100
        - ``family_count_above_10pct``: int, families exceeding 10% area
    """
    if not family_areas:
        return {
            "dominant_family": "",
            "max_family_area_pct": 0.0,
            "flesh_area_pct": 0.0,
            "cool_area_pct": 0.0,
            "warm_area_pct": 0.0,
            "shadow_area_pct": 0.0,
            "family_count_above_10pct": 0,
        }
    dominant = max(family_areas, key=lambda k: family_areas[k])
    return {
        "dominant_family": dominant,
        "max_family_area_pct": float(family_areas[dominant] * 100.0),
        "flesh_area_pct": float(family_areas.get("flesh", 0.0) * 100.0),
        "cool_area_pct": float(family_areas.get("cool", 0.0) * 100.0),
        "warm_area_pct": float(family_areas.get("warm", 0.0) * 100.0),
        "shadow_area_pct": float(family_areas.get("shadow", 0.0) * 100.0),
        "family_count_above_10pct": int(sum(1 for v in family_areas.values() if v > 0.10)),
    }


def suggest_template(*, family_areas: dict[str, float]) -> TemplateSuggestion:
    """Rank the three templates against measurable family-area evidence.

    Returns ``TemplateSuggestion`` with the best match + confidence + reason,
    OR ``template_id=None`` when no template clearly wins. Opus reads this
    + the image preview + picks via ``propose_stack(strategy_template=…)``.
    """
    flesh = family_areas.get("flesh", 0.0)
    cool = family_areas.get("cool", 0.0)
    blue_green = cool + family_areas.get("shadow", 0.0)
    max_area = max(family_areas.values()) if family_areas else 0.0
    n_families = sum(1 for v in family_areas.values() if v > 0.10)

    # Portrait: flesh dominant family area > 15%
    if flesh > 0.15:
        confidence = min(1.0, 0.5 + flesh)
        return TemplateSuggestion(
            template_id="portrait_emma",
            confidence=float(confidence),
            reason=f"flesh family at {flesh * 100:.1f}% — portrait_emma matches",
        )

    # Landscape: green+blue dominant + no significant flesh
    if blue_green > 0.30 and flesh < 0.05:
        confidence = min(1.0, 0.4 + blue_green)
        return TemplateSuggestion(
            template_id="landscape",
            confidence=float(confidence),
            reason=f"cool + shadow families total {blue_green * 100:.1f}% — landscape matches",
        )

    # High-chroma: one family > 35% AND family count ≤ 4
    if max_area > 0.35 and n_families <= 4:
        confidence = min(1.0, 0.3 + max_area)
        return TemplateSuggestion(
            template_id="high_chroma_graphic",
            confidence=float(confidence),
            reason=(
                f"one family dominates ({max_area * 100:.1f}%) with low complexity "
                f"({n_families} significant families) — high_chroma_graphic matches"
            ),
        )

    # No clear winner — Opus picks freely
    return TemplateSuggestion(
        template_id=None,
        confidence=0.0,
        reason="no clear template match (fallback to free pigment_idx); Opus picks via prompt",
    )


__all__ = [
    "TemplateId",
    "HueFamily",
    "TemplateSlot",
    "StrategyTemplate",
    "TemplateSuggestion",
    "TEMPLATES",
    "pick_template_hints",
    "suggest_template",
]
