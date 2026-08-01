"""D8 RED — 3 strategy templates + measurable picker hints (addendum-v2 + fix 5).

Three hardcoded named templates loaded from
:mod:`backend.services.v23.core.templates`:

- ``portrait_emma`` — cream / cool / flesh / warm / shadow / detail
- ``landscape``     — cream / sky_cool / foliage_green / warm_earth / shadow / detail
- ``high_chroma_graphic`` — base / dominant / complement / accent / dark_key

Per addendum-v3 fix 5: backend ``pick_template_hints()`` returns
MEASURABLE features only (family-area dict + dominant-family). Subject
classification is Opus's job, NOT the backend's.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def test_three_templates_loaded() -> None:
    from backend.services.v23.core.templates import TEMPLATES

    assert set(TEMPLATES.keys()) == {"portrait_emma", "landscape", "high_chroma_graphic"}


def test_portrait_emma_has_six_slots_in_order() -> None:
    from backend.services.v23.core.templates import TEMPLATES

    t = TEMPLATES["portrait_emma"]
    families = [slot.family for slot in t.slots]
    assert families == ["cream", "cool", "flesh", "warm", "shadow", "detail"]
    # Light → dark soft order via decreasing typical luminance is the prior
    luminances = [slot.expected_okL for slot in t.slots]
    assert luminances == sorted(luminances, reverse=True), (
        f"portrait_emma slots out of light→dark order: {luminances}"
    )


def test_landscape_has_six_slots() -> None:
    from backend.services.v23.core.templates import TEMPLATES

    t = TEMPLATES["landscape"]
    families = [slot.family for slot in t.slots]
    assert families == ["cream", "cool", "shadow", "warm", "shadow", "detail"]


def test_high_chroma_graphic_has_five_slots() -> None:
    from backend.services.v23.core.templates import TEMPLATES

    t = TEMPLATES["high_chroma_graphic"]
    assert len(t.slots) == 5


def test_picker_hints_returns_measurables_no_subject_label() -> None:
    """Addendum-v3 fix 5: NO subject_label, NO intent classification."""
    from backend.services.v23.core.templates import pick_template_hints

    family_areas = {
        "cream": 0.20,
        "cool": 0.05,
        "flesh": 0.28,
        "warm": 0.12,
        "shadow": 0.05,
        "detail": 0.10,
        "accent": 0.20,
    }
    hints = pick_template_hints(family_areas=family_areas)
    assert "subject_label" not in hints, "subject classification belongs to Opus, not backend"
    assert "intent" not in hints
    # Measurables only
    assert "dominant_family" in hints
    assert "flesh_area_pct" in hints
    assert "max_family_area_pct" in hints
    assert "family_count_above_10pct" in hints


def test_picker_dominant_family_matches_max() -> None:
    from backend.services.v23.core.templates import pick_template_hints

    areas = {"cream": 0.30, "cool": 0.10, "flesh": 0.05}
    hints = pick_template_hints(family_areas=areas)
    assert hints["dominant_family"] == "cream"
    assert hints["max_family_area_pct"] == pytest.approx(30.0, abs=0.01)


def test_template_autoselect_portrait_when_flesh_dominant() -> None:
    """Picker only RECOMMENDS — Opus reads + decides."""
    from backend.services.v23.core.templates import suggest_template

    areas = {
        "cream": 0.20,
        "cool": 0.05,
        "flesh": 0.28,
        "warm": 0.12,
        "shadow": 0.05,
        "detail": 0.10,
        "accent": 0.20,
    }
    suggestion = suggest_template(family_areas=areas)
    assert suggestion.template_id == "portrait_emma"
    assert suggestion.confidence > 0.5
    assert suggestion.reason  # human-readable


def test_template_autoselect_landscape_when_green_blue_dominant() -> None:
    from backend.services.v23.core.templates import suggest_template

    areas = {
        "cream": 0.10,
        "cool": 0.35,
        "flesh": 0.0,
        "warm": 0.15,
        "shadow": 0.15,
        "detail": 0.10,
        "accent": 0.15,
    }
    s = suggest_template(family_areas=areas)
    assert s.template_id == "landscape"


def test_template_autoselect_high_chroma_when_one_family_dominates() -> None:
    from backend.services.v23.core.templates import suggest_template

    areas = {
        "cream": 0.05,
        "cool": 0.05,
        "flesh": 0.0,
        "warm": 0.40,
        "shadow": 0.05,
        "detail": 0.05,
        "accent": 0.40,
    }
    s = suggest_template(family_areas=areas)
    assert s.template_id == "high_chroma_graphic"


def test_template_autoselect_falls_back_to_null_on_no_clear_winner() -> None:
    """Equal distribution → no recommendation, Opus picks freely."""
    from backend.services.v23.core.templates import suggest_template

    areas = {f: 1.0 / 7 for f in ("cream", "cool", "flesh", "warm", "shadow", "detail", "accent")}
    s = suggest_template(family_areas=areas)
    assert s.template_id is None
    assert "no clear" in s.reason.lower() or "fallback" in s.reason.lower()


def test_template_slot_dataclass_is_frozen() -> None:
    from backend.services.v23.core.templates import TemplateSlot

    slot = TemplateSlot(family="cream", role="base", expected_okL=0.95, expected_area_pct=15.0)
    with pytest.raises(FrozenInstanceError):
        slot.family = "cool"  # type: ignore[misc]
