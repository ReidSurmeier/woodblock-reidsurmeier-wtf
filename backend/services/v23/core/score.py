"""D12.a — score_candidate_stack 5-component real breakdown.

Per addendum-v3 fix 4: pure ΔE never claims "underprint correctness".
The combined score makes the trade-offs explicit + Opus-readable:

| Component         | Default weight | Range | Higher is better |
|-------------------|----------------|-------|------------------|
| visual_match      | 0.40 | 0..1 | yes — 1 ⇔ ΔE 0; 0 ⇔ ΔE ≥ target |
| carveability      | 0.20 | 0..1 | yes — based on small-island ratio |
| simplicity        | 0.15 | 0..1 | yes — fewer impressions + low hidden overhead |
| underprint_utility| 0.15 | 0..1 | yes — covered/support coverage from state_summary |
| template_fit      | 0.10 | 0..1 | yes — fraction of impressions matching template slots |

The components are NOT physical proofs of correctness — they're
weighted heuristics. The "notes" string surfaces the weakest component
so Opus can name it directly to the artist.
"""

from __future__ import annotations

from typing import Any

from backend.services.v23 import orchestrator as _orch
from backend.services.v23.core import templates as _templates

_DEFAULT_WEIGHTS: dict[str, float] = {
    "visual_match": 0.40,
    "carveability": 0.20,
    "simplicity": 0.15,
    "underprint_utility": 0.15,
    "template_fit": 0.10,
}

_TARGET_DE: float = 1.5  # mean ΔE2000 target from defaults
_MAX_IMPRESSIONS: int = 10  # plan §10 hard cap


def _visual_match(plan: _orch.PartialPlan) -> float:
    de = plan.reconstruction_dE_mean
    if de is None:
        return 0.0
    return float(max(0.0, 1.0 - de / _TARGET_DE))


def _simplicity(plan: _orch.PartialPlan) -> float:
    """Fewer impressions + low hidden-coverage overhead → higher score."""
    n = len(plan.impressions)
    if n == 0:
        return 0.0
    # 1 impression → ~0.9; max impressions → ~0.4
    base = 1.0 - (n / _MAX_IMPRESSIONS) * 0.5
    # Penalise large total hidden coverage
    hidden_overhead = 0.0
    for entry in plan.state_summary:
        hidden_overhead += entry.get("covered_pct", 0.0) / 100.0
    if plan.state_summary:
        hidden_overhead /= len(plan.state_summary)
    return float(max(0.0, min(1.0, base - hidden_overhead * 0.2)))


def _carveability(plan: _orch.PartialPlan) -> float:
    """Approximate: low-impression count + reasonable block_count → higher score."""
    if not plan.impressions:
        return 0.0
    # Penalty for too many distinct blocks (cap 6 default)
    block_penalty = max(0, plan.block_count - 6) * 0.05
    # State summary gives a rough split — pull pct visible from each impression
    visible_share = 0.0
    n = max(len(plan.state_summary), 1)
    for entry in plan.state_summary:
        visible_share += entry.get("visible_pct", 0.0) / 100.0
    visible_share /= n
    # High visible-share → less "wasted" mask area → more carveable
    score = visible_share * 0.7 + 0.3 - block_penalty
    return float(max(0.0, min(1.0, score)))


def _underprint_utility(plan: _orch.PartialPlan) -> float:
    """Fraction of mask area that contributes as support or covered (underprint)."""
    if not plan.state_summary:
        return 0.0
    total = 0.0
    for entry in plan.state_summary:
        total += entry.get("support_pct", 0.0) + entry.get("covered_pct", 0.0)
    avg = total / (len(plan.state_summary) * 100.0)
    # Sweet spot ~0.3 — too little = unused; too much = wasted carving
    if avg < 0.05:
        return 0.0
    if avg > 0.50:
        return float(max(0.0, 1.0 - (avg - 0.50) * 2))
    return float(min(1.0, avg / 0.30))


def _template_fit(plan: _orch.PartialPlan) -> float:
    """Match impression pigment families to the suggested template's slot families."""
    if plan.suggested_template is None or plan.suggested_template not in _templates.TEMPLATES:
        return 0.0
    template = _templates.TEMPLATES[plan.suggested_template]
    expected_families = [slot.family for slot in template.slots]

    from backend.services.v23.core import forward_render_jax

    pigment_to_family = _pigment_family_map(forward_render_jax.PIGMENT_RGB_255)

    actual_families = [
        pigment_to_family.get(imp["pigment_id"], "accent") for imp in plan.impressions
    ]
    if not actual_families:
        return 0.0

    # Soft fit: fraction of impressions whose family appears anywhere in template
    template_family_set = set(expected_families)
    matches = sum(1 for f in actual_families if f in template_family_set)
    return float(matches / len(actual_families))


def _pigment_family_map(pigment_rgb_255) -> dict[int, str]:
    """Coarse pigment → family lookup for the 13-catalog."""
    # Hand-mapping aligned with the catalog + family taxonomy
    return {
        0: "warm",  # cadmium_yellow
        1: "warm",  # hansa_yellow
        2: "warm",  # cadmium_orange
        3: "warm",  # cadmium_red
        4: "accent",  # quinacridone_magenta
        5: "accent",  # cobalt_violet
        6: "cool",  # ultramarine_blue
        7: "cool",  # cobalt_blue
        8: "shadow",  # viridian_green
        9: "shadow",  # forest_green
        10: "warm",  # burnt_sienna
        11: "shadow",  # raw_umber
        12: "detail",  # ivory_black
    }


def score_plan_real(plan: _orch.PartialPlan) -> dict[str, Any]:
    """Compute the 5-component combined score for a persisted PartialPlan."""
    components = {
        "visual_match": _visual_match(plan),
        "carveability": _carveability(plan),
        "simplicity": _simplicity(plan),
        "underprint_utility": _underprint_utility(plan),
        "template_fit": _template_fit(plan),
    }
    overall = sum(_DEFAULT_WEIGHTS[k] * components[k] for k in _DEFAULT_WEIGHTS)

    # Find weakest component for the plain-language notes
    weakest = min(components, key=lambda k: components[k])
    notes = _explain_weakest(weakest, components[weakest], plan)

    return {
        "plan_id": plan.plan_id,
        "overall": overall,
        **components,
        "component_weights": dict(_DEFAULT_WEIGHTS),
        "notes": notes,
    }


def _explain_weakest(name: str, value: float, plan: _orch.PartialPlan) -> str:
    if name == "visual_match":
        return (
            f"weakest component is visual_match ({value:.2f}). "
            f"Reconstruction ΔE mean is {plan.reconstruction_dE_mean} vs target 1.5. "
            "Calibrate pigments via upload_swatch_overprint_matrix for accuracy."
        )
    if name == "carveability":
        return (
            f"weakest component is carveability ({value:.2f}). "
            "Run simplify_masks_for_carving to remove tiny islands."
        )
    if name == "simplicity":
        return (
            f"weakest component is simplicity ({value:.2f}). "
            f"Plan has {len(plan.impressions)} impressions; consider merging similar ones."
        )
    if name == "underprint_utility":
        return (
            f"weakest component is underprint_utility ({value:.2f}). "
            "Pin a region as underprint via pin_region(action='force') to add structural depth."
        )
    if name == "template_fit":
        return (
            f"weakest component is template_fit ({value:.2f}). "
            "Try a different strategy_template or let the solver pick freely (template=None)."
        )
    return f"weakest component is {name} ({value:.2f})."


__all__ = ["score_plan_real"]
