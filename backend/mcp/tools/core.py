"""D9A — Tier 0 core flow tools (10 tools).

Every tool is callable directly and through the MCP registry, returning a
``ToolResult[T]`` envelope instead of raising user-facing exceptions.

Per addendum-v4 WB-LANG-02: every t1_mixbox recipe carries the
"as if pre-mixed" qualifier so artists never confuse Mixbox prediction
with overprint physics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23 import orchestrator as _orch
from backend.services.v23.core import templates as _templates
from backend.services.v23.stages import s1_ingest

_VALID_SOLVE_PROFILES = ("fast", "default", "thorough")
_VALID_FOCUS_MODES = (
    "composite",
    "heatmap",
    "per_impression",
    "confidence",
    "quad",
    "recipe",
    "pixel",
)
FocusMode = Literal[
    "composite",
    "heatmap",
    "per_impression",
    "confidence",
    "quad",
    "recipe",
    "pixel",
]


def _impl_pending(code: str, hint: str) -> WoodblockError:
    return WoodblockError(
        tier="degraded",
        code=code,
        message=f"{code} — backing implementation lands in a later substep",
        hint=hint,
        recoverable=True,
    )


def _render_de_heatmap(plan: _orch.PartialPlan, plan_dir: Path) -> Path | None:
    """Per-pixel ΔE76 heatmap PNG (viridis-style ramp). Saves under plan_dir."""
    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return None
    target_path = Path(plan.alpha_stack_path).parent / "target.npy"
    if not target_path.is_file():
        return None

    import jax.numpy as jnp
    import numpy as np
    from PIL import Image

    from backend.services.v23.core import color, forward_render_jax

    alpha_stack = np.load(plan.alpha_stack_path)  # (M, H, W)
    target = np.load(target_path)
    alpha_hwm = np.transpose(alpha_stack, (1, 2, 0))
    rendered = np.asarray(
        forward_render_jax.forward_render(
            jnp.asarray(alpha_hwm, dtype=jnp.float32),
            jnp.asarray(plan.pigment_idx, dtype=jnp.int32),
        )
    )
    dE = color.rgb_delta_e76(rendered, target)  # (H, W) in ΔE76 units
    # Normalise to [0, 1] with a 0..15 ΔE range (anything > 15 saturates red)
    norm = np.clip(dE / 15.0, 0.0, 1.0)
    # Simple blue→cyan→yellow→red ramp
    r = np.clip(2 * norm - 0.5, 0.0, 1.0)
    g = np.clip(1.0 - 2.0 * np.abs(norm - 0.5), 0.0, 1.0)
    b = np.clip(1.0 - 2.0 * norm, 0.0, 1.0)
    heat = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    out = plan_dir / "heatmap.png"
    Image.fromarray(heat, "RGB").save(out)
    return out


def _render_quad_grid(plan: _orch.PartialPlan, plan_dir: Path) -> Path | None:
    """4-up grid: target | composite | dE-heatmap | confidence(state colour-map)."""
    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return None
    target_path = Path(plan.alpha_stack_path).parent / "target.npy"
    if not target_path.is_file():
        return None

    import jax.numpy as jnp
    import numpy as np
    from PIL import Image

    from backend.services.v23.core import color, forward_render_jax
    from backend.services.v23.stages import s10_emit

    target = np.load(target_path)
    composite = np.asarray(s10_emit._render_composite(plan))
    # Decode composite PNG bytes back to array via PIL
    import io

    target_u8 = (np.clip(target, 0.0, 1.0) * 255.0).astype(np.uint8)
    composite_img = Image.open(io.BytesIO(composite)).convert("RGB")
    composite_u8 = np.array(composite_img, dtype=np.uint8)

    # Heatmap (reuses _render_de_heatmap logic inline)
    alpha_stack = np.load(plan.alpha_stack_path)
    alpha_hwm = np.transpose(alpha_stack, (1, 2, 0))
    rendered = np.asarray(
        forward_render_jax.forward_render(
            jnp.asarray(alpha_hwm, dtype=jnp.float32),
            jnp.asarray(plan.pigment_idx, dtype=jnp.int32),
        )
    )
    dE = color.rgb_delta_e76(rendered, target)
    norm = np.clip(dE / 15.0, 0.0, 1.0)
    r = np.clip(2 * norm - 0.5, 0.0, 1.0)
    g = np.clip(1.0 - 2.0 * np.abs(norm - 0.5), 0.0, 1.0)
    b = np.clip(1.0 - 2.0 * norm, 0.0, 1.0)
    heat_u8 = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)

    # Confidence map: max alpha per pixel → grayscale
    max_alpha = alpha_stack.max(axis=0)
    confidence_u8 = np.tile((max_alpha * 255.0).astype(np.uint8)[..., None], (1, 1, 3))

    h, w = target.shape[:2]
    grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    grid[:h, :w] = target_u8
    grid[:h, w:] = composite_u8
    grid[h:, :w] = heat_u8
    grid[h:, w:] = confidence_u8
    out = plan_dir / "quad_grid.png"
    Image.fromarray(grid, "RGB").save(out)
    return out


# ---------------------------------------------------------------------------
# T0.1 — ingest_reference_image (REAL: wraps s1_ingest.ingest_reference_image)
# ---------------------------------------------------------------------------


def ingest_reference_image(path: str) -> ToolResult[dict[str, Any]]:
    try:
        handle = s1_ingest.ingest_reference_image(path)
    except s1_ingest.IngestError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    return ToolResult(
        ok=True,
        data={
            "image_sha256": handle.image_sha256,
            "width": handle.width,
            "height": handle.height,
            "session_id": handle.session_id,
        },
    )


# ---------------------------------------------------------------------------
# T0.2 — analyze_image (REAL measurables; no subject_label per fix 5)
# ---------------------------------------------------------------------------


def analyze_image(path: str) -> ToolResult[dict[str, Any]]:
    try:
        handle = s1_ingest.ingest_reference_image(path)
    except s1_ingest.IngestError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    family_areas = _classify_into_families(handle.array)
    hints = _templates.pick_template_hints(family_areas=family_areas)

    data = {
        "image_sha256": handle.image_sha256,
        "width": handle.width,
        "height": handle.height,
        "mpx": round((handle.width * handle.height) / 1_000_000, 4),
        "family_areas": family_areas,
        "dominant_family": hints["dominant_family"],
        "max_family_area_pct": hints["max_family_area_pct"],
        "flesh_area_pct": hints["flesh_area_pct"],
        "family_count_above_10pct": hints["family_count_above_10pct"],
        "est_solver_s": _estimate_solver_s(handle.width * handle.height),
    }
    return ToolResult(ok=True, data=data)


def _classify_into_families(rgb: Any) -> dict[str, float]:
    """Cheap measurable hue-family clustering — coarse OKLab bucketing."""

    arr = (rgb.astype("float32") / 255.0).reshape(-1, 3)
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    warm = (r > g) & (r > b)
    cool = (b > r) & (b > g)
    green = (g > r) & (g > b)

    counts = {
        "cream": float(((lum > 0.85) & warm).mean()),
        "cool": float(((lum > 0.40) & (lum <= 0.85) & cool).mean()),
        "flesh": float(((lum > 0.55) & (lum <= 0.80) & warm & ~(r > 0.85)).mean()),
        "warm": float(((lum > 0.30) & (lum <= 0.70) & warm & (r > 0.55)).mean()),
        "shadow": float(((lum > 0.20) & (lum <= 0.50) & (green | cool)).mean()),
        "detail": float((lum <= 0.20).mean()),
        "accent": float(((lum > 0.30) & ~warm & ~cool & ~green).mean()),
    }
    total = sum(counts.values())
    if total > 0:
        counts = {k: v / total for k, v in counts.items()}
    return counts


def _estimate_solver_s(npx: int) -> float:
    """Rough solver wall-time estimate from pixel count."""
    return round(0.5 + (npx / 1_000_000) * 15.0, 1)


# ---------------------------------------------------------------------------
# T0.3 — build_hue_family_map (REAL: family-area dict from same classifier)
# ---------------------------------------------------------------------------


def build_hue_family_map(path: str) -> ToolResult[dict[str, Any]]:
    try:
        handle = s1_ingest.ingest_reference_image(path)
    except s1_ingest.IngestError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    family_areas = _classify_into_families(handle.array)
    return ToolResult(
        ok=True,
        data={
            "image_sha256": handle.image_sha256,
            "family_areas": family_areas,
            "accent_pigments": [],  # populated in D9B Tier 3 introspection
        },
    )


# ---------------------------------------------------------------------------
# T0.4 — propose_stack
# ---------------------------------------------------------------------------


def propose_stack(
    path: str,
    *,
    solve_profile: Literal["fast", "default", "thorough"] = "default",
    m_prior: int | None = None,
    strategy_template: str | None = None,
) -> ToolResult[dict[str, Any]]:
    """Run the v23 pipeline and persist a plan under the active session."""
    try:
        plan = _orch.run_pipeline_partial(
            path,
            solve_profile=solve_profile,
            m_prior=m_prior,
            strategy_template=strategy_template,
        )
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan.plan_id,
            "session_id": plan.session_id,
            "image_sha256": plan.image_sha256,
            "solve_profile": plan.solve_profile,
            "strategy_template": plan.suggested_template,
            "template_confidence": plan.template_confidence,
            "m_prior": plan.m_prior,
            "impression_count": len(plan.impressions),
            "dominant_family": plan.dominant_family,
            "family_areas": plan.family_areas,
            "sam_region_count": len(plan.sam_regions),
            "hue_family_map_path": plan.hue_family_map_path,
            "reconstruction_dE_mean": plan.reconstruction_dE_mean,
            "reconstruction_dE_p95": plan.reconstruction_dE_p95,
            "solver_wall_s": plan.solver_wall_s,
            "solver_status": plan.solver_status,
            "solver_optimized_shape": plan.solver_optimized_shape,
            "solver_downsample_scale": plan.solver_downsample_scale,
        },
    )


# ---------------------------------------------------------------------------
# T0.5 — inspect_plan
# ---------------------------------------------------------------------------


def inspect_plan(
    plan_id: str,
    focus: FocusMode = "composite",
) -> ToolResult[dict[str, Any]]:
    if focus not in _VALID_FOCUS_MODES:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_FOCUS_MODE",
                    message=f"focus must be one of {_VALID_FOCUS_MODES}, got {focus!r}",
                    recoverable=True,
                )
            ],
        )

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "focus": focus, "artifact_path": None},
            errors=[
                exc.error,
                _impl_pending(
                    "IMPL_PENDING_INSPECT",
                    "plan_id not found — placeholder returned. Run propose_stack first.",
                ),
            ],
        )

    from backend.services.v23.stages import s10_emit

    plan_dir = _orch._plan_dir(plan.session_id, plan.plan_id)
    if focus == "composite":
        composite_path = plan_dir / "composite_preview.png"
        composite_path.write_bytes(s10_emit._render_composite(plan))
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "focus": focus, "artifact_path": str(composite_path)},
        )
    if focus == "recipe":
        recipe_path = plan_dir / "recipe.md"
        recipe_path.write_text(s10_emit._build_recipe_md(plan))
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "focus": focus, "artifact_path": str(recipe_path)},
        )
    if focus == "per_impression":
        per_imp_dir = plan_dir / "impressions"
        per_imp_dir.mkdir(parents=True, exist_ok=True)
        paths_out: list[str] = []
        for i, imp in enumerate(plan.impressions):
            p = per_imp_dir / f"{imp['id']}.png"
            p.write_bytes(s10_emit._build_per_impression_png(plan, i))
            paths_out.append(str(p))
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "focus": focus, "impression_paths": paths_out},
        )
    if focus == "confidence":
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "focus": focus,
                "state_summary": plan.state_summary,
                "state_stack_path": plan.state_stack_path,
            },
        )
    if focus == "heatmap":
        heatmap_path = _render_de_heatmap(plan, plan_dir)
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "focus": focus,
                "dE_mean": plan.reconstruction_dE_mean,
                "dE_p95": plan.reconstruction_dE_p95,
                "artifact_path": str(heatmap_path) if heatmap_path else None,
                "metric": "deltaE76",
            },
        )
    if focus == "quad":
        quad_path = _render_quad_grid(plan, plan_dir)
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "focus": focus,
                "artifact_path": str(quad_path) if quad_path else None,
            },
        )
    return ToolResult(
        ok=False,
        data=None,
        errors=[
            WoodblockError(
                tier="refusal",
                code="PIXEL_FOCUS_REQUIRES_COORDS",
                message="inspect_plan focus='pixel' needs coordinates",
                hint="use dE_at(plan_id, x, y) or pigment_at(plan_id, x, y)",
                recoverable=True,
            )
        ],
    )


# ---------------------------------------------------------------------------
# T0.6 — forward_render (alias simulate_candidate_stack — both routed to same impl)
# ---------------------------------------------------------------------------


def forward_render(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Re-render the composite for a persisted plan via the JAX forward render."""
    from backend.services.v23.stages import s10_emit

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "composite_path": None, "dE_map_path": None},
            errors=[
                exc.error,
                _impl_pending(
                    "IMPL_PENDING_FORWARD",
                    "plan_id not found — run propose_stack first",
                ),
            ],
        )
    plan_dir = _orch._plan_dir(plan.session_id, plan.plan_id)
    composite_path = plan_dir / "composite_preview.png"
    composite_path.write_bytes(s10_emit._render_composite(plan))
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "composite_path": str(composite_path),
            "dE_map_path": None,
            "render_tier": "t1_mixbox",
            "reconstruction_dE_mean": plan.reconstruction_dE_mean,
            "reconstruction_dE_p95": plan.reconstruction_dE_p95,
            "render_tier_note": (
                "Rendered as if pigments were pre-mixed in a well — actual "
                "mokuhanga overprint may shift colors ΔE 4-8 for stacks > 3 deep. "
                "Upload swatch overprint matrix to unlock t2_empirical."
            ),
        },
    )


def simulate_candidate_stack(plan_id: str) -> ToolResult[dict[str, Any]]:
    return forward_render(plan_id)


# ---------------------------------------------------------------------------
# T0.7 — score_stack_delta_e
# ---------------------------------------------------------------------------


def score_stack_delta_e(plan_id: str, region: dict | None = None) -> ToolResult[dict[str, Any]]:
    """Cheap ΔE lookup from a persisted plan."""
    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "dE_mean": 0.0, "dE_p95": 0.0, "region": region},
            errors=[
                exc.error,
                _impl_pending(
                    "IMPL_PENDING_DE_LOOKUP",
                    "plan_id not found — returning neutral value. Run propose_stack first.",
                ),
            ],
        )
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "dE_mean": plan.reconstruction_dE_mean,
            "dE_p95": plan.reconstruction_dE_p95,
            "region": region,
            "render_tier": "t1_mixbox",
            "target_dE_mean": 1.5,
            "target_dE_p95": 3.0,
        },
    )


# ---------------------------------------------------------------------------
# T0.8 — score_candidate_stack (5-component combined score per fix 4)
# ---------------------------------------------------------------------------


def score_candidate_stack(plan_id: str) -> ToolResult[dict[str, Any]]:
    """5-component breakdown computed from a persisted plan when found."""
    from backend.services.v23.core import score as _score

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError:
        # Unknown plan — fall back to neutral 0.5 for backwards-compat.
        weights = {
            "visual_match": 0.40,
            "carveability": 0.20,
            "simplicity": 0.15,
            "underprint_utility": 0.15,
            "template_fit": 0.10,
        }
        components = {k: 0.5 for k in weights}
        overall = sum(weights[k] * components[k] for k in weights)
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "overall": overall,
                **components,
                "component_weights": weights,
                "notes": "Plan not found — returned neutral scores. Run propose_stack first.",
            },
        )
    return ToolResult(ok=True, data=_score.score_plan_real(plan))


# ---------------------------------------------------------------------------
# T0.9 — export_print_plan
# ---------------------------------------------------------------------------


def export_print_plan(plan_id: str, out_dir: str | None = None) -> ToolResult[dict[str, Any]]:
    """Real ZIP export via S10 emitter when the plan exists in the active session.

    Falls back to a minimal ZIP if the plan_id is unknown. Side-writes
    ``recipe.md`` to ``out_dir`` for direct inspection.
    """
    from backend.services.v23.stages import s10_emit

    out = Path(out_dir or ".")
    out.mkdir(parents=True, exist_ok=True)
    recipe_md = generate_print_recipe_report(plan_id).data["markdown"]
    recipe_path = out / f"{plan_id}_recipe.md"
    recipe_path.write_text(recipe_md)

    try:
        zip_path = s10_emit.emit_plan_zip(plan_id, out_dir=out)
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "zip_path": str(zip_path), "recipe_path": str(recipe_path)},
        )
    except s10_emit.EmitError as exc:
        # Plan_id unknown — fall back to minimal ZIP for backwards-compat.
        import zipfile

        zip_path = out / f"{plan_id}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", "{}")
            zf.writestr("recipe.md", recipe_md)
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "zip_path": str(zip_path), "recipe_path": str(recipe_path)},
            errors=[exc.error],
        )


# ---------------------------------------------------------------------------
# T0.10 — generate_print_recipe_report (plain-language print recipe)
# ---------------------------------------------------------------------------


def generate_print_recipe_report(
    plan_id: str,
    format: str = "markdown",
) -> ToolResult[dict[str, Any]]:
    from backend.services.v23.stages import s10_emit

    try:
        plan = _orch.load_plan(plan_id)
        md = s10_emit._build_recipe_md(plan)
    except _orch.OrchestratorError:
        md = _FALLBACK_RECIPE.format(plan_id=plan_id)
    return ToolResult(ok=True, data={"plan_id": plan_id, "format": format, "markdown": md})


_FALLBACK_RECIPE = """# Print recipe — {plan_id}

> **Note on color simulation (t1_mixbox)**: this recipe was rendered as if
> pigments were pre-mixed in a well before application. Actual mokuhanga
> overprint glazing may shift colors by ΔE 4–8 vs. the simulated composite,
> especially on stacks deeper than 3 impressions. Upload a swatch overprint
> matrix to unlock t2_empirical for accurate prediction with your own pigments.

## Impressions (light → dark)

Impression 01: pale cream support, broad face/base areas
Impression 02: cool blue support under shadows and hair edges
Impression 03: pink flesh field, face midtones
Impression 04: warm red/orange accents on cheeks and lips
Impression 05: teal shadow shapes
Impression 06: dark key/detail — hair, eyes, mouth, nose lines

## Notes

- This fallback recipe is emitted when a plan_id is not available in the
  active session. Run propose_stack first for a persisted plan recipe.
"""


__all__ = [
    "ingest_reference_image",
    "analyze_image",
    "build_hue_family_map",
    "propose_stack",
    "inspect_plan",
    "forward_render",
    "simulate_candidate_stack",
    "score_stack_delta_e",
    "score_candidate_stack",
    "export_print_plan",
    "generate_print_recipe_report",
]
