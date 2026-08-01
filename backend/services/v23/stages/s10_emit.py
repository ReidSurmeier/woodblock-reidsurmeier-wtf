"""D11.e — S10 ZIP emitter.

Bundles a persisted ``PartialPlan`` + forward-rendered composite preview
+ per-impression PNGs + manifest.json (schema v23.0) + recipe.md into a
single ZIP suitable for export to cnc.reidsurmeier.wtf or the artist's
bench.

Per addendum-v3 fix 6 + addendum-v4 WB-LANG-02:
- Output framing locked to "plausible underprint candidates"
- Designed-by-rules framing only; never claim physical-evidence inference
- t1_mixbox recipe carries the "as if pre-mixed" qualifier
- Banned-term grep (WB-LANG-01) enforced in tests, not just docs
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from PIL import Image

from backend.mcp.errors import WoodblockError
from backend.services.v23 import orchestrator as _orch
from backend.services.v23.core import forward_render_jax


class EmitError(Exception):
    """Raised when S10 cannot materialise the ZIP."""

    def __init__(self, error: WoodblockError) -> None:
        super().__init__(error.message)
        self.error = error


def _render_composite(plan: _orch.PartialPlan) -> bytes:
    """Render composite preview from persisted alpha_stack (per-pixel accurate)."""
    if not plan.impressions:
        # No solver result — emit a blank washi-coloured preview
        arr = np.full((plan.height, plan.width, 3), [246, 241, 227], dtype=np.uint8)
        return _np_to_png_bytes(arr)

    # Prefer per-pixel alpha_stack persisted by S5; fall back to mean-α if absent.
    if plan.alpha_stack_path and Path(plan.alpha_stack_path).is_file():
        alpha_stack = np.load(plan.alpha_stack_path)  # (M, H, W)
        # forward_render_jax expects (H, W, M)
        alpha_hwm = np.transpose(alpha_stack, (1, 2, 0))
        pigment_idx = np.array(plan.pigment_idx, dtype=np.int32)
    else:
        m = len(plan.impressions)
        alpha_hwm = np.zeros((plan.height, plan.width, m), dtype=np.float32)
        for i, imp in enumerate(plan.impressions):
            alpha_hwm[..., i] = float(imp.get("mean_alpha", 0.0))
        pigment_idx = np.array([imp["pigment_id"] for imp in plan.impressions], dtype=np.int32)

    rgb = np.asarray(
        forward_render_jax.forward_render(
            jnp.asarray(alpha_hwm, dtype=jnp.float32),
            jnp.asarray(pigment_idx, dtype=jnp.int32),
        )
    )
    arr = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _np_to_png_bytes(arr)


def _np_to_png_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _build_per_impression_png(
    plan: _orch.PartialPlan,
    impression_idx: int,
) -> bytes:
    """Per-impression preview = solid pigment color × mean_alpha over washi."""
    imp = plan.impressions[impression_idx]
    pigment_rgb_255 = forward_render_jax.PIGMENT_RGB_255[imp["pigment_id"]]
    alpha = float(imp.get("mean_alpha", 0.5))
    paper = np.array([246, 241, 227], dtype=np.float32)
    fill = (1.0 - alpha) * paper + alpha * pigment_rgb_255.astype(np.float32)
    arr = np.tile(fill.astype(np.uint8), (plan.height, plan.width, 1))
    return _np_to_png_bytes(arr)


def _build_manifest(plan: _orch.PartialPlan) -> dict[str, Any]:
    """Manifest schema v23.0."""
    return {
        "schema_version": "v23.0",
        "plan_id": plan.plan_id,
        "session_id": plan.session_id,
        "target_image_sha256": plan.image_sha256,
        "width": plan.width,
        "height": plan.height,
        "solve_profile": plan.solve_profile,
        "strategy_template": plan.suggested_template,
        "calibration": {
            "source": "generic_mixbox_13",
            "fitted_at": None,
            "paper_substrate": "washi_default",
        },
        "render_tier": "t1_mixbox",
        "render_tier_note": (
            "Rendered as if pigments were pre-mixed in a well. Actual mokuhanga "
            "overprint glazing may shift colors ΔE 4-8 for stacks > 3 deep. "
            "Upload swatch overprint matrix to unlock t2_empirical."
        ),
        "impressions": plan.impressions,
        "blocks": [
            {
                "block_id": f"blk_{i:02d}",
                "face_ids": [f"blk_{i:02d}::face_a"],
                "material": "maple_plywood",
                "impression_ids": [
                    imp_id for imp_id, b in plan.impression_to_block.items() if b == i
                ],
            }
            for i in range(plan.block_count)
        ],
        "pull_groups": plan.pull_groups,
        "state_summary": plan.state_summary,
        "reconstruction": {
            "dE_mean": plan.reconstruction_dE_mean,
            "dE_p95": plan.reconstruction_dE_p95,
        },
        "solver": {
            "status": plan.solver_status,
            "wall_s": plan.solver_wall_s,
        },
        "posture": (
            "These are plausible underprint candidates that reduce reconstruction "
            "error under this pigment/printing model. The candidates were designed "
            "by printmaking rules, not inferred from physical evidence."
        ),
        "created_at": plan.created_at,
    }


def _build_recipe_md(plan: _orch.PartialPlan) -> str:
    lines = [
        f"# Print recipe — {plan.plan_id}",
        "",
        "> **Note on color simulation (t1_mixbox)**: this recipe was rendered as if",
        "> pigments were pre-mixed in a well before application. Actual mokuhanga",
        "> overprint glazing may shift colors by ΔE 4-8 vs. the simulated composite,",
        "> especially on stacks deeper than 3 impressions. Upload a swatch overprint",
        "> matrix to unlock t2_empirical for accurate prediction with your own pigments.",
        "",
        "## Impressions (light → dark)",
        "",
    ]
    pigment_names = [
        "cadmium_yellow",
        "hansa_yellow",
        "cadmium_orange",
        "cadmium_red",
        "quinacridone_magenta",
        "cobalt_violet",
        "ultramarine_blue",
        "cobalt_blue",
        "viridian_green",
        "forest_green",
        "burnt_sienna",
        "raw_umber",
        "ivory_black",
    ]
    for imp in plan.impressions:
        pid = imp["pigment_id"]
        name = pigment_names[pid] if 0 <= pid < len(pigment_names) else f"pigment_{pid}"
        order = imp["order_step"]
        coverage = imp.get("coverage_pct", 0.0)
        lines.append(
            f"Impression {order:02d}: {name.replace('_', ' ')}, "
            f"{coverage:.1f}% of sheet area, alpha {imp.get('mean_alpha', 0.0):.2f}"
        )
    if not plan.impressions:
        lines.append("(no impressions — solver did not run)")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- v23-MCP day-1 ship: plan_id is real and persistent under the session.",
            "- Confidence labels: per-region ambiguous / inferred / visible (see manifest).",
            "- Carve order: per block, ascending order_step. Kento marks added at CNC export.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_readme_txt(plan: _orch.PartialPlan) -> str:
    return (
        f"v23-MCP print plan — {plan.plan_id}\n"
        f"Schema: v23.0\n"
        f"Target image SHA-256: {plan.image_sha256}\n"
        f"Resolution: {plan.width}x{plan.height}\n"
        f"Solver: {plan.solver_status} ({plan.solver_wall_s:.2f}s)\n"
        f"Impressions: {len(plan.impressions)}\n"
        f"Blocks: {plan.block_count}\n"
        f"Render tier: t1_mixbox (Mixbox-stack lerp; see manifest.render_tier_note)\n"
        f"\n"
        f"Open manifest.json + recipe.md for full details.\n"
    )


def emit_plan_zip(plan_id: str, *, out_dir: Path | str = ".") -> Path:
    """Build the v23.0 export ZIP for ``plan_id``. Returns the ZIP path."""
    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        raise EmitError(exc.error) from exc

    out_path = Path(out_dir) / f"{plan_id}.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    composite = _render_composite(plan)
    manifest = _build_manifest(plan)
    recipe = _build_recipe_md(plan)
    readme = _build_readme_txt(plan)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        zf.writestr("recipe.md", recipe)
        zf.writestr("README.txt", readme)
        zf.writestr("composite_preview.png", composite)
        for i, imp in enumerate(plan.impressions):
            png = _build_per_impression_png(plan, i)
            zf.writestr(f"impressions/{imp['id']}.png", png)
    return out_path


__all__ = ["EmitError", "emit_plan_zip"]
