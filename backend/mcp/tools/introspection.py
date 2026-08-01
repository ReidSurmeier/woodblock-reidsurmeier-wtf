"""D9B — Tier 3 introspection tools (6 tools, all REAL read-only)."""

from __future__ import annotations

from typing import Any

from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23.core import forward_render_jax


def get_pigments() -> ToolResult[dict[str, Any]]:
    """Return the 13-pigment Mixbox-anchored catalog."""
    pigments = []
    names = [
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
    for idx, name in enumerate(names):
        r, g, b = forward_render_jax.PIGMENT_RGB_255[idx].tolist()
        pigments.append(
            {
                "pigment_id": name,
                "name": name.replace("_", " ").title(),
                "rgb": [int(r), int(g), int(b)],
                "hex": f"#{r:02x}{g:02x}{b:02x}",
            }
        )
    return ToolResult(
        ok=True, data={"pigments": pigments, "count": len(pigments), "catalog": "generic_mixbox_13"}
    )


def get_emma_priors() -> ToolResult[dict[str, Any]]:
    """Return Emma-derived priors (hue families, accent rule, keyblock rule)."""
    return ToolResult(
        ok=True,
        data={
            "hue_families": ["cream", "cool", "flesh", "warm", "shadow", "detail", "accent"],
            "default_M_prior": 6,
            "M_prior_range": [4, 12],
            "accent_block_rule": "is_accent_block if pigment_count > 4 OR family_count > 2",
            "underprint_area_ratio_range": [1.5, 4.0],
            "keyblock_rule": "darkest detail impression gets highest order_step",
            "pigments_per_block_target": {"mean": 3.0, "cap": 5, "floor": 1},
        },
    )


def get_defaults() -> ToolResult[dict[str, Any]]:
    """Return locked technical defaults from research-v23-mcp-defaults.md."""
    return ToolResult(
        ok=True,
        data={
            "solve_profile_walltime_s": {"fast": 60, "default": 180, "thorough": 600},
            "dE2000_target": {"mean": 1.5, "p95": 3.0},
            "ambiguous_band": {"pixel_dE_gt": 3.0, "mask_mean_gt": 1.5},
            "block_count_target": 6,
            "block_count_cap": 10,
            "min_island_px_at_300dpi": 60,
            "kento_dilation_px_at_300dpi": 3,
            "schema_version": "v23.0",
            "calibration_default": "generic_mixbox_13",
            "render_tier_default": "t1_mixbox",
        },
    )


def solver_telemetry(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Read solver run summary from a persisted plan."""
    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "pyramid_levels_completed": 0,
                "iters_per_level": [],
                "loss_per_level": [],
                "rule_loss_breakdown": {},
                "exit_reason": "unknown_plan",
                "wall_time_s": 0.0,
                "divergence_flags": [],
                "optimized_shape": [],
                "downsample_scale": 1.0,
            },
            errors=[exc.error],
        )
    profile_iters = {"fast": 60, "default": 180, "thorough": 400}
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "pyramid_levels_completed": 1,  # single-level for day-1
            "iters_per_level": [profile_iters.get(plan.solve_profile, 180)],
            "loss_per_level": [],  # full per-iter trace lands at D14.b
            "rule_loss_breakdown": {},
            "exit_reason": plan.solver_status,
            "wall_time_s": plan.solver_wall_s,
            "divergence_flags": [],
            "solve_profile": plan.solve_profile,
            "optimized_shape": plan.solver_optimized_shape,
            "downsample_scale": plan.solver_downsample_scale,
            "impression_count": len(plan.impressions),
            "block_count": plan.block_count,
        },
    )


_PIGMENT_NAMES = [
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


def _delta_e76(rgb_a: tuple[float, float, float], rgb_b: tuple[float, float, float]) -> float:
    """Quick ΔE76 in CIE Lab. Cheap + monotonic with perceptual difference."""

    def _srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
        r, g, b = (max(0.0, min(1.0, c)) for c in rgb)

        def _linearise(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r_l, g_l, b_l = _linearise(r), _linearise(g), _linearise(b)
        # sRGB -> XYZ (D65)
        x = 0.4124564 * r_l + 0.3575761 * g_l + 0.1804375 * b_l
        y = 0.2126729 * r_l + 0.7151522 * g_l + 0.0721750 * b_l
        z = 0.0193339 * r_l + 0.1191920 * g_l + 0.9503041 * b_l
        # XYZ -> Lab (D65 ref white)
        xn, yn, zn = 0.95047, 1.0, 1.08883

        def _f(t: float) -> float:
            return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)

        fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b_lab = 200.0 * (fy - fz)
        return L, a, b_lab

    la, aa, ba = _srgb_to_lab(rgb_a)
    lb, ab, bb = _srgb_to_lab(rgb_b)
    return float(((la - lb) ** 2 + (aa - ab) ** 2 + (ba - bb) ** 2) ** 0.5)


def dE_at(plan_id: str, x: int, y: int) -> ToolResult[dict[str, Any]]:
    """Per-pixel ΔE — render single pixel from persisted alpha_stack vs target."""
    if x < 0 or y < 0:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_COORDS",
                    message=f"x and y must be >= 0, got ({x}, {y})",
                    recoverable=True,
                ),
            ],
        )

    from pathlib import Path

    import numpy as _np

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if y >= plan.height or x >= plan.width:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="OUT_OF_BOUNDS",
                    message=f"({x}, {y}) outside plan bounds {plan.width}x{plan.height}",
                    recoverable=True,
                ),
            ],
        )

    if not plan.alpha_stack_path or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "x": x,
                "y": y,
                "dE": None,
                "target_rgb": None,
                "rendered_rgb": None,
            },
            errors=[
                WoodblockError(
                    tier="degraded",
                    code="NO_SOLVER_OUTPUT",
                    message="plan has no persisted alpha_stack — solver did not run",
                    hint="re-run propose_stack with solver enabled",
                    recoverable=True,
                )
            ],
        )

    target_path = Path(plan.alpha_stack_path).parent / "target.npy"
    if not target_path.is_file():
        return ToolResult(
            ok=True,
            data={
                "plan_id": plan_id,
                "x": x,
                "y": y,
                "dE": None,
                "target_rgb": None,
                "rendered_rgb": None,
            },
            errors=[
                WoodblockError(
                    tier="degraded",
                    code="NO_TARGET_CACHE",
                    message="plan has no persisted target image — older plan?",
                    recoverable=True,
                )
            ],
        )

    alpha_stack = _np.load(plan.alpha_stack_path)  # (M, H, W)
    target = _np.load(target_path)  # (H, W, 3)

    alpha_pixel = alpha_stack[:, y, x]  # (M,)
    pigment_idx = _np.asarray(plan.pigment_idx, dtype=_np.int32)

    import jax.numpy as jnp

    alpha_hwm = jnp.asarray(alpha_pixel[None, None, :], dtype=jnp.float32)  # (1, 1, M)
    rendered = forward_render_jax.forward_render(
        alpha_hwm,
        jnp.asarray(pigment_idx, dtype=jnp.int32),
    )
    rendered_rgb = tuple(float(c) for c in _np.asarray(rendered[0, 0]))
    target_rgb = tuple(float(c) for c in target[y, x])
    dE = _delta_e76(target_rgb, rendered_rgb)

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "x": x,
            "y": y,
            "dE": round(dE, 3),
            "target_rgb": [round(c, 4) for c in target_rgb],
            "rendered_rgb": [round(c, 4) for c in rendered_rgb],
            "metric": "deltaE76",
        },
    )


def pigment_at(plan_id: str, x: int, y: int) -> ToolResult[dict[str, Any]]:
    """Per-pixel impression stack — load alpha_stack at pixel and emit per-impression rows."""
    if x < 0 or y < 0:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="INVALID_COORDS",
                    message=f"x and y must be >= 0, got ({x}, {y})",
                    recoverable=True,
                ),
            ],
        )

    from pathlib import Path

    import numpy as _np

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if y >= plan.height or x >= plan.width:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="OUT_OF_BOUNDS",
                    message=f"({x}, {y}) outside plan bounds {plan.width}x{plan.height}",
                    recoverable=True,
                ),
            ],
        )

    if not plan.alpha_stack_path or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "x": x, "y": y, "impressions": []},
            errors=[
                WoodblockError(
                    tier="degraded",
                    code="NO_SOLVER_OUTPUT",
                    message="plan has no persisted alpha_stack — solver did not run",
                    recoverable=True,
                )
            ],
        )

    alpha_stack = _np.load(plan.alpha_stack_path)  # (M, H, W)
    alpha_pixel = alpha_stack[:, y, x]  # (M,)
    impressions_meta = plan.impressions

    rows = []
    for i, alpha in enumerate(alpha_pixel):
        a = float(alpha)
        if a < 0.01:
            continue  # paper-clear — skip noise
        pid = int(plan.pigment_idx[i]) if i < len(plan.pigment_idx) else None
        meta = impressions_meta[i] if i < len(impressions_meta) else {}
        rows.append(
            {
                "order_step": int(meta.get("order_step", i)),
                "impression_id": meta.get("id", f"imp_{i:02d}"),
                "pigment_id": pid,
                "pigment_name": (
                    _PIGMENT_NAMES[pid]
                    if pid is not None and 0 <= pid < len(_PIGMENT_NAMES)
                    else f"pigment_{pid}"
                ),
                "alpha": round(a, 4),
            }
        )
    rows.sort(key=lambda r: r["order_step"])

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "x": x,
            "y": y,
            "impressions": rows,
            "alpha_threshold": 0.01,
        },
    )


__all__ = [
    "get_pigments",
    "get_emma_priors",
    "get_defaults",
    "solver_telemetry",
    "dE_at",
    "pigment_at",
]
