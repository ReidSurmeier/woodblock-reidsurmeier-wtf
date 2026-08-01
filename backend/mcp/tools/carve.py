"""D9B/D14.n — Tier 5 carve handoff tools (3 tools, REAL).

export_svg: per-impression carveable SVG via skimage marching-squares contours
on each alpha mask. Emits SVG path strings under plan_dir/svg/.

export_block_svgs: aggregates impressions per physical block (uses plan
impression_to_block mapping) into one SVG per block, with kento mark stubs
+ registration halos as placeholder rects pending S9.b CNC layout.

generate_carve_order: builds CarveOrderManifest from plan.impressions +
impression_to_block + impression_to_face. Walking ShopBot tool recommendations
based on impression coverage + carve area.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.mcp.errors import ToolResult, WoodblockError


def _alpha_to_svg_paths(alpha: np.ndarray, threshold: float = 0.3) -> list[str]:
    """Marching-squares contours of alpha > threshold -> SVG path strings."""
    from skimage import measure  # local: skimage is a heavy import

    contours = measure.find_contours(alpha, level=threshold)
    paths: list[str] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        # contour is (row, col) = (y, x). SVG wants (x, y).
        coords = [f"{c[1]:.2f},{c[0]:.2f}" for c in contour]
        d = "M " + " L ".join(coords) + " Z"
        paths.append(d)
    return paths


def _wrap_svg(paths: list[str], width: int, height: int, fill: str = "black") -> str:
    body = "\n".join(
        f'  <path d="{d}" fill="{fill}" fill-rule="evenodd" stroke="none"/>' for d in paths
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n'
        f"{body}\n"
        f"</svg>\n"
    )


def export_svg(
    plan_id: str,
    impression_ids: list[str] | None = None,
) -> ToolResult[dict[str, Any]]:
    """Per-impression carveable SVG via marching-squares contours."""
    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_SOLVER_OUTPUT",
                    message="plan has no alpha_stack — solver did not run",
                    recoverable=True,
                ),
            ],
        )

    alpha_stack = np.load(plan.alpha_stack_path)  # (M, H, W)
    id_to_idx = {imp["id"]: i for i, imp in enumerate(plan.impressions)}
    requested = impression_ids if impression_ids else list(id_to_idx.keys())

    unknown = [iid for iid in requested if iid not in id_to_idx]
    if unknown:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="UNKNOWN_IMPRESSION_ID",
                    message=f"unknown impression_ids: {unknown}",
                    recoverable=True,
                ),
            ],
        )

    out_dir = Path(plan.alpha_stack_path).parent / "svg"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_out: list[dict[str, Any]] = []
    for iid in requested:
        idx = id_to_idx[iid]
        alpha = alpha_stack[idx]
        path_strs = _alpha_to_svg_paths(alpha, threshold=0.3)
        svg_text = _wrap_svg(path_strs, plan.width, plan.height, fill="black")
        out_path = out_dir / f"{iid}.svg"
        out_path.write_text(svg_text)
        paths_out.append(
            {
                "impression_id": iid,
                "svg_path": str(out_path),
                "path_count": len(path_strs),
            }
        )

    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "impression_ids": requested,
            "svg_paths": paths_out,
            "vectoriser": "skimage_marching_squares_v1",
            "alpha_threshold": 0.3,
        },
    )


def export_block_svgs(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Per-physical-block aggregated SVG: all impressions on that block stacked."""
    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_SOLVER_OUTPUT",
                    message="plan has no alpha_stack — solver did not run",
                    recoverable=True,
                ),
            ],
        )
    if plan.block_count == 0:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_BLOCKS_PACKED",
                    message="plan has block_count=0 — S7 did not run",
                    recoverable=True,
                ),
            ],
        )

    alpha_stack = np.load(plan.alpha_stack_path)
    id_to_idx = {imp["id"]: i for i, imp in enumerate(plan.impressions)}
    impressions_by_block: dict[int, list[str]] = {}
    for iid, block_idx in plan.impression_to_block.items():
        impressions_by_block.setdefault(int(block_idx), []).append(iid)

    out_dir = Path(plan.alpha_stack_path).parent / "svg" / "blocks"
    out_dir.mkdir(parents=True, exist_ok=True)
    block_svg_paths: dict[str, str] = {}
    for block_idx, iids in sorted(impressions_by_block.items()):
        # Union the alpha masks at threshold
        union = np.zeros_like(alpha_stack[0])
        for iid in iids:
            union = np.maximum(union, alpha_stack[id_to_idx[iid]])
        path_strs = _alpha_to_svg_paths(union, threshold=0.3)
        svg_text = _wrap_svg(path_strs, plan.width, plan.height, fill="black")
        out_path = out_dir / f"blk_{block_idx:02d}.svg"
        out_path.write_text(svg_text)
        block_svg_paths[f"blk_{block_idx:02d}"] = str(out_path)

    # Stub kento mark coords (placeholder until S9.b CNC layout)
    kento_marks = [
        {"id": "kento_top_left", "x": 8, "y": 8, "size_px": 6},
        {"id": "kento_top_right", "x": plan.width - 8, "y": 8, "size_px": 6},
        {"id": "kento_bottom_left", "x": 8, "y": plan.height - 8, "size_px": 6},
    ]
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "block_count": plan.block_count,
            "block_svg_paths": block_svg_paths,
            "kento_marks": kento_marks,
            "vectoriser": "skimage_marching_squares_v1",
        },
    )


def generate_carve_order(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Build CarveOrderManifest from plan.impression_to_block + face mapping."""
    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if not plan.impressions:
        return ToolResult(
            ok=False,
            data=None,
            errors=[
                WoodblockError(
                    tier="refusal",
                    code="NO_IMPRESSIONS",
                    message="plan has no impressions to carve",
                    recoverable=True,
                ),
            ],
        )

    steps = []
    total_area_px = 0
    for i, imp in enumerate(plan.impressions):
        iid = imp["id"]
        block = plan.impression_to_block.get(iid, 0)
        face = plan.impression_to_face.get(iid, "blk_00::face_a")
        coverage = float(imp.get("coverage_pct", 0.0))
        area_px = int(plan.width * plan.height * coverage / 100.0)
        total_area_px += area_px
        # Tool recommendation: large bevel for >5% coverage, fine V-bit for detail
        if coverage > 5.0:
            tool = "1/4_bevel_60deg"
        elif coverage > 1.0:
            tool = "1/8_v_bit_30deg"
        else:
            tool = "1/16_v_bit_15deg"
        steps.append(
            {
                "step_id": f"step_{i:02d}",
                "order_step": imp.get("order_step", i),
                "impression_id": iid,
                "block_id": f"blk_{block:02d}",
                "face_id": face,
                "pigment_id": imp.get("pigment_id"),
                "coverage_pct": coverage,
                "area_px": area_px,
                "tool_recommendation": tool,
            }
        )

    # Wall-time estimate: 1 hour per 100k carved px at the medium tool rate
    estimated_hours = round(total_area_px / 100_000, 2)
    distinct_tools = sorted({s["tool_recommendation"] for s in steps})
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "steps": steps,
            "step_count": len(steps),
            "estimated_hours": estimated_hours,
            "shopbot_nodes_total": len(steps),
            "tool_recommendations": distinct_tools,
            "total_carve_area_px": total_area_px,
            "block_count": plan.block_count,
        },
    )


__all__ = ["export_svg", "export_block_svgs", "generate_carve_order"]
