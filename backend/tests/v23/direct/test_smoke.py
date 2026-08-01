"""Ring 1 smoke — the golden-path MCP tools are importable.

Golden-path tool surface:
  ingest_reference_image, analyze_image, build_hue_family_map,
  propose_stack, inspect_plan, alternative_stacks, compare_plans,
  simplify_masks_for_carving, score_candidate_stack,
  generate_print_recipe_report, export_print_plan.
"""

from __future__ import annotations

import importlib

DAY1_TOOLS: tuple[str, ...] = (
    "ingest_reference_image",
    "analyze_image",
    "build_hue_family_map",
    "propose_stack",
    "inspect_plan",
    "alternative_stacks",
    "compare_plans",
    "simplify_masks_for_carving",
    "score_candidate_stack",
    "generate_print_recipe_report",
    "export_print_plan",
)

TOOL_MODULES: tuple[str, ...] = (
    "backend.mcp.tools.core",
    "backend.mcp.tools.hitl",
    "backend.mcp.tools.carve",
)


def test_day1_tools_importable() -> None:
    found: set[str] = set()
    for mod_name in TOOL_MODULES:
        mod = importlib.import_module(mod_name)
        for name in DAY1_TOOLS:
            if hasattr(mod, name):
                found.add(name)
    missing = set(DAY1_TOOLS) - found
    assert not missing, f"day-1 tools not importable: {sorted(missing)}"
