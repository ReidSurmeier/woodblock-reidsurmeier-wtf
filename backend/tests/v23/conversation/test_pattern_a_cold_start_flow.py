"""Ring 3 — Pattern-A cold-start flow.

Pattern A (research-v23-mcp-testing.md §4): the user hands Opus an
image and asks for a print plan. Canonical tool sequence:

    analyze_image → propose_stack → inspect_plan → export_print_plan
"""

from __future__ import annotations

import importlib
from pathlib import Path

EXPECTED_FLOW: tuple[str, ...] = (
    "analyze_image",
    "propose_stack",
    "inspect_plan",
    "export_print_plan",
)


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WOODBLOCK_HOME", str(tmp_path))
    from backend.mcp import paths

    importlib.reload(paths)
    from backend.services.v23 import session as _sess

    importlib.reload(_sess)
    from backend.services.v23 import orchestrator as _orch

    importlib.reload(_orch)


def test_pattern_a_cold_start_flow(mock_opus, monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    image_path = (
        "/home/reidsurmeier/src/woodblock-reidsurmeier-wtf/corpus/reid_untitled_01/original.png"
    )

    analyze = mock_opus.step("analyze_image", {"path": image_path})
    assert analyze.ok is True, analyze.errors
    proposed = mock_opus.step(
        "propose_stack",
        {"image_path": image_path, "solve_profile": "fast"},
    )
    assert proposed.ok is True, proposed.errors
    plan_id = proposed.data["plan_id"]
    mock_opus.step("inspect_plan", {"plan_id": plan_id, "focus": "heatmap"})
    exported = mock_opus.step("export_print_plan", {"plan_id": plan_id, "out_dir": str(tmp_path)})
    assert exported.ok is True, exported.errors

    actual = tuple(name for name, _args, _r in mock_opus.transcript)
    assert actual == EXPECTED_FLOW, f"flow drifted: {actual}"
