"""Ring 2 — JSON-RPC ``tools/list`` / ``tools/call`` over stdio."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image


def test_server_responds_to_initialize(mcp_stdio_client) -> None:
    response = mcp_stdio_client.call(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        },
    )
    assert response["jsonrpc"] == "2.0"
    assert response["result"]["serverInfo"]["name"] == "woodblock_stack"
    assert "tools" in response["result"]["capabilities"]


def test_server_responds_to_tools_list(mcp_stdio_client) -> None:
    response = mcp_stdio_client.call("tools/list")
    assert isinstance(response, dict) and response, "no JSON-RPC response received"
    assert response.get("jsonrpc") == "2.0"
    assert "result" in response, response
    tools = response["result"].get("tools", [])
    names = {t.get("name") for t in tools}
    expected_subset = {"analyze_image", "propose_stack", "export_print_plan", "export_svg"}
    assert expected_subset <= names, f"missing tools: {expected_subset - names}"
    assert all("inputSchema" in t for t in tools)


def test_server_executes_tool_call(mcp_stdio_client) -> None:
    response = mcp_stdio_client.call(
        "tools/call",
        {
            "name": "get_defaults",
            "arguments": {},
        },
    )
    assert response["jsonrpc"] == "2.0"
    result = response["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is True
    assert "solve_profile_walltime_s" in payload["data"]


def test_server_reports_unknown_tool(mcp_stdio_client) -> None:
    response = mcp_stdio_client.call(
        "tools/call",
        {
            "name": "does_not_exist",
            "arguments": {},
        },
    )
    result = response["result"]
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["errors"][0]["code"] == "UNKNOWN_TOOL"


def test_server_executes_propose_stack_image_path_alias(tmp_path, mcp_stdio_client) -> None:
    img_path = tmp_path / "alias.png"
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :4] = (220, 180, 120)
    arr[:, 4:] = (40, 80, 120)
    Image.fromarray(arr, "RGB").save(img_path)

    response = mcp_stdio_client.call(
        "tools/call",
        {
            "name": "propose_stack",
            "arguments": {"image_path": str(img_path), "solve_profile": "fast"},
        },
    )
    result = response["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is True
    assert payload["data"]["plan_id"].startswith("plan_")
