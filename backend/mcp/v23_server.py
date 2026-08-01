"""woodblock_stack v23 MCP stdio server.

The production target is FastMCP, but the local venv does not currently
include that optional dependency. This module implements the small MCP
JSON-RPC surface needed for validation and execution: initialize,
tools/list, and tools/call over Content-Length framed stdio.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from backend.mcp.registry import TOOLS, call_mcp_tool, list_mcp_tools, tool_result_to_jsonable

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "woodblock_stack"
SERVER_VERSION = "0.0.1"


@dataclass(frozen=True)
class _StdioMCPApp:
    name: str = SERVER_NAME

    @property
    def tool_count(self) -> int:
        return len(TOOLS)


app = _StdioMCPApp()


def main() -> None:
    """Run the MCP server on stdin/stdout."""
    run_stdio_loop()


def run_stdio_loop() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        try:
            request = _read_message(stdin)
        except json.JSONDecodeError as exc:
            _write_message(stdout, _jsonrpc_error(None, -32700, "parse error", str(exc)))
            continue
        if request is None:
            return
        try:
            response = _handle_request(request)
        except SystemExit:
            return
        except Exception as exc:
            response = _jsonrpc_error(request.get("id"), -32603, "internal error", str(exc))
        if response is not None:
            _write_message(stdout, response)


def _handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if request_id is None:
        if method == "notifications/initialized":
            return None
        if method == "exit":
            raise SystemExit
        return None

    if method == "initialize":
        return _jsonrpc_result(
            request_id,
            {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if method == "tools/list":
        return _jsonrpc_result(request_id, {"tools": list_mcp_tools()})
    if method == "tools/call":
        return _jsonrpc_result(request_id, _call_tool_result(params))
    if method == "ping":
        return _jsonrpc_result(request_id, {})
    if method == "resources/list":
        return _jsonrpc_result(request_id, {"resources": []})
    if method == "prompts/list":
        return _jsonrpc_result(request_id, {"prompts": []})
    if method == "shutdown":
        return _jsonrpc_result(request_id, None)
    return _jsonrpc_error(request_id, -32601, f"method not found: {method}")


def _call_tool_result(params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(name, str):
        payload = tool_result_to_jsonable(call_mcp_tool("", {}))
        payload["errors"][0]["message"] = "tools/call requires string param 'name'"
    elif not isinstance(arguments, dict):
        payload = tool_result_to_jsonable(call_mcp_tool(name, {}))
        payload["ok"] = False
        payload["data"] = None
        payload["errors"] = [
            {
                "tier": "refusal",
                "code": "INVALID_TOOL_ARGUMENTS",
                "message": "tools/call param 'arguments' must be an object",
                "hint": "pass arguments as a JSON object",
                "recoverable": True,
                "retry_with": None,
                "context": {},
            }
        ]
    else:
        payload = tool_result_to_jsonable(call_mcp_tool(name, arguments))
    return {
        "content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
        "structuredContent": payload,
        "isError": not bool(payload.get("ok")),
    }


def _read_message(stream: Any) -> dict[str, Any] | None:
    first = _read_nonempty_line(stream)
    if first is None:
        return None
    if first.lower().startswith(b"content-length:"):
        length = _content_length(first)
        while True:
            header = stream.readline()
            if header in (b"\r\n", b"\n", b""):
                break
            if header.lower().startswith(b"content-length:"):
                length = _content_length(header)
        body = stream.read(length)
        return json.loads(body.decode("utf-8"))
    return json.loads(first.decode("utf-8"))


def _read_nonempty_line(stream: Any) -> bytes | None:
    while True:
        line = stream.readline()
        if line == b"":
            return None
        if line.strip():
            return line


def _content_length(header: bytes) -> int:
    try:
        return int(header.split(b":", 1)[1].strip())
    except Exception as exc:
        raise json.JSONDecodeError(f"invalid Content-Length header: {header!r}", "", 0) from exc


def _write_message(stream: Any, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    stream.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
    stream.write(payload)
    stream.flush()


def _jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(
    request_id: Any,
    code: int,
    message: str,
    data: Any | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


if __name__ == "__main__":
    main()
