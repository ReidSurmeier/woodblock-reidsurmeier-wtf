"""Ring 2 fixtures — stdio MCP server subprocess + tiny JSON-RPC client."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class MCPStdioClient:
    """Minimal JSON-RPC framing over a subprocess' stdin/stdout.

    Lives here in the conftest so tests can import the shape without
    pulling in FastMCP machinery. Real impl matches MCP framing
    (LSP-style ``Content-Length:`` headers) once D19.1 lands.
    """

    proc: subprocess.Popen
    _id: int = 0

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        self._id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": method,
            "params": params or {},
        }
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        payload = json.dumps(req).encode("utf-8")
        self.proc.stdin.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        return self._read_response(timeout=timeout)

    def _read_response(self, timeout: float = 30.0) -> dict[str, Any]:
        assert self.proc.stdout is not None
        headers: dict[str, str] = {}
        while True:
            line = self.proc.stdout.readline()
            if line in (b"", b"\r\n", b"\n"):
                break
            key, _, value = line.decode("ascii").partition(":")
            headers[key.lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length == 0:
            return {}
        body = self.proc.stdout.read(length)
        return json.loads(body.decode("utf-8"))

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


@pytest.fixture
def mcp_stdio_client() -> Any:
    """Start the v23 server as a subprocess and yield a stdio client.

    Launches the local Python module directly so the transport ring tests
    validate the same console entry point as ``woodblock-mcp``.
    """
    cmd = [sys.executable, "-m", "backend.mcp.v23_server"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )
    client = MCPStdioClient(proc=proc)
    try:
        yield client
    finally:
        client.close()
