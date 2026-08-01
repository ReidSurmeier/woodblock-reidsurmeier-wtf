# 0003 — MCP transport: stdio over SSH (synthesizer reconciliation)

Status: superseded (2026-07-31 by ADR-0006)
Authority: `research-v23-mcp-plan-v2.1.md` §11 + `research-v23-mcp-protocol.md` §2

This ADR preserves the May 2026 day-one decision and its reasoning. It is no
longer the current operator path. ADR-0006 moves execution into the
Orca-managed Pugnet workspace and removes SSH registration from the maintained
runbook.

Claude Code (Opus 4.7) runs on the Linux dev box. The JAX solver + SAM cache + GPU semaphore live on the Windows GPU box under WSL2 Ubuntu. Transport choice connects them. Two specialist briefs disagreed:

- **`research-v23-mcp-protocol.md` §2 (protocol agent):** stdio over SSH. `claude mcp add woodblock_stack --scope user -- ssh reidsurmeier2@100.67.23.102 "wsl -d Ubuntu -- /opt/woodblock-mcp/bin/woodblock-mcp"`. Subprocess lifecycle managed by Claude Code; stderr → logs, stdout → JSON-RPC framing.
- **`research-v23-mcp-reuse.md` §6 (reuse agent):** MCP-over-SSE on Tailscale. Native multi-process, bearer-token auth in header, no SSH key dance.

Synthesizer (plan v2.1 §11) reconciled: **stdio over SSH wins for day-1**. SSE is the v23.x upgrade once multi-client is required.

## Decision

```bash
claude mcp add woodblock_stack --scope user -- \
  ssh reidsurmeier2@100.67.23.102 \
  "wsl -d Ubuntu -- /home/reidsurmeier2/.venv-v23/bin/woodblock-mcp"
```

FastMCP 2.x. `mcp.run(transport="stdio")`. Same code switches to `mcp.run(transport="http", port=8765)` for the v23.x SSE/streamable-HTTP upgrade — only the run call differs, tool implementations are transport-agnostic.

## Alternatives considered

**B. streamable-HTTP on Tailscale with `BearerAuthProvider`.** Rejected day-1: adds bearer token rotation, requires bind only to Tailscale interface (`host="100.67.23.102"`, never `0.0.0.0`), needs systemd unit. None of these are hard, but they are surface area v23 does not need to ship through. Reuse agent's framing assumed multi-client was a day-1 need; user clarified it isn't (single artist, single GPU box).

**C. Local FastMCP on Linux dev box, JAX solver over HTTP to Windows.** Rejected because JAX compile cost (~30 s first call, ~50 ms subsequent) must be paid once per Claude Code session, not per tool call. Embedding JAX in the MCP subprocess gives a free warm-start across an entire conversation; remote JAX via HTTP would re-pay compile per session boundary or require a separate persistent JAX worker (which is what the WSL2 subprocess effectively becomes for free).

## Trade-off accepted

- **Gained:** JAX warm-start is automatic (subprocess lives for the chat session), no port exposure, no bearer token, dies cleanly on Claude Code restart, 3-command install, GPU semaphore is a process-local lock.
- **Lost:** single client at a time (Mac + Linux can't both drive the GPU simultaneously — punt to v23.x SSE per `research-v23-mcp-protocol.md` §2 note 3); GPU on Windows over SSH adds ~150 ms RTT per progress emit (mitigated via batching per protocol §2 note 2); debugging is harder than `curl /health` — use `claude mcp get woodblock_stack` + tail subprocess stderr (note 7).
- **Failure mode noted:** Claude Code restart kills the subprocess and ssh-agent can expire mid-session. Documented in `research-v23-mcp-edges.md` §7 graceful-cascade.

## Consequence

`pyproject.toml` declares the entrypoint `woodblock-mcp = "backend.mcp.v23_server:main"` so `uv tool install` produces a stable PATH binary the SSH command can resolve. Env vars `JAX_COMPILATION_CACHE_DIR`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `WB_DATA_DIR` configured via the FastMCP lifespan entrypoint. No `~/.claude.json` manual edits required for users.
