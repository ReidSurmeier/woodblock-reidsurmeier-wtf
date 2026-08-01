# 0006 — Orca-managed Pugnet execution

Status: accepted (2026-07-31)
Supersedes: ADR-0003 as the current operator path

## Context

ADR-0003 selected stdio over SSH for a May 2026 day-one experiment. The
workspace has since moved to Orca remote-server ownership. The machine known in
the workspace as Pugnet is connected through the Orca server and worker
services, and this repository has a registered canonical checkout and
task-specific Orca worktrees there.

Fresh inspection found no Woodblock MCP deployment route. Two isolated legacy
`woodblock-*` containers remain as evidence of an earlier review frontend, but
they publish no host ports and own no Tailscale Serve route. They are not the
MCP runtime and are not promoted by this decision.

## Decision

Pugnet is the primary development and GPU execution host for Woodblock MCP.
Operators enter the registered repository through Orca, create or resume an
Orca worktree, and run the repository-local `woodblock-mcp` entry point in an
Orca-managed terminal. Orca owns the remote connection and session lifecycle;
the repository does not instruct users to create a separate SSH-backed Claude
MCP registration.

The MCP process remains local and on demand:

```bash
cd /home/reidsurmeier/src/woodblock-reidsurmeier-wtf
.venv-v23/bin/woodblock-mcp
```

This does not declare the inherited Next.js frontend, Docker Compose stack, or
an HTTP/SSE endpoint to be deployed. A persistent or multi-client MCP service
requires a separate runtime component, health endpoint, rollback packet, and
tailnet exposure decision.

## Consequences

- Orca, rather than an operator-maintained SSH command, owns access to Pugnet.
- Worktree identity and task history stay visible in the Orca project.
- GPU validation runs on the machine that owns the compatible environment and
  hardware.
- The stdio entry point remains useful without exposing another network port.
- Pairing material, machine tokens, and other secrets stay outside repository
  documentation and command examples.
- ADR-0003 remains available as historical rationale, but its registration
  command is not a current runbook.

## Verification boundary

The remote Orca services and registered checkout were verified live when this
decision was adopted. Woodblock MCP itself remains an on-demand process:
repository tests, an Orca connection, or legacy containers do not constitute a
deployment. A future persistent runtime must be inventoried and verified
independently on Pugnet and in the Droplet Platform registry.
