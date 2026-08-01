# Woodblock MCP agent guide

This repository is the MCP-first Woodblock / Chuck planning experiment. Read
`PROJECT.md`, then `CONTEXT.md`, then the relevant ADRs before changing code.
The inherited Next.js and Compose surface is review/source-lineage
infrastructure, not the authoritative solve path or an active deployment.

## Mission

Build a mokuhanga planning tool that ingests one reference image and produces a
plausible Block / Impression / Mask Plan designed by printmaking rules. Never
present its output as evidence of an artist's historical physical process.

## Boundaries

- Preserve the vocabulary and confidence limits in `CONTEXT.md`.
- Keep the MCP server authoritative per ADR-0001.
- Keep Overprint distinct from Mixing per ADR-0002.
- Do not edit corpus originals or publish reference media as a side effect of
  tests.
- Keep plan sessions, calibration data, generated runs, virtual environments,
  and GPU caches out of Git.
- Do not stop, rebuild, remove, or expose legacy Pugnet containers unless the
  task explicitly owns a reviewed deployment or cleanup plan.
- Do not claim GPU, corpus, CNC, carving, registration, Pigment, paper, or
  physical-print validation from the portable CPU gate.
- Never commit credentials, pairing codes, private keys, token values, or raw
  secret-scan output.

## TDD

Every behavior change follows red, green, refactor:

1. add the smallest failing test under `backend/tests/v23/` or `tests/`;
2. run it and confirm the failure represents the intended missing behavior;
3. implement only enough to make it pass;
4. rerun the focused test and the relevant portable ring;
5. run the complete applicable validation before merge; and
6. record slower GPU, corpus, or physical gates as separate evidence.

Plan-mutating MCP tools create a new `plan_id` in the same session and never
overwrite the parent Plan. Tool implementations return structured
`ToolResult` errors instead of raising across the MCP boundary.

## Commands

Repository contract:

```bash
python3 -m unittest tests.test_repository_contract -v
```

Pinned lint:

```bash
uvx --from ruff==0.15.20 \
  ruff check backend/mcp backend/services/v23 backend/tests/v23 tests
```

Portable Python gate:

```bash
WOODBLOCK_HOME="$(mktemp -d)" \
  python -m pytest backend/tests/v23/scaffold backend/tests/v23/unit \
  -m "not solver" -q
```

Complete solver-bearing suite:

```bash
WOODBLOCK_HOME="$(mktemp -d)" \
  timeout 900 python -m pytest backend/tests/v23 -q --tb=line
```

Inherited review frontend:

```bash
npm ci
npm run check
```

See `TESTING.md` for dependency setup, the portable/full-suite distinction,
and the measured resolution of the former 300-second timeout.

## Architecture

The v23 pipeline is under `backend/services/v23/`; MCP registration and tools
are under `backend/mcp/`. The canonical domain model is in `CONTEXT.md`.

The ten processing stages are:

1. input canonicalization;
2. optional SAM region prior;
3. hue-family classification;
4. geometry warm start;
5. inverse solve;
6. three-state Mask classification;
7. Block packing and pull-group assignment;
8. topology repair;
9. per-Impression vectorization; and
10. Plan export.

Before changing a stage, read its implementation, tests, and any ADR that
defines its boundary.

## Agent skills

### Issue tracker

Issues and PRDs live in GitHub Issues for
`ReidSurmeier/woodblock-reidsurmeier-wtf`. Always pass that repository
explicitly to `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and
`wontfix` as the five workflow states. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. Root `CONTEXT.md` and `docs/adr/` are the
durable domain sources. See `docs/agents/domain.md`.

## Resume boundary

`PROJECT.md` owns current state, verified evidence, limitations, and next work.
Do not copy volatile commit IDs, test counts, or machine paths into this file;
update the Project Resume Packet and GitHub issue instead. The current
third-party JAXopt deprecation is documented there; repository-owned
scikit-image deprecations have been repaired.
