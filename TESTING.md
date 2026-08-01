# Woodblock MCP testing

The repository has two validation boundaries. The portable gate is required on
every pull request. The complete solver-bearing suite is a separate local/GPU
gate and must be reported with its actual backend, runtime, skips, and timeout.

## Environment

Python 3.11 and 3.12 are supported. Create a disposable environment and install
the CPU validation extras:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[mcp,io,solver-cpu,dev]"
```

The `solver` extra installs the CUDA 13 JAX stack for a compatible Pugnet GPU
environment. The `solver-cpu` extra is the portable CI boundary. Do not install
both to imply that the CPU gate validates GPU behavior.

## Red-green-refactor

For each change:

1. add one focused test;
2. run it and retain the expected red output in the working notes;
3. implement the smallest change;
4. rerun the focused test;
5. run the relevant ring; and
6. run the complete applicable gate before merge.

Repository and documentation contracts live under `tests/`. Product tests live
under `backend/tests/v23/` and are grouped by scaffold, unit, stage, direct MCP,
transport, conversation, solver-smoke, and corpus rings.

## Required portable gate

```bash
python -m unittest tests.test_repository_contract -v

ruff check backend/mcp backend/services/v23 backend/tests/v23 tests

WOODBLOCK_HOME="$(mktemp -d)" \
  python -m pytest \
  backend/tests/v23/scaffold \
  backend/tests/v23/unit \
  -m "not solver" \
  -q

python -m compileall -q backend tests
```

The GitHub workflow pins Ruff 0.15.20 and runs this gate on Python 3.11 and
3.12. A solver-marked test is excluded only from the portable job; it remains
part of the complete suite.

## Complete solver-bearing gate

Use a fresh plan root and state the observed JAX backend:

```bash
python - <<'PY'
import jax

print(jax.__version__)
print(jax.default_backend())
print(jax.devices())
PY

validation_root="$(mktemp -d)"
WOODBLOCK_HOME="$validation_root" \
  timeout 900 python -m pytest backend/tests/v23 -q --tb=line
find "$validation_root" -depth -delete
```

The pre-tracer handoff command used a 300-second timeout and reached only part
of the 297-test collection. The repair reused one expensive HITL plan pair
within its module instead of rebuilding it for every assertion. On 2026-07-31
the complete gate then reported 279 passed and 18 skipped in 224.94 seconds
(229.97 seconds wall-clock) with JAX 0.10.0 on one CUDA device.

The ring split remains useful evidence: direct MCP took 140.52 seconds, stages
67.40 seconds, transport/conversation/corpus 4.28 seconds, and solver-smoke
7.23 seconds. Preserve the 900-second outer bound and continue reporting actual
backend, skips, and runtime.

The corpus gate may require media that is intentionally absent from a clean
checkout. Missing private media should be an explicit skip or refusal, never a
download from an undocumented location.

## Inherited review frontend

The Next.js package is retained as an inherited review frontend, not as the
MCP product or an active deployment. Validate changes to that surface with:

```bash
npm ci
npm run check
```

Playwright tests require an explicitly started local frontend and backend.
They must not target the public Color Separator application as evidence for
this repository.

## Live checks

Live Pugnet, Tailscale, Docker, and Droplet checks are read-only unless a task
owns an explicit deployment plan. Verify separately:

- container process health;
- published host ports;
- Docker network attachment;
- Tailscale Serve ownership;
- GitHub Pages and deployment records; and
- Droplet Runtime Component ownership.

An internally responsive legacy container without a route, release identity,
and rollback packet is not a deployment.
