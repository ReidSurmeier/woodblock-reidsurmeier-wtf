# Chuck MCP

Chuck MCP is a local MCP server for building mokuhanga-style print plans from a
single input image. It runs a JAX inverse-stack solver, emits cumulative pull
previews, and can export carving-oriented SVG artifacts for testing.

This repository is the Chuck experiment. It is intentionally separate from
`emma-mokuhanga-mcp`; do not mix the two repos or tool surfaces.

## Current Status

The current `main` branch preserves the role-based solver experiment and the
later v23 MCP tool work. It is a maintenance-stage, local, on-demand research
tool. It is not final CNC-ready printmaking advice, and portable CPU validation
does not prove GPU, carving, registration, Pigment, paper, or physical-print
behavior.

`PROJECT.md` is the current Resume Packet. It records live GitHub, Pugnet, and
Droplet ownership separately from the source tree.

### Inherited review frontend

The checked-in Next.js, Playwright, and Docker Compose surface descends from the
earlier Color Separator application. It is retained as an inherited review
frontend and source-lineage scaffold; ADR-0001 keeps the MCP server
authoritative. It is not an active deployment, does not own the public Color
Separator route, and must not be used as evidence that this repository is
serving users.

## What It Does

Chuck MCP takes one image and generates:

- an ordered stack of translucent/semi-opaque impressions
- cumulative print previews after every pull
- per-impression alpha masks
- a final composite preview
- plan metadata for MCP tools
- optional SVG/carving exports

The system does **not** recover historical or true underlayers from an image.
It designs plausible underprint candidates under the current pigment/rendering
model.

## Current Solver

The current S5 solver uses:

- JAX + JAXopt L-BFGS on CUDA
- light-to-dark print ordering, with black/key detail last
- bounded internal solve grids for 2K images on 12 GB GPUs
- a role layout after print ordering:
  - early pulls: broad underlayer controls
  - middle pulls: regional color/shadow controls
  - final pulls: detail/key controls
- different parameterization by role:
  - underlayers use a 4x coarser control grid
  - middle impressions use a 2x coarser control grid
  - detail impressions remain full solve-grid
- edge-weighted RGB loss
- low-pass target loss
- layer-weighted TV
- local-support/speckle penalty
- dark-on-bright penalty

The staged warm-up solver is available but not enabled by default:

```bash
WOODBLOCK_ROLE_WARMUP=1
```

Validation showed the warm-up stages over-shaped early pulls on the Emma test
image. The default is therefore joint optimization over role-parameterized
groups.

## Solve Profiles

`propose_stack` accepts `solve_profile` and optional `m_prior`.

| Profile | Max L-BFGS iterations | Default `m_prior` | Internal pixel budget |
|---|---:|---:|---:|
| `fast` | 60 | 6 | 256k |
| `default` | 180 | 8 | 512k |
| `thorough` | 400 | 10 | 768k |

`m_prior` is validated from 4 through 12.

For large images, the solver optimizes on a bounded internal grid and returns
full-resolution masks/previews. Override the grid cap with:

```bash
WOODBLOCK_SOLVER_MAX_PIXELS=1200000
```

## Latest Reference Run

Input:

```text
/srv/woodblock-share/input-images/close_emma_2002_2048.jpg
```

Latest role-based output:

```text
/srv/woodblock-share/output-images/chuck-rolebased-joint-main-20260512-160644
```

Clean image-only output folder:

```text
/srv/woodblock-share/chuck-clean-outputs/chuck-rolebased-joint-main-20260512-160644
```

Metrics:

| Metric | Value |
|---|---:|
| mean DeltaE76 | 7.277 |
| p95 DeltaE76 | 18.258 |
| solver wall time | 42.86s |
| optimized shape | 974 x 789 |
| downsample scale | 0.47558 |
| impressions | 9 |
| first 3 pull components at alpha >= 0.30 | 1364 |

Print order:

1. cadmium yellow
2. hansa yellow
3. cadmium orange
4. quinacridone magenta
5. burnt sienna
6. cobalt violet
7. viridian green
8. cobalt blue
9. ivory black

Comparison to the saved M10 build:

| Metric | Saved M10 | Role-Based Current |
|---|---:|---:|
| mean DeltaE76 | 6.941 | 7.277 |
| p95 DeltaE76 | 17.772 | 18.258 |
| first 3 pull components | 2847 | 1364 |
| solver wall time | 27.7s | 42.9s |

Interpretation: the role-based solver makes the early pulls less fragmented,
but pays a modest reconstruction cost. This is now the main branch because it is
closer to the intended printmaking structure, not because it is finished.

## Setup

Use Python 3.11+ on Linux/WSL2 with an NVIDIA GPU. The current tested host uses
JAX 0.10.0 with CUDA 13 packages.

```bash
cd /home/reidsurmeier/src/woodblock-reidsurmeier-wtf

python3 -m venv .venv-v23
. .venv-v23/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[solver,mcp,io,dev]"
```

Verify GPU JAX:

```bash
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-v23/bin/python - <<'PY'
import jax
print(jax.__version__)
print(jax.default_backend())
print(jax.devices())
PY
```

Expected backend:

```text
gpu
```

## Run Through MCP Registry

This runs the same in-process tool path the MCP server exposes:

```bash
WOODBLOCK_DISABLE_SAM=1 \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv-v23/bin/python - <<'PY'
from backend.mcp.registry import call_mcp_tool

image = "/srv/woodblock-share/input-images/close_emma_2002_2048.jpg"

result = call_mcp_tool("propose_stack", {
    "path": image,
    "solve_profile": "thorough",
})
print(result.model_dump(mode="json"))
PY
```

Useful follow-up tools:

```python
call_mcp_tool("inspect_plan", {"plan_id": plan_id, "focus": "composite"})
call_mcp_tool("inspect_plan", {"plan_id": plan_id, "focus": "per_impression"})
call_mcp_tool("forward_render", {"plan_id": plan_id})
call_mcp_tool("score_stack_delta_e", {"plan_id": plan_id})
call_mcp_tool("score_candidate_stack", {"plan_id": plan_id})
call_mcp_tool("solver_telemetry", {"plan_id": plan_id})
call_mcp_tool("export_svg", {"plan_id": plan_id})
call_mcp_tool("export_block_svgs", {"plan_id": plan_id})
```

## Run On The Orca-managed Pugnet Host

Console entry point:

```bash
.venv-v23/bin/woodblock-mcp
```

Pugnet is the primary development and GPU execution host. Open the registered
repository or a task worktree through Orca, then run the entry point in its
Orca-managed terminal. Orca owns the remote connection and session lifecycle;
do not add a separate SSH-backed MCP registration.

This is an on-demand local MCP process, not a persistent deployment. The
inherited frontend containers do not expose it, and no Woodblock tailnet route
currently exists. See ADR-0006 for the current transport decision and ADR-0003
for the superseded day-one SSH rationale.

## Tool Surface

The registered tool surface is generated from:

```text
backend/mcp/registry.py
```

Primary tools:

- `ingest_reference_image`
- `analyze_image`
- `build_hue_family_map`
- `propose_stack`
- `inspect_plan`
- `forward_render`
- `simulate_overprint`
- `score_stack_delta_e`
- `score_candidate_stack`
- `solver_telemetry`
- `export_print_plan`
- `export_svg`
- `export_block_svgs`
- `generate_carve_order`

Additional modules provide HITL edits, calibration, session, carve, and overlay
tools.

## Output Locations

Default plan/session artifacts:

```text
~/.woodblock/v23/
```

Shared full outputs from validation runs:

```text
/srv/woodblock-share/output-images/
```

Clean image-only outputs:

```text
/srv/woodblock-share/chuck-clean-outputs/
```

Use the clean folder when reviewing visuals. It contains only:

- final composite
- incremental pull contact sheet
- impressions + cumulative pull contact sheet
- individual cumulative pull images

## Architecture

```text
MCP client
  -> backend/mcp/v23_server.py
  -> backend/mcp/registry.py
  -> backend/mcp/tools/*
  -> backend/services/v23/orchestrator.py
  -> S1 ingest
  -> S2 optional SAM gateway
  -> S3 hue family map
  -> S4 Tan RGB warm start
  -> S5 JAX role-based inverse solver
  -> S6 three-state mask classifier
  -> S7 block packing
  -> S8 topology diagnostics/repair hooks
  -> S9 SVG vectorization
  -> S10 manifest and ZIP emit
```

## Render Model

Current default rendering is Tier 1:

- RGB/JAX forward stack
- pigment anchors from the 13-pigment catalog
- rendered as if pigments were pre-mixed in a well

This is directionally useful but not physically final mokuhanga overprint
simulation. T2 empirical swatch correction exists as a local path after
`upload_swatch_overprint_matrix`. Spectral/two-flux rendering remains future
work.

## What Is Still Not Fixed

The current role-based solver is better aligned with printmaking intent, but it
still needs:

- SLIC/superpixel brushed-zone parameters for middle impressions
- hard pre-vectorization machinability scoring
- component-count acceptance thresholds in normal tool output
- better persistence of actual JAXopt `iter_num`, not just max iteration budget
- broader test coverage across `/Volumes/woodblock/Examples`
- more careful visual gates before accepting SVGs as carving geometry

The build plan for the role-based solver lives here:

```text
docs/solver-role-based-build-plan.md
```

## Tests

Focused checks used for the current main:

```bash
.venv-v23/bin/python -m ruff check \
  backend/services/v23/stages/s5_solver.py \
  backend/tests/v23/stages/test_s5_solver.py \
  backend/services/v23/orchestrator.py \
  backend/tests/v23/stages/test_orchestrator.py \
  backend/mcp/tools/core.py

.venv-v23/bin/python -m pytest \
  backend/tests/v23/stages/test_s5_solver.py \
  backend/tests/v23/stages/test_orchestrator.py \
  backend/tests/v23/direct/test_core_tools.py::test_propose_stack_is_real_solver_post_d14h \
  backend/tests/v23/direct/test_core_tools.py::test_propose_stack_rejects_invalid_solve_profile \
  -q
```

Last focused result:

```text
24 passed in 50.85s
```

## License

MIT. Copyright 2026 Reid Surmeier.
