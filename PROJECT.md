# Project status

## Purpose

Build an MCP-first mokuhanga planning tool that turns one reference image into
a plausible, inspectable Block / Impression / Mask plan without claiming to
recover an artist's historical process.

## Current status

Status: maintenance-stage experiment; local, on-demand MCP runtime.

The Python `woodblock_stack` package is the maintained product boundary. It
owns the v23 stage pipeline, the MCP tool surface, plan persistence, forward
rendering, and carving-oriented exports. The checked-in Next.js and Compose
files descend from the earlier Color Separator application and are retained as
an inherited review frontend and source-lineage scaffold. ADR-0001 keeps that
frontend outside the authoritative solve path.

Deployment ownership: none.

Pugnet is the primary development and GPU execution host. The current operator
path is an Orca-managed Pugnet repository/worktree and terminal, not a separate
SSH-backed Claude MCP registration. ADR-0006 records that decision. This
connection boundary does not make the MCP process a persistent deployment.

GitHub has no Pages site and no deployment records for this repository. Pugnet
has two legacy `woodblock-*` containers whose processes answer internally, but
they are isolated from their declared Docker networks, publish no host ports,
and own no Tailscale Serve route. The frontend container's health status is a
false negative caused by a `curl`-based probe in an image without `curl`.
Pugnet's active tailnet port 8001 belongs to the separate Color Separator
backend. The Droplet Platform inventory has no Woodblock MCP Runtime Component.

## Verified evidence

Verified at the start of GitHub issue 3:

- the clean Canonical Checkout and GitHub `main` both point to `70a26e6`;
- GitHub describes this repository as a fork of the Color Separator lineage
  for the Woodblock / Chuck MCP pipeline;
- the repository contains 297 collected v23 tests;
- the previous GitHub workflow fails in its Ruff job and has never completed a
  successful maintained validation run;
- Ruff 0.15.20 reports 72 current findings under the repository's declared
  `E`, `F`, `I`, `UP`, and `B` rule set;
- the repository's documented full-suite command exceeded its 300-second
  bound locally instead of completing in the claimed one minute;
- the two legacy Pugnet containers answer HTTP 200 internally, but they are
  not reachable through a host port or tailnet route;
- Pugnet's Orca remote services and this repository's Orca registration are
  active, while the Woodblock MCP entry point remains on demand;
- the live Color Separator backend is a separate container and source tree;
  and
- a fresh 44-component Droplet snapshot reconciles with zero issues and has no
  matching runtime.

These observations are a baseline, not a claim that the validation or
container residue is already repaired.

Verified after the issue 3 repair:

- Ruff 0.15.20 passes the maintained Python, service, and test trees;
- the complete v23 suite reports 279 passed and 18 skipped in 224.94 seconds
  (229.97 seconds wall-clock) with JAX 0.10.0 on one CUDA device;
- the portable scaffold/unit ring reports 81 passed and 6 solver-marked tests
  deselected;
- the inherited frontend reports zero npm audit vulnerabilities, passes ESLint
  and TypeScript, and completes its warning-free production build in 35.34
  seconds with the maintained webpack command;
- Next.js/Sentry configuration uses the current instrumentation and proxy file
  conventions; and
- the Docker frontend probe uses Node, matching the installed runtime instead
  of requiring an absent `curl` binary.

## Known limitations

- The inherited frontend still contains Color Separator domain and UI code;
  its repository-local review role does not make it the primary product.
- The legacy Pugnet containers have no verified release commit, rollback
  packet, or active route and must not be promoted as a deployment.
- Physical carving, registration, transfer, Pigment, paper, and Overprint
  tolerances remain unverified.
- JAXopt is unmaintained and emits its upstream deprecation warning; replacing
  it is a distinct solver decision, not a formatting change.
- Full-size reference and run artifacts remain external to the Git checkout
  and require separate rights and custody decisions.

## Next work

GitHub issue 3 owns this completed documentation and validation tracer. The
next implementation slice should come from the remaining domain or
physical-validation gaps, while a separate follow-up owns the decision about
the isolated legacy Pugnet containers. Do not rebuild or remove those
containers as an incidental repository-maintenance step.
