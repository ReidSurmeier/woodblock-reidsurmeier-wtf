# Domain documentation

## Required reading

Before changing product behavior, read:

- root `CONTEXT.md` for the canonical vocabulary;
- root `PROJECT.md` for the current runtime and deployment boundary; and
- the relevant decisions under `docs/adr/`.

This is a single-context repository. Do not invent a second glossary for the
inherited frontend.

## Preferred language

Use **Block**, **Impression**, **Mask**, **Pigment**, **Order**, **Underprint**,
**Plan**, **Stack**, **Overprint**, **Mixing**, and **Render tier** as defined in
`CONTEXT.md`. Preserve its distinction between a designed plausible Plan and
evidence about a historical physical process.

## Architecture boundary

ADR-0001 makes the MCP server authoritative. The inherited frontend may review
or visualize artifacts, but it must not silently become a second solve path.
ADR-0002 distinguishes Overprint from Mixing. ADR-0003 records the historical
stdio-over-SSH transport decision; current Orca/Pugnet operation must be
documented separately if that transport changes.

If a proposed issue, test, or implementation conflicts with those decisions,
surface the conflict and add an ADR when the new decision is hard to reverse,
surprising, and based on a real trade-off.
