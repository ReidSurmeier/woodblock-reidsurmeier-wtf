# Security policy

## Supported boundary

Security fixes target the current `main` branch. This is a local, on-demand
MCP experiment; no public upload service or hosted Woodblock MCP deployment is
supported.

The inherited Next.js and Docker Compose files are review/source-lineage
infrastructure. Their presence does not authorize exposing the application,
binding it to a public interface, or treating legacy Pugnet containers as a
release.

## Reporting

Report a vulnerability privately through GitHub's private vulnerability
reporting interface when it is available. If that interface is unavailable,
contact the repository owner privately. Do not open a public issue containing
credentials, private image paths, unpublished reference media, or raw
secret-scan output.

## Secrets and local state

- Never commit API keys, pairing codes, SSH keys, bearer tokens, or machine
  credentials.
- Keep plan sessions, calibration data, run artifacts, virtual environments,
  and GPU caches outside Git.
- Use secret-manager references rather than literal values in docs or scripts.
- Bind experimental HTTP surfaces to loopback unless a documented deployment
  contract explicitly requires a tailnet listener.
- Treat uploaded reference images and generated plans as private by default
  until publication rights are recorded.

## Dependency and workflow policy

GitHub Actions must use commit-pinned actions. Python and Node dependency
changes require their normal lock/metadata updates, the portable validation
gate, and an active-tree secret scan before merge. The inherited
`backend/requirements.txt` manifest must also pass pip-audit and its image
upload route test. GPU or physical-print success must not be inferred from a
portable CPU workflow.
