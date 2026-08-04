# Security Policy

## Reporting a Vulnerability

Please open a private security advisory on the repository or contact the maintainers.
Do not open a public issue for security-sensitive reports.

## Production hardening

- Never expose `HOST=0.0.0.0` without `MEMGRAPHRAG_API_KEY` or `AUTH_ACCOUNTS`.
- Prefer binding to `127.0.0.1` for local-only access.
- Rotate `TOKEN_SECRET` and API keys regularly.
- Keep PostgreSQL and Neo4j credentials out of source control (use `.env`).
