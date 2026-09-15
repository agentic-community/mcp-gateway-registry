# go-validate

A Go **fast-path sidecar** for the auth-server's `GET /validate` endpoint.

nginx fires `auth_request /validate` on every proxied request through the gateway.
The Python handler runs on a single uvicorn worker and is the throughput ceiling of
the authenticated data path. `go-validate` serves the steady-state RS256 bearer path
in Go and **reverse-proxies everything else to the unchanged Python auth-server**, so
it is byte-identical where it answers and zero-risk where it does not.

Design, testing plan, and expert review: `.scratchpad/ant-hackathon-aug-2026/final/`
(lld.md, testing.md, review.md). Issue: agentic-community/mcp-gateway-registry#1652.

## What it does

- **Fast path:** verifies an RS256 JWT from the configured IdP against a cached JWKS,
  maps claims to identity, mints the HS256 `X-Internal-Token-Registry`, and writes the
  identity headers nginx consumes.
- **Fallback:** cookies, other IdPs, opaque tokens, unknown `kid`, or an unset fast
  path are reverse-proxied to the Python auth-server.
- **Fail closed:** recognized-but-invalid token -> 401; unrecognized -> Python.

## Configuration (environment)

| Variable | Default | Notes |
|----------|---------|-------|
| `GOVALIDATE_LISTEN` | `:8899` | listen address |
| `SECRET_KEY` | (required for fast path) | shared HS256 key; validated for missing/weak at startup |
| `JWKS_URL` | (required for fast path) | configured IdP JWKS endpoint |
| `VALIDATE_ISSUER` | (required for fast path) | expected `iss` (config-driven, not from the token) |
| `VALIDATE_AUDIENCE` | (required for fast path) | expected `aud` (enforced against config) |
| `AUTH_FALLBACK_URL` | `http://auth-server:8888` | Python auth-server for fallback |
| `JWKS_REFRESH_SECONDS` | `300` | background JWKS refresh interval |

If `SECRET_KEY` / `JWKS_URL` / `VALIDATE_ISSUER` / `VALIDATE_AUDIENCE` are not all set,
the sidecar runs in **fallback-only** mode (every request proxied to Python) - still
correct, just not accelerated.

## Endpoints

- `GET /validate` - the auth_request handler
- `GET /health` - readiness (degraded when JWKS is unhealthy)
- `GET /metrics` - plaintext counters (fastpath_ok / unauthorized / fallback)

## Build & run

```bash
go build -o go-validate .
SECRET_KEY=... JWKS_URL=... VALIDATE_ISSUER=... VALIDATE_AUDIENCE=... \
  AUTH_FALLBACK_URL=http://127.0.0.1:8888 ./go-validate
```

## Status

Hackathon scope: RS256 fast path + fallback. Not yet ported (stays on Python via
fallback): federation/admin static tokens, session cookies, OBO exchange, per-tool
ACL, rate limiting, audit. See the LLD "Non-Goals" and review blockers before
production use.
