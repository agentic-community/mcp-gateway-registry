# Testing Guide — Go `/validate` Fast-Path Sidecar

This guide lets anyone confirm the go-validate fast path (issue #1652, PR #1653) works end to end: functional correctness (e2e suite), that a real IdP token is actually accelerated (not just proxied), and throughput under load (stress test) using the bundled `pingmcp` fast upstream.

It is written for the **docker-compose** deployment (Keycloak or Entra/Cognito). The same checks apply on ECS/EKS; only how you reach `/metrics` differs (noted where relevant).

---

## 0. What you are validating

1. **Non-breaking:** with the sidecar deployed, the gateway behaves exactly as before (e2e suite passes).
2. **Fast path engages:** a real IdP **RS256 access token** is verified and served by the Go sidecar itself (not reverse-proxied to Python).
3. **Fail-safe:** anything the fast path can't verify (cookies, self-signed HS256 tokens, other IdPs) transparently falls back to Python — correct results, no errors.
4. **Throughput:** end-to-end RPS/latency through the gateway against a fast upstream (`pingmcp`), so the measurement reflects the auth check, not a slow backend.

> **Key fact before you start:** the fast path only accelerates **RS256 access tokens** from the configured IdP (Keycloak / Cognito / Entra / Okta). A repo `.token` file is usually a **self-signed HS256** registry token — it will **fall back to Python by design** and will NOT move the `fastpath_ok` counter. To exercise the fast path you must mint a real IdP token (Part D).

---

## 1. Prerequisites

- Docker + Docker Compose v2.24+ (`docker compose version`)
- `uv` (for the Python e2e script), `go` 1.24+ (only if you build the load tool), `curl`, `jq`
- The repo checked out on the PR branch, with a working `.env` (copy from `.env.example` and fill IdP config)
- The stack running: `./build_and_run.sh` (compose builds the `go-validate` sidecar automatically)

---

## 2. Confirm the sidecar is deployed and healthy

The compose `registry` routes nginx `/validate` to `go-validate:8899` by default (`VALIDATE_UPSTREAM_URL`). Confirm the sidecar came up in fast-path mode:

```bash
# Container is running
docker compose ps go-validate

# Startup log: mode + provider + the accepted issuers/audiences it derived
docker compose logs go-validate | grep -E "listening on|accepted (issuers|client_ids)"
# Expect e.g.:
#   go-validate listening on :8899 | mode=fast-path | provider=keycloak | fallback=http://auth-server:8888
#   accepted issuers=[...] | accepted audiences=[...]

# Health + metrics (reached inside the compose network)
docker compose exec go-validate /go-validate -healthcheck && echo "health OK"
docker compose exec registry curl -s http://go-validate:8899/metrics
```

In `/metrics`, before any traffic you should see:
```
govalidate_fastpath_ready 1          # configured to accelerate
govalidate_jwks_healthy 1            # IdP JWKS loaded
govalidate_jwks_refresh_failures_total 0
govalidate_fastpath_ok 0
govalidate_fallback 0
```

- **`fastpath_ready 0`** ⇒ the sidecar is proxying only (missing config). The startup log prints a loud `WARN ... FALLBACK-ONLY mode: missing ...` naming exactly what to set.
- **`ready 1` + `jwks_healthy 0`** ⇒ degraded (IdP JWKS unreachable); it still falls back safely.

> On ECS/EKS the sidecar has no exposed port; read `/metrics` via `aws ecs execute-command ... --container auth-server --command "curl -s http://localhost:8899/metrics"` (or `kubectl exec`).

---

## 3. Part A — End-to-end functional suite (the pre-release gate)

This is the same suite run before every release. It exercises registry CRUD, search, security scan, and an external MCP server through the gateway — all of whose auth checks pass through the fast-path sidecar (falling back for the self-signed `.token`).

```bash
uv run python tests/e2e_release_test.py \
  --registry-url http://localhost \
  --token-file .token
```

**Expected:** `*** ALL TESTS PASSED ***` (8/8) on a clean compose stack.
- If you test against a shared/managed deployment where the token lacks the `publish_skill` permission, test 5 (Skill CRUD) may return `403`, and a strict-MCP upstream may return `405` on test 8 — those are deployment/permission specifics, not fast-path regressions. On a local compose stack it should be a clean 8/8.

This proves **non-breaking**: the sidecar in front of `/validate` did not change any observable behavior.

---

## 4. Part B — Register the `pingmcp` fast upstream (+ SSRF allowlist)

`pingmcp` is a minimal, dependency-free Go streamable-http MCP server with a single `echo` tool. It exists so the load test is bounded by the **auth check**, not a slow upstream. Source: `servers/pingmcp/` (canonical: https://github.com/aarora79/pingmcp).

### 4a. Start pingmcp (opt-in `benchmark` profile)

```bash
docker compose --profile benchmark up -d --build pingmcp-server
# It listens on pingmcp-server:8100 in-cluster (also published to 127.0.0.1:8100).
```

### 4b. Allow it through the SSRF guard (REQUIRED)

The registry runs every registered `proxy_pass_url` through an SSRF guard that blocks private/loopback hosts. `pingmcp-server` is an internal host, so you must allowlist it or **registration/health-check will be blocked**. In your `.env`:

```bash
# comma-separated exact hostnames/IPs (least privilege)
SSRF_ALLOWED_HOSTS=pingmcp-server
```

Then restart the registry so it picks up the new allowlist:
```bash
docker compose up -d registry
```

### 4c. Register it in the gateway

A ready-made registration payload ships at `cli/examples/pingmcp.json` (path `/pingmcp/`, `proxy_pass_url http://pingmcp-server:8100/`, `auth_scheme: none`).

```bash
uv run python api/registry_management.py \
  --registry-url http://localhost \
  --token-file .token \
  register --config cli/examples/pingmcp.json
```

> Note: the global flags (`--registry-url`, `--token-file`) come **before** the `register` subcommand.

Confirm it's reachable through the gateway (any authenticated request works; `405`/`403` still means the request traversed `/validate`):
```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost/pingmcp/mcp \
  -H "Authorization: Bearer $(cat .token)" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1.0"}}}'
```

---

## 5. Part C — Mint a REAL IdP token (this is what exercises the fast path)

The `.token` file is self-signed HS256 → it always falls back. To trigger the fast path you need an RS256 access token from the configured IdP. Pick the one matching your `AUTH_PROVIDER`.

### Keycloak (default compose)
```bash
CID=$(grep -E '^KEYCLOAK_M2M_CLIENT_ID=' .env | cut -d= -f2-)
CSEC=$(grep -E '^KEYCLOAK_M2M_CLIENT_SECRET=' .env | cut -d= -f2-)
REALM=$(grep -E '^KEYCLOAK_REALM=' .env | cut -d= -f2-)
TOK=$(curl -s -X POST "http://localhost:8080/realms/$REALM/protocol/openid-connect/token" \
  -d grant_type=client_credentials -d "client_id=$CID" --data-urlencode "client_secret=$CSEC" \
  | jq -r .access_token)
echo "$TOK" > /tmp/idp.token
```

### Cognito (M2M)
```bash
# Needs the M2M app-client id + secret and the Cognito domain; scope = a resource-server scope.
TOK=$(curl -s -X POST "https://<domain>.auth.<region>.amazoncognito.com/oauth2/token" \
  -d grant_type=client_credentials -d "client_id=<m2m-client-id>" \
  --data-urlencode "client_secret=<m2m-secret>" \
  --data-urlencode "scope=mcp-servers-unrestricted/read mcp-servers-unrestricted/execute" \
  | jq -r .access_token)
echo "$TOK" > /tmp/idp.token
```

Sanity-check it is RS256 and from the expected issuer:
```bash
python3 -c "import sys,json,base64; h,p,_=open('/tmp/idp.token').read().strip().split('.'); \
d=lambda s: json.loads(base64.urlsafe_b64decode(s+'='*(-len(s)%4))); \
print('alg=',d(h)['alg'],'iss=',d(p).get('iss'),'aud=',d(p).get('aud'))"
# alg should be RS256; iss/aud must match the sidecar's 'accepted issuers/audiences' from step 2.
```

---

## 6. Part D — Prove the fast path serves a real token, then measure RPS

### 6a. One request → confirm `fastpath_ok` increments

```bash
# capture the counter before
docker compose exec registry curl -s http://go-validate:8899/metrics | grep -E "fastpath_ok|fallback"

# one authenticated request through the gateway with the REAL IdP token
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost/pingmcp/mcp \
  -H "Authorization: Bearer $(cat /tmp/idp.token)" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"1.0"}}}'

# capture again — fastpath_ok should have gone UP
docker compose exec registry curl -s http://go-validate:8899/metrics | grep -E "fastpath_ok|fallback"
```

**Pass criteria:** `govalidate_fastpath_ok` increases. That means the Go sidecar verified the token, resolved scopes, minted the internal token, and answered `/validate` itself — no fallback to Python.

> Cross-check the negative: repeat with `--token-file .token` (HS256) and confirm `govalidate_fallback` increments instead — proving the fail-safe path.

### 6b. Load test (RPS)

Any concurrent HTTP driver works. Example with [`hey`](https://github.com/rakyll/hey) (`go install github.com/rakyll/hey@latest`):

```bash
hey -n 2000 -c 50 -m POST \
  -H "Authorization: Bearer $(cat /tmp/idp.token)" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"lt","version":"1.0"}}}' \
  http://localhost/pingmcp/mcp
```

No-tools fallback (pure bash, lower throughput but zero deps):
```bash
TOK=$(cat /tmp/idp.token)
time (for i in $(seq 1 500); do
  curl -s -o /dev/null -X POST http://localhost/pingmcp/mcp \
    -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' &
  [ $((i % 25)) -eq 0 ] && wait
done; wait)
```

After the run, confirm the load went through the fast path (not fallback):
```bash
docker compose exec registry curl -s http://go-validate:8899/metrics | grep -E "fastpath_ok|fallback|unauthorized"
```
`fastpath_ok` should account for the bulk of the requests.

### 6c. Compare against Python (optional, the deck's apples-to-apples)

The headline numbers (Python 65–551 rps → Go 3k–18k rps, p99 ~2s → ~130ms) were measured by hitting **`/validate` directly on the host** (isolating the auth-check CPU cost), not through nginx/CloudFront. To reproduce that comparison:
- Point the driver at `http://auth-server:8888/validate` (Python) vs `http://go-validate:8899/validate` (Go) from inside the compose network, sending the IdP token in the `X-Authorization` header plus `X-Original-URL: http://localhost/pingmcp/mcp`.
- A run through the public gateway URL instead measures network/RTT, so its absolute RPS will be far lower and is **not** comparable to those figures.

---

## 7. Expected results summary

| Check | Pass criteria |
|---|---|
| Sidecar health (step 2) | `fastpath_ready 1`, `jwks_healthy 1`, `jwks_refresh_failures 0` |
| E2E suite (Part A) | 8/8 on a clean compose stack (non-breaking) |
| pingmcp reachable (Part B) | request traverses `/validate` (200/403/405 all count) |
| Real token fast-pathed (Part D 6a) | `govalidate_fastpath_ok` increments |
| Fail-safe (Part D 6a negative) | HS256 `.token` increments `govalidate_fallback`, request still succeeds |
| Load test (Part D 6b) | `fastpath_ok` accounts for the load; no `unauthorized`/errors; latency stable |

---

## 8. Gotchas

- **`.token` is HS256 → always falls back.** Use a real IdP token (Part C) to see `fastpath_ok` move. This is the single most common source of "it's not accelerating" confusion.
- **Register a server before load-testing its path**, and **allowlist its host** in `SSRF_ALLOWED_HOSTS` (Part B) or registration/health-check is blocked.
- **`iss`/`aud` must match** the sidecar's `accepted issuers/audiences` (step 2 startup log). A mismatch fails safe (fallback) with no error — check the log, not just the HTTP status.
- If you **recreate the `go-validate` container**, its IP changes and nginx may cache the old one — `docker compose exec registry nginx -s reload` (or restart registry) after recreating the sidecar.
- **Turn it off** to A/B: set `VALIDATE_UPSTREAM_URL=http://auth-server:8888` in `.env` and restart registry — nginx then routes `/validate` straight to Python (no sidecar), useful for a before/after comparison.
