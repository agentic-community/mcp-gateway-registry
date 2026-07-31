# Testing: MCP access-token TTL

Manual test plan for the configurable MCP access-token lifetime ([#1477](https://github.com/agentic-community/mcp-gateway-registry/issues/1477)). It exercises the two UI entry points that mint a token — the **Get JWT Token** button in the left sidebar and the **MCP Configuration** modal (server card → *Get Config* / connect), plus the **Generate Token** page where the lifetime is selectable — and the `MCP_TOKEN_DEFAULT_TTL_HOURS` / `MCP_TOKEN_MAX_TTL_HOURS` parameters behind them.

## What each entry point does (important)

The three token entry points do NOT all let you pick a lifetime:

| Entry point | Where | Lifetime it requests | Configurable per-request? |
|-------------|-------|----------------------|---------------------------|
| **Get JWT Token** (sidebar) | left nav "Get JWT Token" | hardcoded `expires_in_hours: 8` ([Sidebar.tsx](../frontend/src/components/Sidebar.tsx)) | No — always the default |
| **MCP Configuration modal** | server card, the dialog in the screenshot | hardcoded `expires_in_hours: 8` ([ServerConfigModal.tsx](../frontend/src/components/ServerConfigModal.tsx)) | No — always the default; the "Token expires in 8 hours" text is static |
| **Generate Token page** | left nav "Generate Token" | selectable 1h / 8h / 24h ([TokenGeneration.tsx](../frontend/src/pages/TokenGeneration.tsx)) | Yes |

So the sidebar button and the modal always request the **default** lifetime. That is why `MCP_TOKEN_DEFAULT_TTL_HOURS` is the parameter that changes their behavior — raise the default and both of them mint longer tokens without any UI change. `MCP_TOKEN_MAX_TTL_HOURS` bounds what the Generate Token page (and the raw API) may request.

> The two settings are read by BOTH the registry (request validation) and the auth-server (minting), so both services must pick up an `.env` change (restart both). Defaults: 8h default, 24h max, with a hardcoded 7-day (168h) absolute ceiling on the max.

## Prerequisites

- A running stack (`./build_and_run.sh`) with a logged-in user that has the `token-generation` permission (needed to see the Generate Token page and the sidebar button).
- `REG` set to the gateway URL: `export REG=http://localhost`.
- A helper to decode a JWT's lifetime without verifying the signature:

```bash
# Prints iat, exp, and (exp-iat) in hours for a JWT on stdin.
decode_ttl() {
  python3 - "$1" <<'PY'
import base64, json, sys
tok = sys.argv[1].split(".")
pad = lambda s: s + "=" * (-len(s) % 4)
claims = json.loads(base64.urlsafe_b64decode(pad(tok[1])))
iat, exp = claims["iat"], claims["exp"]
print(f"iat={iat} exp={exp} ttl_hours={(exp-iat)/3600:.2f} sub={claims.get('sub')}")
PY
}
```

---

## Test 1 — Default TTL via the Get JWT Token button and the modal (shipped 8h)

**Goal:** confirm both hardcoded-default entry points mint an 8h token out of the box.

### 1a. Sidebar "Get JWT Token" button

1. In the UI, click **Get JWT Token** in the left sidebar.
2. In the token modal, copy the token.
3. Decode it and confirm ~8h:
   ```bash
   decode_ttl "<paste token>"    # expect ttl_hours=8.00
   ```

### 1b. MCP Configuration modal (the dialog in the screenshot)

1. Open a server card and launch **MCP Configuration** (the "MCP Configuration for ... tools" dialog).
2. It shows *"Token Ready — Copy and Paste! ... Token expires in 8 hours."*
3. Copy the token out of the `X-Authorization: Bearer <...>` line in the Configuration JSON, and decode it:
   ```bash
   decode_ttl "<paste token>"    # expect ttl_hours=8.00
   ```
4. Click **Refresh** and confirm a new token is issued, still ~8h.

### 1c. Equivalent API call (what both buttons do under the hood)

```bash
curl -sS -X POST "$REG/api/tokens/generate" \
  -H "Authorization: Bearer $(cat .token)" \
  -H "Content-Type: application/json" \
  -d '{"expires_in_hours": 8}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print("expires_in(s):", d["tokens"]["expires_in"])'
# expect expires_in(s): 28800
```

**Pass:** all three yield an ~8h (28800s) token.

---

## Test 2 — Selectable TTL via the Generate Token page (up to the max)

**Goal:** confirm the page honors the selected lifetime up to the configured max.

1. Open **Generate Token** in the sidebar.
2. Set **Expires In** to **24 hours**, generate, copy the token.
3. Decode: `decode_ttl "<token>"` → expect `ttl_hours=24.00`.
4. Repeat with **1 hour** → expect `ttl_hours=1.00`.

API equivalent (any value 1..max):
```bash
curl -sS -X POST "$REG/api/tokens/generate" \
  -H "Authorization: Bearer $(cat .token)" -H "Content-Type: application/json" \
  -d '{"expires_in_hours": 12}' | python3 -c 'import sys,json; print("expires_in(s):", json.load(sys.stdin)["tokens"]["expires_in"])'
# expect 43200  (12h) — no config change needed for any value <= max
```

**Pass:** the minted token's TTL equals the requested hours.

---

## Test 3 — Requesting above the max is rejected

**Goal:** confirm the server-side cap.

```bash
curl -sS -o /dev/null -w "%{http_code}\n" -X POST "$REG/api/tokens/generate" \
  -H "Authorization: Bearer $(cat .token)" -H "Content-Type: application/json" \
  -d '{"expires_in_hours": 999}'
# expect 400  (detail: "expires_in_hours must be an integer between 1 and 24")
```

**Pass:** HTTP 400; no token issued.

---

## Test 4 — Raising the default lifts the sidebar + modal tokens

**Goal:** confirm `MCP_TOKEN_DEFAULT_TTL_HOURS` changes the two hardcoded-8h entry points without a UI change.

1. Set in `.env`:
   ```bash
   MCP_TOKEN_DEFAULT_TTL_HOURS=12
   ```
2. Restart the registry and auth-server so both re-read the setting:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.override.yml up -d \
     --no-deps --force-recreate registry auth-server
   ```
3. Repeat Test 1a and 1b (sidebar button + MCP Configuration modal), decode the token → expect `ttl_hours=12.00`.

> The modal's on-screen text still reads "expires in 8 hours" (it is static), but the actual token now carries 12h. Decoding is the source of truth.

**Pass:** both entry points now mint ~12h tokens.

---

## Test 5 — Raising the max allows a longer selectable/requestable TTL

**Goal:** confirm `MCP_TOKEN_MAX_TTL_HOURS` raises the ceiling, and the absolute 7-day cap holds.

1. Set in `.env` and restart registry + auth-server (as in Test 4):
   ```bash
   MCP_TOKEN_MAX_TTL_HOURS=72
   ```
2. A 72h request now succeeds:
   ```bash
   curl -sS -X POST "$REG/api/tokens/generate" \
     -H "Authorization: Bearer $(cat .token)" -H "Content-Type: application/json" \
     -d '{"expires_in_hours": 72}' | python3 -c 'import sys,json; print("expires_in(s):", json.load(sys.stdin)["tokens"]["expires_in"])'
   # expect 259200  (72h)
   ```
3. **Absolute-ceiling guard:** set `MCP_TOKEN_MAX_TTL_HOURS=9999`, restart, and check the registry startup log:
   ```bash
   docker logs mcp-gateway-registry-registry-1 2>&1 | grep -i "MCP token TTL setting"
   # expect: MCP token TTL setting 9999 clamped to 168 (allowed range 1..168 hours)
   ```
   A request for `expires_in_hours: 200` is then still rejected with 400 (effective max is 168, not 9999).

**Pass:** the ceiling rises to the configured value, but never above the hardcoded 168h.

---

## Cleanup

Remove the test overrides from `.env` (or reset to defaults) and restart:
```bash
# delete MCP_TOKEN_DEFAULT_TTL_HOURS / MCP_TOKEN_MAX_TTL_HOURS lines from .env, then:
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d \
  --no-deps --force-recreate registry auth-server
```

## Related

- [How do I generate an MCP access token that lasts longer than 8 hours?](faq/generate-token-longer-than-8-hours.md)
- [Unified parameter reference](unified-parameter-reference.md) (`MCP_TOKEN_DEFAULT_TTL_HOURS`, `MCP_TOKEN_MAX_TTL_HOURS`)
