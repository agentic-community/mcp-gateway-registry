# EnforceAI Management (Stage 6)

This document describes the self-service management surface for EnforceAI:
- Management HTTP APIs served by `auth_server`
- CLI usage (`cli/enforceai_cli.py`)
- Required configuration (environment variables + secret file paths)
- Operational model (“bootstrap”)

## What Management Does
Users can manage their own EnforceAI resources:
- Agents: create/list/get/update/revoke
- API keys: create/list/revoke (secret returned once on creation)
- Gateway tokens: mint; revoke by `jti`; revoke-all via `tokens_valid_after`

All operations are strictly ownership-scoped by `user_id`.

## Required Configuration (Auth Server)
EnforceAI management endpoints are enabled when `ENFORCEAI_DB_PATH` is set.

### Core
- `ENFORCEAI_DB_PATH`: SQLite DB path for EnforceAI state (agents, keys, revocations, audit).
- `ENFORCEAI_SCOPES_CATALOG_PATH` (or `SCOPES_CATALOG_PATH`): path to the `scopes.yml` catalog used for validating scopes.

### Optional (Upstream OAuth Credential Storage)
- `ENFORCEAI_UPSTREAM_KEK_PATH`: path to a hex-encoded 32-byte KEK file used to encrypt stored upstream OAuth client secrets at rest.

### Auth Mode
Pick one:
- `ENFORCEAI_AUTH_PROVIDER=oidc`
- `ENFORCEAI_AUTH_PROVIDER=gateway-token`
- `ENFORCEAI_AUTH_PROVIDER=api-key`
- `ENFORCEAI_AUTH_PROVIDER=mixed`

#### OIDC (`ENFORCEAI_AUTH_PROVIDER=oidc` or `mixed`)
- `OIDC_ISSUERS`: JSON mapping keyed by `iss` with issuer config.
  - Minimum keys per issuer:
    - `jwks_uri` (or `jwks_url`)
    - `audiences` (or `audience`)

#### API keys (`ENFORCEAI_AUTH_PROVIDER=api-key` or `mixed`)
- `ENFORCEAI_API_KEY_PEPPER_PATH`: file containing pepper bytes for hashing API key secrets.

#### Gateway tokens (minting, and auth when `gateway-token`/`mixed`)
- `ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH`: PEM private key used for minting tokens.
- `ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR`: directory containing `<kid>.pem` public keys.
- `ENFORCEAI_GATEWAY_ACTIVE_KID`: active key id (must match a `<kid>.pem` filename).
- `ENFORCEAI_GATEWAY_ISSUER`: gateway token `iss` claim to mint/accept.

## Management HTTP API (Auth Server)
Base prefix: `/enforceai`

- `GET /enforceai/agents`
- `POST /enforceai/agents`
- `GET /enforceai/agents/{agent_id}`
- `PATCH /enforceai/agents/{agent_id}`
- `POST /enforceai/agents/{agent_id}/revoke`
- `POST /enforceai/agents/{agent_id}/tokens/revoke-all`
- `POST /enforceai/agents/{agent_id}/api-keys`
- `GET /enforceai/agents/{agent_id}/api-keys`
- `POST /enforceai/api-keys/{key_id}/revoke`
- `POST /enforceai/agents/{agent_id}/tokens/mint`
- `POST /enforceai/tokens/revoke`

### Credentials
All endpoints require a valid identity resolved by the Stage 4 resolver:
- OIDC: `Authorization: Bearer <oidc_jwt>` and `X-Agent-Id: <uuidv4>`
- Gateway token: `Authorization: Bearer <gateway_token>` or `X-Gateway-Token: <gateway_token>`
- API key: `X-API-Key: eak_<key_id>.<secret>`

Exactly one credential source must be provided per request.

## CLI (Stage 6.3)
CLI lives at `cli/enforceai_cli.py`.

### CLI environment variables
- `ENFORCEAI_AUTH_SERVER_URL` (default: `http://localhost:8888`)
- `ENFORCEAI_AUTHORIZATION` (for OIDC or bearer gateway tokens)
- `ENFORCEAI_X_AGENT_ID` (required for OIDC)
- `ENFORCEAI_X_GATEWAY_TOKEN` (alternative to `ENFORCEAI_AUTHORIZATION` for gateway tokens)
- `ENFORCEAI_X_API_KEY`

### Examples

List agents (OIDC):
```bash
uv run python cli/enforceai_cli.py \
  --base-url http://localhost:8888 \
  --authorization "$ENFORCEAI_AUTHORIZATION" \
  --x-agent-id "$ENFORCEAI_X_AGENT_ID" \
  agents list
```

Create an agent:
```bash
uv run python cli/enforceai_cli.py \
  --authorization "$ENFORCEAI_AUTHORIZATION" \
  --x-agent-id "$ENFORCEAI_X_AGENT_ID" \
  agents create --scope scope-mgmt --alias my-agent
```

Create an API key for an agent (secret returned once):
```bash
uv run python cli/enforceai_cli.py \
  --authorization "$ENFORCEAI_AUTHORIZATION" \
  --x-agent-id "$ENFORCEAI_X_AGENT_ID" \
  api-keys create <agent_id> --scope scope-mgmt
```

Mint a gateway token for an agent:
```bash
uv run python cli/enforceai_cli.py \
  --authorization "$ENFORCEAI_AUTHORIZATION" \
  --x-agent-id "$ENFORCEAI_X_AGENT_ID" \
  tokens mint <agent_id> --scope scope-mgmt --ttl-seconds 3600
```

Revoke all tokens for an agent:
```bash
uv run python cli/enforceai_cli.py \
  --authorization "$ENFORCEAI_AUTHORIZATION" \
  --x-agent-id "$ENFORCEAI_X_AGENT_ID" \
  tokens revoke-all <agent_id>
```

## Operational Model (“Bootstrap”)
Management is self-service and requires at least one existing credential to authenticate.

Recommended bootstrap approach:
1. Create an initial agent record for the user out-of-band (admin bootstrap or one-time provisioning step).
2. Use that agent’s credential (OIDC + `X-Agent-Id`, gateway token, or API key) to call management APIs.
3. Create additional agents and credentials as needed.

This avoids introducing a separate “admin” identity surface in Phase 1 while keeping all management strictly ownership-scoped.

## Audit Retention (Stage 7)
Audit events are emitted to stdout and persisted best-effort to SQLite. Retention cleanup is out-of-band via `cli/enforceai_audit_cleanup.py`.

See `enforceai/instructions/ENFORCEAI_AUDIT_RETENTION.md`.
