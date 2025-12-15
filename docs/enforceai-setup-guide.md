# EnforceAI Setup and Usage Guide

This guide explains how to run and use the EnforceAI functionality (identity resolver, agent-scoped FGAC, management API/CLI, and audit retention) in this repository.

It complements:
- `docs/complete-setup-guide.md` (full EC2 + Docker stack)
- `docs/macos-setup-guide.md` (local macOS dev stack)

## What You Get

When EnforceAI is enabled (`ENFORCEAI_DB_PATH` is set), the Auth Server (`auth_server/server.py`) adds:
- **Identity resolution** (Stage 4): OIDC JWT, gateway tokens, API keys (configurable via `ENFORCEAI_AUTH_PROVIDER`).
- **Agent binding**: For OIDC, requests must include `X-Agent-Id: <uuidv4>`; ownership is validated in the EnforceAI DB.
- **FGAC inputs**: an `IdentityContext` with `user_id`, `agent_id`, `scopes`, `provider`, and metadata.
- **Management API** (Stage 6): `/enforceai/*` endpoints for agents, API keys, and token revocation/minting.
- **Audit retention tooling** (Stage 7): operator cleanup via `cli/enforceai_audit_cleanup.py`.

## 1. Prerequisites

Use the same prerequisites as `docs/macos-setup-guide.md` or `docs/complete-setup-guide.md`:
- Python `3.12+`
- `uv`
- `openssl`
- `jq` (recommended for inspecting JSON output)

Create the local virtualenv:

```bash
uv sync
source .venv/bin/activate
```

All commands below assume the virtualenv is active (so `python` points at `.venv/bin/python`).

## 2. One-Command Bootstrap

Run:

```bash
./scripts/enforceai_dev_bootstrap.sh
```

This generates EnforceAI dev state under a local state directory (default: `.enforceai/` in the repo).

For full gateway via Docker Compose (recommended), run:

```bash
ENFORCEAI_STATE_DIR="$HOME/mcp-gateway/enforceai" ./scripts/enforceai_dev_bootstrap.sh
```

The state includes:
- SQLite DB (`enforceai.db`) with migrations applied
- RSA keypair for gateway token mint/verify (`secrets/`)
- API key pepper (`secrets/api_key_pepper`)
- A bootstrap `user_id` and `agent_id`
- A bootstrap gateway token (`bootstrap_gateway_token.txt`)
- A sourceable env file for local host runs (`enforceai.env`)
- A sourceable env file for Docker Compose (`enforceai.compose.env`)

The repo-local `.enforceai/` is gitignored.

If you need to overwrite generated secrets/ids (without deleting the DB), use:

```bash
./scripts/enforceai_dev_bootstrap.sh --force
```

Load the environment in your current shell for local (host) usage:

```bash
source .enforceai/enforceai.env  # or $HOME/mcp-gateway/enforceai/enforceai.env
```

## 3. Full Gateway (Docker Compose)

For full gateway enforcement (nginx + registry + auth-server + MCP servers), use the standard stack:

```bash
./build_and_run.sh --prebuilt
```

Then enable EnforceAI in the `auth-server` container:

```bash
source $HOME/mcp-gateway/enforceai/enforceai.compose.env
docker compose up -d --force-recreate auth-server
```

Verify auth server health:

```bash
curl -s http://127.0.0.1:8888/health | jq .
```

## 4. Use the Gateway with an EnforceAI Token

Use the bootstrap gateway token as the client credential:

```bash
export ENFORCEAI_STATE_DIR="${ENFORCEAI_STATE_DIR:-$HOME/mcp-gateway/enforceai}"
export ENFORCEAI_TOKEN="$(cat $ENFORCEAI_STATE_DIR/bootstrap_gateway_token.txt)"
```

Example: call the gateway MCP endpoint (streamable HTTP):

```bash
curl http://localhost/mcpgw/mcp -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ENFORCEAI_TOKEN" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | jq .
```

Notes:
- `/mcpgw/mcp` is a JSON-RPC endpoint; a browser `GET` is not a valid request (you should see `405`).
- Required headers: `Content-Type: application/json` and `Authorization: Bearer <token>` (the gateway also accepts `X-Authorization` for legacy clients).
- If you see `401`, the token/header is missing or malformed.
- If you see `403`, you authenticated but were denied by FGAC (or the agent is revoked).
- If you see `500` when hitting the gateway, the most common cause is nginx being unable to reach the Auth Server for `auth_request` (e.g., `auth-server` container not running/recreated). Check `docker compose ps` and `docker compose logs registry auth-server`.

Tool visibility enforcement (Stage 5) should filter `tools/list` results based on the token’s scopes and any agent `allowed_tools` restrictions.

## 5. Auth Server Only (Local Development)

Run the Auth Server on localhost:

```bash
uvicorn auth_server.server:app --host 127.0.0.1 --port 8888 --reload
```

Health check:

```bash
curl -s http://127.0.0.1:8888/health | jq .
```

## 6. Use the Management API via the EnforceAI CLI

The EnforceAI CLI wraps the `/enforceai/*` management API:
- `cli/enforceai_cli.py`
- Base URL default: `http://localhost:8888`

### 4.1 Configure CLI environment variables

```bash
export ENFORCEAI_STATE_DIR="${ENFORCEAI_STATE_DIR:-.enforceai}"
export ENFORCEAI_AUTHORIZATION="Bearer $(cat $ENFORCEAI_STATE_DIR/bootstrap_gateway_token.txt)"
```

### 4.2 List agents

```bash
python cli/enforceai_cli.py --pretty agents list
```

### 4.3 Create a new agent (scopes + optional allowed_tools)

```bash
python cli/enforceai_cli.py --pretty agents create \
  --scope registry-users-lob1 \
  --alias "my-agent" \
  --metadata '{"purpose":"demo"}'
```

### 4.4 Mint a gateway token for an agent

Use the `agent_id` returned by `agents create` (or any agent you own):

```bash
python cli/enforceai_cli.py --pretty tokens mint "<agent_id>" \
  --scope registry-users-lob1 \
  --ttl-seconds 3600
```

### 4.5 Create an API key for an agent (requires `ENFORCEAI_API_KEY_PEPPER_PATH`)

Switch to `mixed` mode if you want the Auth Server to accept API keys as credentials:

```bash
export ENFORCEAI_AUTH_PROVIDER="mixed"
```

Restart the Auth Server process so it picks up the new `ENFORCEAI_AUTH_PROVIDER` value.

Then create an API key:

```bash
python cli/enforceai_cli.py --pretty api-keys create "<agent_id>" --scope registry-users-lob1
```

## 7. Optional: OIDC Mode (External IdP)

If you want to use OIDC JWTs, set:
- `ENFORCEAI_AUTH_PROVIDER=oidc` (or `mixed`)
- `OIDC_ISSUERS` to a JSON map keyed by `iss`

Example shape:

```bash
export ENFORCEAI_AUTH_PROVIDER="oidc"
export OIDC_ISSUERS='{
  "https://issuer.example": {
    "jwks_uri": "https://issuer.example/.well-known/jwks.json",
    "audiences": ["mcp-registry"]
  }
}'
```

For OIDC requests:
- Send `Authorization: Bearer <oidc_jwt>`
- Send `X-Agent-Id: <uuidv4>` (required; validated against the EnforceAI agent registry for that `user_id`)

## 8. Audit Retention Cleanup (Operator)

Audit events are emitted to stdout and persisted best-effort to the EnforceAI SQLite DB.

Run cleanup manually:

```bash
export ENFORCEAI_DB_PATH="$(pwd)/.enforceai/enforceai.db"
export ENFORCEAI_AUDIT_RETENTION_DAYS=30
export ENFORCEAI_AUDIT_MAX_DB_BYTES=500000000

python -m cli.enforceai_audit_cleanup
```

Dry run:

```bash
python -m cli.enforceai_audit_cleanup --dry-run
```

## 9. Notes on Docker Compose

`docker-compose.yml` now passes through `ENFORCEAI_*` and mounts `$HOME/mcp-gateway/enforceai` into the `auth-server` container at `/app/enforceai_state` (so it does not shadow the `enforceai` Python package).
