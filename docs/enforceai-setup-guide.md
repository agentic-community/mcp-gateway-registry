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
- `jq` (recommended for inspecting JSON output)

Create the local virtualenv:

```bash
uv sync
source .venv/bin/activate
```

## 2. Choose a Scopes Catalog

EnforceAI loads and validates a scope catalog (`scopes.yml`) using `auth_server/enforceai/fgac/catalog.py`. The simplest starting point is the repo’s default:

- `auth_server/scopes.yml`

For local development, set:

```bash
export ENFORCEAI_SCOPES_CATALOG_PATH="$(pwd)/auth_server/scopes.yml"
```

## 3. Create EnforceAI Local State (DB + Secrets)

Pick a local directory for EnforceAI runtime state:

```bash
mkdir -p .enforceai/secrets/gateway_public_keys
```

### 3.1 EnforceAI DB

EnforceAI uses SQLite for persistence (agents, API keys, revocations, audit events).

```bash
export ENFORCEAI_DB_PATH="$(pwd)/.enforceai/enforceai.db"
```

DB migrations are applied automatically when the Auth Server first loads EnforceAI stores.

### 3.2 Gateway token keys (required for `gateway-token` / `mixed`)

Generate an RSA keypair and a public key file named by `kid`:

```bash
export ENFORCEAI_GATEWAY_ACTIVE_KID="kid-local-1"
export ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH="$(pwd)/.enforceai/secrets/gateway_private.pem"
export ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR="$(pwd)/.enforceai/secrets/gateway_public_keys"
export ENFORCEAI_GATEWAY_ISSUER="enforceai-gateway"

openssl genpkey -algorithm RSA -out "$ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH" -pkeyopt rsa_keygen_bits:2048
openssl pkey -in "$ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH" -pubout -out "$ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR/$ENFORCEAI_GATEWAY_ACTIVE_KID.pem"
```

### 3.3 API key pepper (required for `api-key` / `mixed`)

The pepper is an opaque byte sequence used as part of the API key secret hashing scheme.

```bash
export ENFORCEAI_API_KEY_PEPPER_PATH="$(pwd)/.enforceai/secrets/api_key_pepper"
python -c 'import secrets; print(secrets.token_hex(32))' > "$ENFORCEAI_API_KEY_PEPPER_PATH"
chmod 600 "$ENFORCEAI_API_KEY_PEPPER_PATH"
```

## 4. Pick an EnforceAI Auth Mode

Set `ENFORCEAI_AUTH_PROVIDER` to one of:
- `gateway-token`: only gateway tokens (via `Authorization: Bearer <token>` or `X-Gateway-Token`)
- `api-key`: only API keys (via `X-API-Key: eak_<key_id>.<secret>`)
- `oidc`: only OIDC JWTs (via `Authorization: Bearer <jwt>` plus `X-Agent-Id`)
- `mixed`: accept API keys, gateway tokens, and OIDC bearer tokens; bearer routing uses token `iss`

For a fully local development setup without an external IdP, start with `gateway-token`:

```bash
export ENFORCEAI_AUTH_PROVIDER="gateway-token"
```

## 5. Bootstrap: Create the First Agent (Out-of-Band)

Management endpoints are ownership-scoped by `user_id`, and OIDC requests also require an existing agent binding. This means you must create the initial agent record out-of-band once per user.

For a fully local setup (no IdP), choose a local user id in EnforceAI canonical format:

```bash
export ENFORCEAI_BOOTSTRAP_USER_ID="local|admin"
export ENFORCEAI_BOOTSTRAP_AGENT_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
```

Create the initial agent record using the EnforceAI data layer (runs migrations if needed):

```bash
uv run python - <<'PY'
from pathlib import Path
import os

from auth_server.enforceai.db.data_layer import EnforceAIDataLayer

db_path = Path(os.environ["ENFORCEAI_DB_PATH"])
user_id = os.environ["ENFORCEAI_BOOTSTRAP_USER_ID"]
agent_id = os.environ["ENFORCEAI_BOOTSTRAP_AGENT_ID"]

data_layer = EnforceAIDataLayer(db_path=db_path)
data_layer.initialize()
stores = data_layer.build_stores()

stores.agent_store.create_agent(
    user_id=user_id,
    agent_id=agent_id,
    scopes=["registry-admins"],
    allowed_tools=None,
    alias="bootstrap",
    metadata={"bootstrap": True},
)

print(f"Bootstrapped agent_id={agent_id} for user_id={user_id}")
PY
```

## 6. Bootstrap: Mint a Gateway Token (Out-of-Band)

Mint a gateway token that you can use to call the management API:

```bash
export ENFORCEAI_BOOTSTRAP_TOKEN="$(uv run python - <<'PY'
from pathlib import Path
import os

from auth_server.enforceai.crypto.keyring import GatewayKeyring
from auth_server.enforceai.tokens.mint import mint_gateway_token

keyring = GatewayKeyring.load(
    private_key_path=Path(os.environ["ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH"]),
    public_keys_dir=Path(os.environ["ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR"]),
    active_kid=os.environ["ENFORCEAI_GATEWAY_ACTIVE_KID"],
)

token = mint_gateway_token(
    keyring=keyring,
    issuer=os.environ["ENFORCEAI_GATEWAY_ISSUER"],
    user_id=os.environ["ENFORCEAI_BOOTSTRAP_USER_ID"],
    agent_id=os.environ["ENFORCEAI_BOOTSTRAP_AGENT_ID"],
    scopes=["registry-admins"],
    ttl_seconds=3600,
)

print(token)
PY)"
```

## 7. Run the Auth Server (Local)

Run the Auth Server on localhost:

```bash
uvicorn auth_server.server:app --host 127.0.0.1 --port 8888 --reload
```

Health check:

```bash
curl -s http://127.0.0.1:8888/health | jq .
```

## 8. Use the Management API via the EnforceAI CLI

The EnforceAI CLI wraps the `/enforceai/*` management API:
- `cli/enforceai_cli.py`
- Base URL default: `http://localhost:8888`

### 8.1 Configure CLI environment variables

```bash
export ENFORCEAI_AUTH_SERVER_URL="http://127.0.0.1:8888"
export ENFORCEAI_AUTHORIZATION="Bearer $ENFORCEAI_BOOTSTRAP_TOKEN"
```

### 8.2 List agents

```bash
uv run python cli/enforceai_cli.py --pretty agents list
```

### 8.3 Create a new agent (scopes + optional allowed_tools)

```bash
uv run python cli/enforceai_cli.py --pretty agents create \
  --scope registry-users-lob1 \
  --alias "my-agent" \
  --metadata '{"purpose":"demo"}'
```

### 8.4 Mint a gateway token for an agent

Use the `agent_id` returned by `agents create` (or any agent you own):

```bash
uv run python cli/enforceai_cli.py --pretty tokens mint "<agent_id>" \
  --scope registry-users-lob1 \
  --ttl-seconds 3600
```

### 8.5 Create an API key for an agent (requires `ENFORCEAI_API_KEY_PEPPER_PATH`)

Switch to `mixed` mode if you want the Auth Server to accept API keys as credentials:

```bash
export ENFORCEAI_AUTH_PROVIDER="mixed"
```

Restart the Auth Server process so it picks up the new `ENFORCEAI_AUTH_PROVIDER` value.

Then create an API key:

```bash
uv run python cli/enforceai_cli.py --pretty api-keys create "<agent_id>" --scope registry-users-lob1
```

## 9. Optional: OIDC Mode (External IdP)

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

## 10. Audit Retention Cleanup (Operator)

Audit events are emitted to stdout and persisted best-effort to the EnforceAI SQLite DB.

Run cleanup manually:

```bash
export ENFORCEAI_DB_PATH="$(pwd)/.enforceai/enforceai.db"
export ENFORCEAI_AUDIT_RETENTION_DAYS=30
export ENFORCEAI_AUDIT_MAX_DB_BYTES=500000000

uv run python -m cli.enforceai_audit_cleanup
```

Dry run:

```bash
uv run python -m cli.enforceai_audit_cleanup --dry-run
```

## 11. Running in Docker Compose (Optional)

The default `docker-compose.yml` does not set EnforceAI environment variables for the `auth-server` service. For local experimentation, create a `docker-compose.override.yml` (do not commit secrets) that:
- Adds `ENFORCEAI_*` env vars to `auth-server`
- Mounts the EnforceAI DB and secret files into the container
- Sets `ENFORCEAI_SCOPES_CATALOG_PATH` to the mounted `scopes.yml` path

Start the stack with:

```bash
docker compose up -d
```
