#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STATE_DIR_DEFAULT="$ROOT_DIR/.enforceai"
STATE_DIR="${ENFORCEAI_STATE_DIR:-$STATE_DIR_DEFAULT}"

MODE_DEFAULT="gateway-token"
MODE="${ENFORCEAI_AUTH_PROVIDER:-$MODE_DEFAULT}"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  shift
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  scripts/enforceai_dev_bootstrap.sh [--force]

Creates a local EnforceAI dev environment:
  - EnforceAI SQLite DB
  - gateway token RSA keys (kid-based rotation layout)
  - API key pepper
  - bootstrap user_id + agent_id
  - bootstrap gateway token
  - a sourceable env file: .enforceai/enforceai.env

Options:
  --force  Overwrite existing generated files where safe (does not delete DB).

Environment overrides:
  ENFORCEAI_STATE_DIR          Root directory for generated state (default: ./.enforceai)
  ENFORCEAI_AUTH_PROVIDER      gateway-token | api-key | oidc | mixed (default: gateway-token)
  ENFORCEAI_DB_PATH            Path to SQLite DB (default: $ENFORCEAI_STATE_DIR/enforceai.db)
  ENFORCEAI_SCOPES_CATALOG_PATH Path to scopes.yml (default: ./auth_server/scopes.yml)

Notes:
  - This is a dev convenience script. Do not use it for production provisioning.
  - mixed/oidc modes require OIDC_ISSUERS to be set separately.
  - For full gateway via docker-compose, set ENFORCEAI_STATE_DIR=$HOME/mcp-gateway/enforceai.
EOF
  exit 0
fi

PYTHON_BIN="${ENFORCEAI_PYTHON:-}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]] && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] No Python interpreter found." >&2
  echo "Run 'uv sync' to create .venv, or set ENFORCEAI_PYTHON=/path/to/python." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python interpreter is not executable: $PYTHON_BIN" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[WARN] 'uv' not found. Continuing with $PYTHON_BIN (expected: repo deps already installed)." >&2
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "[ERROR] 'openssl' not found. Install openssl first." >&2
  exit 2
fi

SCOPES_CATALOG_DEFAULT="$ROOT_DIR/auth_server/scopes.yml"
SCOPES_CATALOG_PATH="${ENFORCEAI_SCOPES_CATALOG_PATH:-$SCOPES_CATALOG_DEFAULT}"

DB_PATH_DEFAULT="$STATE_DIR/enforceai.db"
DB_PATH="${ENFORCEAI_DB_PATH:-$DB_PATH_DEFAULT}"

SECRETS_DIR="$STATE_DIR/secrets"
GATEWAY_PUBLIC_KEYS_DIR_DEFAULT="$SECRETS_DIR/gateway_public_keys"
GATEWAY_PUBLIC_KEYS_DIR="${ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR:-$GATEWAY_PUBLIC_KEYS_DIR_DEFAULT}"
GATEWAY_PRIVATE_KEY_PATH_DEFAULT="$SECRETS_DIR/gateway_private.pem"
GATEWAY_PRIVATE_KEY_PATH="${ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH:-$GATEWAY_PRIVATE_KEY_PATH_DEFAULT}"
GATEWAY_ACTIVE_KID="${ENFORCEAI_GATEWAY_ACTIVE_KID:-kid-local-1}"
GATEWAY_ISSUER="${ENFORCEAI_GATEWAY_ISSUER:-enforceai-gateway}"

API_KEY_PEPPER_PATH_DEFAULT="$SECRETS_DIR/api_key_pepper"
API_KEY_PEPPER_PATH="${ENFORCEAI_API_KEY_PEPPER_PATH:-$API_KEY_PEPPER_PATH_DEFAULT}"

BOOTSTRAP_USER_ID_PATH="$STATE_DIR/bootstrap_user_id"
BOOTSTRAP_AGENT_ID_PATH="$STATE_DIR/bootstrap_agent_id"
BOOTSTRAP_TOKEN_PATH="$STATE_DIR/bootstrap_gateway_token.txt"

ENV_FILE_PATH="$STATE_DIR/enforceai.env"
COMPOSE_ENV_FILE_PATH="$STATE_DIR/enforceai.compose.env"

# Export the runtime config so the inline Python snippets can read it.
export ENFORCEAI_DB_PATH="$DB_PATH"
export ENFORCEAI_SCOPES_CATALOG_PATH="$SCOPES_CATALOG_PATH"
export ENFORCEAI_AUTH_PROVIDER="$MODE"

export ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH="$GATEWAY_PRIVATE_KEY_PATH"
export ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR="$GATEWAY_PUBLIC_KEYS_DIR"
export ENFORCEAI_GATEWAY_ACTIVE_KID="$GATEWAY_ACTIVE_KID"
export ENFORCEAI_GATEWAY_ISSUER="$GATEWAY_ISSUER"

export ENFORCEAI_API_KEY_PEPPER_PATH="$API_KEY_PEPPER_PATH"

echo "------------------------------------------------------"
echo " EnforceAI Dev Bootstrap"
echo "------------------------------------------------------"
echo ""
echo "Root dir:       $ROOT_DIR"
echo "State dir:      $STATE_DIR"
echo "Auth mode:      $MODE"
echo "Scopes catalog: $SCOPES_CATALOG_PATH"
echo "DB path:        $DB_PATH"
echo ""

mkdir -p "$STATE_DIR" "$SECRETS_DIR" "$GATEWAY_PUBLIC_KEYS_DIR"
chmod 700 "$STATE_DIR" "$SECRETS_DIR" "$GATEWAY_PUBLIC_KEYS_DIR" || true

if [[ ! -f "$SCOPES_CATALOG_PATH" ]]; then
  echo "[ERROR] scopes catalog not found: $SCOPES_CATALOG_PATH" >&2
  exit 2
fi

if [[ ! -f "$BOOTSTRAP_USER_ID_PATH" || $FORCE -eq 1 ]]; then
  echo "local|admin" >"$BOOTSTRAP_USER_ID_PATH"
  chmod 600 "$BOOTSTRAP_USER_ID_PATH" || true
  echo "[OK] bootstrap user_id file: $BOOTSTRAP_USER_ID_PATH"
else
  echo "[OK] bootstrap user_id file exists: $BOOTSTRAP_USER_ID_PATH"
fi
BOOTSTRAP_USER_ID="$(cat "$BOOTSTRAP_USER_ID_PATH")"
export ENFORCEAI_BOOTSTRAP_USER_ID="$BOOTSTRAP_USER_ID"

if [[ ! -f "$BOOTSTRAP_AGENT_ID_PATH" || $FORCE -eq 1 ]]; then
  "$PYTHON_BIN" - <<'PY' >"$BOOTSTRAP_AGENT_ID_PATH"
import uuid
print(uuid.uuid4())
PY
  chmod 600 "$BOOTSTRAP_AGENT_ID_PATH" || true
  echo "[OK] bootstrap agent_id file: $BOOTSTRAP_AGENT_ID_PATH"
else
  echo "[OK] bootstrap agent_id file exists: $BOOTSTRAP_AGENT_ID_PATH"
fi
BOOTSTRAP_AGENT_ID="$(cat "$BOOTSTRAP_AGENT_ID_PATH")"
export ENFORCEAI_BOOTSTRAP_AGENT_ID="$BOOTSTRAP_AGENT_ID"

if [[ ! -f "$API_KEY_PEPPER_PATH" || $FORCE -eq 1 ]]; then
  "$PYTHON_BIN" - <<'PY' >"$API_KEY_PEPPER_PATH"
import secrets
print(secrets.token_hex(32))
PY
  chmod 600 "$API_KEY_PEPPER_PATH" || true
  echo "[OK] API key pepper: $API_KEY_PEPPER_PATH"
else
  echo "[OK] API key pepper exists: $API_KEY_PEPPER_PATH"
fi

PUBLIC_KEY_PATH="$GATEWAY_PUBLIC_KEYS_DIR/$GATEWAY_ACTIVE_KID.pem"
if [[ ! -f "$GATEWAY_PRIVATE_KEY_PATH" || ! -f "$PUBLIC_KEY_PATH" || $FORCE -eq 1 ]]; then
  if [[ -f "$GATEWAY_PRIVATE_KEY_PATH" && $FORCE -eq 0 ]]; then
    echo "[SKIP] gateway private key exists: $GATEWAY_PRIVATE_KEY_PATH"
  else
    openssl genpkey \
      -algorithm RSA \
      -out "$GATEWAY_PRIVATE_KEY_PATH" \
      -pkeyopt rsa_keygen_bits:2048 \
      >/dev/null 2>&1
    chmod 600 "$GATEWAY_PRIVATE_KEY_PATH" || true
    echo "[OK] gateway private key: $GATEWAY_PRIVATE_KEY_PATH"
  fi

  if [[ -f "$PUBLIC_KEY_PATH" && $FORCE -eq 0 ]]; then
    echo "[SKIP] gateway public key exists: $PUBLIC_KEY_PATH"
  else
    openssl pkey \
      -in "$GATEWAY_PRIVATE_KEY_PATH" \
      -pubout \
      -out "$PUBLIC_KEY_PATH" \
      >/dev/null 2>&1
    chmod 644 "$PUBLIC_KEY_PATH" || true
    echo "[OK] gateway public key: $PUBLIC_KEY_PATH"
  fi
else
  echo "[OK] gateway keys exist (kid=$GATEWAY_ACTIVE_KID)"
fi

cat >"$ENV_FILE_PATH" <<EOF
export ENFORCEAI_DB_PATH="$DB_PATH"
export ENFORCEAI_SCOPES_CATALOG_PATH="$SCOPES_CATALOG_PATH"
export ENFORCEAI_AUTH_PROVIDER="$MODE"

export ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH="$GATEWAY_PRIVATE_KEY_PATH"
export ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR="$GATEWAY_PUBLIC_KEYS_DIR"
export ENFORCEAI_GATEWAY_ACTIVE_KID="$GATEWAY_ACTIVE_KID"
export ENFORCEAI_GATEWAY_ISSUER="$GATEWAY_ISSUER"

export ENFORCEAI_API_KEY_PEPPER_PATH="$API_KEY_PEPPER_PATH"

export ENFORCEAI_BOOTSTRAP_USER_ID="$BOOTSTRAP_USER_ID"
export ENFORCEAI_BOOTSTRAP_AGENT_ID="$BOOTSTRAP_AGENT_ID"

export ENFORCEAI_AUTH_SERVER_URL="http://127.0.0.1:8888"
EOF
chmod 600 "$ENV_FILE_PATH" || true
echo "[OK] wrote env file: $ENV_FILE_PATH"

"$PYTHON_BIN" - <<'PY' >"$COMPOSE_ENV_FILE_PATH"
import os

mode = os.environ.get("ENFORCEAI_AUTH_PROVIDER", "gateway-token")

print('export ENFORCEAI_AUTH_PROVIDER="' + mode + '"')
print('export ENFORCEAI_DB_PATH="/app/enforceai_state/enforceai.db"')
print('export ENFORCEAI_SCOPES_CATALOG_PATH="/app/scopes.yml"')
print('export ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH="/app/enforceai_state/secrets/gateway_private.pem"')
print('export ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR="/app/enforceai_state/secrets/gateway_public_keys"')
print('export ENFORCEAI_GATEWAY_ACTIVE_KID="' + os.environ.get("ENFORCEAI_GATEWAY_ACTIVE_KID", "kid-local-1") + '"')
print('export ENFORCEAI_GATEWAY_ISSUER="' + os.environ.get("ENFORCEAI_GATEWAY_ISSUER", "enforceai-gateway") + '"')
print('export ENFORCEAI_API_KEY_PEPPER_PATH="/app/enforceai_state/secrets/api_key_pepper"')
print('export ENFORCEAI_AUDIT_RETENTION_DAYS="' + os.environ.get("ENFORCEAI_AUDIT_RETENTION_DAYS", "30") + '"')
print('export ENFORCEAI_AUDIT_MAX_DB_BYTES="' + os.environ.get("ENFORCEAI_AUDIT_MAX_DB_BYTES", "500000000") + '"')
print('export OIDC_ISSUERS="' + os.environ.get("OIDC_ISSUERS", "{}") + '"')
PY
chmod 600 "$COMPOSE_ENV_FILE_PATH" || true
echo "[OK] wrote compose env file: $COMPOSE_ENV_FILE_PATH"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

from auth_server.enforceai.db.data_layer import EnforceAIDataLayer

db_path = Path(os.environ["ENFORCEAI_DB_PATH"])
user_id = os.environ["ENFORCEAI_BOOTSTRAP_USER_ID"]
agent_id = os.environ["ENFORCEAI_BOOTSTRAP_AGENT_ID"]

data_layer = EnforceAIDataLayer(db_path=db_path)
data_layer.initialize()
stores = data_layer.build_stores()

existing = stores.agent_store.get_agent_by_id(agent_id=agent_id)
if existing is None:
    stores.agent_store.create_agent(
        user_id=user_id,
        agent_id=agent_id,
        scopes=["registry-admins"],
        allowed_tools=None,
        alias="bootstrap",
        metadata={"bootstrap": True},
    )
    print(f"[OK] bootstrapped agent_id={agent_id} user_id={user_id}")
else:
    print(f"[OK] agent already exists agent_id={agent_id} user_id={existing.user_id}")
PY

"$PYTHON_BIN" - <<'PY' >"$BOOTSTRAP_TOKEN_PATH"
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
    ttl_seconds=2592000,  # 30 days
)

print(token)
PY
chmod 600 "$BOOTSTRAP_TOKEN_PATH" || true
echo "[OK] wrote bootstrap token: $BOOTSTRAP_TOKEN_PATH"

echo ""
echo "Next:"
echo "Local (run auth server on host):"
echo "  1) source \"$ENV_FILE_PATH\""
echo "  2) uvicorn auth_server.server:app --host 127.0.0.1 --port 8888 --reload"
echo "  3) export ENFORCEAI_AUTHORIZATION=\"Bearer \$(cat '$BOOTSTRAP_TOKEN_PATH')\""
echo "  4) \"$PYTHON_BIN\" cli/enforceai_cli.py --pretty agents list"
echo ""
echo "Docker Compose (full gateway):"
echo "  1) source \"$COMPOSE_ENV_FILE_PATH\""
echo "  2) docker compose up -d --force-recreate auth-server"
