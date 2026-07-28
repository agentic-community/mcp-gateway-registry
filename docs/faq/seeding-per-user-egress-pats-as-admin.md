# How does an admin seed per-user egress PATs for a `pat` server?

For third-party MCP servers that authenticate with a static Personal Access Token / API key (not OAuth), the gateway supports a **`pat` egress mode**: each user's PAT is vaulted per-user and injected on egress, so the token never lives on the user's laptop. Normally each user submits their own PAT on the **Connected Accounts** page. This FAQ covers the admin path: **an admin seeding a PAT on another user's behalf** through the API, so you can script bulk provisioning without every user self-submitting.

For the full egress model and where `pat` fits among the modes, see [Per-User Egress Auth](../design/egress-auth-design.md#the-egress-modes) and the [Per-User Egress Credential Vault](../egress-credential-vault.md).

## Prerequisites

- Egress auth is enabled on the deployment (`EGRESS_AUTH_ENABLED=true`).
- The target server is registered and configured for `pat` mode (see below).
- You have an **admin** token (a bearer JWT with the admin scope). The commands below load it from a `.token` file via `--token-file`; substitute your own path.

## 1. Register the server and set it to `pat` mode

Register the server with a **Backend Authentication** scheme that matches how the upstream expects the token, because `pat` inherits the inject header from it:

- `auth_scheme: "bearer"` → the PAT is injected as `Authorization: Bearer <PAT>` (GitHub, most APIs).
- `auth_scheme: "api_key"` with an `auth_header_name` → the PAT is injected as `<header>: <PAT>` (bare, no prefix; e.g. GitLab `PRIVATE-TOKEN`).

Example registration config (`server.json`):

```json
{
  "server_name": "GitHub MCP (PAT)",
  "path": "/github-pat",
  "proxy_pass_url": "https://api.githubcopilot.com/mcp/",
  "auth_scheme": "bearer",
  "supported_transports": ["streamable-http"],
  "status": "active",
  "visibility": "public"
}
```

Register it, then switch egress to `pat`:

```bash
# Register the server
uv run python api/registry_management.py \
  --registry-url http://localhost --token-file .token \
  register --config server.json

# Configure egress auth = pat. The provider slug is just a vault-namespace /
# display key (lowercase letters, digits, hyphen, underscore); it does NOT need
# to match an OAuth provider. There is no header config here -- the inject
# header is inherited from the server's Backend Authentication above.
uv run python api/registry_management.py \
  --registry-url http://localhost --token-file .token \
  egress-configure --path /github-pat --mode pat --provider github-pat
```

## 2. Seed a PAT for a specific user (admin on-behalf)

The API is `PUT /api/servers/{server_path}/egress-pat`. An admin supplies two extra fields to target another user:

- `sub` — the target user's **canonical egress id**. This is the user's OIDC `sub` (stable per user + gateway), NOT their display name. See [Finding a user's `sub` and `auth_method`](#finding-a-users-sub-and-auth_method).
- `auth_method` — the target user's **ingress auth method** (how they log into the gateway, e.g. `oauth2`). This is the first segment of the vault key `(auth_method, user_id, provider, server_path)`, so it must match what the target vends with, or the PAT lands in a partition the target never reads. It is **required** with `sub`; omitting it is rejected (`400`), and a non-admin supplying `sub` is rejected (`403`).

### CLI

```bash
uv run python api/registry_management.py \
  --registry-url http://localhost --token-file .token \
  egress-pat-set \
    --path /github-pat \
    --secret "ghp_theusers_personal_access_token" \
    --ttl-value 30 --ttl-unit days \
    --sub "<target-user-oidc-sub>" \
    --auth-method oauth2
```

The TTL is **mandatory and bounded** (`minutes` | `hours` | `days`, capped at 30 days); there is no "never expires". The response reports `configured`, `sub`, and `expires_at` — it never echoes the secret.

### Direct HTTP (curl)

```bash
curl -sS -X PUT \
  "http://localhost/api/servers/github-pat/egress-pat" \
  -H "Authorization: Bearer $(python3 -c "import json;print(json.load(open('.token'))['tokens']['access_token'])")" \
  -H "Content-Type: application/json" \
  -d '{
    "secret": "ghp_theusers_personal_access_token",
    "ttl_value": 30,
    "ttl_unit": "days",
    "sub": "<target-user-oidc-sub>",
    "auth_method": "oauth2"
  }'
```

### Python (registry client)

```python
from api.registry_client import RegistryClient

client = RegistryClient(registry_url="http://localhost", token=admin_token)

client.set_egress_pat(
    server_path="/github-pat",
    secret="ghp_theusers_personal_access_token",
    ttl_value=30,
    ttl_unit="days",
    sub="<target-user-oidc-sub>",   # admin-only on-behalf
    auth_method="oauth2",            # required with sub
)
```

## 3. Seed PATs for many users (copy-paste loop)

If everyone logs in through the same IdP, `auth_method` is the same for all of them (commonly `oauth2`). Put the per-user rows in a CSV `sub,pat` and loop the CLI:

```bash
#!/usr/bin/env bash
# users.csv format (no header): <oidc-sub>,<pat>
#   4b1c...e9,ghp_aaa...
#   9f22...01,ghp_bbb...
set -euo pipefail

REGISTRY_URL="http://localhost"
TOKEN_FILE=".token"
SERVER_PATH="/github-pat"
AUTH_METHOD="oauth2"     # the target users' ingress auth method
TTL_VALUE=30
TTL_UNIT="days"

while IFS=, read -r SUB PAT; do
  [ -z "${SUB}" ] && continue
  echo "Seeding PAT for ${SUB} on ${SERVER_PATH}"
  uv run python api/registry_management.py \
    --registry-url "${REGISTRY_URL}" --token-file "${TOKEN_FILE}" \
    egress-pat-set \
      --path "${SERVER_PATH}" \
      --secret "${PAT}" \
      --ttl-value "${TTL_VALUE}" --ttl-unit "${TTL_UNIT}" \
      --sub "${SUB}" \
      --auth-method "${AUTH_METHOD}"
done < users.csv
```

> Secrets on the command line are visible in the process list (`ps`) and shell history. For a real bulk run, read each PAT from a file or a secrets manager rather than embedding it in the CSV, and avoid logging the value.

## 4. Verify and delete

Status is presence + expiry only (never the secret). Both take the same admin `--sub` + `--auth-method` on-behalf arguments:

```bash
# Is a PAT vaulted for this user, and when does it expire?
uv run python api/registry_management.py \
  --registry-url http://localhost --token-file .token \
  egress-pat-status --path /github-pat --sub "<target-user-oidc-sub>" --auth-method oauth2

# Delete a user's PAT (idempotent)
uv run python api/registry_management.py \
  --registry-url http://localhost --token-file .token \
  egress-pat-delete --path /github-pat --sub "<target-user-oidc-sub>" --auth-method oauth2
```

## Finding a user's `sub` and `auth_method`

- **`auth_method`** is the ingress auth method the user logs in with. For a single-IdP deployment this is one value for everyone (e.g. `oauth2` for Keycloak/Cognito). It is NOT the egress mode.
- **`sub`** is the user's canonical egress id — the OIDC `sub` claim, which is stable per user and gateway. It is what the vend path keys the vault on. List users with:

  ```bash
  uv run python api/registry_management.py \
    --registry-url http://localhost --token-file .token \
    user-list --search "<name-or-email>"
  ```

  When in doubt, have the target user self-submit once from the **Connected Accounts** page (which resolves their `sub` automatically) and confirm the vend works; after that, admin on-behalf updates use the same `sub`.

## How the injected credential is chosen

At vend time the gateway derives the header from the server's Backend Authentication (one upstream, one header contract):

| Backend `auth_scheme` | Injected header |
|-----------------------|-----------------|
| `bearer` | `<auth_header_name or "Authorization">: Bearer <PAT>` |
| `api_key` | `<auth_header_name or "X-API-Key">: <PAT>` (bare, no prefix) |
| `none` / unset | `Authorization: Bearer <PAT>` (safe default) |

So configure the header **once**, in Backend Authentication, and `pat` egress inherits it. A common mistake is storing a secret that already includes the `Bearer` prefix — for a `bearer` server that produces `Authorization: Bearer Bearer <PAT>` and the upstream returns `400`. Store the **raw** token; the gateway adds the prefix.

## Security notes

- The PAT is **write-only**: no endpoint ever returns or logs the secret value.
- The lifetime is **mandatory and bounded** (≤ 30 days) and re-checked at vend, so an expired PAT is a miss (fail-closed) and the user is prompted to re-submit.
- Only an **admin** may seed on another user's behalf, and only with an explicit target `auth_method`; there is no silent fall-back to the admin's own identity.
- Mutations (submit, delete) are CSRF-protected; a bearer/CLI caller satisfies this automatically.
