# How do I register an M2M client and assign it groups without an IdP Admin API token?

**Short answer**: use the direct M2M client registration API at `/api/iam/m2m-clients`. Create the M2M client in your IdP (Keycloak, Okta, Entra, Auth0) as you normally would, then register its `client_id` with the registry and assign groups. The registry writes directly to its own `idp_m2m_clients` collection -- no `OKTA_API_TOKEN` or equivalent IdP Admin API credentials required.

## When to use this

- Your enterprise gates IdP Admin API tokens (e.g. Okta requires approval for Admin API access) and getting one is disproportionate overhead.
- You already know the M2M `client_id` you want to register (it lives in the IdP).
- You want to assign groups so the registry's auth server can enrich M2M tokens with those groups during authorization.

If `OKTA_API_TOKEN` / `AUTH0_M2M_CLIENT_ID` etc. are available, the existing `/api/iam/okta/m2m/*` or `/api/iam/auth0/m2m/*` sync endpoints cover the same ground. This FAQ is for the case where those credentials are not available.

## Prerequisites

- A user JWT (or static API token) with **admin** scope on the registry.
- The `client_id` of an M2M client you have already created in your IdP.
- `M2M_DIRECT_REGISTRATION_ENABLED=true` on the registry (this is the default).

## Step 1: Create the M2M client in your IdP

In Keycloak Admin UI (example; equivalent steps apply in Okta/Entra/Auth0):

1. Navigate to **Clients > Create client**.
2. Client type: **OpenID Connect**. Client ID: e.g. `my-automation-pipeline`. **Save**.
3. Enable **Client authentication** and **Service accounts roles**. Disable standard/direct flows. **Save**.
4. Copy the **Client Secret** from the `Credentials` tab. Your application will use this pair to request tokens from Keycloak.

You do **not** need to assign groups inside the IdP. The registry resolves groups from its own `idp_m2m_clients` collection, which you will populate in the next step.

## Step 2: Register the client with the registry

Using the bundled CLI (`api/registry_management.py`):

```bash
export REGISTRY_URL=http://localhost
export TOKEN_FILE=~/repos/mcp-gateway-registry/.token   # admin user token

uv run python api/registry_management.py \
  --registry-url $REGISTRY_URL --token-file $TOKEN_FILE \
  m2m-client-create \
  --client-id my-automation-pipeline \
  --client-name "My Automation Pipeline" \
  --groups pipeline-operators,registry-readonly \
  --description "CI/CD pipeline service account"
```

Expected output:

```
M2M client registered successfully

Client ID:    my-automation-pipeline
Name:         My Automation Pipeline
Provider:     manual
Enabled:      True
Groups:       pipeline-operators, registry-readonly
Description:  CI/CD pipeline service account
Created by:   admin
Created at:   ...
Updated at:   ...
```

`Provider: manual` means the record was created via this API (rather than synced from an IdP). Manual records are the only ones this API can modify or delete later.

## Step 3: Verify from your application

1. Request an M2M access token from Keycloak using client credentials:

   ```bash
   curl -X POST \
     "http://localhost:8080/realms/mcp-gateway/protocol/openid-connect/token" \
     -d "grant_type=client_credentials" \
     -d "client_id=my-automation-pipeline" \
     -d "client_secret=<from-keycloak-ui>"
   ```

2. Call the registry with that token. The registry's auth server looks up `client_id=my-automation-pipeline` in `idp_m2m_clients`, enriches the token with groups `["pipeline-operators", "registry-readonly"]`, and authorization proceeds based on those groups.

   ```bash
   curl -H "Authorization: Bearer $M2M_TOKEN" "$REGISTRY_URL/api/servers"
   ```

## Managing registered clients

All commands below use the same `--registry-url`/`--token-file` prefix as above.

**List**:

```bash
uv run python api/registry_management.py \
  --registry-url $REGISTRY_URL --token-file $TOKEN_FILE \
  m2m-client-list --provider manual
```

Supports `--limit`, `--skip`, `--json`.

**Get one**:

```bash
uv run python api/registry_management.py \
  --registry-url $REGISTRY_URL --token-file $TOKEN_FILE \
  m2m-client-get --client-id my-automation-pipeline
```

**Update** (partial -- fields you omit are left unchanged; pass `--groups ""` to clear groups):

```bash
# Change groups only
uv run python api/registry_management.py \
  --registry-url $REGISTRY_URL --token-file $TOKEN_FILE \
  m2m-client-update \
  --client-id my-automation-pipeline \
  --groups registry-readonly

# Disable the client (kill switch)
uv run python api/registry_management.py \
  --registry-url $REGISTRY_URL --token-file $TOKEN_FILE \
  m2m-client-update \
  --client-id my-automation-pipeline \
  --enabled false
```

**Delete**:

```bash
uv run python api/registry_management.py \
  --registry-url $REGISTRY_URL --token-file $TOKEN_FILE \
  m2m-client-delete --client-id my-automation-pipeline --force
```

## HTTP endpoints (for reference)

| Method | Path | Auth |
|--------|------|------|
| POST | `/api/iam/m2m-clients` | admin |
| GET | `/api/iam/m2m-clients` | any authenticated user (paginated) |
| GET | `/api/iam/m2m-clients/{client_id}` | any authenticated user |
| PATCH | `/api/iam/m2m-clients/{client_id}` | admin |
| DELETE | `/api/iam/m2m-clients/{client_id}` | admin |

## Things to know

- **Ownership guard**: records created by this API have `provider: "manual"`. Records written by the existing Okta/Auth0 sync services (`provider: "okta"`, `provider: "auth0"`) are visible via `GET` but return `HTTP 403` on `PATCH` or `DELETE` from this API, to prevent conflicts with IdP sync.
- **Duplicate `client_id`**: returns `HTTP 409 Conflict`. One `client_id` can only have one record across all providers.
- **Admins grant privilege directly**: any admin calling this API can assign any group to any `client_id`. The audit log records every mutation with the calling admin's identity for accountability. Treat the registry admin role accordingly.
- **Feature flag**: `M2M_DIRECT_REGISTRATION_ENABLED` (default `true`) disables the whole router if set to `false`. Surface the flag on the System Config page under **Authentication**.

## Related FAQs

- [How do I restrict which MCP servers a user can see based on their Entra ID group?](restrict-server-visibility-by-entra-group.md)
- [Can I use an Entra ID token to call the registry API instead of the UI-generated token?](use-entra-token-for-registry-api.md)
- [How do I register and manage MCP servers that require authentication?](registering-auth-protected-servers.md)
