# PingFederate Setup Guide

This guide walks through configuring PingFederate as the identity provider for the MCP Gateway Registry.

There are three deployment modes. **Docker Compose** is the only mode where the registry can also start a PingFederate container for you (under a profile gate, for local development). For **Helm** and **Terraform/ECS**, the customer brings their own PingFederate instance and supplies its endpoint and credentials via configuration. The application code is identical across all three modes; only the deployment surface differs.

## Prerequisites

- A PingFederate 11.x or later instance with OIDC enabled.
- For local Docker Compose dev only: a free Ping Identity DevOps account from [developer.pingidentity.com/devops](https://developer.pingidentity.com/devops/how-to/devopsRegistration.html) and Docker Compose v2.24+.
- The OAuth client and JWT Access Token Manager (ATM) configured in PingFederate. See [Admin Console Configuration](#admin-console-configuration) below; this part is the same regardless of deployment mode.

## Variable reference

The same set of variables is set across all three deployment modes; only the file you edit and the variable name casing differ. Every variable maps 1:1 across surfaces.

| What it is | `.env` (Docker Compose) | Terraform variable | Helm value |
|---|---|---|---|
| Active provider switch | `AUTH_PROVIDER=pingfederate` | `pingfederate_enabled = true` (no `auth_provider` variable; see note) | `global.authProvider.type: pingfederate` |
| Show login button | `PINGFEDERATE_ENABLED=true` | `pingfederate_enabled = true` | `pingfederate.enabled: true` |
| Server-to-server URL (auth-server reaches PF) | `PINGFEDERATE_BASE_URL` | `pingfederate_base_url` | `pingfederate.baseUrl` |
| Browser-facing URL (used in redirects) | `PINGFEDERATE_EXTERNAL_URL` | `pingfederate_external_url` | `pingfederate.externalUrl` |
| OAuth client (web login) ID | `PINGFEDERATE_CLIENT_ID` | `pingfederate_client_id` | `pingfederate.clientId` |
| OAuth client secret (web login, **secret**) | `PINGFEDERATE_CLIENT_SECRET` | `pingfederate_client_secret` | `pingfederate.clientSecret` (or `pingfederate.clientSecretExistingSecret`) |
| OAuth client ID for M2M tokens (optional) | `PINGFEDERATE_M2M_CLIENT_ID` | `pingfederate_m2m_client_id` | `pingfederate.m2mClientId` |
| OAuth client secret for M2M (**secret**) | `PINGFEDERATE_M2M_CLIENT_SECRET` | `pingfederate_m2m_client_secret` | `pingfederate.m2mClientSecret` (or `pingfederate.m2mClientSecretExistingSecret`) |
| Static audience override (optional) | `PINGFEDERATE_APPLICATION_ID_URI` | `pingfederate_application_id_uri` | `pingfederate.applicationIdUri` |
| JWT claim name for groups | `PINGFEDERATE_GROUPS_CLAIM` | `pingfederate_groups_claim` | `pingfederate.groupsClaim` |
| Admin API URL (registry creates clients/users) | `PF_ADMIN_URL` | `pf_admin_url` | `pingfederateAdmin.url` |
| Admin API username | `PF_ADMIN_USER` | `pf_admin_user` | `pingfederateAdmin.user` |
| Admin API password (**secret**) | `PF_ADMIN_PASS` | `pf_admin_pass` | `pingfederateAdmin.password` (or `pingfederateAdmin.passwordExistingSecret`) |
| User-to-group fallback allowlist | `IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS=pingfederate` | `idp_user_group_fallback_enabled_providers = "pingfederate"` | `idpUserGroupFallbackEnabledProviders: "pingfederate"` (registry + auth-server) |

Note on the provider switch: Docker (`AUTH_PROVIDER`) and Helm (`authProvider.type`) take a provider-name string. The Terraform module has no `auth_provider` variable; you enable exactly one provider by setting its boolean `*_enabled` flag to true (leave the others false), and the module derives the `AUTH_PROVIDER` value for the containers. If you enable none, Keycloak is the default. So for Terraform, `pingfederate_enabled = true` is the only switch needed.

The "secret" rows must be sourced from a secrets store in production (AWS Secrets Manager for Terraform, a Kubernetes Secret for Helm via `*ExistingSecret`). Don't paste secrets into `terraform.tfvars` or `values.yaml` checked into git.

For full cross-surface parameter reference, see [docs/unified-parameter-reference.md](../unified-parameter-reference.md).

## Mode 1: Docker Compose (registry can run PingFederate locally)

This is the only mode where the registry stack can spin up a PingFederate container for you. PingFederate is gated behind a Docker Compose profile so it doesn't run unless you opt in.

### Step 1: Set DevOps credentials and PingFederate config in `.env`

```bash
# Required for the bundled PingFederate container to fetch its trial license
PING_IDENTITY_ACCEPT_EULA=YES
PING_IDENTITY_DEVOPS_USER=you@example.com
PING_IDENTITY_DEVOPS_KEY=<your-uuid-key>

# Tell the registry to use PingFederate
AUTH_PROVIDER=pingfederate
PINGFEDERATE_ENABLED=true

# Endpoints (the bundled container exposes these)
PINGFEDERATE_BASE_URL=https://pingfederate:9031
PINGFEDERATE_EXTERNAL_URL=https://localhost:9031

# OAuth client created by init-pingfederate.sh
PINGFEDERATE_CLIENT_ID=mcp-gateway
PINGFEDERATE_CLIENT_SECRET=<picked by you, also in init script>

# Admin API (defaults match the bundled container)
PF_ADMIN_URL=https://pingfederate:9999
PF_ADMIN_USER=administrator
PF_ADMIN_PASS=2FederateM0re

# User-to-group fallback enabled for PingFederate
IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS=pingfederate
```

### Step 2: Start the stack with the PingFederate profile

```bash
docker compose --profile pingfederate up -d
```

This brings up the PingFederate container alongside the registry, auth-server, and the rest. Without `--profile pingfederate`, the PingFederate container does not start.

### Step 3: Bootstrap PingFederate

Wait 2-3 minutes for PingFederate to come up (license activation + profile init), then run the bootstrap script. It creates the OAuth client, the `groups` scope, the JWT ATM, two test users (admin and testuser), and seeds the registry's `idp_user_groups` collection with their group mappings.

```bash
bash pingfederate/setup/init-pingfederate.sh
```

After this you can log in to the registry at `https://localhost` with `admin / admin123` (admin) or `testuser / changeme` (read-only).

### Step 4 (optional): switch to BYO PingFederate

If you have an external PingFederate you'd rather point at, simply omit `--profile pingfederate` from `docker compose up` and update `PINGFEDERATE_BASE_URL`, `PINGFEDERATE_EXTERNAL_URL`, `PF_ADMIN_URL`, `PINGFEDERATE_CLIENT_ID`, `PINGFEDERATE_CLIENT_SECRET`, and `PF_ADMIN_PASS` to your own values. You'll then create the OAuth client and ATM in your PingFederate admin console manually, as described in [Admin Console Configuration](#admin-console-configuration) below.

## Mode 2: Terraform / AWS ECS (BYO PingFederate)

The Terraform module does NOT start a PingFederate task. The customer is expected to operate their own PingFederate (on-prem, in another VPC, in a separate ECS cluster, etc.) and provide its endpoint and credentials. The Terraform module wires those values into the registry and auth-server task definitions, and the secrets through AWS Secrets Manager.

### Step 1: Create the OAuth client and ATM in your PingFederate

Follow [Admin Console Configuration](#admin-console-configuration) below. Note the client ID, client secret, M2M client ID/secret, and the URLs of your PingFederate runtime and admin endpoints.

### Step 2: Set the variables in `terraform.tfvars`

Use a separate non-committed `*.auto.tfvars` file (e.g. `secrets.auto.tfvars`) for the secret values, OR populate the AWS Secrets Manager secret values out-of-band via `aws secretsmanager update-secret` after `terraform apply` — the resources have `lifecycle { ignore_changes = [secret_string] }` so future plans won't drift.

Note: the Terraform module has no `auth_provider` variable (that name is the
Docker `.env` / Helm `values.yaml` switch). On this surface the provider is
selected by the boolean flag below; setting `pingfederate_enabled = true` is
what makes the module emit `AUTH_PROVIDER=pingfederate` to the containers.

```hcl
# terraform.tfvars
pingfederate_enabled           = true

# Endpoints — both should point at your PingFederate
pingfederate_base_url          = "https://pf.internal.example.com:9031"
pingfederate_external_url      = "https://pf.example.com"

# OAuth client for web login
pingfederate_client_id         = "mcp-gateway"
pingfederate_client_secret     = "<set in secrets.auto.tfvars or via aws secretsmanager>"

# Optional separate M2M client (defaults to web client)
pingfederate_m2m_client_id     = "mcp-gateway-m2m"
pingfederate_m2m_client_secret = "<set in secrets.auto.tfvars or via aws secretsmanager>"

# JWT shape
pingfederate_groups_claim      = "groups"

# Admin API for the registry to create OAuth clients and PCV users from the UI
pf_admin_url                   = "https://pf-admin.internal.example.com:9999"
pf_admin_user                  = "<service-account-username>"
pf_admin_pass                  = "<set in secrets.auto.tfvars or via aws secretsmanager>"

# User-to-group fallback (registry consults idp_user_groups collection
# when JWT carries no groups claim)
idp_user_group_fallback_enabled_providers = "pingfederate"
```

### Step 3: Apply

```bash
terraform plan
terraform apply
```

Three Secrets Manager entries are created (`pingfederate_client_secret`, `pingfederate_m2m_client_secret`, `pf_admin_pass`) and the registry/auth-server tasks read them via `valueFrom` at boot. The plain (non-secret) values are wired as regular environment variables.

### Step 4: bootstrap your PingFederate (if not already done)

The Terraform module does not run `init-pingfederate.sh` against your PingFederate. Run the OAuth client and ATM creation steps once in your PingFederate admin console (see [Admin Console Configuration](#admin-console-configuration)). For the registry's `idp_user_groups` user-to-group fallback rows, use the registry's IAM > User Groups page after the stack is up.

## Mode 3: Helm / Kubernetes (BYO PingFederate)

The Helm chart does NOT include a PingFederate Pod or StatefulSet. The customer brings their own PingFederate (running in the same cluster, a separate cluster, or off-cluster) and supplies its endpoint and credentials.

### Step 1: Create the OAuth client and ATM in your PingFederate

Same as Terraform mode — follow [Admin Console Configuration](#admin-console-configuration) once in your PingFederate admin console.

### Step 2: Create the Kubernetes secrets

Either let the chart manage secrets via `values.yaml` (fine for dev), or pre-create your own k8s Secrets and point the chart at them via `*ExistingSecret` keys (recommended for production):

```bash
kubectl create secret generic pingfederate-credentials \
  --from-literal=PINGFEDERATE_CLIENT_SECRET='<your secret>' \
  --from-literal=PINGFEDERATE_M2M_CLIENT_SECRET='<your m2m secret>' \
  --from-literal=PF_ADMIN_PASS='<your admin password>'
```

### Step 3: Configure values

```yaml
# values.yaml override
global:
  authProvider:
    type: pingfederate

# OIDC configuration (registry + auth-server both read these)
pingfederate:
  enabled: true
  baseUrl: "https://pf.internal.example.com:9031"
  externalUrl: "https://pf.example.com"
  clientId: "mcp-gateway"
  clientSecretExistingSecret: "pingfederate-credentials"     # or set clientSecret directly
  clientSecretExistingSecretKey: "PINGFEDERATE_CLIENT_SECRET"
  m2mClientId: "mcp-gateway-m2m"
  m2mClientSecretExistingSecret: "pingfederate-credentials"
  m2mClientSecretExistingSecretKey: "PINGFEDERATE_M2M_CLIENT_SECRET"
  groupsClaim: "groups"

# Admin API (registry only — auth-server doesn't read these)
pingfederateAdmin:
  url: "https://pf-admin.internal.example.com:9999"
  user: "<service-account-username>"
  passwordExistingSecret: "pingfederate-credentials"
  passwordExistingSecretKey: "PF_ADMIN_PASS"

# User-to-group fallback allowlist
idpUserGroupFallbackEnabledProviders: "pingfederate"
```

For the umbrella stack chart (`charts/mcp-gateway-registry-stack`), the same keys live under both `registry:` and `auth-server:` stanzas. The two stanzas must agree on `idpUserGroupFallbackEnabledProviders` — the chart's parent `values.yaml` includes a comment reminding you of this.

### Step 4: Install

```bash
helm install mcp-gateway-registry charts/mcp-gateway-registry-stack \
  --namespace mcp-gateway --create-namespace \
  -f values.yaml
```

### Step 5: bootstrap your PingFederate (if not already done)

Just like the Terraform mode — the Helm chart does not run `init-pingfederate.sh`. Configure your PingFederate's OAuth client and ATM once via the admin console, then use the registry's IAM > User Groups page to populate `idp_user_groups` records as needed.

## Admin Console Configuration

These are the one-time steps you (or `init-pingfederate.sh` in dev mode) must perform inside your PingFederate admin console. They are the same across all three deployment modes.

### 1. Create OAuth Client

1. Navigate to **Applications > OAuth > Clients**.
2. Click **Add Client**.
3. Configure:
   - **Client ID:** `mcp-gateway` (or whatever you set `PINGFEDERATE_CLIENT_ID` to)
   - **Client Authentication:** Client Secret
   - **Client Secret:** generate, save into your secret store
   - **Redirect URIs:**
     - Local Docker Compose: `http://localhost:8888/oauth2/callback/pingfederate`
     - Production: `https://your-gateway.example.com/oauth2/callback/pingfederate`
   - **Allowed Grant Types:** Authorization Code, Client Credentials, Refresh Token
   - **Scopes:** `openid`, `email`, `profile`, `groups` (create the `groups` scope if it doesn't exist; see step 2)

### 2. Create the `groups` scope

PingFederate has no built-in `groups` scope. Create one:

1. Navigate to **OAuth Settings > Scopes**.
2. Add scope: `groups`.
3. Description: `Access to group memberships`.

### 3. Configure the JWT Access Token Manager (ATM)

1. Navigate to **Applications > OAuth > Access Token Management**.
2. Select your JWT ATM instance (or create one).
3. Under **Attribute Contract**, add:
   - **Attribute Name:** `groups`
   - **Multi-valued:** Yes
4. Under **Attribute Mapping**, map `groups` to a source:
   - LDAP: `memberOf`
   - JDBC: your groups query
   - Or use the PingFederate expression language

### 4. Wire the OIDC Policy

1. Navigate to **Applications > OAuth > OpenID Connect Policy**.
2. Under **Attribute Contract**, ensure `groups` is mapped.
3. Source it from the ATM's `groups` attribute.

### 5. (Optional) Create an M2M-only client

If you want a separate client for service-to-service tokens (so service-token revocations don't affect web logins), create a second client with grant type **Client Credentials only**, scopes `openid groups`, and set `PINGFEDERATE_M2M_CLIENT_ID` / `PINGFEDERATE_M2M_CLIENT_SECRET` accordingly. If you skip this, M2M tokens use the web client.

## What to do when JWTs come back with empty groups

PingFederate's default user store (Simple PCV) has no groups concept. Even after you do step 3 above, JWTs may come back with `groups: []` because the user has no group memberships in the source directory.

The registry handles this with a fallback: it consults the `idp_user_groups` MongoDB collection (per-username group mappings) when an enabled IdP returns an empty groups claim. Set `IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS=pingfederate` (already the default) to enable the fallback for PingFederate. Then use the registry's **IAM > User Groups** page to map usernames to groups.

For production deployments using LDAP or AD as the user store, you can populate groups inside PingFederate directly via the ATM's attribute mapping (step 3 above) and bypass the fallback. See your PingFederate documentation on Password Credential Validators for the full set of user-store options (LDAP, AD, JDBC, PingDirectory, etc.).

## TLS / Self-Signed Certificate Handling (local dev only)

The PingFederate dev container uses a self-signed certificate on port 9031. The auth-server's PingFederate provider does NOT support `verify=False`; instead, mount a CA bundle.

```bash
# Extract the dev container's certificate
openssl s_client -connect localhost:9031 -showcerts < /dev/null 2>/dev/null \
  | openssl x509 -outform PEM > pf-cert.pem

# In .env (Docker Compose mounts this into auth-server)
REQUESTS_CA_BUNDLE=/path/to/pf-cert.pem
```

`init-pingfederate.sh` does this automatically; the bundle ends up at `pingfederate/setup/pingfederate-ca-bundle.pem` and `docker-compose.yml` mounts it into the auth-server container.

For production with a properly-signed certificate (Let's Encrypt, internal PKI, etc.), no special handling is needed.

## M2M / Client Credentials Flow

```bash
curl -X POST https://pf.example.com/as/token.oauth2 \
  -d "grant_type=client_credentials" \
  -d "client_id=mcp-gateway-m2m" \
  -d "client_secret=<secret>" \
  -d "scope=openid groups"
```

Use the resulting JWT as `X-Authorization: Bearer <token>` against the gateway.

## Troubleshooting

### Empty groups in JWT

**Symptom:** Auth-server logs `PingFederate token has no 'groups' claim for sub=...`

**Fix:** Either complete step 3 above (ATM extended attribute contract for `groups`), OR rely on the `idp_user_groups` MongoDB fallback (set up via the IAM > User Groups page in the registry). For production, do step 3.

### Discovery fetch timeout

**Symptom:** `OpenID configuration retrieval failed: ...timeout`

**Fix:** `PINGFEDERATE_BASE_URL` must be reachable from inside the auth-server container. In Docker Compose, use the service name (`https://pingfederate:9031`), not `localhost`. In Kubernetes, use the in-cluster DNS name. In ECS, use the Service Connect name or a VPC-routable DNS name.

### Redirect URI mismatch

**Symptom:** `invalid_request: redirect_uri does not match`

**Fix:** The redirect URI registered in PingFederate must exactly match `<protocol>://<auth-server-host>/oauth2/callback/pingfederate`. Confirm the protocol (http vs https), the host, and the path.

### License activation failure (Docker Compose only)

**Symptom:** PingFederate container exits with license-related errors.

**Fix:** Verify `PING_IDENTITY_DEVOPS_USER` and `PING_IDENTITY_DEVOPS_KEY` are set correctly in `.env`. The container needs internet access to fetch the trial license on first start. Once activated, the `pingfederate-data` volume persists the license across restarts.

### `400 Invalid client_id` on login

**Symptom:** PingFederate's login screen shows `400 - Invalid client_id`.

**Fix:** The OAuth client doesn't exist yet, or its `client_id` doesn't match `PINGFEDERATE_CLIENT_ID`. Run `bash pingfederate/setup/init-pingfederate.sh` (Docker Compose dev mode) or create the client in your PingFederate admin console (BYO modes).

### Logout returns the registry SPA shell instead of the PingFederate "Sign Off Successful" page

**Symptom:** Clicking logout in the registry redirects to a URL like `https://your-gw/idp/startSLO.ping?...` and the page shows the registry SPA instead of PF's signoff page.

**Fix:** The nginx `/idp/` proxy block is gated on `AUTH_PROVIDER=pingfederate`. If you set `AUTH_PROVIDER=pingfederate` and rebuilt, but logout still doesn't work, verify:
1. The registry container was rebuilt without cache after the AUTH_PROVIDER change.
2. `docker exec mcp-gateway-registry-registry-1 grep -c "location /idp/" /app/docker/nginx_rev_proxy_http_and_https.conf` returns `2` (or `1` for HTTP-only).
