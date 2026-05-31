# PingFederate Setup Guide

This guide walks through configuring PingFederate as the identity provider for MCP Gateway Registry.

## Prerequisites

- PingFederate 11.x or later with OIDC enabled
- For local development: a free Ping Identity DevOps account from
  [developer.pingidentity.com/devops](https://developer.pingidentity.com/devops/how-to/devopsRegistration.html)
- Docker Compose v2.24+ (for the `profiles:` field)

## Local Container Quickstart

Start PingFederate locally alongside the gateway:

```bash
# Set your Ping DevOps credentials in .env
PING_IDENTITY_ACCEPT_EULA=YES
PING_IDENTITY_DEVOPS_USER=you@example.com
PING_IDENTITY_DEVOPS_KEY=<your-uuid-key>

# Start with the pingfederate profile
docker compose --profile pingfederate up -d
```

PingFederate will be available at:
- Runtime (OIDC): `https://localhost:9031`
- Admin console: `https://localhost:9999/pingfederate/app`
  - Default admin credentials: `administrator` / `2FederateM0re`

First startup takes 2-3 minutes for license activation and profile initialization.

## Admin Console Configuration

### 1. Create OAuth Client

1. Navigate to **Applications > OAuth > Clients**
2. Click **Add Client**
3. Configure:
   - **Client ID:** `mcp-gateway`
   - **Client Authentication:** Client Secret
   - **Client Secret:** (generate and save for `.env`)
   - **Redirect URIs:**
     - Local dev: `http://localhost:8888/oauth2/callback/pingfederate`
     - Production: `https://your-gateway.example.com:8888/oauth2/callback/pingfederate`
   - **Allowed Grant Types:** Authorization Code, Client Credentials, Refresh Token
   - **Scopes:** openid, email, profile, groups (create `groups` if it doesn't exist)

### 2. Configure Custom Groups Scope

PingFederate has no built-in `groups` scope. You must create one:

1. Navigate to **OAuth Settings > Scopes**
2. Add scope: `groups`
3. Description: "Access to group memberships"

### 3. Configure JWT Access Token Manager (ATM)

1. Navigate to **Applications > OAuth > Access Token Management**
2. Select your JWT ATM instance (or create one)
3. Under **Attribute Contract**, add:
   - **Attribute Name:** `groups`
   - **Multi-valued:** Yes
4. Under **Attribute Mapping**, map `groups` to a source:
   - LDAP: `memberOf` attribute
   - JDBC: your groups query
   - Expression: PingFederate expression language

### 4. Wire the OIDC Policy

1. Navigate to **Applications > OAuth > OpenID Connect Policy**
2. Under **Attribute Contract**, ensure `groups` is mapped
3. Source it from the ATM's `groups` attribute

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AUTH_PROVIDER` | Yes | Set to `pingfederate` |
| `PINGFEDERATE_BASE_URL` | Yes | Internal URL (server-to-server), e.g. `http://pingfederate:9031` |
| `PINGFEDERATE_EXTERNAL_URL` | Yes | Browser-facing URL, e.g. `https://localhost:9031` |
| `PINGFEDERATE_CLIENT_ID` | Yes | OAuth client ID (`mcp-gateway`) |
| `PINGFEDERATE_CLIENT_SECRET` | Yes | OAuth client secret |
| `PINGFEDERATE_M2M_CLIENT_ID` | No | Separate M2M client (defaults to web client) |
| `PINGFEDERATE_M2M_CLIENT_SECRET` | No | Separate M2M secret (defaults to web secret) |
| `PINGFEDERATE_APPLICATION_ID_URI` | No | Static audience value if ATM is configured with one |
| `PINGFEDERATE_GROUPS_CLAIM` | No | JWT claim name for groups (default: `groups`) |
| `PINGFEDERATE_ENABLED` | No | Show login button (default: `false`) |

## Docker Compose Deployment

```bash
# In .env:
AUTH_PROVIDER=pingfederate
PINGFEDERATE_BASE_URL=http://pingfederate:9031
PINGFEDERATE_EXTERNAL_URL=https://localhost:9031
PINGFEDERATE_CLIENT_ID=mcp-gateway
PINGFEDERATE_CLIENT_SECRET=<your-secret>
PINGFEDERATE_ENABLED=true

# Start (with local PF container):
docker compose --profile pingfederate up -d
```

## Terraform / ECS Deployment

```hcl
# In terraform.tfvars:
pingfederate_enabled            = true
pingfederate_base_url           = "https://pf.internal.example.com:9031"
pingfederate_external_url       = "https://pf.example.com:9031"
pingfederate_client_id          = "mcp-gateway"
pingfederate_client_secret      = "your-secret"  # use .auto.tfvars (not committed)
pingfederate_groups_claim       = "groups"
```

Secrets (`client_secret`, `m2m_client_secret`) are stored in AWS Secrets Manager and injected via `valueFrom`.

## Helm / Kubernetes Deployment

```yaml
# values.yaml override:
authProvider:
  type: pingfederate

pingfederate:
  enabled: true
  baseUrl: "https://pf.internal.example.com:9031"
  externalUrl: "https://pf.example.com:9031"
  clientId: "mcp-gateway"
  clientSecretExistingSecret: "pingfederate-credentials"
  groupsClaim: "groups"
```

## TLS / Self-Signed Certificate Handling

The PingFederate dev container uses a self-signed certificate on port 9031.

**Correct approach:**

```bash
# Extract the dev container's certificate
openssl s_client -connect localhost:9031 -showcerts < /dev/null 2>/dev/null \
  | openssl x509 -outform PEM > pf-cert.pem

# Mount it into the auth-server container and set:
REQUESTS_CA_BUNDLE=/path/to/pf-cert.pem
```

**Do not** disable TLS verification (e.g. with `PYTHONHTTPSVERIFY=0` or by patching the provider). The provider intentionally exposes no `verify=False` knob. If you cannot make verification work, file a ticket.

## M2M / Client Credentials Flow

```bash
curl -X POST https://pf.example.com:9031/as/token.oauth2 \
  -d "grant_type=client_credentials" \
  -d "client_id=mcp-gateway" \
  -d "client_secret=<secret>" \
  -d "scope=openid"
```

The resulting JWT can be used with `X-Authorization: Bearer <token>` against the gateway.

## Troubleshooting

### Empty groups in JWT

**Symptom:** Users log in but have no group memberships; the auth-server logs:
`PingFederate token has no 'groups' claim for sub=...`

**Fix:** Configure the JWT ATM extended attribute contract to include a `groups` attribute, and map it to your user store's group membership attribute. See "Configure JWT Access Token Manager" above.

### Discovery fetch timeout

**Symptom:** Auth-server logs `OpenID configuration retrieval failed: ...timeout`

**Fix:** Verify `PINGFEDERATE_BASE_URL` is reachable from the auth-server container. In Docker, use the service name (`http://pingfederate:9031`), not `localhost`.

### Redirect URI mismatch

**Symptom:** PingFederate returns `invalid_request: redirect_uri does not match`

**Fix:** The redirect URI registered in PingFederate must exactly match `http(s)://<auth-server-host>:8888/oauth2/callback/pingfederate`. Check both protocol and port.

### License activation failure

**Symptom:** PingFederate container exits with license-related errors

**Fix:** Verify `PING_IDENTITY_DEVOPS_USER` and `PING_IDENTITY_DEVOPS_KEY` are set correctly in `.env`. The container needs internet access to fetch the trial license on first start.
