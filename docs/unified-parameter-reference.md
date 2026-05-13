# Unified Configuration Parameter Reference

*Created: 2026-05-10*
*Status: Living document — update on every PR that adds, renames, or removes a configuration parameter. See also [`configuration.md`](configuration.md) for full parameter semantics.*

---

## Purpose

The MCP Gateway Registry is configured identically across **three deployment surfaces**. The *same logical parameter* carries a different name, lives in a different file, and is verified with a different command depending on how you deploy. This document maps every parameter across all three surfaces so that:

- Operators can find the right variable name for their deployment.
- Reviewers can confirm a new parameter was wired through all three surfaces.
- The naming drift between `SCREAMING_SNAKE_CASE`, `snake_case`, and `camelCase` is visible and intentional, not accidental.

---

## The Three Deployment Surfaces

| # | Surface | Primary Naming | Where you SET the parameter | Where you VERIFY the parameter was applied |
|---|---------|----------------|-----------------------------|--------------------------------------------|
| 1 | **Docker / docker-compose** (local dev + single-host prod) | `SCREAMING_SNAKE_CASE` env vars | `.env` (copy from [`.env.example`](../.env.example)). Also wired into: [`docker-compose.yml`](../docker-compose.yml), [`docker-compose.podman.yml`](../docker-compose.podman.yml), [`docker-compose.prebuilt.yml`](../docker-compose.prebuilt.yml), [`docker-compose.dhi.yml`](../docker-compose.dhi.yml) | `docker exec <container> env \| grep <VAR>`, and the UI at **Settings → System Config → Configuration** (also [`GET /api/config/full`](../registry/api/config_routes.py)) |
| 2 | **Terraform / AWS ECS** (managed AWS deployment) | `snake_case` Terraform variables | `terraform/aws-ecs/terraform.tfvars` (copy from [`terraform.tfvars.example`](../terraform/aws-ecs/terraform.tfvars.example)). Also wired into: [`variables.tf`](../terraform/aws-ecs/variables.tf), [`main.tf`](../terraform/aws-ecs/main.tf), [`modules/mcp-gateway/variables.tf`](../terraform/aws-ecs/modules/mcp-gateway/variables.tf), [`modules/mcp-gateway/ecs-services.tf`](../terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf) | `terraform plan`, ECS console task-definition env block, CloudWatch logs, or hit `/api/config/full` on the deployed ALB |
| 3 | **Helm / Kubernetes (EKS)** | `camelCase` Helm values | [`charts/mcp-gateway-registry-stack/values.yaml`](../charts/mcp-gateway-registry-stack/values.yaml) for the full stack, or per-subchart: [`charts/registry/values.yaml`](../charts/registry/values.yaml), [`charts/auth-server/values.yaml`](../charts/auth-server/values.yaml), [`charts/mcpgw/values.yaml`](../charts/mcpgw/values.yaml), [`charts/mongodb-configure/values.yaml`](../charts/mongodb-configure/values.yaml), [`charts/keycloak-configure/values.yaml`](../charts/keycloak-configure/values.yaml). Container env wired in each chart's `templates/deployment.yaml` and `templates/secret.yaml`. | `helm template charts/mcp-gateway-registry-stack \| grep <VAR>`, `kubectl describe pod <pod> \| grep <VAR>`, or hit `/api/config/full` on the ingress |

### Verifying configuration via the API

The registry already exposes a configuration dump endpoint — we do **not** need to build one for this effort:

| Endpoint | Access | Purpose |
|----------|--------|---------|
| `GET /api/config` | Any authenticated caller (Bearer JWT or cookie session). Nginx gates `/api/*` via `auth_request /validate`. | Minimal config: deployment mode, feature flags |
| `GET /api/config/full` | Admin only, **cookie session only** — Bearer JWT does not currently work even for admin users (see note below) | Grouped full configuration. Sensitive values masked. Wired in [`registry/api/config_routes.py`](../registry/api/config_routes.py) |
| `GET /api/config/export?format={env\|json\|tfvars\|yaml}` | Admin only, **cookie session only** — same limitation as `/api/config/full` | Export config in the surface-native format. Use `include_sensitive=true` with caution. |

These endpoints are the **authoritative source** for what the running registry actually sees — always use them to verify before filing a "config not applied" bug.

#### Current limitation: Bearer token does not work on `/api/config/full` or `/api/config/export` (2026-05-10)

Verified against a deployed registry (`https://<registry-host>`) using a signed JWT from the "Generate JWT Token" UI flow (a self-signed admin token with `is_admin=true`, the same token family `registry_management.py` uses):

| Endpoint | Auth | Result |
|----------|------|--------|
| `GET /api/config` | `Authorization: Bearer <jwt>` | **200 OK** — returns deployment_mode, registry_mode, nginx_updates_enabled, registration_gate_enabled, asset_lifecycle_statuses, features |
| `GET /api/auth/me` | `Authorization: Bearer <jwt>` | **200 OK** — confirms `is_admin: true`, admin groups, admin scopes |
| `GET /api/config/full` | `Authorization: Bearer <jwt>` | **401** `{"detail":"Authentication required"}` — despite token being admin |
| `GET /api/config/export?format=env` | `Authorization: Bearer <jwt>` | **401** `{"detail":"Authentication required"}` — same |

Root cause: the 401 text matches the session-cookie-only helpers in [`registry/auth/dependencies.py:30,74`](../registry/auth/dependencies.py) (`get_current_user` and `get_user_session_data`), not the 403 "Admin access required" that the handler itself would raise. That means the `Depends(enhanced_auth)` chain for these two endpoints resolves into a cookie-session-only path on this deployment and never reaches the `is_admin` check.

**Workarounds for now:**
- Use the UI at **Settings → System Config → Configuration** (logged in as admin via browser).
- Or `curl -b cookies.txt` after logging into the UI and saving the session cookie.
- Or hit `/api/config` (works with Bearer) for the subset of fields exposed there.

**Cookie-based curl attempt (2026-05-10).** Retested with a manually captured `mcp_gateway_session` cookie (1,737 bytes, Starlette-signed). nginx returned HTTP 401 with `WWW-Authenticate: Bearer`, meaning the auth-server's `/validate` rejected the cookie via `validate_session_cookie()` ([auth_server/server.py:630](../auth_server/server.py#L630)). Most likely causes: cookie expired, `SECRET_KEY` rotated since the cookie was issued, or the cookie was issued on a different host. Net effect: even the documented cookie-only path is brittle for out-of-band testing — browser session is the only reliable path right now.

**Follow-up work:** `/api/config/full` and `/api/config/export` should accept Bearer JWTs for admin callers too. The fix is in `registry/auth/dependencies.py` — widen the `enhanced_auth` path so Bearer admin tokens are accepted on these endpoints. **Separate issue to be filed** once we confirm intent (some deployments may deliberately keep these cookie-only as defense-in-depth).

---

## How to read the tables

Each logical-group table has columns:

| Column | Meaning |
|--------|---------|
| **Parameter** | Human-readable name of the setting. |
| **Docker (`.env`)** | Variable name in [`.env.example`](../.env.example) / `.env`. |
| **Terraform (`.tfvars`)** | Variable name in [`terraform/aws-ecs/terraform.tfvars.example`](../terraform/aws-ecs/terraform.tfvars.example). Blank = not exposed on this surface (deployment-agnostic or not yet wired). |
| **Helm (`values.yaml`)** | YAML path in the stack chart, e.g. `registry.app.deploymentMode`. Blank = not exposed on this surface. |
| **Purpose** | One-line description. See [`configuration.md`](configuration.md) for full semantics. |

Conventions:
- A blank cell means the parameter is **not configurable** on that surface — either it is deployment-specific (e.g. CloudFront only applies to ECS) or the wiring is missing (flag this in a PR).
- `—` in **Purpose** means the parameter mirrors the row directly above.
- Secrets/sensitive values are flagged with **(secret)** — these must use AWS Secrets Manager (Terraform) or `existingSecret` / `secretKeyRef` (Helm), never plain values in version control.

---

## Group 1 — Registry Identity & Card

Registry metadata used for federation, discovery, and the header UI.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Public registry URL | `REGISTRY_URL` | — (derived from `base_domain` / `use_regional_domains`) | `global.domain` (plus `registry.registryCard.url`) | Public URL the registry is reachable at. |
| Registry display name | `REGISTRY_NAME` | `registry_name` | `registry.registryCard.name` | Human-readable name (falls back to a random docker-style name). |
| Operating organization | `REGISTRY_ORGANIZATION_NAME` | `registry_organization_name` | `registry.registryCard.organizationName` | Organization that runs this registry. |
| Description | `REGISTRY_DESCRIPTION` | `registry_description` | `registry.registryCard.description` | Federation-visible description. |
| Admin contact email | `REGISTRY_CONTACT_EMAIL` | `registry_contact_email` | `registry.registryCard.contactEmail` | Optional contact. |
| Contact / docs URL | `REGISTRY_CONTACT_URL` | `registry_contact_url` | `registry.registryCard.contactUrl` | Optional link. |
| Federation registry id | — (set at runtime via API) | `registry_id` | `global.federation.registryId` / `registry.app.registryId` | Unique identifier for this instance in peer federation. |

---

## Group 2 — Deployment Mode & UI Visibility

Controls registry-vs-gateway integration and which tabs render in the UI. See [`configuration.md#deployment-mode-configuration`](configuration.md).

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Deployment mode | `DEPLOYMENT_MODE` | `deployment_mode` | `registry.app.deploymentMode` | `with-gateway` (nginx integration) or `registry-only`. |
| Registry mode | `REGISTRY_MODE` | `registry_mode` | `registry.app.registryMode` | `full`, `mcp-servers-only`, `agents-only`, `skills-only`. |
| Show Servers tab | `SHOW_SERVERS_TAB` | `show_servers_tab` | `registry.app.showServersTab` | UI tab toggle (AND-ed with `REGISTRY_MODE`). |
| Show Virtual Servers tab | `SHOW_VIRTUAL_SERVERS_TAB` | `show_virtual_servers_tab` | `registry.app.showVirtualServersTab` | — |
| Show Skills tab | `SHOW_SKILLS_TAB` | `show_skills_tab` | `registry.app.showSkillsTab` | — |
| Show Agents tab | `SHOW_AGENTS_TAB` | `show_agents_tab` | `registry.app.showAgentsTab` | — |
| Disable built-in demo server | `DISABLE_AI_REGISTRY_TOOLS_SERVER` | `disable_ai_registry_tools_server` | `registry.app.disableAiRegistryToolsServer` | Prevent auto-registration of the built-in `airegistry-tools` demo server. |

---

## Group 3 — Auth Server URLs & JWT

Internal and external URLs for the auth server, plus internal JWT signing.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Auth server internal URL | `AUTH_SERVER_URL` | — (constructed by module) | `registry.app.authServerUrl` | Server-to-server URL inside the container network. |
| Auth server external URL | `AUTH_SERVER_EXTERNAL_URL` | — (from domain config) | `auth-server.app.externalUrl` | Public URL for browser redirects. |
| Internal JWT issuer | (constant in code) | — | `auth-server.app.jwtIssuer` | `iss` claim on internal service JWTs. |
| Internal JWT audience | (constant in code) | — | `auth-server.app.jwtAudience` | `aud` claim on internal service JWTs. |
| App secret key **(secret)** | `SECRET_KEY` | (via `TF_VAR_*` / secrets manager) | `global.secretKey` (auto-generated if unset) | JWT signing + credential encryption. Rotating invalidates stored creds. |

---

## Group 4 — Gateway Host Configuration

Affects only `with-gateway` deployments (nginx reverse proxy).

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Extra nginx `server_name` entries | `GATEWAY_ADDITIONAL_SERVER_NAMES` | — | — (ingress annotations handle this) | Space-separated list of additional hostnames / IPs to accept. |
| Server bind address (IPv6 opt-in) | `BIND_HOST` (and `HOST` for currenttime/mcpgw) | `bind_host` | `mcpgw.app.bindHost` | Default `0.0.0.0` (IPv4) works everywhere. Set to `::` only for IPv6-only deployments — requires `net.ipv6.bindv6only=0` on the host AND an IPv6 loopback in the container. Issue #863 / PR #864. Local-dev `uvicorn` direct invocation and `servers/currenttime` keep the safer `127.0.0.1` default. |

---

## Group 5 — Registry API Auth (Static Tokens)

Enterprise-perimeter auth for registry APIs without full IdP validation. See [`docs/registry-api-auth.md`](../docs/registry-api-auth.md).

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable static token auth | `REGISTRY_STATIC_TOKEN_AUTH_ENABLED` | `registry_static_token_auth_enabled` | `auth-server.app.registryStaticTokenAuthEnabled` | Master switch for static-token API auth. |
| Legacy single API token **(secret)** | `REGISTRY_API_TOKEN` | `registry_api_token` (use `TF_VAR_*`) | `auth-server.app.registryApiToken` | Single admin-level key. |
| Scoped multi-key JSON map **(secret)** | `REGISTRY_API_KEYS` | `registry_api_keys` (use `TF_VAR_*`) | `registry.app.registryApiKeys` + `registryApiKeysExistingSecret` | Named keys with per-key group assignments. |
| Max tokens / user / hour | — | `max_tokens_per_user_per_hour` | `auth-server.app.maxTokensPerUserPerHour` | Rate limit for token vending. |

---

## Group 6 — Registration Webhook (Issue #742)

Fire-and-forget POST on register/delete.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Webhook URL | `REGISTRATION_WEBHOOK_URL` | `registration_webhook_url` | `registry.app.registrationWebhookUrl` | POST target. Empty disables. |
| Auth header name | `REGISTRATION_WEBHOOK_AUTH_HEADER` | `registration_webhook_auth_header` | `registry.app.registrationWebhookAuthHeader` | `Authorization` auto-prefixes `Bearer `. |
| Auth token **(secret)** | `REGISTRATION_WEBHOOK_AUTH_TOKEN` | `registration_webhook_auth_token` | `registry.app.registrationWebhookAuthToken` | Token value. |
| HTTP timeout | `REGISTRATION_WEBHOOK_TIMEOUT_SECONDS` | `registration_webhook_timeout_seconds` | `registry.app.registrationWebhookTimeoutSeconds` | Seconds. Default 10. |

---

## Group 7 — Registration Gate / Admission Control (Issue #809)

Fail-closed external approval of registrations.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable | `REGISTRATION_GATE_ENABLED` | `registration_gate_enabled` | `registry.app.registrationGateEnabled` | Master switch. |
| Gate URL | `REGISTRATION_GATE_URL` | `registration_gate_url` | `registry.app.registrationGateUrl` | Required when enabled. |
| Auth type | `REGISTRATION_GATE_AUTH_TYPE` | `registration_gate_auth_type` | `registry.app.registrationGateAuthType` | `none`, `api_key`, `bearer`, `oauth2_client_credentials`. |
| Auth credential **(secret)** | `REGISTRATION_GATE_AUTH_CREDENTIAL` | `registration_gate_auth_credential` | `registry.app.registrationGateAuthCredential` | Used with `api_key` / `bearer`. |
| Auth header name | `REGISTRATION_GATE_AUTH_HEADER_NAME` | `registration_gate_auth_header_name` | `registry.app.registrationGateAuthHeaderName` | Used with `api_key`. |
| HTTP timeout | `REGISTRATION_GATE_TIMEOUT_SECONDS` | `registration_gate_timeout_seconds` | `registry.app.registrationGateTimeoutSeconds` | Per-attempt seconds. |
| Max retries | `REGISTRATION_GATE_MAX_RETRIES` | `registration_gate_max_retries` | `registry.app.registrationGateMaxRetries` | Retries after first attempt. |
| OAuth2 token URL | `REGISTRATION_GATE_OAUTH2_TOKEN_URL` | `registration_gate_oauth2_token_url` | `registry.app.registrationGateOauth2TokenUrl` | For `oauth2_client_credentials`. |
| OAuth2 client id | `REGISTRATION_GATE_OAUTH2_CLIENT_ID` | `registration_gate_oauth2_client_id` | `registry.app.registrationGateOauth2ClientId` | — |
| OAuth2 client secret **(secret)** | `REGISTRATION_GATE_OAUTH2_CLIENT_SECRET` | `registration_gate_oauth2_client_secret` | `registry.app.registrationGateOauth2ClientSecret` | — |
| OAuth2 scope | `REGISTRATION_GATE_OAUTH2_SCOPE` | `registration_gate_oauth2_scope` | `registry.app.registrationGateOauth2Scope` | e.g. `api://app-id/.default`. |

---

## Group 8 — Federation (Peer Registries)

Static-token and OAuth2 config for peer-to-peer federation.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable static token auth | `FEDERATION_STATIC_TOKEN_AUTH_ENABLED` | `federation_static_token_auth_enabled` | `global.federation.staticTokenAuthEnabled` | Allow peers to use static Bearer. |
| Federation static token **(secret)** | `FEDERATION_STATIC_TOKEN` | `federation_static_token` | `global.federation.staticToken` | Auto-generated if empty. |
| Encryption key **(secret)** | `FEDERATION_ENCRYPTION_KEY` | `federation_encryption_key` | `global.federation.encryptionKey` | Fernet key for storing peer tokens in MongoDB. |
| Federation token endpoint | `FEDERATION_TOKEN_ENDPOINT` | — | `registry.app.federationTokenEndpoint` | OAuth2 token endpoint for outbound peer auth. |
| Federation client id | `FEDERATION_CLIENT_ID` | — | `registry.app.federationClientId` | — |
| Federation client secret **(secret)** | `FEDERATION_CLIENT_SECRET` | — | `registry.app.federationClientSecret` | — |

---

## Group 9 — Workday ASOR Federation

Single URL; disables itself when unset.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Workday OAuth2 token URL | `WORKDAY_TOKEN_URL` | — | `registry.app.workdayTokenUrl` | Disables ASOR if placeholder. |
| ASOR access token **(secret)** | — | — | `registry.app.asorAccessToken` | Pre-obtained token (bypasses token URL). |

---

## Group 10 — AWS Agent Registry Federation

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable AWS Agent Registry federation | `AWS_REGISTRY_FEDERATION_ENABLED` | `aws_registry_federation_enabled` | `registry.awsRegistry.federationEnabled` | Overrides the `aws_registry.enabled` flag stored in MongoDB. |

---

## Group 11 — M2M Direct Client Registration (Issue #851)

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable M2M admin router | `M2M_DIRECT_REGISTRATION_ENABLED` | `m2m_direct_registration_enabled` | `registry.app.m2mDirectRegistrationEnabled` | Exposes `/api/iam/m2m-clients` without an IdP Admin API token. Default on. |

---

## Group 12 — Auth Provider Selection

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Provider type | `AUTH_PROVIDER` | Derived from `entra_enabled` / `okta_enabled` / `auth0_enabled` flags | `global.authProvider.type` | `keycloak`, `cognito`, `entra`, `okta`, `auth0`. |
| IdP group filter prefixes | `IDP_GROUP_FILTER_PREFIX` | `idp_group_filter_prefix` | `registry.idpGroupFilterPrefix` | Comma-separated prefixes for IAM > Groups. |

### 12a — Keycloak

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Internal URL | `KEYCLOAK_URL` | — (templated) | `auth-server.keycloak.externalUrl` + Helm service DNS | Inside container network. |
| External URL | `KEYCLOAK_EXTERNAL_URL` | — (from `keycloak_domain` / `base_domain`) | — (templated from `global.domain`) | Browser-reachable URL. |
| Admin URL | `KEYCLOAK_ADMIN_URL` | — | — | Used by setup scripts. |
| Realm | `KEYCLOAK_REALM` | — | `global.authProvider.keycloak.realm` / `auth-server.keycloak.realm` | e.g. `mcp-gateway`. |
| Admin username | `KEYCLOAK_ADMIN` | `keycloak_admin` | `global.authProvider.keycloak.adminUsername` | — |
| Admin password **(secret)** | `KEYCLOAK_ADMIN_PASSWORD` | `keycloak_admin_password` | (auto-generated, stored in `<release>-keycloak` secret) | — |
| DB password **(secret)** | `KEYCLOAK_DB_PASSWORD` | `keycloak_database_password` | (auto-generated, stored in `<release>-keycloak-postgresql` secret) | — |
| DB username | — | `keycloak_database_username` | — | — |
| Web client id | `KEYCLOAK_CLIENT_ID` | — | — | Populated by `init-keycloak.sh`. |
| Web client secret **(secret)** | `KEYCLOAK_CLIENT_SECRET` | — | — | — |
| M2M client id | `KEYCLOAK_M2M_CLIENT_ID` | — | `auth-server.keycloak.m2mClientId` | — |
| M2M client secret **(secret)** | `KEYCLOAK_M2M_CLIENT_SECRET` | — | `auth-server.keycloak.m2mClientSecret` | — |
| Enabled flag | `KEYCLOAK_ENABLED` | — | `auth-server.keycloak.enabled` | Enable Keycloak in OAuth2 providers. |
| Initial admin password | `INITIAL_ADMIN_PASSWORD` | — | — | First-run user. |
| Initial user password | `INITIAL_USER_PASSWORD` | — | — | First-run test user. |
| Log level | — | `keycloak_log_level` | — | Keycloak container log level. |
| DB min ACU | — | `keycloak_database_min_acu` | — | Aurora Serverless floor. |
| DB max ACU | — | `keycloak_database_max_acu` | — | Aurora Serverless ceiling. |

### 12b — Amazon Cognito

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| User Pool ID | `COGNITO_USER_POOL_ID` | — | `auth-server.cognito.userPoolId` | — |
| Client ID | `COGNITO_CLIENT_ID` | — | `auth-server.cognito.clientId` | — |
| Client secret **(secret)** | `COGNITO_CLIENT_SECRET` | — | `auth-server.cognito.clientSecret` | — |
| Enabled | `COGNITO_ENABLED` | — | — | — |
| Custom domain | `COGNITO_DOMAIN` | — | `auth-server.cognito.domain` | Optional. |
| Region | `AWS_REGION` | `aws_region` | `auth-server.cognito.region` | — |

### 12c — Microsoft Entra ID

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable | — (implicit via `AUTH_PROVIDER`) | `entra_enabled` | (set `global.authProvider.type: entra`) | Flag. |
| Tenant ID | `ENTRA_TENANT_ID` | `entra_tenant_id` | `auth-server.entra.tenantId` / `registry.entra.tenantId` | — |
| Client ID | `ENTRA_CLIENT_ID` | `entra_client_id` | `auth-server.entra.clientId` / `registry.entra.clientId` | — |
| Client secret **(secret)** | `ENTRA_CLIENT_SECRET` | `entra_client_secret` | `auth-server.entra.clientSecret` / `registry.entra.clientSecret` | — |
| Enabled flag | `ENTRA_ENABLED` | — | — | — |
| Login base URL | `ENTRA_LOGIN_BASE_URL` | `entra_login_base_url` | `auth-server.entra.loginBaseUrl` | Sovereign clouds. |
| Admin group id | `ENTRA_GROUP_ADMIN_ID` | — | `global.authProvider.entra.adminGroupId` | — |
| Users group id | `ENTRA_GROUP_USERS_ID` | — | — | — |

### 12d — Okta

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable | — (implicit via `AUTH_PROVIDER`) | `okta_enabled` | (set `global.authProvider.type: okta`) | Flag. |
| Domain | `OKTA_DOMAIN` | `okta_domain` | `auth-server.okta.domain` | — |
| Client ID | `OKTA_CLIENT_ID` | `okta_client_id` | `auth-server.okta.clientId` | — |
| Client secret **(secret)** | `OKTA_CLIENT_SECRET` | `okta_client_secret` | `auth-server.okta.clientSecret` | — |
| M2M client id | `OKTA_M2M_CLIENT_ID` | `okta_m2m_client_id` | `auth-server.okta.m2mClientId` | Defaults to web client. |
| M2M client secret **(secret)** | `OKTA_M2M_CLIENT_SECRET` | `okta_m2m_client_secret` | `auth-server.okta.m2mClientSecret` | — |
| API token **(secret)** | `OKTA_API_TOKEN` | `okta_api_token` | `auth-server.okta.apiToken` | For IAM operations. |
| Auth server id | `OKTA_AUTH_SERVER_ID` | `okta_auth_server_id` | `auth-server.okta.authServerId` | Custom AS; defaults to Org AS. |

### 12e — Auth0

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable | — | `auth0_enabled` | (set `global.authProvider.type: auth0`) | Flag. |
| Domain | `AUTH0_DOMAIN` | `auth0_domain` | `auth-server.auth0.domain` | — |
| Client ID | `AUTH0_CLIENT_ID` | `auth0_client_id` | `auth-server.auth0.clientId` | — |
| Client secret **(secret)** | `AUTH0_CLIENT_SECRET` | `auth0_client_secret` | `auth-server.auth0.clientSecret` | — |
| API audience | `AUTH0_AUDIENCE` | `auth0_audience` | `auth-server.auth0.audience` | — |
| Groups claim URI | `AUTH0_GROUPS_CLAIM` | `auth0_groups_claim` | `auth-server.auth0.groupsClaim` | Default `https://mcp-gateway/groups`. |
| Enabled flag | `AUTH0_ENABLED` | — | — | — |
| M2M client id | `AUTH0_M2M_CLIENT_ID` | `auth0_m2m_client_id` | `auth-server.auth0.m2mClientId` | For IAM Management. |
| M2M client secret **(secret)** | `AUTH0_M2M_CLIENT_SECRET` | `auth0_m2m_client_secret` | `auth-server.auth0.m2mClientSecret` | — |
| Management API token **(secret)** | `AUTH0_MANAGEMENT_API_TOKEN` | `auth0_management_api_token` | `auth-server.auth0.managementApiToken` | Static alternative (24h expiry). |

### 12f — GitHub / Google OAuth Apps (login, not SKILL.md fetching)

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| GitHub client ID | `GITHUB_CLIENT_ID` | — | — | OAuth App login. |
| GitHub client secret **(secret)** | `GITHUB_CLIENT_SECRET` | — | — | — |
| GitHub enabled | `GITHUB_ENABLED` | — | — | — |
| Google client ID | `GOOGLE_CLIENT_ID` | — | — | OAuth App login. |
| Google client secret **(secret)** | `GOOGLE_CLIENT_SECRET` | — | — | — |
| Google enabled | `GOOGLE_ENABLED` | — | — | — |

---

## Group 13 — Session Cookie Security

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Secure flag | `SESSION_COOKIE_SECURE` | `session_cookie_secure` | `auth-server.app.sessionCookieSecure` | Must be `true` in HTTPS, `false` on plain-HTTP localhost. |
| Cookie domain | `SESSION_COOKIE_DOMAIN` | `session_cookie_domain` | `auth-server.app.sessionCookieDomain` | Leading dot for cross-subdomain; empty is safest. |
| Store OAuth tokens in session | `OAUTH_STORE_TOKENS_IN_SESSION` | `oauth_store_tokens_in_session` | `auth-server.app.oauthStoreTokensInSession` | Disable for Entra (large tokens). |

---

## Group 13a — Tool-level Access Enforcement (Issue #1026)

Controls the per-user tool allowlist filter applied at the registry REST endpoints and the MCP `tools/list` JSON-RPC response. The allowlist is resolved from the `mcp-scopes` collection used by existing server-level access checks. All three values are non-sensitive.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable MCP tools/list filter | `MCP_TOOLS_LIST_FILTER_ENABLED` | `mcp_tools_list_filter_enabled` | `auth-server.app.mcpToolsListFilterEnabled`, `registry.app.mcpToolsListFilterEnabled` | Master switch for the MCP protocol tools/list filter. Default `true`. REST endpoints always filter regardless of this flag. |
| MCP proxy max body bytes | `MCP_PROXY_MAX_BODY_BYTES` | `mcp_proxy_max_body_bytes` | `auth-server.app.mcpProxyMaxBodyBytes` | Upper bound (bytes) the auth-server proxy hop buffers when filtering tools/list; oversize returns HTTP 413. Default `2097152` (2 MiB). |
| Tool-filter audit log level | `TOOL_FILTER_AUDIT_LOG_LEVEL` | `tool_filter_audit_log_level` | `auth-server.app.toolFilterAuditLogLevel`, `registry.app.toolFilterAuditLogLevel` | Launch-window log level for prune audit lines: `DEBUG`, `INFO`, or `WARNING`. Default `INFO`; flip to `DEBUG` after two quiet weeks in production. |

---

## Group 14 — GitHub Private Repo Access (for SKILL.md fetching)

Only the Helm `mcpgw` subchart and Docker expose these today.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| PAT **(secret)** | `GITHUB_PAT` | `github_pat` | `mcpgw.app.githubPat` / `mcpgw.app.githubPatExistingSecret` | Personal Access Token. |
| App ID | `GITHUB_APP_ID` | `github_app_id` | `mcpgw.app.githubAppId` | GitHub App alternative (preferred for orgs). |
| App installation ID | `GITHUB_APP_INSTALLATION_ID` | `github_app_installation_id` | `mcpgw.app.githubAppInstallationId` | — |
| App private key **(secret)** | `GITHUB_APP_PRIVATE_KEY` | `github_app_private_key` | `mcpgw.app.githubAppPrivateKey` / `mcpgw.app.githubAppPrivateKeyExistingSecret` | PEM. |
| Extra GitHub hosts | `GITHUB_EXTRA_HOSTS` | `github_extra_hosts` | `mcpgw.app.githubExtraHosts` | For GitHub Enterprise Server. |
| API base URL | `GITHUB_API_BASE_URL` | `github_api_base_url` | `mcpgw.app.githubApiBaseUrl` | GHES: `https://<ghes>/api/v3`. |

---

## Group 15 — Storage Backend

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Storage backend | `STORAGE_BACKEND` | `storage_backend` | `mongodb-configure.mongodb.storage_backend` | `file` (deprecated), `documentdb`, `mongodb-ce`, `mongodb`, `mongodb-atlas`. |
| Host | `DOCUMENTDB_HOST` | — (derived from module) | `mongodb-configure.mongodb.host` | — |
| Port | `DOCUMENTDB_PORT` | — | `mongodb-configure.mongodb.port` | Default 27017. |
| Database | `DOCUMENTDB_DATABASE` | — | `mongodb-configure.mongodb.database` | Default `mcp_registry`. |
| Username | `DOCUMENTDB_USERNAME` | `documentdb_admin_username` | `mongodb-configure.mongodb.username` / `mongodb.user` | — |
| Password **(secret)** | `DOCUMENTDB_PASSWORD` | `documentdb_admin_password` | `mongodb-configure.mongodb.password` / `mongodb.password` / `global.existingMongoCredentialsSecret` | — |
| TLS | `DOCUMENTDB_USE_TLS` | — | `mongodb-configure.mongodb.use_tls` | Set `true` for AWS DocumentDB. |
| TLS CA file | `DOCUMENTDB_TLS_CA_FILE` | — | — | For DocumentDB (`global-bundle.pem`). |
| IAM auth | `DOCUMENTDB_USE_IAM` | — | — | DocumentDB-only. |
| Replica set | `DOCUMENTDB_REPLICA_SET` | — | `mongodb-configure.mongodb.replica_set` | — |
| Read preference | `DOCUMENTDB_READ_PREFERENCE` | — | — | e.g. `secondaryPreferred`. |
| Namespace | `DOCUMENTDB_NAMESPACE` | — | `mongodb-configure.mongodb.namespace` | Multi-tenant segmentation. |
| Full connection string override **(secret)** | `MONGODB_CONNECTION_STRING` | `mongodb_connection_string` / `mongodb_connection_string_secret_arn` | `mongodb.connectionString` / `global.existingMongoCredentialsSecret` | Wins over discrete vars; required for Atlas / external MongoDB. |
| DocDB shard vCPU | — | `documentdb_shard_capacity` | — | AWS DocumentDB Elastic only. |
| DocDB shard count | — | `documentdb_shard_count` | — | — |

---

## Group 16 — AI / LLM

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Anthropic API key **(secret)** | `ANTHROPIC_API_KEY` | — | — | Required for Claude-backed agent functionality. |
| Smithery API key **(secret)** | `SMITHERY_API_KEY` | — | — | Access Smithery-hosted MCP servers. |

---

## Group 17 — MCP Server Security Scanning

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable scanning | `SECURITY_SCAN_ENABLED` | — | — | — |
| Scan on registration | `SECURITY_SCAN_ON_REGISTRATION` | — | — | — |
| Block unsafe | `SECURITY_BLOCK_UNSAFE_SERVERS` | — | — | — |
| Analyzers | `SECURITY_ANALYZERS` | — | — | `yara`, `llm`, `api` (comma-separated). |
| Scan timeout | `SECURITY_SCAN_TIMEOUT` | — | — | Seconds. |
| Add pending tag | `SECURITY_ADD_PENDING_TAG` | — | — | Tag servers that fail scan. |
| LLM scanner API key **(secret)** | `MCP_SCANNER_LLM_API_KEY` | — | — | OpenAI for `llm` analyzer. |

---

## Group 18 — A2A Agent Security Scanning

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable scanning | `AGENT_SECURITY_SCAN_ENABLED` | — | — | — |
| Scan on registration | `AGENT_SECURITY_SCAN_ON_REGISTRATION` | — | — | — |
| Block unsafe | `AGENT_SECURITY_BLOCK_UNSAFE_AGENTS` | — | — | — |
| Analyzers | `AGENT_SECURITY_ANALYZERS` | — | — | `yara`, `spec`, `heuristic`, `llm`, `endpoint`. |
| Scan timeout | `AGENT_SECURITY_SCAN_TIMEOUT` | — | — | Seconds. |
| Add pending tag | `AGENT_SECURITY_ADD_PENDING_TAG` | — | — | — |
| LLM scanner API key **(secret)** | `A2A_SCANNER_LLM_API_KEY` | — | — | Azure OpenAI for `llm` analyzer. |

---

## Group 19 — Skill Security Scanning

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable skill scanning | — | — | `registry.app.skillSecurityScanEnabled` | — |
| Skill analyzers | — | — | `registry.app.skillSecurityAnalyzers` | `static`, `behavioral`, `llm`, `meta`, `virustotal`, `ai-defense`. |

---

## Group 20 — Embeddings / Vector Search

Used by `registry` and `mcpgw` services.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Provider | `EMBEDDINGS_PROVIDER` | `embeddings_provider` | `mcpgw.app.embeddingsProvider` | `sentence-transformers` or `litellm`. |
| Model name | `EMBEDDINGS_MODEL_NAME` | `embeddings_model_name` | `mcpgw.app.embeddingsModelName` | — |
| Model dimensions | `EMBEDDINGS_MODEL_DIMENSIONS` | `embeddings_model_dimensions` | `mcpgw.app.embeddingsModelDimensions` | Must match model output. |
| API key **(secret)** | `EMBEDDINGS_API_KEY` | `embeddings_api_key` | `mcpgw.app.embeddingsApiKey` / `mcpgw.app.embeddingsApiKeyExistingSecret` | For `litellm` cloud providers. |
| Custom API base | `EMBEDDINGS_API_BASE` | — | `mcpgw.app.embeddingsApiBase` | — |
| AWS region | `EMBEDDINGS_AWS_REGION` | `embeddings_aws_region` | `mcpgw.app.embeddingsAwsRegion` | Bedrock. |

---

## Group 21 — ANS (Agent Naming Service)

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable integration | `ANS_INTEGRATION_ENABLED` | `ans_integration_enabled` | `registry.ans.enabled` | — |
| API endpoint | `ANS_API_ENDPOINT` | `ans_api_endpoint` | `registry.ans.apiEndpoint` | Default GoDaddy. |
| API key **(secret)** | `ANS_API_KEY` | `ans_api_key` | `registry.ans.apiKey` / `apiKeyExistingSecret` | — |
| API secret **(secret)** | `ANS_API_SECRET` | `ans_api_secret` | `registry.ans.apiSecret` / `apiSecretExistingSecret` | — |
| Timeout | `ANS_API_TIMEOUT_SECONDS` | `ans_api_timeout_seconds` | `registry.ans.apiTimeoutSeconds` | — |
| Sync interval | `ANS_SYNC_INTERVAL_HOURS` | `ans_sync_interval_hours` | `registry.ans.syncIntervalHours` | Re-verify cadence. |
| Cache TTL | `ANS_VERIFICATION_CACHE_TTL_SECONDS` | `ans_verification_cache_ttl_seconds` | `registry.ans.verificationCacheTtlSeconds` | — |

---

## Group 22 — External Registry Tags

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| External registry tag list | `EXTERNAL_REGISTRY_TAGS` | — | — | Comma-separated tags shown under "External Registries". |

---

## Group 23 — MCPGW Server (MCP Gateway server component)

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Enable OIDC | `OIDC_ENABLED` | — | — | Blocked pending issue #895. |
| OIDC client id | `OIDC_CLIENT_ID` | — | — | — |
| OIDC client secret **(secret)** | `OIDC_CLIENT_SECRET` | — | — | — |
| Keycloak internal URL | `KEYCLOAK_INTERNAL_URL` | — | — | Container-network URL. |
| M2M client id | `M2M_CLIENT_ID` | — | — | For registry API calls. |
| M2M client secret **(secret)** | `M2M_CLIENT_SECRET` | — | — | — |
| MCPGW base URL | `MCPGW_BASE_URL` | — | — | OAuth redirect URIs. |
| Bind host | `HOST` | — | — | `127.0.0.1` vs `0.0.0.0`. |
| Registry URL | — | — | `mcpgw.app.registryUrl` | Where MCPGW talks to registry. |

---

## Group 24 — Audit & Application Logging

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Audit log enabled | `AUDIT_LOG_ENABLED` | `audit_log_enabled` | — | — |
| Audit TTL (days) | `AUDIT_LOG_MONGODB_TTL_DAYS` | `audit_log_ttl_days` | — | TTL index. |
| App log max bytes | `APP_LOG_MAX_BYTES` | — | `registry.app.appLogMaxBytes` / `auth-server.app.appLogMaxBytes` | Rotating file size. |
| App log backup count | `APP_LOG_BACKUP_COUNT` | — | `*.app.appLogBackupCount` | — |
| Centralized log enabled | `APP_LOG_CENTRALIZED_ENABLED` | `app_log_centralized_enabled` | `*.app.appLogCentralizedEnabled` | Write to MongoDB. |
| Centralized TTL (days) | `APP_LOG_CENTRALIZED_TTL_DAYS` | `app_log_centralized_ttl_days` | `*.app.appLogCentralizedTtlDays` | — |
| Mongo buffer size | `APP_LOG_MONGODB_BUFFER_SIZE` | — | `*.app.appLogMongodbBufferSize` | Records before flush. |
| Flush interval (s) | `APP_LOG_MONGODB_FLUSH_INTERVAL_SECONDS` | — | `*.app.appLogMongodbFlushIntervalSeconds` | — |
| App log level | `APP_LOG_LEVEL` | `app_log_level` | `*.app.appLogLevel` | `DEBUG`, `INFO`, etc. |
| Excluded loggers | `APP_LOG_EXCLUDED_LOGGERS` | `app_log_excluded_loggers` | `*.app.appLogExcludedLoggers` | Comma-separated. |

---

## Group 25 — OTLP / OpenTelemetry Export

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| OTLP endpoint | `OTEL_OTLP_ENDPOINT` | `otel_otlp_endpoint` | `registry.app.otelOtlpEndpoint` | Empty disables. |
| OTLP headers **(secret)** | `OTEL_EXPORTER_OTLP_HEADERS` | `otel_exporter_otlp_headers` | `registry.app.otelExporterOtlpHeaders` | API-key-bearing. Use Secrets Manager on ECS. |
| Export interval (ms) | `OTEL_OTLP_EXPORT_INTERVAL_MS` | `otel_otlp_export_interval_ms` | `registry.app.otelOtlpExportIntervalMs` | Default 30000. |
| Metrics temporality | `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE` | `otel_exporter_otlp_metrics_temporality_preference` | `registry.app.otelExporterOtlpMetricsTemporalityPreference` | `cumulative` (default) or `delta` (Datadog). |

---

## Group 26 — Grafana / Observability Pipeline

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Grafana admin password **(secret)** | `GRAFANA_ADMIN_PASSWORD` | `grafana_admin_password` | — | Required if observability is on. |
| Enable AMP/ADOT/Grafana pipeline | — | `enable_observability` | — | ECS-specific. |
| Metrics service image | — | `metrics_service_image_uri` | — | ECR URI. |
| Grafana image | — | `grafana_image_uri` | — | ECR URI. |

---

## Group 27 — Anonymous Usage Telemetry

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Disable all telemetry | `MCP_TELEMETRY_DISABLED` | `mcp_telemetry_disabled` | `registry.app.mcpTelemetryDisabled` | `1` / `true` opts out. |
| Disable heartbeat only | `MCP_TELEMETRY_OPT_OUT` | `mcp_telemetry_opt_out` | `registry.app.mcpTelemetryOptOut` | Startup ping still sent. |
| Heartbeat interval (min) | `MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES` | `mcp_telemetry_heartbeat_interval_minutes` | `registry.app.telemetryHeartbeatIntervalMinutes` | Default 1440. |
| Collector endpoint | `MCP_TELEMETRY_ENDPOINT` | — | — | Self-hosted override. |
| Debug mode | `TELEMETRY_DEBUG` | `telemetry_debug` | `registry.app.telemetryDebug` | Log payloads instead of send. |
| Disable IMDS probe (cloud detection) | `MCP_TELEMETRY_IMDS_PROBE_DISABLED` | `mcp_telemetry_imds_probe_disabled` | `registry.app.mcpTelemetryImdsProbeDisabled` | Issue #986. Env/DMI/ECS/k8s tiers still run. |

---

## Group 28 — AgentCore Token Refresher

OAuth per-client-id secrets. These are dynamic and named after the client_id.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Per-client secret **(secret)** | `OAUTH_CLIENT_SECRET_<client_id>` | — | — | Overrides Cognito auto-retrieval. Vendor-level fallbacks (`AUTH0_CLIENT_SECRET`, `OKTA_CLIENT_SECRET`, `ENTRA_CLIENT_SECRET`, `KEYCLOAK_CLIENT_SECRET`) live in the provider groups above. |

---

## Group 29 — Container Registry Credentials (CI only)

Used only by the publish workflow; not by the running registry.

| Parameter | Docker (`.env`) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------|-----------------------|----------------------|---------|
| Docker Hub username | `DOCKERHUB_USERNAME` | — | — | — |
| Docker Hub token **(secret)** | `DOCKERHUB_TOKEN` | — | — | — |
| Docker Hub org | `DOCKERHUB_ORG` | — | — | — |
| GitHub username | `GITHUB_USERNAME` | — | — | — |
| GitHub token **(secret)** | `GITHUB_TOKEN` | — | — | — |
| GitHub org | `GITHUB_ORG` | — | — | — |

---

## Group 29 — Extra Environment Variable Injection (Issue #1000)

User-supplied environment variables passed to the registry, auth-server, and mcpgw containers *in addition to* the chart-managed variables. Each surface enforces a reserved-name list so users cannot override chart-managed values; the canonical list lives in `charts/<subchart>/reserved-env-names.txt` and is the shared source of truth across all three surfaces.

| Parameter | Docker (`.env` / extra_env) | Terraform (`.tfvars`) | Helm (`values.yaml`) | Purpose |
|-----------|-----------------------------|-----------------------|----------------------|---------|
| Registry extra env | File: `extra_env/registry.env` at the repo root (override with `$MCP_EXTRA_ENV_DIR`); key=value per line, picked up via `env_file:` with `required: false` | `registry_extra_env = [{ name, value }, ...]` **(sensitive)** | `registry.extraEnv: [{ name, value }]` / `registry.extraEnvFrom: [...]` | Inject custom env vars into the registry container. |
| Auth-server extra env | File: `extra_env/auth-server.env` (same directory as above) | `auth_server_extra_env = [{ name, value }, ...]` **(sensitive)** | `auth-server.extraEnv: [...]` / `auth-server.extraEnvFrom: [...]` | Inject custom env vars into the auth-server container. |
| mcpgw extra env | File: `extra_env/mcpgw.env` (same directory as above) | `mcpgw_extra_env = [{ name, value }, ...]` **(sensitive)** | `mcpgw.extraEnv: [...]` / `mcpgw.extraEnvFrom: [...]` | Inject custom env vars into the mcpgw container. |

**Reserved names (shared across all three surfaces):**
- `charts/registry/reserved-env-names.txt`
- `charts/auth-server/reserved-env-names.txt`
- `charts/mcpgw/reserved-env-names.txt`

**Collision enforcement per surface:**
- **Docker / Docker Compose**: `build_and_run.sh` runs `validate_extra_env` preflight on every start; rejects reserved-name collisions with the exact file and line number.
- **Terraform / ECS**: `terraform plan` uses a `validation` block on each `*_extra_env` variable that reads the same `reserved-env-names.txt` via `file()` and rejects reserved names with `contains()`.
- **Helm**: `registry.validateExtraEnv` / `auth-server.validateExtraEnv` / `mcpgw.validateExtraEnv` helpers in `_helpers.tpl` fail `helm template`/`install` with a clear error if a reserved name is supplied via `extraEnv`.

**Secret handling:** For production secrets, prefer Kubernetes `extraEnvFrom` (Helm) or AWS Secrets Manager ARNs wired into the task definition's `secrets` block (Terraform; see `mongodb_connection_string_secret_arn` as a reference pattern) rather than passing plaintext values via `*_extra_env`. The `extra_env/*.env` files on the Docker surface are plaintext on disk and should not be used for production secrets.

---

## Group 30 — Infrastructure-Only (Terraform and Helm) Parameters

These have no `.env` equivalent because they describe the infrastructure, not the running registry.

### Terraform / ECS infrastructure

| Terraform variable | Purpose |
|--------------------|---------|
| `ingress_cidr_blocks` | CIDRs allowed to reach the main ALB. |
| `use_regional_domains` | Regional subdomain pattern. |
| `base_domain` | Root domain for regional pattern. |
| `keycloak_domain` | Custom Keycloak hostname. |
| `root_domain` | Custom root hostname. |
| `enable_cloudfront` | Create CloudFront distributions. |
| `enable_route53_dns` | Create Route53 + ACM. |
| `cloudfront_prefix_list_name` | Restrict ALB to CloudFront origin IPs. |
| `registry_image_uri` | ECR image for registry. |
| `auth_server_image_uri` | ECR image for auth-server. |
| `currenttime_image_uri`, `mcpgw_image_uri`, `realserverfaketools_image_uri` | ECR images for built-in MCP servers. |
| `flight_booking_agent_image_uri`, `travel_assistant_agent_image_uri` | ECR images for A2A demo agents. |
| `aws_region` | Deploy region. |
| `name` | Deployment name prefix. |
| `vpc_cidr` | VPC CIDR. |
| `enable_monitoring` | CloudWatch dashboards. |
| `alarm_email` | SNS destination. |
| `currenttime_replicas`, `mcpgw_replicas`, `realserverfaketools_replicas`, `flight_booking_agent_replicas`, `travel_assistant_agent_replicas` | ECS service desired counts. |

### Helm / chart-only

| Helm value | Purpose |
|------------|---------|
| `global.image.registry`, `tag`, `pullPolicy` | Image defaults shared across subcharts. |
| `global.chartVersion` | Chart version stamp (CI sets this). |
| `global.sharedSecretName`, `existingSharedSecret` | Naming of the stack-level shared secret. |
| `global.oauthProviderSecretName`, `existingOauthProviderSecret` | Naming of the OAuth-provider-secret. |
| `global.existingMongoCredentialsSecret` | External Mongo URI secret. |
| `global.ingress.className`, `tls`, `routingMode`, `paths.*`, `inboundCidrs` | ALB ingress shape. |
| `keycloak.create` | Deploy Keycloak in-chart vs external. |
| `keycloak.httpRelativePath` | Keycloak base path. |
| `keycloakIngress.enabled` | Create a Keycloak ingress. |
| `<subchart>.service.type`, `.service.port` | K8s service shape. |
| `<subchart>.resources.{requests,limits}` | Pod resource sizing. |
| `<subchart>.nodeSelector` | Pod scheduling. |
| `<subchart>.app.replicas` | Deployment replica count. |
| `<subchart>.ingress.*` | Per-subchart ingress overrides. |
| `mongodb-kubernetes.operator.*` | MongoDB Community operator knobs. |
| `mongodb-configure.*`, `keycloak-configure.*` | One-shot job configuration for the init jobs. |

---

## Checklist for new parameters

When you add a new configuration parameter:

- [ ] Add to [`.env.example`](../.env.example) with description and default.
- [ ] Wire through `docker-compose.yml` (and `.podman.yml`, `.prebuilt.yml`, `.dhi.yml`) service env blocks.
- [ ] Add Terraform variable in [`terraform/aws-ecs/variables.tf`](../terraform/aws-ecs/variables.tf), pass through [`main.tf`](../terraform/aws-ecs/main.tf) and the module, map to ECS task env in [`modules/mcp-gateway/ecs-services.tf`](../terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf), and document in [`terraform.tfvars.example`](../terraform/aws-ecs/terraform.tfvars.example).
- [ ] Add Helm values default in [`charts/<subchart>/values.yaml`](../charts) AND the stack [`charts/mcp-gateway-registry-stack/values.yaml`](../charts/mcp-gateway-registry-stack/values.yaml); wire into the subchart's `templates/deployment.yaml` and (if sensitive) `templates/secret.yaml`.
- [ ] Register the field in [`registry/api/config_routes.py`](../registry/api/config_routes.py) `CONFIG_GROUPS` so it appears on **Settings → System Config** and in `GET /api/config/full`. Mark sensitive values with `is_sensitive=True`.
- [ ] Add a new row to the appropriate group in **this file**. If it belongs in a brand-new group, add a new group section. Confirmed by reviewer before merge.

If one of the three surfaces legitimately does not apply, leave the cell blank and explain in the PR description — do not silently omit.
