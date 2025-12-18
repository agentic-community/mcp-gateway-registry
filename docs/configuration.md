# Configuration Reference

This document provides a comprehensive reference for all configuration files in the MCP Gateway Registry project. Each configuration file serves a specific purpose in the authentication and operation of the system.

## Configuration Files Overview

| File | Purpose | Type | Location | Example File | User Modification |
|------|---------|------|----------|--------------|-------------------|
| [`.env`](#main-environment-configuration) | Main project environment variables | Environment | Project root | `.env.example` | **Yes** - Required |
| [`.env` (OAuth)](#oauth-environment-configuration) | OAuth provider credentials | Environment | `credentials-provider/oauth/` | `.env.example` | **Yes** - Required |
| [`oauth2_providers.yml`](#oauth2-providers-configuration) | OAuth2 provider definitions | YAML | `auth_server/` | - | **No** - Pre-configured |
| [`scopes.yml`](#scopes-configuration) | Fine-grained access control scopes | YAML | `auth_server/` | - | **Rarely** - Only for custom permissions |
| [`docker-compose.yml`](#docker-compose-configuration) | Container orchestration | YAML | Project root | - | **Rarely** - Only for custom deployments |

---

## Main Environment Configuration

**File:** `.env` (Project root)
**Purpose:** Core project settings, registry URLs, and primary authentication credentials.

### Authentication Provider Selection

The MCP Gateway Registry supports multiple authentication providers. Choose one by setting the `AUTH_PROVIDER` environment variable:

- **`keycloak`**: Enterprise-grade open-source identity and access management with individual agent audit trails
- **`cognito`**: Amazon managed authentication service

Based on your selection, configure the corresponding provider-specific variables below.

### EnforceAI (Optional)

EnforceAI is enabled when `ENFORCEAI_DB_PATH` is set. In EnforceAI mode, ingress authentication supports:
- OIDC JWTs (generic multi-issuer) via `OIDC_ISSUERS`
- Gateway-issued tokens (RS256) via `ENFORCEAI_GATEWAY_*`
- API keys via `ENFORCEAI_API_KEY_PEPPER_PATH`

Core EnforceAI configuration variables:

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `ENFORCEAI_DB_PATH` | SQLite DB path for EnforceAI state | `/var/lib/enforceai/enforceai.db` | ✅ (to enable EnforceAI) |
| `ENFORCEAI_AUTH_PROVIDER` | EnforceAI auth mode (`oidc`, `gateway-token`, `api-key`, `mixed`) | `mixed` | ✅ |
| `OIDC_ISSUERS` | JSON map keyed by `iss` with OIDC issuer config | `{"https://issuer": {"jwks_uri": "https://issuer/jwks.json", "audiences": ["mcp-registry"]}}` | If using `oidc`/`mixed` |
| `ENFORCEAI_SCOPES_CATALOG_PATH` | Path to `scopes.yml` used for scope validation/catalog | `/etc/mcp/scopes.yml` | ✅ |
| `ENFORCEAI_API_KEY_PEPPER_PATH` | File path to pepper bytes for API key hashing | `/etc/mcp/api_key_pepper` | If using `api-key`/`mixed` |
| `ENFORCEAI_GATEWAY_PRIVATE_KEY_PATH` | PEM private key for minting gateway tokens | `/etc/mcp/gateway_private.pem` | If minting/using gateway tokens |
| `ENFORCEAI_GATEWAY_PUBLIC_KEYS_DIR` | Directory containing `<kid>.pem` public keys | `/etc/mcp/gateway_public_keys` | If using gateway tokens |
| `ENFORCEAI_GATEWAY_ACTIVE_KID` | Active key id (matches a `<kid>.pem` filename) | `kid-1` | If minting gateway tokens |
| `ENFORCEAI_GATEWAY_ISSUER` | Expected/minted gateway token issuer (`iss`) | `enforceai-gateway` | If using gateway tokens |

For EnforceAI operational docs (management API + CLI), see `enforceai/instructions/ENFORCEAI_CONTEXT.md` and `enforceai/instructions/ENFORCEAI_MANAGEMENT.md`.

### Core Variables

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `REGISTRY_URL` | Public URL of the MCP Gateway Registry | `https://mcpgateway.ddns.net` | ✅ |
| `ADMIN_USER` | Registry admin username | `admin` | ✅ |
| `ADMIN_PASSWORD` | Registry admin password | `your-secure-password` | ✅ |
| `AUTH_PROVIDER` | Authentication provider (`cognito` or `keycloak`) | `keycloak` | ✅ |
| `AWS_REGION` | AWS region for services | `us-east-1` | ✅ |

### Keycloak Configuration (if AUTH_PROVIDER=keycloak)

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `KEYCLOAK_URL` | Keycloak server URL (internal/Docker network) | `http://keycloak:8080` | ✅ |
| `KEYCLOAK_EXTERNAL_URL` | Keycloak server URL (external/browser access) | `https://mcpgateway.ddns.net` (production)<br/>`http://localhost:8080` (local development) | ✅ |
| `KEYCLOAK_ADMIN_URL` | Keycloak admin URL (for setup scripts) | `http://localhost:8080` | ✅ |
| `KEYCLOAK_REALM` | Keycloak realm name | `mcp-gateway` | ✅ |
| `KEYCLOAK_ADMIN` | Keycloak admin username | `admin` | ✅ |
| `KEYCLOAK_ADMIN_PASSWORD` | Keycloak admin password | `SecureKeycloakAdmin123!` | ✅ |
| `KEYCLOAK_DB_PASSWORD` | Keycloak database password | `SecureKeycloakDB123!` | ✅ |
| `KEYCLOAK_CLIENT_ID` | Keycloak web client ID (see note below) | `mcp-gateway-web` | ✅ |
| `KEYCLOAK_CLIENT_SECRET` | Keycloak web client secret (auto-generated) | `0tiBtgQFcaBiwHXIxDws...` | ✅ |
| `KEYCLOAK_M2M_CLIENT_ID` | Keycloak M2M client ID (see note below) | `mcp-gateway-m2m` | ✅ |
| `KEYCLOAK_M2M_CLIENT_SECRET` | Keycloak M2M client secret (auto-generated) | `ZJqbsamnQs79hbUbkJLB...` | ✅ |
| `KEYCLOAK_ENABLED` | Enable Keycloak in OAuth2 providers | `true` | ✅ |
| `INITIAL_ADMIN_PASSWORD` | Initial admin user password | `changeme` | For setup |
| `INITIAL_USER_PASSWORD` | Initial test user password | `testpass` | For setup |

**Note: Getting Keycloak Client IDs and Secrets**

The client IDs and secrets are automatically generated when you run the Keycloak initialization script:

```bash
cd keycloak/setup
./init-keycloak.sh
```

The script will:
1. Create the clients with the IDs you specify (`mcp-gateway-web` and `mcp-gateway-m2m`)
2. Generate secure random secrets for each client
3. Display the generated secrets at the end of the script output
4. Save them to a file for your reference

**To retrieve existing client secrets from a running Keycloak instance:**

```bash
# Method 1: Use the helper script (Recommended)
cd keycloak/setup
export KEYCLOAK_ADMIN_PASSWORD="your-admin-password"
./get-all-client-credentials.sh
# This will display the secrets and save them to .oauth-tokens/keycloak-client-secrets.txt

# Method 2: Using Keycloak Admin Console (Web UI)
# 1. Navigate to https://your-keycloak-url/admin
# 2. Login with admin credentials
# 3. Select your realm (mcp-gateway)
# 4. Go to Clients → Select your client
# 5. Go to Credentials tab
# 6. Copy the Secret value

# Method 3: Check the original initialization output
# The init-keycloak.sh script saves secrets to keycloak-client-secrets.txt
cat keycloak/setup/keycloak-client-secrets.txt
```

### Amazon Cognito Configuration (if AUTH_PROVIDER=cognito)

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `COGNITO_USER_POOL_ID` | Amazon Cognito User Pool ID | `us-east-1_vm1115QSU` | ✅ |
| `COGNITO_CLIENT_ID` | Amazon Cognito App Client ID | `3aju04s66t...` | ✅ |
| `COGNITO_CLIENT_SECRET` | Amazon Cognito App Client Secret | `85ps32t55df39hm61k966fqjurj...` | ✅ |
| `COGNITO_DOMAIN` | Cognito domain (optional) | `auto` | Optional |

### Session Cookie Security Configuration

**CRITICAL:** These settings control how session cookies are transmitted and shared. Incorrect configuration will cause login failures.

| Variable | Description | Example | Required | Default |
|----------|-------------|---------|----------|---------|
| `SESSION_COOKIE_SECURE` | Enable HTTPS-only cookie transmission | `false` (localhost)<br/>`true` (production) | ✅ | `false` |
| `SESSION_COOKIE_DOMAIN` | Cookie domain for cross-subdomain sharing | `""` (single domain)<br/>`.example.com` (cross-subdomain) | ❌ | Empty |

#### SESSION_COOKIE_SECURE - Critical for Your Environment

**YOU MUST SET THIS CORRECTLY OR LOGIN WILL FAIL:**

**For Local Development (localhost via HTTP):**
```bash
SESSION_COOKIE_SECURE=false  # MUST be false
```
- Localhost runs over HTTP (not HTTPS)
- Cookies with `secure=true` are ONLY sent over HTTPS
- Setting this to `true` on localhost = **login will fail** ❌

**For Production with HTTPS:**
```bash
SESSION_COOKIE_SECURE=true  # MUST be true
```
- Production deployments use HTTPS
- Cookies must have `secure=true` to prevent session hijacking
- Setting this to `false` in production = **security vulnerability** ❌

#### SESSION_COOKIE_DOMAIN - When to Set This

**Most deployments should leave this EMPTY** (default behavior = safest):

```bash
SESSION_COOKIE_DOMAIN=  # Empty string or unset
```

**Only set this if you need cross-subdomain authentication:**

| Deployment Type | Example Domains | SESSION_COOKIE_DOMAIN |
|----------------|-----------------|----------------------|
| **Single domain** | `mcpgateway.ddns.net` | `""` (empty) |
| **Cross-subdomain** | `auth.example.com`<br/>`registry.example.com` | `.example.com` |
| **Multi-level domains** | `registry.region-1.corp.company.internal` | `.corp.company.internal` |

**Important Security Notes:**
- Empty domain = cookie scoped to exact host only (safest)
- Set domain only when you control ALL subdomains
- Never set to public suffixes (`.com`, `.net`, `.ddns.net`)
- Domain must start with a dot (`.example.com`)

**See Also:** [Cookie Security Design Documentation](design/cookie-security-design.md) for detailed security analysis and deployment scenarios.

### Optional Variables

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| `AUTH_SERVER_URL` | Internal auth server URL | `http://auth-server:8888` | - |
| `AUTH_SERVER_EXTERNAL_URL` | External auth server URL | `https://mcpgateway.ddns.net` | - |
| `SECRET_KEY` | Application secret key | Auto-generated if not provided | Auto-generated |
| `ATLASSIAN_AUTH_TOKEN` | Atlassian OAuth token | Auto-populated from credentials | - |
| `SRE_GATEWAY_AUTH_TOKEN` | SRE Gateway auth token | Auto-populated from credentials | - |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | `sk-ant-api03-...` | For AI functionality |

### Container Registry Configuration (Optional - for CI/CD and local builds)

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `DOCKERHUB_USERNAME` | Docker Hub username for publishing containers | `your_dockerhub_username` | **Optional** |
| `DOCKERHUB_TOKEN` | Docker Hub access token | `your_dockerhub_access_token` | **Optional** |
| `GITHUB_USERNAME` | GitHub username for GHCR publishing | `your_github_username` | **Optional** |
| `GITHUB_TOKEN` | GitHub Personal Access Token with packages:write scope | `ghp_your_token_here` | **Optional** |
| `DOCKERHUB_ORG` | Docker Hub organization name (leave empty for personal account) | `mcpgateway` or empty | **Optional** |
| `GITHUB_ORG` | GitHub organization name (leave empty for personal account) | `agentic-community` or empty | **Optional** |

**Note: Container Registry Credentials (Completely Optional)**

These credentials are **entirely optional** and only needed if you want to:
- **Publish container images**: Automatically via GitHub Actions or manually via scripts
- **Contribute pre-built containers**: For easier deployment by other users

**What happens if these are not configured:**
- ✅ **The MCP Gateway Registry will work perfectly** - all core functionality remains intact
- ✅ **GitHub Actions will succeed** - builds will complete successfully, just without publishing to Docker Hub
- ✅ **Local development is unaffected** - no scripts will fail or produce errors
- ✅ **Only container publishing is skipped** - everything else continues normally

**When you might want to configure these:**
- **Contributing to the project**: Publishing official container images
- **Custom deployments**: Creating your own container registry for internal use
- **Development workflow**: Testing container builds locally

**How to obtain credentials (only if needed):**
- **Docker Hub**: Get access token from [Docker Hub Security Settings](https://hub.docker.com/settings/security)
- **GitHub Container Registry**: Generate Personal Access Token with `packages:write` scope from [GitHub Token Settings](https://github.com/settings/tokens)

**Setup instructions (only if publishing containers):**
- **In GitHub Actions**: Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` as repository secrets
- **For local builds**: Add credentials to your `.env` file and use `scripts/publish_containers.sh`
- **GITHUB_TOKEN**: Automatically provided in GitHub Actions, manually generated for local use

**Organization vs Personal Account Publishing:**
- **Personal Account** (Free): Leave `DOCKERHUB_ORG` and `GITHUB_ORG` empty
  - Images published as: `username/image-name`
  - Example: `aarora79/registry:latest`
- **Organization Account** (Paid for Docker Hub): Set organization names
  - Images published as: `organization/image-name`
  - Example: `mcpgateway/registry:latest`

---

## Keycloak Setup and Configuration

When using Keycloak as your authentication provider, the system provides comprehensive setup scripts and configuration options:

### Initial Setup

Run the Keycloak initialization script to set up the realm, clients, and groups:

```bash
cd keycloak/setup
./init-keycloak.sh
```

This script will:
1. Create the `mcp-gateway` realm
2. Set up web and M2M clients with proper configurations
3. Create necessary groups (`mcp-servers-unrestricted`, `mcp-servers-restricted`)
4. Configure group mappers for JWT token claims
5. Create initial admin and test users

### Service Account Management

For individual AI agent audit trails, create service accounts:

```bash
# Create individual agent service account
./setup-agent-service-account.sh --agent-id sre-agent --group mcp-servers-unrestricted

# Create shared M2M service account
./setup-m2m-service-account.sh
```

### Token Generation

Generate tokens for Keycloak authentication:

```bash
# Generate tokens for a single agent service account
uv run python credentials-provider/keycloak/generate_tokens.py --agent-id sre-agent

# Generate tokens for all agents
uv run python credentials-provider/keycloak/generate_tokens.py --all-agents

# Generate a full local credential/config set (tokens + editor configs)
./credentials-provider/generate_creds.sh
```

For detailed Keycloak integration documentation, see [Keycloak Integration Guide](keycloak-integration.md).

---

## OAuth Environment Configuration

**File:** `credentials-provider/oauth/.env`
**Purpose:** OAuth credentials for gateway ingress authentication (Keycloak/Cognito).

### Ingress Authentication

#### For Keycloak (if AUTH_PROVIDER=keycloak)

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `KEYCLOAK_URL` | Keycloak server URL | `https://mcpgateway.ddns.net` | ✅ |
| `KEYCLOAK_REALM` | Keycloak realm | `mcp-gateway` | ✅ |
| `KEYCLOAK_M2M_CLIENT_ID` | M2M client ID | `mcp-gateway-m2m` | ✅ |
| `KEYCLOAK_M2M_CLIENT_SECRET` | M2M client secret | `ZJqbsamnQs79hbUbkJLB...` | ✅ |

#### For Cognito (if AUTH_PROVIDER=cognito)

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `INGRESS_OAUTH_USER_POOL_ID` | Cognito User Pool for ingress auth | `us-east-1_vm1115QSU` | ✅ |
| `INGRESS_OAUTH_CLIENT_ID` | Cognito client ID for ingress | `5v2rav1v93...` | ✅ |
| `INGRESS_OAUTH_CLIENT_SECRET` | Cognito client secret for ingress | `1i888fnolv6k5sa1b8s5k839pdm...` | ✅ |

### Upstream Authentication (Gateway-Managed)

Upstream authentication (API keys, OAuth provider tokens, JWTs) is terminated at the gateway:

- Agent/client configuration contains only gateway ingress credentials.
- Upstream credentials are configured and stored by the gateway per server (and per binding rules, e.g., per-user or per-service).

Do not configure upstream provider secrets in `credentials-provider/oauth/.env` and do not distribute upstream tokens to agents.

---

## OAuth2 Providers Configuration

**File:** `auth_server/oauth2_providers.yml`
**Purpose:** OAuth2 provider definitions for web-based authentication flows.

### Keycloak Provider Configuration

When using Keycloak as the authentication provider, the following configuration is used:

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `display_name` | Human-readable name | ✅ | `"Keycloak"` |
| `client_id` | OAuth client ID | ✅ | `"${KEYCLOAK_CLIENT_ID}"` |
| `client_secret` | OAuth client secret | ✅ | `"${KEYCLOAK_CLIENT_SECRET}"` |
| `auth_url` | Authorization endpoint | ✅ | `"${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/auth"` |
| `token_url` | Token endpoint | ✅ | `"${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/token"` |
| `user_info_url` | User info endpoint | ✅ | `"${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/userinfo"` |
| `logout_url` | Logout endpoint | ✅ | `"${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/logout"` |
| `scopes` | OAuth scopes | ✅ | `["openid", "email", "profile"]` |
| `groups_claim` | JWT claim for groups | ✅ | `"groups"` |
| `enabled` | Provider enabled | ✅ | `true` |

### General Provider Configuration Fields

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `display_name` | Human-readable provider name | ✅ | `"Amazon Cognito"` |
| `client_id` | OAuth client ID (can use env vars) | ✅ | `"${COGNITO_CLIENT_ID}"` |
| `client_secret` | OAuth client secret (can use env vars) | ✅ | `"${COGNITO_CLIENT_SECRET}"` |
| `auth_url` | Authorization endpoint URL | ✅ | `"https://domain.auth.region.amazoncognito.com/oauth2/authorize"` |
| `token_url` | Token endpoint URL | ✅ | `"https://domain.auth.region.amazoncognito.com/oauth2/token"` |
| `user_info_url` | User info endpoint URL | ✅ | `"https://domain.auth.region.amazoncognito.com/oauth2/userInfo"` |
| `logout_url` | Logout endpoint URL | ✅ | `"https://domain.auth.region.amazoncognito.com/logout"` |
| `scopes` | OAuth scopes array | ✅ | `["openid", "email", "profile"]` |
| `response_type` | OAuth response type | ✅ | `"code"` |
| `grant_type` | OAuth grant type | ✅ | `"authorization_code"` |
| `username_claim` | JWT claim for username | ✅ | `"email"` |
| `groups_claim` | JWT claim for groups | ❌ | `"cognito:groups"` |
| `email_claim` | JWT claim for email | ✅ | `"email"` |
| `name_claim` | JWT claim for name | ✅ | `"name"` |
| `enabled` | Whether provider is enabled | ✅ | `true` |

### Supported Providers

- **Keycloak**: Enterprise-grade open-source identity and access management
- **Amazon Cognito**: Amazon managed authentication service
- **GitHub**: Repository and development services (planned)
- **Google**: Google Workspace and consumer accounts

To enable Google login:

- Set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
- Set `GOOGLE_ENABLED=true`
- Configure the Google OAuth client redirect URI as `${AUTH_SERVER_EXTERNAL_URL}/oauth2/callback/google`

---

## Scopes Configuration

**File:** `auth_server/scopes.yml`
**Purpose:** Fine-grained access control (FGAC) scope definitions.

### Scope Categories

- **MCP Servers**: Individual server access (`mcp-servers-{name}/read`, `mcp-servers-{name}/execute`)
- **Unrestricted**: Global access (`mcp-servers-unrestricted/read`, `mcp-servers-unrestricted/execute`)
- **Admin**: Administrative functions (`admin/registry`, `admin/users`)

---

## Docker Compose Configuration

**File:** `docker-compose.yml`
**Purpose:** Container orchestration for development and deployment.

### Services

- **registry**: Main MCP Gateway Registry service
- **auth-server**: OAuth2 authentication server
- **frontend**: Web interface (React application)

### Key Configuration

- Environment variable injection from `.env` files
- Port mappings for local development
- Volume mounts for persistent data
- Health checks and restart policies

---

## Configuration Security

### Best Practices

1. **Never commit real credentials** to version control
2. **Use environment variables** for sensitive data
3. **Rotate credentials regularly** especially for production
4. **Limit scope permissions** to minimum required access
5. **Monitor credential usage** through logging and audit trails

### File Permissions

- `.env` files should have `600` permissions (readable only by owner)
- Configuration directories should have `700` permissions
- Generated token files are automatically secured with `600` permissions

---

## Troubleshooting

### Common Issues

1. **Login redirects back to login page**
   - **Most Common Cause:** `SESSION_COOKIE_SECURE=true` but accessing via HTTP
   - **Solution for localhost:** Set `SESSION_COOKIE_SECURE=false` in `.env`
   - **Solution for production:** Ensure HTTPS is properly configured
   - **Check:** Browser dev tools → Application → Cookies (cookie should be present)
   - **Check:** Server logs for `Auth server setting session cookie: secure=...`

2. **Missing environment variables**: Check that all required variables are set in the appropriate `.env` files

3. **Invalid credentials**: Verify OAuth client IDs and secrets with providers

4. **Network connectivity**: Ensure firewall rules allow OAuth callback URLs

5. **Token expiration**: Use the credential refresh scripts to update expired tokens

6. **Scope mismatches**: Verify requested OAuth scopes match provider configurations

7. **Session cookie not being sent by browser**
   - Check cookie domain matches your hostname
   - Verify `SESSION_COOKIE_DOMAIN` is empty for single-domain deployments
   - Check browser third-party cookie settings
   - Inspect cookie attributes in browser dev tools

### Validation Commands

```bash
# Validate OAuth configuration
cd credentials-provider
./generate_creds.sh --verbose

# Test MCP gateway connectivity
cd tests
./tests/mcp_cmds.sh ping

# Check configuration files
python -c "import yaml; yaml.safe_load(open('file.yml'))"  # YAML validation
```

### Log Files

- **OAuth flows**: `.oauth-tokens/` directory contains generated tokens and logs
- **Registry operations**: Check `registry.log` for service-level issues
- **Authentication**: Check `auth.log` for OAuth and FGAC issues
