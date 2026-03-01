# Configuration Reference

This document provides a comprehensive reference for all configuration files in the MCP Gateway Registry project. Each configuration file serves a specific purpose in the authentication and operation of the system.

## Configuration Files Overview

| File | Purpose | Type | Location | Example File | User Modification |
|------|---------|------|----------|--------------|-------------------|
| [`.env`](#main-environment-configuration) | Main project environment variables | Environment | Project root | `.env.example` | **Yes** - Required |
| [`.env` (OAuth)](#oauth-environment-configuration) | OAuth provider credentials | Environment | `credentials-provider/oauth/` | `.env.example` | **Yes** - Required |
| [`.env` (AgentCore)](#agentcore-environment-configuration) | AgentCore authentication config | Environment | `credentials-provider/agentcore-auth/` | `.env.example` | **Optional** - Only if using AgentCore |
| [`oauth2_providers.yml`](#oauth2-providers-configuration) | OAuth2 provider definitions | YAML | `auth_server/` | - | **No** - Pre-configured |
| [`scopes.yml`](#scopes-configuration) | Fine-grained access control scopes | YAML | `auth_server/` | - | **Rarely** - Only for custom permissions |
| [`oauth_providers.yaml`](#oauth-providers-mapping) | Provider-specific OAuth configurations | YAML | `credentials-provider/oauth/` | - | **No** - Pre-configured |
| [`docker-compose.yml`](#docker-compose-configuration) | Container orchestration | YAML | Project root | - | **Rarely** - Only for custom deployments |

---

## Main Environment Configuration

**File:** `.env` (Project root)
**Purpose:** Core project settings, registry URLs, and primary authentication credentials.

### Authentication Provider Selection

The MCP Gateway Registry supports multiple authentication providers. Choose one by setting the `AUTH_PROVIDER` environment variable:

- **`keycloak`**: Open-source identity and access management with individual agent audit trails
- **`cognito`**: Amazon managed authentication service

Based on your selection, configure the corresponding provider-specific variables below.

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
| `SRE_GATEWAY_AUTH_TOKEN` | SRE Gateway auth token | Auto-populated from credentials | - |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | `sk-ant-api03-...` | For AI functionality |

### Storage Backend Configuration

The MCP Gateway Registry supports three storage backends for servers, agents, and scopes management.

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `STORAGE_BACKEND` | Storage backend for registry data | `file`, `mongodb-ce`, or `documentdb` | `file` |

> **⚠️ DEPRECATION WARNING:** File-based storage is deprecated and will be removed in a future release. MongoDB CE is now the recommended backend for local development and testing.

**Backend Options:**

#### File Backend (Deprecated)
- **Status**: **DEPRECATED** - Will be removed in a future release
- **Migration Path**: Switch to MongoDB CE for local development or DocumentDB for production
- **Pros**: Simple, no external dependencies, human-readable JSON files
- **Cons**: Limited concurrent writes, no distributed access, FAISS-based vector search, **deprecated**

```bash
STORAGE_BACKEND=file  # DEPRECATED - Use mongodb-ce instead
```

**Data stored in:**
- Servers: `~/mcp-gateway/servers/*.json`
- Agents: `~/mcp-gateway/agents/*.json`
- Scopes: `auth_server/scopes.yml`
- Security scans: `~/mcp-gateway/security_scans/*.json`

#### MongoDB CE Backend (Recommended for Local Development)
- **Status**: **RECOMMENDED** for all local development and testing
- **Best for**: Local development, feature development, testing, CI/CD pipelines
- **Pros**: Docker-based, no cloud dependencies, replica set support, application-level vector search, production-like environment
- **Cons**: Limited to ~10,000 documents, O(n) vector search performance (acceptable for development)

```bash
STORAGE_BACKEND=mongodb-ce
DOCUMENTDB_HOST=mongodb       # Docker service name
DOCUMENTDB_PORT=27017
DOCUMENTDB_DATABASE=mcp_registry
DOCUMENTDB_NAMESPACE=default
DOCUMENTDB_USE_TLS=false      # No TLS for local dev
```

**MongoDB Collections Created:**
- `mcp_servers_{namespace}` - Server definitions
- `mcp_agents_{namespace}` - A2A agent cards
- `mcp_scopes_{namespace}` - Authorization scopes
- `mcp_embeddings_1536_{namespace}` - Vector embeddings (1536 dimensions)
- `mcp_security_scans_{namespace}` - Security scan results
- `mcp_federation_config_{namespace}` - Federation configuration

**First-Time MongoDB CE Setup:**

```bash
# 1. Start MongoDB container
docker-compose up -d mongodb
sleep 5

# 2. Initialize collections and indexes
docker-compose up mongodb-init

# 3. Verify setup
docker exec mcp-mongodb mongosh --eval "use mcp_registry; show collections"

# 4. Switch backend and restart
export STORAGE_BACKEND=mongodb-ce
docker-compose restart registry
```

#### DocumentDB Backend (Production, Recommended)
- **Best for**: Production deployments, high concurrency, large-scale systems
- **Pros**: Native HNSW vector search, distributed storage, AWS-managed, clustering support
- **Cons**: Requires AWS infrastructure, uses AWS pricing

```bash
STORAGE_BACKEND=documentdb
DOCUMENTDB_HOST=cluster.docdb.amazonaws.com
DOCUMENTDB_PORT=27017
DOCUMENTDB_DATABASE=mcp_registry
DOCUMENTDB_NAMESPACE=production
DOCUMENTDB_USERNAME=admin
DOCUMENTDB_PASSWORD=<secure-password>
DOCUMENTDB_USE_TLS=true
DOCUMENTDB_TLS_CA_FILE=global-bundle.pem
DOCUMENTDB_REPLICA_SET=rs0
```

**DocumentDB Collections Created:**
Same as MongoDB CE (above), but with native HNSW vector indexes for sub-100ms semantic search.

**First-Time DocumentDB Setup:**

```bash
# 1. Deploy DocumentDB cluster via Terraform
cd terraform/aws-ecs
terraform apply

# 2. Collections and indexes are created automatically on first application startup

# 3. Verify setup (from bastion host or EC2 with access)
mongosh --host <cluster-endpoint> \
        --username admin \
        --password <password> \
        --tls \
        --tlsCAFile global-bundle.pem \
        --eval "use mcp_registry; show collections"
```

**Important Notes:**
- MongoDB CE uses application-level vector search (Python cosine similarity)
- DocumentDB uses native HNSW vector indexes for production performance
- Both backends use the same repository code (`DocumentDBServerRepository`, etc.)
- `auth_server/scopes.yml` is no longer the source of truth when using mongodb-ce or documentdb backend

**Switching Between Backends:**

You can switch between backends at any time by changing `STORAGE_BACKEND`:

```bash
# Switch to file backend
export STORAGE_BACKEND=file
docker-compose restart registry

# Switch to MongoDB CE backend
export STORAGE_BACKEND=mongodb-ce
docker-compose restart registry

# Switch to DocumentDB backend
export STORAGE_BACKEND=documentdb
docker-compose restart registry
```

**For AWS ECS Deployments:** See [terraform/aws-ecs/README.md](../terraform/aws-ecs/README.md) for automated Terraform deployment with DocumentDB.

**For Detailed Architecture:** See [Storage Architecture: MongoDB CE & AWS DocumentDB](design/storage-architecture-mongodb-documentdb.md) for comprehensive implementation details.

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

### Federation Configuration

#### WORKDAY_TOKEN_URL (Optional)

Configuration for Workday ASOR (Agent Service Orchestrator) federation integration.

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `WORKDAY_TOKEN_URL` | Workday OAuth2 token endpoint URL | `https://services.wd101.myworkday.com/ccx/oauth2/production_instance/token` | **Optional** |

**Required only if using Workday ASOR federation**

- **Default**: `https://your-tenant.workday.com/ccx/oauth2/your_instance/token` (placeholder)
- **Format**: `https://<tenant>.workday.com/ccx/oauth2/<instance>/token`
- **Example**: `https://services.wd101.myworkday.com/ccx/oauth2/production_instance/token`
- **Security**: Must use HTTPS in production environments
- **Behavior**: If not configured with a valid URL, ASOR federation will be automatically disabled with a warning logged

**Getting your Workday token URL:**

Replace the placeholder values with your actual Workday tenant identifiers:
- `<tenant>`: Your Workday tenant domain (e.g., `services.wd101.myworkday.com`)
- `<instance>`: Your Workday instance name (e.g., `production_instance`, `sandbox_instance`)

**Configuration example:**

```bash
# For production Workday instance
WORKDAY_TOKEN_URL=https://services.wd101.myworkday.com/ccx/oauth2/production_instance/token

# For sandbox/testing instance
WORKDAY_TOKEN_URL=https://services.wd101.myworkday.com/ccx/oauth2/sandbox_instance/token
```

**Troubleshooting:**

- If ASOR federation is not working, check the registry logs for warnings about WORKDAY_TOKEN_URL
- Ensure the URL uses HTTPS (HTTP will fail in production)
- Verify your Workday tenant and instance names are correct
- Contact your Workday administrator if you're unsure about your instance configuration

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
# Generate M2M token for ingress
uv run python credentials-provider/token_refresher.py

# Generate agent-specific token
uv run python credentials-provider/token_refresher.py --agent-id sre-agent
```

For detailed Keycloak integration documentation, see [Keycloak Integration Guide](keycloak-integration.md).

---

## OAuth Environment Configuration

**File:** `credentials-provider/oauth/.env`
**Purpose:** OAuth provider credentials for ingress and egress authentication flows.

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

### Egress Authentication (Optional)

Support for multiple OAuth provider configurations using numbered suffixes (`_1`, `_2`, `_3`, etc.):

| Variable Pattern | Description | Example | Required |
|------------------|-------------|---------|----------|
| `EGRESS_OAUTH_CLIENT_ID_N` | OAuth client ID for provider N | `cNYWTFwyZB...` | For each provider |
| `EGRESS_OAUTH_CLIENT_SECRET_N` | OAuth client secret for provider N | `ATOAubT-N-lAzpT05RDFq9dxcVr...` | For each provider |
| `EGRESS_OAUTH_REDIRECT_URI_N` | OAuth redirect URI for provider N | `http://localhost:8080/callback` | For each provider |
| `EGRESS_OAUTH_SCOPE_N` | OAuth scopes for provider N | Uses provider defaults if not set | Optional |
| `EGRESS_PROVIDER_NAME_N` | Provider name (google, github, etc.) | `google` | For each provider |
| `EGRESS_MCP_SERVER_NAME_N` | MCP server name for provider N | `google` | For each provider |

### Supported Providers

- **Google**: Gmail, Drive, Calendar services
- **GitHub**: Repository and issue management
- **Microsoft**: Office 365, Teams integration
- **Bedrock AgentCore**: AWS AgentCore services

---

## AgentCore Environment Configuration

**File:** `credentials-provider/agentcore-auth/.env`
**Purpose:** Amazon Bedrock AgentCore authentication configuration with support for multiple gateways.

### Shared Configuration

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `COGNITO_DOMAIN` | AgentCore Cognito domain URL | `https://your-cognito-domain.auth.region.amazoncognito.com` | ✅ |
| `COGNITO_USER_POOL_ID` | Cognito User Pool ID | `region_your_pool_id` | ✅ |

### Gateway-Specific Configurations

Support for multiple gateways using numbered suffixes (`_1`, `_2`, `_3`, etc., up to `_100`). Each configuration set requires all four parameters:

| Variable Pattern | Description | Example | Required |
|------------------|-------------|---------|----------|
| `AGENTCORE_CLIENT_ID_N` | AgentCore Cognito client ID for gateway N | `your_client_id_here` | ✅ |
| `AGENTCORE_CLIENT_SECRET_N` | AgentCore Cognito client secret for gateway N | `your_client_secret_here` | ✅ |
| `AGENTCORE_GATEWAY_ARN_N` | Amazon Bedrock AgentCore Gateway ARN for gateway N | `arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/my-gateway-1` | ✅ |
| `AGENTCORE_SERVER_NAME_N` | MCP server name for AgentCore gateway N | `my-gateway-1` | ✅ |

**Example Configuration:**
```bash
# Configuration Set 1
AGENTCORE_CLIENT_ID_1=your_client_id_here
AGENTCORE_CLIENT_SECRET_1=your_client_secret_here
AGENTCORE_GATEWAY_ARN_1=arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/my-gateway-1
AGENTCORE_SERVER_NAME_1=my-gateway-1

# Configuration Set 2
AGENTCORE_CLIENT_ID_2=your_client_id_here
AGENTCORE_CLIENT_SECRET_2=your_client_secret_here
AGENTCORE_GATEWAY_ARN_2=arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/my-gateway-2
AGENTCORE_SERVER_NAME_2=my-gateway-2
```

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

- **Keycloak**: Open-source identity and access management
- **Amazon Cognito**: Amazon managed authentication service
- **GitHub**: Repository and development services (planned)
- **Google**: Google Workspace and consumer services (planned)

---

## Scopes Configuration

**File:** `auth_server/scopes.yml`
**Purpose:** Fine-grained access control (FGAC) scope definitions.

### Scope Categories

- **MCP Servers**: Individual server access (`mcp-servers-{name}/read`, `mcp-servers-{name}/execute`)
- **Unrestricted**: Global access (`mcp-servers-unrestricted/read`, `mcp-servers-unrestricted/execute`)
- **Admin**: Administrative functions (`admin/registry`, `admin/users`)

---

## OAuth Providers Mapping

**File:** `credentials-provider/oauth/oauth_providers.yaml`
**Purpose:** Provider-specific OAuth endpoint configurations and metadata.

### Provider Fields

| Field | Description | Example |
|-------|-------------|---------|
| `auth_url` | OAuth authorization URL | `https://accounts.google.com/o/oauth2/v2/auth` |
| `token_url` | OAuth token exchange URL | `https://oauth2.googleapis.com/token` |
| `scopes` | Default OAuth scopes | `["https://www.googleapis.com/auth/drive.readonly"]` |
| `client_credentials_supported` | Whether provider supports client credentials flow | `false` |

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

---

## Viewing Configuration via UI

Administrators can view and export the current system configuration through the web interface.

### Accessing the Configuration Viewer

1. Navigate to **Settings** from the main dashboard
2. Select **System Config** > **Configuration** from the sidebar

![System Configuration Viewer](img/system-config.png)

### Features

The Configuration Viewer provides:

- **Grouped View**: Configuration parameters organized into 11 categories:
  - Deployment Mode
  - Storage Backend
  - Authentication
  - Embeddings / Vector Search
  - Health Checks
  - WebSocket Settings
  - Security Scanning (MCP Servers)
  - Security Scanning (Agents)
  - Audit Logging
  - Federation
  - Well-Known Discovery

- **Search**: Filter configuration parameters by name or value
- **Expand/Collapse**: View all groups or focus on specific categories
- **Sensitive Value Masking**: Passwords, API keys, and secrets are automatically masked
- **Statistics**: Quick overview showing total, enabled, disabled, and issue counts

### Export Options

Click the **Export** button to download configuration in multiple formats:

| Format | Description | Use Case |
|--------|-------------|----------|
| ENV | Shell environment variables | Docker/shell deployment |
| JSON | Structured JSON format | Programmatic access |
| TFVARS | Terraform variables | Infrastructure as Code |
| YAML | YAML format | Kubernetes ConfigMaps |

**Note**: Sensitive values are masked by default in exports. Use `include_sensitive=true` with caution.

---

## Configuration API

The registry provides REST API endpoints for programmatic configuration access.

### GET /api/config

Returns basic configuration information (public endpoint).

```bash
curl -X GET "https://your-registry/api/config"
```

**Response:**
```json
{
  "deployment_mode": "with-gateway",
  "registry_mode": "full",
  "nginx_updates_enabled": true,
  "features": {
    "mcp_servers": true,
    "agents": true,
    "skills": true
  }
}
```

### GET /api/config/full

Returns complete configuration grouped by category (admin only).

```bash
curl -X GET "https://your-registry/api/config/full" \
  -H "Cookie: session=<session-cookie>"
```

**Response:**
```json
{
  "groups": {
    "deployment": {
      "title": "Deployment Mode",
      "order": 1,
      "fields": {
        "deployment_mode": {
          "label": "Deployment Mode",
          "value": { "raw": "with-gateway", "display": "with-gateway", "is_masked": false }
        }
      }
    }
  },
  "generated_at": "2025-01-15T10:30:00Z"
}
```

### GET /api/config/export

Export configuration in various formats (admin only).

```bash
# Export as ENV format
curl -X GET "https://your-registry/api/config/export?format=env" \
  -H "Cookie: session=<session-cookie>"

# Export as JSON with sensitive values
curl -X GET "https://your-registry/api/config/export?format=json&include_sensitive=true" \
  -H "Cookie: session=<session-cookie>"
```

**Query Parameters:**

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `format` | `env`, `json`, `tfvars`, `yaml` | `env` | Export format |
| `include_sensitive` | `true`, `false` | `false` | Include sensitive values (use with caution) |

**Rate Limiting**: These endpoints are rate-limited to 10 requests per minute per user.