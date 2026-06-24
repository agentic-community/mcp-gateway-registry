# Authentication and User Management Guide

This guide describes how to manage groups, users, and M2M (machine-to-machine) service accounts in the MCP Gateway Registry, and how to generate JWT tokens for authentication.

> **SECURITY WARNING**
>
> The examples in this document use placeholder credentials for demonstration purposes only.
> **NEVER use these example values in production.**
>
> Always generate unique, secure credentials and store them in:
> - AWS Secrets Manager (production)
> - Environment variables (development)
> - `.env` files (local only, never commit)

## Table of Contents

1. [Overview](#overview)
2. [Bootstrap State](#bootstrap-state)
3. [Creating Groups](#creating-groups)
4. [Creating Human Users](#creating-human-users)
5. [Creating M2M Service Accounts](#creating-m2m-service-accounts)
6. [Generating JWT Tokens](#generating-jwt-tokens)
   - [For Human Users (via UI)](#for-human-users-via-ui)
   - [For M2M Accounts (via generate_creds.sh)](#for-m2m-accounts-via-generate_credssh)
7. [Provider-Specific Notes](#provider-specific-notes)

---

## Overview

The MCP Gateway Registry supports two identity providers:
- **Keycloak** - Self-hosted identity provider with full automation support
- **Microsoft Entra ID** - Enterprise Azure AD integration

Both providers use the same CLI interface (`registry_management.py`) for user and group management, with minor differences in configuration.

---

## Bootstrap State

When the system is first deployed, it is bootstrapped with **minimal configuration**:

### Initial Bootstrap (Both Providers)

| Component | Description |
|-----------|-------------|
| **registry-admins** group | Administrative group with full registry access |
| **Admin user** | Initial administrator account |
| **Admin scopes** | `registry-admins` scope mapped to the admin group |

### Keycloak Bootstrap

For Keycloak deployments, the `init-keycloak.sh` script automatically creates:
- The `mcp-gateway` realm
- `mcp-gateway-web` client (for web UI)
- `mcp-gateway-m2m` client (for M2M authentication)
- Initial admin user and `registry-admins` group

### Entra ID Bootstrap

For Entra ID deployments:
- The `registry-admins` group **must be created manually** in Azure Portal
- The Group Object ID is required when running the DocumentDB initialization script:
  ```bash
  ./terraform/aws-ecs/scripts/run-documentdb-init.sh --entra-group-id "your-group-object-id"
  ```

See [Entra ID Setup Guide](./entra-id-setup.md) for detailed Entra ID configuration instructions.

**All additional groups, users, and M2M accounts must be created as described below.**

---

## Creating Groups

Groups control access to MCP servers and registry resources. Users and M2M accounts are assigned to groups to receive their permissions.

### Prerequisites

You need an admin token to create groups. You can obtain one by:
- **UI Method**: Log in to the registry web UI and click the **"Get JWT Token"** button in the top-left sidebar. Save the token to `api/.token`.
- **M2M Method**: Create an M2M account with admin permissions and generate a token using `generate_creds.sh` (see [Generating JWT Tokens](#generating-jwt-tokens)).

### Create a Group Definition File

Create a JSON file defining the group (e.g., `my-group.json`):

```json
{
  "scope_name": "public-mcp-users",
  "description": "Users with access to public MCP servers",
  "servers": [
    {
      "server_name": "currenttime",
      "tools": ["get_current_time"],
      "access_level": "execute"
    },
    {
      "server_name": "mcpgw",
      "tools": ["*"],
      "access_level": "execute"
    }
  ],
  "create_in_idp": true
}
```

**Key fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `scope_name` | Yes | Unique identifier for the group/scope |
| `description` | Yes | Human-readable description |
| `servers` | Yes | List of server access configurations |
| `create_in_idp` | No | If `true`, creates the group in the identity provider (Keycloak/Entra ID) |

### Import the Group

```bash
uv run python api/registry_management.py \
  --token-file api/.token \
  --registry-url https://registry.us-east-1.example.com \
  import-group --file my-group.json
```

### Example Group Definitions

See the [cli/examples/](../cli/examples/) directory for sample group definitions:
- [public-mcp-users.json](../cli/examples/public-mcp-users.json) - Public access group with access to context7, cloudflare-docs servers and flight-booking agent
- `currenttime-users.json` - Access to currenttime server only

### Bootstrap Admin Scope

The `registry-admins` scope is automatically loaded during database initialization from [scripts/registry-admins.json](../scripts/registry-admins.json). This file defines full administrative access:

```json
{
  "_id": "registry-admins",
  "group_mappings": ["registry-admins"],
  "server_access": [
    {
      "server": "*",
      "methods": ["all"],
      "tools": ["all"]
    }
  ]
}
```

This is loaded by the database initialization scripts:
- **Local (MongoDB CE)**: `docker compose up mongodb-init` runs `scripts/init-mongodb-ce.py`
- **Production (DocumentDB)**: `./terraform/aws-ecs/scripts/run-documentdb-init.sh` runs `scripts/init-documentdb-indexes.py`
- **Entra ID**: `./terraform/aws-ecs/scripts/run-documentdb-init.sh --entra-group-id "your-group-object-id"`

For Entra ID, the `--entra-group-id` parameter adds the Entra ID Group Object ID to the `group_mappings` array so that members of that Azure AD group receive admin permissions.

---

## Creating Human Users

Human users can log in via the web UI using OAuth2 authentication (Keycloak or Entra ID).

### Create a Human User

```bash
uv run python api/registry_management.py \
  --token-file api/.token \
  --registry-url https://registry.us-east-1.example.com \
  user-create-human \
  --username jsmith \
  --email jsmith@example.com \
  --first-name John \
  --last-name Smith \
  --groups public-mcp-users \
  --password "SecurePassword123!"
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--username` | Yes | Unique username for the user |
| `--email` | Yes | Email address |
| `--first-name` | Yes | First name |
| `--last-name` | Yes | Last name |
| `--groups` | Yes | Comma-separated list of groups to assign |
| `--password` | Yes | Initial password (user should change on first login) |

### Multiple Groups

To assign a user to multiple groups:

```bash
uv run python api/registry_management.py \
  --token-file api/.token \
  --registry-url https://registry.us-east-1.example.com \
  user-create-human \
  --username analyst \
  --email analyst@example.com \
  --first-name Data \
  --last-name Analyst \
  --groups "public-mcp-users,analytics-team" \
  --password "SecurePassword123!"
```

---

## Creating M2M Service Accounts

M2M (machine-to-machine) accounts are used for programmatic API access by AI coding assistants, agents, and automated systems.

### Create an M2M Service Account

```bash
uv run python api/registry_management.py \
  --token-file api/.token \
  --registry-url https://registry.us-east-1.example.com \
  user-create-m2m \
  --name my-ai-agent \
  --groups public-mcp-users \
  --description "AI coding assistant service account"
```

**Output:**

```
Client ID: my-ai-agent
Client Secret: sqFaOkF8un1tAfKXjlgm2xjGQBfLlNS3
Groups: public-mcp-users

IMPORTANT: Save the client secret securely - it cannot be retrieved later.
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--name` | Yes | Unique name for the M2M account (becomes the Client ID) |
| `--groups` | Yes | Comma-separated list of groups to assign |
| `--description` | No | Description of the service account's purpose |

### Save the Credentials

**The client secret is only displayed once.** Save it immediately to a secure location:

```bash
# Create an agent configuration file for use with generate_creds.sh
cat > .oauth-tokens/agent-my-ai-agent.json << 'EOF'
{
  "client_id": "my-ai-agent",
  "client_secret": "sqFaOkF8un1tAfKXjlgm2xjGQBfLlNS3",
  "keycloak_url": "https://kc.us-east-1.example.com",
  "keycloak_realm": "mcp-gateway",
  "auth_provider": "keycloak"
}
EOF
```

For Entra ID, add the identity to `.oauth-tokens/entra-identities.json`:

```json
[
  {
    "identity_name": "my-ai-agent",
    "tenant_id": "your-tenant-id",
    "client_id": "client-id-from-output",
    "client_secret": "client-secret-from-output",
    "scope": "api://your-app-client-id/.default"
  }
]
```

---

## Generating JWT Tokens

### For Human Users (via UI)

Human users generate JWT tokens through the MCP Gateway Registry web interface:

1. **Log in** to the registry at `https://registry.us-east-1.example.com`
2. Click the **"Get JWT Token"** button in the top-left sidebar
3. **Copy the generated token**

These self-signed tokens:
- Are signed with HS256 using the server's `SECRET_KEY`
- Include the user's groups and scopes
- Can be used for programmatic API access
- Have a configurable expiration time

**Using the token:**

```bash
# Save to a token file
echo '{"access_token": "eyJhbGciOi..."}' > api/.token

# Use with registry_management.py
uv run python api/registry_management.py \
  --token-file api/.token \
  --registry-url https://registry.us-east-1.example.com \
  list
```

### For M2M Accounts (via generate_creds.sh)

M2M accounts generate tokens using the OAuth2 client credentials flow via the `generate_creds.sh` script.

#### Step 1: Configure the Agent

Create an agent configuration file in `.oauth-tokens/`:

**For Keycloak:**

```json
{
  "client_id": "my-ai-agent",
  "client_secret": "sqFaOkF8un1tAfKXjlgm2xjGQBfLlNS3",
  "keycloak_url": "https://kc.us-east-1.example.com",
  "keycloak_realm": "mcp-gateway",
  "auth_provider": "keycloak"
}
```

Save as `.oauth-tokens/agent-my-ai-agent.json`

**For Entra ID:**

Edit `.oauth-tokens/entra-identities.json`:

```json
[
  {
    "identity_name": "my-ai-agent",
    "tenant_id": "6e6ee81b-6bf3-495d-a7fc-d363a551f765",
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "scope": "api://1bd17ba1-aad3-447f-be0b-26f8f9ee859f/.default"
  }
]
```

#### Step 2: Generate the Token

**For Keycloak:**

```bash
./credentials-provider/generate_creds.sh \
  -a keycloak \
  -k https://kc.us-east-1.example.com
```

**For Entra ID:**

```bash
./credentials-provider/generate_creds.sh \
  -a entra \
  -i .oauth-tokens/entra-identities.json
```

#### Step 3: Use the Generated Token

The script saves tokens to `.oauth-tokens/agent-<name>-token.json`:

```bash
# List servers using the generated token
uv run python api/registry_management.py \
  --token-file .oauth-tokens/agent-my-ai-agent-token.json \
  --registry-url https://registry.us-east-1.example.com \
  list
```

### generate_creds.sh Options

```
./credentials-provider/generate_creds.sh [OPTIONS]

OPTIONS:
    --auth-provider, -a PROVIDER       Auth provider: 'keycloak' or 'entra' (required)
    --keycloak-url, -k URL             Keycloak server URL (required for keycloak)
    --keycloak-realm, -r REALM         Keycloak realm name (default: mcp-gateway)
    --entra-tenant-id TENANT_ID        Entra tenant ID
    --entra-client-id CLIENT_ID        Entra client ID
    --entra-client-secret SECRET       Entra client secret
    --entra-login-url URL              Entra login base URL (default: https://login.microsoftonline.com)
    --identities-file, -i FILE         Custom path to identities JSON file (for entra)
    --verbose, -v                      Enable verbose debug logging
    --help, -h                         Show help message

EXAMPLES:
    # Keycloak
    ./generate_creds.sh -a keycloak -k https://kc.example.com

    # Entra ID with identities file
    ./generate_creds.sh -a entra -i .oauth-tokens/entra-identities.json

    # Keycloak with verbose output
    ./generate_creds.sh -a keycloak -k https://kc.example.com -v
```

### Manual Token Generation (curl)

You can also generate tokens directly using curl:

**Keycloak:**

```bash
curl -s -X POST "https://kc.us-east-1.example.com/realms/mcp-gateway/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=my-ai-agent" \
  -d "client_secret=sqFaOkF8un1tAfKXjlgm2xjGQBfLlNS3" \
  -d "grant_type=client_credentials"
```

**Entra ID:**

```bash
curl -s -X POST "https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id={M2M_CLIENT_ID}" \
  -d "client_secret={M2M_CLIENT_SECRET}" \
  -d "scope=api://{APP_CLIENT_ID}/.default" \
  -d "grant_type=client_credentials"
```

---

## Provider-Specific Notes

### Keycloak

- Groups are identified by **name** (e.g., `registry-admins`)
- M2M accounts are created as Keycloak clients with service accounts
- Token lifetime is configurable in Keycloak realm settings (default: 5 minutes)
- Supports automatic group and user creation via API

### Entra ID

- Groups are identified by **Object ID** (UUID, e.g., `16c7e67e-e8ae-498c-ba2e-0593c0159e43`)
- M2M accounts are Azure App Registrations with client credentials
- Token lifetime is typically 1 hour
- Groups must be created manually in Azure Portal before use
- Group Object IDs are required for scope mappings in the `mcp_scopes` collection in DocumentDB

### Token Comparison

| Aspect | Human User Token | M2M Token |
|--------|------------------|-----------|
| **Generation** | UI "Get JWT Token" button | `generate_creds.sh` or curl |
| **Algorithm** | HS256 (self-signed) | RS256 (IdP-signed) |
| **Validation** | Server SECRET_KEY | JWKS from IdP |
| **Use Case** | Interactive/programmatic access | Automated systems, AI agents |
| **Refresh** | Generate new via UI | Use client credentials flow |

---

## See Also

- [Entra ID Setup Guide](./entra-id-setup.md) - Complete Entra ID configuration
- [Complete Setup Guide](./complete-setup-guide.md) - Initial system setup
- [Terraform AWS ECS README](../terraform/aws-ecs/README.md) - Production deployment
