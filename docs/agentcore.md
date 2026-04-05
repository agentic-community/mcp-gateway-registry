# Registering Amazon Bedrock AgentCore Assets

This guide covers how to register AgentCore Gateways and Agent Runtimes in the MCP Gateway Registry. There are two approaches: bulk auto-registration (scan an entire AWS account) or per-server manual registration (one resource at a time).

## Two Ways to Register AgentCore Assets

| Approach | Best For | Token Management |
|----------|----------|-----------------|
| **Method 1: Bulk Scanner** | Discovering and registering all resources in an AWS account at once | Automated via `token_refresher.py` |
| **Method 2: Per-Server Registration** | Registering individual gateways or agents manually (same as any other MCP server/agent) | Manual token generation |

---

## Method 1: Bulk Scanner (Auto-Registration)

The AgentCore scanner CLI automates the discovery and registration of all AgentCore Gateways and Agent Runtimes in your AWS account. Instead of manually creating JSON configuration files for each resource, the CLI scans your account, builds registrations, and writes a token refresh manifest -- all in one command. A separate token refresher process then keeps egress tokens up to date.

The scanner and token refresher work with any OIDC-compliant identity provider -- Cognito, Auth0, Okta, Entra ID, Keycloak, or any custom provider. The IdP is auto-detected from the OIDC discovery URL in each gateway's configuration.

> **Prerequisites:** Before using auto-registration, complete the setup steps in the [Auto-Registration Prerequisites Guide](agentcore-auto-registration-prerequisites.md).

### Step 1: Scan and Register

```bash
# Discover resources without registering (preview)
uv run python -m cli.agentcore sync --dry-run

# Register all gateways and runtimes
uv run python -m cli.agentcore sync

# Overwrite existing registrations (update metadata if changed)
uv run python -m cli.agentcore sync --overwrite

# List discovered resources without registering
uv run python -m cli.agentcore list
```

The sync command:
1. Discovers all READY gateways and runtimes via AWS Bedrock AgentCore API
2. Registers each gateway as an MCP Server and each runtime as an MCP Server (protocol=MCP) or A2A Agent (protocol=HTTP/A2A)
3. Writes `token_refresh_manifest.json` listing all CUSTOM_JWT gateways that need token refresh

> **Note:** Agents imported from runtimes are registered with an empty skills array. To add skills after import, use the agent edit dialog in the UI or the `PUT /api/agents/{path}` API endpoint.

### Step 2: Configure Client Secrets (for CUSTOM_JWT Gateways)

CUSTOM_JWT gateways need OAuth2 client secrets to generate egress tokens. Add them to your `.env` file:

```bash
# Per-client secret (highest priority) -- use the client_id from allowed_clients
OAUTH_CLIENT_SECRET_49ujl0b9ser72gnp6q1ph9v6vs=your-secret-here

# Or vendor-level secrets (shared across all gateways for that IdP)
AUTH0_CLIENT_SECRET=your-auth0-secret
OKTA_CLIENT_SECRET=your-okta-secret
ENTRA_CLIENT_SECRET=your-entra-secret
KEYCLOAK_CLIENT_SECRET=your-keycloak-secret
```

**Cognito gateways need no configuration** -- the token refresher auto-retrieves client secrets via the AWS API (`describe_user_pool_client`).

Secret resolution priority:
1. Per-client env var: `OAUTH_CLIENT_SECRET_<client_id>`
2. Cognito auto-retrieval via AWS API (Cognito only)
3. Vendor-specific env var: `AUTH0_CLIENT_SECRET`, etc.

### Step 3: Run Token Refresher

The token refresher reads the manifest, resolves secrets, fetches OAuth2 tokens, PATCHes them into the registry, and triggers a security rescan for each updated server (enabled by default, requires admin privileges on the registry token):

```bash
# One-time refresh
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token

# Continuous mode (sidecar -- refreshes every 45 minutes)
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token \
    --loop --interval 2700
```

Or set up a cron job:

```bash
# Refresh every 45 minutes (tokens typically expire in 60 min)
*/45 * * * * cd /app && uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token \
    >> /var/log/token-refresher.log 2>&1
```

### Scanner CLI Reference

#### Sync -- Discover and Register

```bash
# Basic sync
uv run python -m cli.agentcore sync

# Dry-run preview
uv run python -m cli.agentcore sync --dry-run

# Overwrite existing registrations
uv run python -m cli.agentcore sync --overwrite

# Register only gateways (skip runtimes)
uv run python -m cli.agentcore sync --gateways-only

# Register only runtimes (skip gateways)
uv run python -m cli.agentcore sync --runtimes-only

# Also register individual mcpServer gateway targets as separate MCP Servers
uv run python -m cli.agentcore sync --include-mcp-targets

# Set visibility for registered resources
uv run python -m cli.agentcore sync --visibility public

# JSON output for CI/CD pipelines
uv run python -m cli.agentcore sync --output json

# Specify region and registry URL
uv run python -m cli.agentcore sync --region us-west-2 --registry-url https://registry.example.com

# Custom token file and timeout
uv run python -m cli.agentcore sync --token-file .token --timeout 60

# Enable debug logging
uv run python -m cli.agentcore sync --debug
```

#### List -- Discover and Display

```bash
# List all discovered resources
uv run python -m cli.agentcore list

# List only gateways
uv run python -m cli.agentcore list --gateways-only

# List only runtimes
uv run python -m cli.agentcore list --runtimes-only

# JSON output
uv run python -m cli.agentcore list --output json

# Specify region
uv run python -m cli.agentcore list --region eu-west-1
```

#### Token Refresher

```bash
# One-time refresh
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token

# With per-client env vars
OAUTH_CLIENT_SECRET_49ujl0b9ser72gnp6q1ph9v6vs=secret \
    uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token

# With vendor-level env vars
AUTH0_CLIENT_SECRET=xxx OKTA_CLIENT_SECRET=yyy \
    uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token

# Continuous mode (sidecar)
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token \
    --loop --interval 2700

# Enable debug logging
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --token-file .token \
    --debug
```

### Scanner CLI Arguments

| Argument | Subcommand | Default | Description |
|----------|------------|---------|-------------|
| `--region` | sync, list | `AWS_REGION` env or `us-east-1` | AWS region to scan |
| `--registry-url` | sync, list | `REGISTRY_URL` env or `http://localhost` | Registry base URL |
| `--token-file` | sync, list | `REGISTRY_TOKEN_FILE` env or `.token` | Path to registry auth token file |
| `--timeout` | sync, list | `30` | AWS API call timeout in seconds |
| `--gateways-only` | sync, list | `false` | Only process gateways |
| `--runtimes-only` | sync, list | `false` | Only process runtimes |
| `--output` | sync, list | `text` | Output format: `text` or `json` |
| `--accounts` | sync, list | `AGENTCORE_ACCOUNTS` env or empty | Comma-separated AWS account IDs for cross-account scanning |
| `--assume-role-name` | sync, list | `AGENTCORE_ASSUME_ROLE_NAME` env or `AgentCoreSyncRole` | IAM role name to assume in each target account |
| `--debug` | sync, list | `false` | Enable DEBUG logging |
| `--dry-run` | sync | `false` | Preview without registering |
| `--overwrite` | sync | `false` | Overwrite existing registrations |
| `--visibility` | sync | `internal` | Registration visibility: `public`, `internal`, `group-restricted` |
| `--include-mcp-targets` | sync | `false` | Register mcpServer gateway targets as separate MCP Servers |
| `--manifest` | sync | `token_refresh_manifest.json` | Output path for token refresh manifest |

### Token Refresher Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest` | `token_refresh_manifest.json` | Path to manifest file |
| `--registry-url` | `REGISTRY_URL` env or `http://localhost` | Registry base URL |
| `--token-file` | `REGISTRY_TOKEN_FILE` env or `.token` | Registry auth token file |
| `--loop` | `false` | Run continuously |
| `--interval` | `2700` (45 min) | Refresh interval in seconds |
| `--scan` / `--no-scan` | `--scan` (enabled) | Trigger security rescan after each credential update. Requires admin privileges on the registry token. Use `--no-scan` to disable. |
| `--debug` | `false` | Enable DEBUG logging |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region to scan |
| `REGISTRY_URL` | `http://localhost` | MCP Gateway Registry URL |
| `REGISTRY_TOKEN_FILE` | `.token` | Path to registry auth token |
| `OAUTH_CLIENT_SECRET_<client_id>` | -- | Per-client OAuth2 secret (highest priority) |
| `AUTH0_CLIENT_SECRET` | -- | Client secret for Auth0 gateways |
| `OKTA_CLIENT_SECRET` | -- | Client secret for Okta gateways |
| `ENTRA_CLIENT_SECRET` | -- | Client secret for Entra gateways |
| `KEYCLOAK_CLIENT_SECRET` | -- | Client secret for Keycloak gateways |
| `AGENTCORE_ACCOUNTS` | -- | Comma-separated AWS account IDs for cross-account scanning |
| `AGENTCORE_ASSUME_ROLE_NAME` | `AgentCoreSyncRole` | IAM role name to assume in each target account |

### Cross-Account Scanning

The CLI can scan multiple AWS accounts in a single run. It assumes an IAM role in each target account to discover and register resources.

```bash
# Scan two accounts
uv run python -m cli.agentcore sync --accounts 111111111111,222222222222

# Scan with a custom role name
uv run python -m cli.agentcore sync --accounts 111111111111,222222222222 --assume-role-name MyCrossAccountRole

# List resources across accounts
uv run python -m cli.agentcore list --accounts 111111111111,222222222222

# Or use environment variables
export AGENTCORE_ACCOUNTS=111111111111,222222222222
export AGENTCORE_ASSUME_ROLE_NAME=AgentCoreSyncRole
uv run python -m cli.agentcore sync
```

How it works:

1. The CLI parses the `--accounts` flag (or `AGENTCORE_ACCOUNTS` env var) into a list of account IDs.
2. For each account, it calls `sts:AssumeRole` on `arn:aws:iam::{account_id}:role/{role_name}` to obtain temporary credentials.
3. A boto3 session is created with those temporary credentials and passed to the scanner and registration builder.
4. Discovery and registration proceed as normal, scoped to each account's resources.
5. If `--accounts` is not provided, the CLI scans only the current account (default behavior).

If `AssumeRole` fails for any account, the CLI stops and reports the error.

Each target account needs an IAM role that trusts the caller's account and has AgentCore discovery permissions. See the [Cross-Account IAM Prerequisites](agentcore-auto-registration-prerequisites.md#cross-account-scanning) for setup details.

### Troubleshooting Auto-Registration

#### `AccessDeniedException` during discovery

The IAM user or role lacks required permissions. Attach the discovery policy from the [prerequisites guide](agentcore-auto-registration-prerequisites.md#iam-permissions-for-discovery).

#### "Already registered - skipping (use --overwrite)"

The resource is already registered in the registry. Use `--overwrite` to update the existing registration with current metadata.

#### Token refresher returns HTTP 500

If the token refresher logs `HTTP 500 from nginx -- registry token may be expired`, regenerate the registry auth token and retry:

```bash
# Regenerate the registry ingress token
python credentials-provider/oauth/ingress_oauth.py

# Retry token refresh
uv run python -m cli.agentcore.token_refresher --manifest token_refresh_manifest.json --token-file .token
```

#### Token file not found

The registry auth token file (default: `.token`) does not exist. Generate it with:

```bash
python credentials-provider/oauth/ingress_oauth.py
```

#### Dry-run shows resources but sync registers nothing

In `--dry-run` mode, the CLI performs discovery but does not register. Remove the `--dry-run` flag to perform actual registration.

#### Timeout errors on AWS API calls

Increase the timeout with `--timeout 60` (or higher). The default is 30 seconds.

---

## Method 2: Per-Server Manual Registration

For registering individual AgentCore gateways or agents one at a time, use the same registration process as any other MCP server or agent in the registry. This is no different from how you register any MCP server or A2A agent -- you create a JSON configuration file and use the service management CLI.

This approach is useful when:
- You want to register a single gateway without scanning the entire account
- You need custom configuration (specific tool lists, descriptions, tags)
- You are integrating a specific AgentCore sample (e.g., Customer Support Assistant)

### How It Works

1. **Create a JSON config file** describing the gateway or agent (path, proxy URL, auth scheme, tags, tool list)
2. **Register with the CLI**: `./cli/service_mgmt.sh add gateway-config.json`
3. **Provide a JWT token** when calling the gateway -- the IdP does not matter, just provide a valid bearer token at call time via `--token-file`
4. **Refresh the token** when it expires -- how you obtain the token is up to you (curl, SDK, script)

The identity provider is irrelevant for manual registration. The registry uses passthrough authentication for `auth_provider: "bedrock-agentcore"` -- it forwards the bearer token to the AgentCore gateway, which validates it against whatever IdP is configured.

### Example JSON Configuration

```json
{
  "server_name": "customer-support-assistant",
  "description": "Amazon Bedrock AgentCore Gateway for customer support operations",
  "path": "/customer-support-assistant",
  "proxy_pass_url": "https://<YOUR-GATEWAY-ID>.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp/",
  "auth_provider": "bedrock-agentcore",
  "auth_scheme": "bearer",
  "supported_transports": ["streamable-http"],
  "tags": ["bedrock", "agentcore", "customer-support"],
  "headers": [
    {
      "Authorization": "Bearer $CUSTOMER_SUPPORT_AUTH_TOKEN"
    }
  ],
  "num_tools": 2,
  "is_python": false,
  "tool_list": [
    {
      "name": "LambdaUsingSDK___check_warranty_status",
      "parsed_description": {
        "main": "Check the warranty status of a product using its serial number"
      },
      "schema": {
        "type": "object",
        "properties": {
          "serial_number": {"type": "string", "description": "Product serial number"}
        },
        "required": ["serial_number"]
      }
    }
  ]
}
```

**Key Configuration Parameters:**

| Parameter | Description |
|-----------|-------------|
| `path` | URL path where this service is accessible through the registry |
| `proxy_pass_url` | Backend AgentCore Gateway URL. Replace `<YOUR-GATEWAY-ID>` with your actual Gateway ID |
| `auth_provider` | Set to `bedrock-agentcore` for passthrough authentication -- the registry forwards the bearer token without validating it |
| `tags` | Searchable tags used by `intelligent_tool_finder` for hybrid search (semantic + tag-based) |
| `tool_list` | Tool definitions with names, descriptions, and JSON schemas. Enables the registry to catalog tools for dynamic discovery by AI agents |

### Register and Call

```bash
# Register the gateway
./cli/service_mgmt.sh add gateway-config.json

# Call a tool through the registry (provide a valid JWT from any IdP)
uv run cli/mcp_client.py \
  --url http://localhost/customer-support-assistant/mcp \
  --token-file .cognito_access_token \
  call --tool LambdaUsingSDK___check_warranty_status \
  --args '{"serial_number":"MNO33333333"}'
```

### When to Choose Each Method

| | Method 1: Bulk Scanner | Method 2: Manual Registration |
|---|---|---|
| **Discovery** | Automatic (scans AWS account) | Manual (you provide the config) |
| **Token refresh** | Automated (`token_refresher.py`) | Manual (you manage token lifecycle) |
| **Customization** | Standard metadata from AWS API; skills must be added manually after import | Full control (tool lists, descriptions, tags, skills) |
| **Scale** | All gateways/runtimes at once | One resource at a time |
| **IdP** | Auto-detected from discovery URL | Any -- just provide a valid JWT |

---

## Troubleshooting

### 404 Not Found Error

Verify:
1. Service is registered: `uv run cli/mcp_client.py --url http://localhost/mcpgw/mcp call --tool list_services --args '{}'`
2. Path matches (use trailing slash for bedrock-agentcore services)
3. Health status is healthy in the UI

### 401 Authentication Error

1. Refresh your egress access token (regenerate from your IdP)
2. Verify token file path is correct
3. Check token has not expired (TTL varies by IdP)

### Service Not Showing as Healthy

1. Verify AgentCore gateway is accessible from the registry container
2. Check network connectivity
3. Review registry logs: `docker logs mcp-gateway-registry-registry-1`
