# Registry Deployment and Registry Mode Configuration

This guide explains the `DEPLOYMENT_MODE` and `REGISTRY_MODE` environment variables that control how the MCP Gateway Registry operates.

## Overview

The registry supports two configuration settings that control its behavior:

| Setting | Purpose | Options |
|---------|---------|---------|
| `DEPLOYMENT_MODE` | Controls nginx/gateway integration | `with-gateway`, `registry-only` |
| `REGISTRY_MODE` | Controls which features are enabled (informational) | `full`, `skills-only`, `mcp-servers-only`, `agents-only` |

## DEPLOYMENT_MODE

The `DEPLOYMENT_MODE` setting determines whether the registry operates as a full gateway with nginx reverse proxy integration, or as a standalone catalog/discovery service.

### Mode: with-gateway (Default)

```bash
DEPLOYMENT_MODE=with-gateway
```

**Behavior:**
- Nginx configuration is regenerated when MCP servers are registered or deleted
- Frontend shows gateway authentication instructions (Authorization: Bearer token)
- MCP proxy requests are routed through nginx to backend servers
- Full gateway functionality enabled

**Use when:**
- Running the registry as part of the MCP Gateway infrastructure
- MCP servers are accessed through the nginx reverse proxy
- You need centralized authentication and routing

### Mode: registry-only

```bash
DEPLOYMENT_MODE=registry-only
```

**Behavior:**
- Nginx configuration is NOT updated when servers are registered/deleted
- Frontend shows "Direct Connection Mode" with `proxy_pass_url`
- MCP proxy requests return 503 Service Unavailable with JSON error
- Registry operates as a catalog/discovery service only

**Use when:**
- Registry is separate from gateway infrastructure
- Clients connect directly to MCP servers (not through gateway)
- You only need server/agent discovery and metadata management

## REGISTRY_MODE

The `REGISTRY_MODE` setting controls which feature flags are returned in the `/api/config` endpoint. This is informational and intended for frontend UI feature gating.

**Note:** Currently, all APIs remain active regardless of this setting. The feature flags are for UI display purposes only.

### Mode Comparison Table

| Mode | MCP Servers | Agents | Skills | Federation | Gateway Proxy |
|------|-------------|--------|--------|------------|---------------|
| `full` | Enabled | Enabled | Enabled | Enabled | Based on DEPLOYMENT_MODE |
| `skills-only` | Disabled | Disabled | Enabled | Disabled | Disabled |
| `mcp-servers-only` | Enabled | Disabled | Disabled | Disabled | Based on DEPLOYMENT_MODE |
| `agents-only` | Disabled | Enabled | Disabled | Disabled | Based on DEPLOYMENT_MODE |

### Mode: full (Default)

```bash
REGISTRY_MODE=full
```

All features enabled. The `gateway_proxy` flag depends on `DEPLOYMENT_MODE`.

### Mode: skills-only

```bash
REGISTRY_MODE=skills-only
```

Only the skills feature flag is enabled. Intended for deployments focused solely on Agent Skills management.

### Mode: mcp-servers-only

```bash
REGISTRY_MODE=mcp-servers-only
```

Only the MCP servers feature flag is enabled.

### Mode: agents-only

```bash
REGISTRY_MODE=agents-only
```

Only the A2A agents feature flag is enabled.

## Configuration Combinations

### Valid Combinations

| DEPLOYMENT_MODE | REGISTRY_MODE | Use Case |
|-----------------|---------------|----------|
| `with-gateway` | `full` | Full MCP Gateway with all features |
| `with-gateway` | `mcp-servers-only` | Gateway for MCP servers only |
| `with-gateway` | `agents-only` | Gateway for A2A agents only |
| `registry-only` | `full` | Standalone catalog with all metadata |
| `registry-only` | `skills-only` | Skills catalog only |
| `registry-only` | `mcp-servers-only` | MCP server catalog only |
| `registry-only` | `agents-only` | Agent catalog only |

### Invalid Combination (Auto-Corrected)

| DEPLOYMENT_MODE | REGISTRY_MODE | Auto-Corrected To |
|-----------------|---------------|-------------------|
| `with-gateway` | `skills-only` | `registry-only` + `skills-only` |

**Rationale:** Skills-only mode doesn't require gateway proxy functionality. The system automatically corrects this invalid combination and logs a warning.

## API Configuration Endpoint

The `/api/config` endpoint returns the current configuration:

```bash
curl http://localhost/api/config
```

**Example Response (with-gateway + full):**
```json
{
  "deployment_mode": "with-gateway",
  "registry_mode": "full",
  "nginx_updates_enabled": true,
  "features": {
    "mcp_servers": true,
    "agents": true,
    "skills": true,
    "federation": true,
    "gateway_proxy": true
  }
}
```

**Example Response (registry-only + skills-only):**
```json
{
  "deployment_mode": "registry-only",
  "registry_mode": "skills-only",
  "nginx_updates_enabled": false,
  "features": {
    "mcp_servers": false,
    "agents": false,
    "skills": true,
    "federation": false,
    "gateway_proxy": false
  }
}
```

## Environment Configuration

### Docker Compose

In your `.env` file:

```bash
# Deployment mode: with-gateway (default) or registry-only
DEPLOYMENT_MODE=registry-only

# Registry mode: full (default), skills-only, mcp-servers-only, or agents-only
REGISTRY_MODE=skills-only
```

### Terraform (AWS ECS)

In `terraform.tfvars`:

```hcl
# Deployment mode
deployment_mode = "registry-only"

# Registry mode (optional, defaults to "full")
registry_mode = "skills-only"
```

Or via environment variables:

```bash
export TF_VAR_deployment_mode="registry-only"
export TF_VAR_registry_mode="skills-only"
```

## Frontend Behavior

### ServerConfigModal

The `ServerConfigModal` component adapts based on `deployment_mode`:

**with-gateway mode:**
- Shows gateway URL constructed from current hostname
- Displays "Authentication Required" warning
- Shows `[YOUR_AUTH_TOKEN]` placeholder in configuration

**registry-only mode:**
- Shows `proxy_pass_url` (direct server URL)
- Displays "Direct Connection Mode" banner
- No gateway authentication headers in configuration

### Feature Flags (Future)

The `features` object in `/api/config` is intended for frontend navigation gating:

```typescript
const { config } = useRegistryConfig();

// Hide navigation items based on features
{config?.features.mcp_servers && <NavItem>MCP Servers</NavItem>}
{config?.features.agents && <NavItem>A2A Agents</NavItem>}
{config?.features.skills && <NavItem>Skills</NavItem>}
{config?.features.federation && <NavItem>Federation</NavItem>}
```

**Note:** This frontend gating is not yet implemented. Currently all navigation items are visible regardless of mode.

## Startup Logging

The registry logs its configuration at startup:

```
INFO: Registry Configuration:
INFO:   DEPLOYMENT_MODE: registry-only
INFO:   REGISTRY_MODE: skills-only
INFO:   Nginx updates: DISABLED
```

If an invalid combination is detected:

```
WARNING: ============================================================
WARNING: Invalid configuration detected!
WARNING: DEPLOYMENT_MODE=with-gateway is incompatible with REGISTRY_MODE=skills-only
WARNING: Auto-correcting to DEPLOYMENT_MODE=registry-only
WARNING: ============================================================
```

## Nginx Behavior in Registry-Only Mode

When `DEPLOYMENT_MODE=registry-only`:

1. **Server Registration:** Nginx configuration is NOT updated
2. **Server Deletion:** Nginx configuration is NOT updated
3. **MCP Proxy Requests:** Return 503 with JSON error:

```json
{
  "error": "gateway_proxy_disabled",
  "message": "Gateway proxy is disabled in registry-only mode. Use proxy_pass_url from server metadata for direct connection."
}
```

The 503 response applies to all paths except:
- `/api/*` - Registry API endpoints
- `/oauth2/*` - Authentication endpoints
- `/keycloak/*`, `/realms/*`, `/resources/*` - Keycloak paths
- `/v0.1/*` - Anthropic-compatible API
- `/health` - Health check
- `/static/*`, `/assets/*`, `/_next/*` - Static assets
- `/validate` - Token validation

## CLI Testing

Use the registry management CLI to check configuration:

```bash
# Check current configuration
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    config --json

# Output formatted for readability
uv run python api/registry_management.py \
    --registry-url http://localhost \
    --token-file .token \
    config
```

## Related Documentation

- [Configuration Reference](configuration.md) - All environment variables
- [AWS ECS Deployment](../terraform/aws-ecs/README.md) - Production deployment guide
- [Static Token Auth](static-token-auth.md) - API authentication without IdP
