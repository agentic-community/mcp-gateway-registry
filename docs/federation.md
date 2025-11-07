# Registry Federation

This guide explains how to configure the MCP Gateway Registry to federate with external registries like Anthropic MCP Registry and Workday ASOR.

## Overview

Registry federation allows your MCP Gateway to:
- **Discover and display servers** from external registries
- **Maintain read-only access** to federated servers
- **Cache server data** to reduce API calls
- **Sync automatically** on startup and at configured intervals
- **Support multiple registries** simultaneously

### Supported Registries

1. **Anthropic MCP Registry** - Public registry of community MCP servers
2. **ASOR (Agent Service Operating Registry)** - Enterprise agent registry from Workday (coming soon)

## Configuration

### Configuration File

Federation is configured via `config/federation.json`. An example configuration is provided in `config/federation.example.json`.

**Default Location:**
- Production: `/app/config/federation.json`
- Custom path: Set `FEDERATION_CONFIG_PATH` environment variable

### Basic Structure

```json
{
  "anthropic": {
    "enabled": true,
    "endpoint": "https://registry.modelcontextprotocol.io",
    "api_version": "v0.1",
    "servers": [...],
    "cache_ttl_seconds": 3600,
    "sync_interval_seconds": 300,
    "display_options": {...}
  },
  "asor": {
    "enabled": false,
    ...
  }
}
```

## Anthropic MCP Registry Federation

### Enable Anthropic Federation

1. **Copy Example Configuration:**
   ```bash
   cp config/federation.example.json config/federation.json
   ```

2. **Edit Configuration:**
   ```bash
   nano config/federation.json
   ```

3. **Set `enabled: true`** in the `anthropic` section

### Configure Servers to Federate

Add servers you want to federate in the `servers` array:

```json
{
  "anthropic": {
    "enabled": true,
    "servers": [
      {
        "name": "ai.smithery/smithery-ai-github",
        "requires_auth": true,
        "auth_type": "api-key",
        "auth_env_var": "SMITHERY_API_KEY",
        "enabled": true,
        "metadata": {
          "description": "GitHub API access",
          "category": "development"
        }
      },
      {
        "name": "io.github.jgador/websharp",
        "requires_auth": false,
        "enabled": true,
        "metadata": {
          "description": "Web search and article extraction",
          "category": "search"
        }
      }
    ]
  }
}
```

### Authentication Setup

For servers requiring authentication (like Smithery servers):

1. **Add API Key to Environment:**
   ```bash
   echo "SMITHERY_API_KEY=your-api-key-here" >> .env
   ```

2. **Configure in federation.json:**
   ```json
   {
     "name": "ai.smithery/example-server",
     "requires_auth": true,
     "auth_type": "api-key",
     "auth_env_var": "SMITHERY_API_KEY"
   }
   ```

### Configuration Options

#### Server Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Server name in Anthropic format (e.g., `ai.smithery/github`) |
| `endpoint` | string | No | Custom API endpoint (if not using default pattern) |
| `requires_auth` | boolean | No | Whether authentication is required (default: `false`) |
| `auth_type` | string | No | Authentication type: `api-key`, `oauth`, `bearer` |
| `auth_env_var` | string | Conditional | Environment variable with auth credentials (required if `requires_auth: true`) |
| `enabled` | boolean | No | Whether to sync this server (default: `true`) |
| `metadata` | object | No | Additional metadata for organization |

#### Federation Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable Anthropic federation |
| `endpoint` | string | `https://registry.modelcontextprotocol.io` | Anthropic API base URL |
| `api_version` | string | `v0.1` | API version to use |
| `cache_ttl_seconds` | integer | `3600` | Cache lifetime in seconds (1 hour) |
| `sync_interval_seconds` | integer | `300` | Sync interval in seconds (5 minutes) |
| `sync_on_startup` | boolean | `true` | Sync on registry startup |
| `timeout_seconds` | integer | `30` | HTTP request timeout |
| `retry_attempts` | integer | `3` | Number of retry attempts for failed requests |

#### Display Options

```json
{
  "display_options": {
    "mark_as_federated": true,
    "attribution_label": "Anthropic MCP Registry",
    "separate_section": true,
    "read_only": true
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mark_as_federated` | boolean | `true` | Show visual indicator for federated servers |
| `attribution_label` | string | `"Anthropic MCP Registry"` | Label showing source registry |
| `separate_section` | boolean | `true` | Display in separate UI section |
| `read_only` | boolean | `true` | Federated servers are read-only |

## Finding Servers to Federate

### Browse Anthropic Registry

Visit [registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io/) to discover available servers.

### List Servers via API

```bash
# List all available servers
curl https://registry.modelcontextprotocol.io/v0.1/servers | jq '.servers[] | .name'

# Get details for a specific server
curl https://registry.modelcontextprotocol.io/v0.1/servers/ai.smithery%2Fsmithery-ai-github/versions/latest | jq '.'
```

### Pre-configured Servers

The repository includes a curated list in `config/federation.json` with popular servers:
- GitHub API access
- Web search and extraction
- Obsidian integration
- Google Forms management
- GitHub PR/issue analysis
- And more...

## ASOR Federation

> **Note:** ASOR federation is currently planned but not yet implemented. The configuration structure is in place for future support.

### Configuration Structure

```json
{
  "asor": {
    "enabled": false,
    "endpoint": "https://api.asor.workday.com/v1",
    "agents": [
      {
        "id": "agent-123",
        "endpoint": "https://asor-instance/agents/123",
        "enabled": true,
        "metadata": {
          "description": "Example ASOR agent"
        }
      }
    ],
    "cache_ttl_seconds": 3600,
    "sync_interval_seconds": 300,
    "auth_type": "oauth2",
    "auth_env_var": "ASOR_API_KEY"
  }
}
```

## Operation

### Startup Behavior

When the registry starts:

1. **Loads configuration** from `config/federation.json`
2. **Checks if federation is enabled**
3. **Syncs servers** from enabled registries (if `sync_on_startup: true`)
4. **Caches server data** for fast access
5. **Logs sync results** with success/failure counts

Example startup logs:
```
🔗 Initializing federation service...
Federation enabled for: anthropic
🔄 Syncing servers from federated registries on startup...
Fetching server ai.smithery/smithery-ai-github from Anthropic Registry
✅ Synced 6 servers from anthropic
```

### Runtime Behavior

- **Cache Management:** Federated servers are cached with configurable TTL
- **Automatic Refresh:** Cache automatically refreshes when expired
- **Read-Only Access:** Federated servers cannot be modified locally
- **Attribution:** UI displays source registry for each federated server

### Accessing Federated Servers

Federated servers appear in the registry alongside local servers:

```python
from registry.services.server_service import server_service

# Get all servers (including federated)
all_servers = server_service.get_all_servers(include_federated=True)

# Get only local servers
local_servers = server_service.get_all_servers(include_federated=False)
```

## Cache Management

### Cache Location

- Memory cache: In-memory for fast access
- Disk cache: `/app/.cache/federation/` for persistence across restarts
- Custom path: Set `FEDERATION_CACHE_DIR` environment variable

### Cache Files

```
/app/.cache/federation/
├── anthropic_cache.json
└── asor_cache.json (when implemented)
```

### Manual Cache Operations

The federation service provides methods for cache management:

```python
from registry.services.federation_service import get_federation_service

federation_service = get_federation_service()

# Force refresh from source
servers = federation_service.get_federated_servers(force_refresh=True)

# Clear cache for specific source
federation_service.clear_cache("anthropic")

# Clear all caches
federation_service.clear_cache()

# Get cache statistics
stats = federation_service.get_cache_stats()
```

## Troubleshooting

### Federation Not Working

**Check logs for initialization:**
```bash
docker compose logs registry | grep -i federation
```

**Expected output:**
```
🔗 Initializing federation service...
Federation enabled for: anthropic
```

### Server Sync Failures

**Problem:** Servers not syncing from Anthropic

**Check:**
1. Federation enabled in config: `"enabled": true`
2. Network connectivity to `registry.modelcontextprotocol.io`
3. Authentication credentials if required

**Logs:**
```bash
docker compose logs registry | grep -i "anthropic"
```

### Authentication Errors

**Problem:** `Environment variable SMITHERY_API_KEY not found`

**Solution:**
1. Add API key to `.env` file
2. Restart registry: `docker compose restart registry`
3. Verify environment variable: `docker compose exec registry env | grep SMITHERY`

### Cache Issues

**Problem:** Stale data or old servers showing

**Solution:**
```bash
# Clear federation cache
docker compose exec registry rm -rf /app/.cache/federation/*

# Restart registry to re-sync
docker compose restart registry
```

### Server Not Appearing in UI

**Check:**
1. Server enabled in config: `"enabled": true`
2. Sync completed successfully (check logs)
3. Cache not expired
4. User has permissions to view federated servers

## Best Practices

### Security

1. **Protect API Keys:** Never commit `.env` to version control
2. **Use Read-Only Keys:** Use minimal permission API keys when possible
3. **Review Servers:** Only federate servers you trust
4. **Monitor Logs:** Check logs for suspicious activity

### Performance

1. **Adjust TTL:** Balance freshness vs. API call frequency
   - Higher TTL = Fewer API calls, less fresh data
   - Lower TTL = More API calls, fresher data

2. **Selective Federation:** Only federate servers you need
3. **Monitor Cache Size:** Large cache = more memory usage

### Maintenance

1. **Regular Updates:** Periodically review and update server list
2. **Test New Servers:** Verify servers before adding to production
3. **Monitor Health:** Check federated server health status
4. **Update Documentation:** Document custom configurations

## Examples

### Minimal Configuration

Federation disabled (default):
```json
{
  "anthropic": {
    "enabled": false
  },
  "asor": {
    "enabled": false
  }
}
```

### Development Configuration

Fast sync for testing:
```json
{
  "anthropic": {
    "enabled": true,
    "cache_ttl_seconds": 300,
    "sync_interval_seconds": 60,
    "servers": [
      {
        "name": "io.github.jgador/websharp",
        "requires_auth": false
      }
    ]
  }
}
```

### Production Configuration

Balanced settings:
```json
{
  "anthropic": {
    "enabled": true,
    "cache_ttl_seconds": 3600,
    "sync_interval_seconds": 300,
    "timeout_seconds": 30,
    "retry_attempts": 3,
    "servers": [
      {
        "name": "ai.smithery/smithery-ai-github",
        "requires_auth": true,
        "auth_type": "api-key",
        "auth_env_var": "SMITHERY_API_KEY"
      },
      {
        "name": "io.github.jgador/websharp",
        "requires_auth": false
      }
    ]
  }
}
```

## Related Documentation

- [Anthropic MCP Registry API](anthropic_registry_api.md)
- [Anthropic Registry Import (Legacy)](anthropic-registry-import.md)
- [Service Management](service-management.md)
- [Authentication Setup](../README.md#authentication)

## Migration from Legacy Import Script

The federation approach replaces the legacy shell script import method (`cli/import_from_anthropic_registry.sh`).

### Key Differences

| Legacy Script | Federation |
|---------------|------------|
| Manual execution | Automatic on startup |
| Shell script | Python service |
| One-time import | Continuous sync with caching |
| No caching | Built-in cache with TTL |
| No UI attribution | Clear source labeling |
| Permanent imports | Read-only federated servers |

### Migration Steps

1. **Review Current Imports:** Check which servers were imported via script
2. **Create Federation Config:** Add those servers to `config/federation.json`
3. **Add Authentication:** Move API keys from script to `.env`
4. **Test Configuration:** Start registry and verify sync
5. **Remove Old Imports:** Optionally remove script-imported servers

## Support

For issues or questions:
- GitHub Issues: [mcp-gateway-registry/issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- Related Issue: [#204 - Implement ASOR Federation](https://github.com/agentic-community/mcp-gateway-registry/issues/204)
