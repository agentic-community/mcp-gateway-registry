# Importing Servers from Anthropic MCP Registry

This guide explains how to import MCP servers from [Anthropic's official MCP Registry](https://registry.modelcontextprotocol.io/) into your MCP Gateway.

## Overview

The Anthropic MCP Registry is an open, collaboratively governed directory of Model Context Protocol (MCP) servers. It is maintained by Anthropic in partnership with GitHub and the wider community through an open-source contribution model. Its API lets MCP clients and gateways discover and import server configurations.

The gateway imports these servers through federation. You add a server name to the Anthropic section of the federation configuration, then run a federation sync, and the registry fetches the server's definition and registers it. Imported servers carry the `ANTHROPIC` badge in the UI.

> **Changed in 1.34.0:** the shell script `cli/import_from_anthropic_registry.sh` and its list file `cli/import_server_list.txt` were removed, along with `cli/service_mgmt.sh`. Use the commands below instead.

## Prerequisites

- MCP Gateway up and running
- The Anthropic federation source enabled, as described in the [Federation Guide](federation.md#anthropic-mcp-registry-integration) (`ANTHROPIC_REGISTRY_ENABLED=true` and an `anthropic` block in the federation configuration)
- A token for a user with federation management rights, for example `.token` on a local deployment

> **Note**: The Anthropic API version is defined in `registry/constants.py` as `ANTHROPIC_API_VERSION`.

## Quick Start

### Import a Single Server

```bash
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
  federation-add-anthropic-server --server-name ai.smithery/smithery-ai-github

uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
  federation-sync --source anthropic
```

The first command adds the server to the federation configuration. The second fetches it from the Anthropic registry and registers it.

The interactive CLI does both in one step with its **Anthropic import** task, or directly:

```bash
uv run python cli/registry_cli_wrapper.py --base-url http://localhost --token-file .token \
  anthropic import ai.smithery/smithery-ai-github
```

### Import Several Servers

Add each server, then sync once:

```bash
for server in ai.smithery/smithery-ai-github io.github.jgador/websharp; do
  uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
    federation-add-anthropic-server --server-name "$server"
done

uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
  federation-sync --source anthropic
```

The [Federation Guide](federation.md#import-specific-servers-for-anthropic) also shows how to list the servers directly in the federation configuration.

### Remove an Imported Server

```bash
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
  federation-remove-anthropic-server --server-name ai.smithery/smithery-ai-github
```

Removing a server from the federation configuration also removes the server it imported.

## Finding Servers to Import

Visit [registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io/) to browse servers, their tools and their authentication requirements, or query its API:

```bash
# List available servers
curl https://registry.modelcontextprotocol.io/v0.1/servers | jq '.servers[] | .name'

# Details for one server
curl https://registry.modelcontextprotocol.io/v0.1/servers/ai.smithery%2Fsmithery-ai-github/versions/latest | jq '.'
```

## Authentication for Imported Servers

The import never copies credentials from a remote server definition: imported servers are registered with no authentication headers. A server that needs an API key or token reports unhealthy, and its tools fail until it receives credentials, so import servers that work without them, or register servers that need credentials yourself with `register --config`.

## Verifying Imported Servers

```bash
# Find the imported server
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
  list --query smithery

# Check health
uv run python api/registry_management.py --registry-url http://localhost --token-file .token \
  healthcheck
```

An imported server's registry path is its Anthropic name with `/` replaced by `-`, so `ai.smithery/smithery-ai-github` becomes `/ai.smithery-smithery-ai-github`. In the web UI, imported servers appear on the MCP Servers tab with the `ANTHROPIC` badge.

## Troubleshooting

### The server is not imported after a sync

Check the registry logs for the federation sync:

```bash
docker compose logs registry | grep -i anthropic
```

Confirm the Anthropic source is enabled (see Prerequisites) and that the server name matches the Anthropic registry exactly, including the namespace before the `/`.

### "already exists in configuration"

The server is already in the federation configuration. Run `federation-sync --source anthropic` to refresh it, or remove and add it again.

### The server shows as unhealthy

A server that needs credentials fails its health checks, because the import does not copy any (see Authentication for Imported Servers). Otherwise, check that the server's URL is reachable from the registry.

## Related Documentation

- [Federation Guide](federation.md)
- [Federation Operational Guide](federation-operational-guide.md)
- [Security Scanner](security-scanner.md)
