# Global Chat MCP Server

Cross-protocol AI agent discovery server for the MCP Gateway & Registry.

## Overview

[Global Chat](https://global-chat.io) is the discovery layer for AI agents — a protocol-agnostic directory that aggregates agents across MCP, A2A, agents.txt, ACDP, and 9+ other protocols. This MCP server exposes Global Chat's agent discovery capabilities as MCP tools.

## Tools

| Tool | Description |
|------|-------------|
| `search_agents` | Search for agents by keyword across all 15+ protocols |
| `get_agent` | Get detailed metadata for a specific agent |
| `list_protocols` | List all supported agent discovery protocols with counts |
| `validate_agents_txt` | Validate an agents.txt file for spec compliance |

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
python server.py --port 9010 --transport streamable-http
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_LISTEN_PORT` | `9010` | Port to listen on |
| `MCP_TRANSPORT` | `streamable-http` | Transport type (`sse` or `streamable-http`) |

## Links

- **Website**: https://global-chat.io
- **npm package**: [@global-chat/mcp-server](https://www.npmjs.com/package/@global-chat/mcp-server)
- **GitHub**: https://github.com/geetchoubey/global-chat
- **agents.txt Validator**: https://global-chat.io/validator
