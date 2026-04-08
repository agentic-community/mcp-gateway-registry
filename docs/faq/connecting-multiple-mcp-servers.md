# How do I connect my agent to multiple MCP servers through the gateway?

The gateway provides a single endpoint with path-based routing:

```python
# Connect to different servers via the gateway using SSE client
from mcp import ClientSession
from mcp.client.sse import sse_client

async def connect_to_server(server_url):
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Use the session for tool calls
            return session

# Example server URLs through the gateway
server_url = f"https://your-gateway.com/currenttime/sse"
time_session = await connect_to_server(server_url)

# Or use the registry's tool discovery
registry_url = f"https://your-gateway.com/mcpgw/sse"
registry_session = await connect_to_server(registry_url)
```

All requests go through the same gateway with authentication handled centrally.

## Related Documentation

- [Authentication](../auth.md) -- authentication modes and headers
- [Installation](../installation.md) -- gateway deployment and configuration
