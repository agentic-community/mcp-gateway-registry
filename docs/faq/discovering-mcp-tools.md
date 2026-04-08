# How do I discover available MCP tools for my AI agent?

You can discover tools in several ways:

1. **Dynamic Tool Discovery** (Recommended): Use the [`intelligent_tool_finder`](../dynamic-tool-discovery.md) tool with natural language queries:
   ```python
   tools = await intelligent_tool_finder(
       natural_language_query="get current time in different timezones",
       session_cookie="your_session_cookie"
   )
   ```

2. **Web Interface**: Browse available tools at `https://your-gateway-url` after authentication.

3. **Direct MCP Connection**: Connect to the registry MCP server at `/mcpgw/sse` and use standard MCP `tools/list` calls.

## Related Documentation

- [Dynamic Tool Discovery](../dynamic-tool-discovery.md) -- full guide on autonomous tool discovery
- [API Reference](../api-reference.md) -- search and listing endpoints
