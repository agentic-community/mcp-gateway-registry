# How do I handle tool discovery when I don't know what tools are available?

Use the Dynamic Tool Discovery feature:

1. **In your agent code**:
   ```python
   # Let your agent discover tools autonomously
   tools = await intelligent_tool_finder(
       natural_language_query="I need to get stock market data",
       session_cookie=session_cookie,
       top_n_tools=3
   )
   
   # Then invoke the discovered tool
   if tools:
       result = await invoke_mcp_tool(
           server_name=tools[0]["service_path"],
           tool_name=tools[0]["tool_name"],
           arguments={"symbol": "AAPL"},
           # ... auth parameters
       )
   ```

2. **Configure your agent** with tool discovery capabilities as shown in the [Dynamic Tool Discovery guide](../dynamic-tool-discovery.md).

## Related Documentation

- [Dynamic Tool Discovery](../dynamic-tool-discovery.md) -- complete guide with configuration details
- [AI Registry Tools](../ai-registry-tools.md) -- available registry tools for agents
