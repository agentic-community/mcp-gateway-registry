# How do I monitor the health of MCP servers?

The registry provides built-in health monitoring:

1. **Web Interface**: View server status at `https://your-gateway`
   - Green: Healthy servers
   - Red: Servers with issues
   - Gray: Disabled servers

2. **Manual Health Checks**: Click the refresh icon on any server card in the dashboard

3. **Logs**: Monitor service logs:
   ```bash
   # View all service logs
   docker compose logs -f
   
   # View specific service logs
   docker compose logs -f registry
   docker compose logs -f auth-server
   ```

4. **API Endpoint**: Programmatic health checks via `/health` endpoints

## Related Documentation

- [Service Management](../service-management.md) -- managing MCP server lifecycle
- [Observability](../OBSERVABILITY.md) -- monitoring and telemetry
