# How do I test my agent's integration with the MCP Gateway locally?

Follow these steps:

1. **Set up local environment**:
   ```bash
   git clone https://github.com/agentic-community/mcp-gateway-registry.git
   cd mcp-gateway-registry
   cp .env.template .env
   # Configure your .env file
   ./build_and_run.sh
   ```

2. **Test authentication**:
   ```bash
   # For user identity mode
   cd agents/
   python cli_user_auth.py
   python agent.py --use-session-cookie --message "test message"
   
   # For agent identity mode
   python agent.py --message "test message"
   ```

3. **Access the web interface** at `http://localhost` to verify server registration and tool availability.

## Related Documentation

- [Quick Start Guide](../quickstart.md) -- getting started with the registry
- [Installation Guide](../installation.md) -- detailed deployment instructions
- [Authentication](../auth.md) -- authentication modes and configuration
