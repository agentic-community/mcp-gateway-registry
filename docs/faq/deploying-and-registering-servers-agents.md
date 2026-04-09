# How do I deploy and register MCP servers and agents?

MCP servers and agents are built and deployed **out of band** -- the MCP Gateway Registry does not host or run them. You build and deploy your servers and agents using whatever framework and infrastructure you prefer, then register them in the registry so they can be discovered and accessed through the gateway.

## Building and Deploying MCP Servers

You can build MCP servers using any MCP-compatible framework and deploy them on any infrastructure:

**Frameworks:**
- [FastMCP](https://github.com/PrefectHQ/fastmcp) -- Python framework for building MCP servers
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) -- Official TypeScript SDK
- Any framework that implements the [MCP specification](https://modelcontextprotocol.io/specification)

**Deployment options:**
- [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) -- managed runtime for MCP servers
- Amazon EKS or any Kubernetes cluster
- AWS ECS, Azure Container Apps, Google Cloud Run
- A standalone Linux instance with Docker or systemd
- Any cloud or on-premises infrastructure that can serve HTTP endpoints

Your deployed server needs to expose an MCP-compatible endpoint (typically `/mcp` for Streamable HTTP) that the gateway can reach over the network.

## Building and Deploying A2A Agents

Similarly, agents are built using any agent framework and deployed independently:

**Frameworks:**
- [A2A Python SDK](https://github.com/a2aproject/a2a-python) -- reference implementation for A2A protocol
- [LangGraph](https://github.com/langchain-ai/langgraph) with A2A adapter
- [CrewAI](https://github.com/crewAIInc/crewAI) -- multi-agent orchestration
- Any framework that exposes an [A2A-compatible agent card](https://a2a-protocol.org/) at `/.well-known/agent-card.json`

**Deployment options:**
- [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) -- managed runtime for A2A agents
- Amazon EKS or any Kubernetes cluster
- AWS Lambda behind API Gateway
- Any cloud infrastructure that can serve HTTP endpoints

Your deployed agent needs to expose its agent card at `/.well-known/agent-card.json` and handle A2A protocol requests.

## Registering in the Registry

Once your server or agent is deployed and accessible, register it in the MCP Gateway Registry using one of these methods:

### Option 1: Register through the Web UI

1. Open the registry dashboard
2. Click **Register** on the MCP Servers or Agents tab
3. Fill in the form with your server/agent details (URL, name, description, etc.)
4. Click Submit

### Option 2: Generate JSON cards using Claude Code skills

Use the built-in Claude Code skills to generate registration JSON by analyzing your source code:

```bash
# For MCP servers -- analyzes source code and generates a server card JSON
/generate-server-card

# For A2A agents -- analyzes source code and generates an agent card JSON
/generate-agent-card
```

These skills produce JSON files that can be uploaded through the UI or used with the API.

### Option 3: Register programmatically via API

Use the [Registry Management CLI](../../api/registry_management.py) to register from the command line.

To get a token, click the **"Get JWT Token"** button in the top-left corner of the registry UI, then click **"Copy JSON"** and save it to a `.token` file:

```bash
# Create .token file with the copied JSON from the registry UI
cat > .token << 'EOF'
<paste the copied JSON here>
EOF

# Register an MCP server from a JSON config file
uv run python api/registry_management.py \
    --registry-url https://your-registry-url \
    --token-file .token \
    register --config my-server-card.json

# Register an A2A agent from a JSON config file
uv run python api/registry_management.py \
    --registry-url https://your-registry-url \
    --token-file .token \
    agent-register --config my-agent-card.json
```

You can also call the REST API directly. See the [OpenAPI specification](../../api/openapi.json) for the full API reference, available at `/openapi.json` on your running registry instance.

### Example JSON files

See the [cli/examples/](../../cli/examples/) directory for complete registration examples:

**MCP Servers:**
- `currenttime.json` -- minimal server example
- `cloudflare-docs-server-config.json` -- server with full configuration
- `complete-server-example.json` -- all available fields documented

**A2A Agents:**
- `flight_booking_agent_card.json` -- agent with multiple skills
- `code_reviewer_agent.json` -- agent with JWT auth and verified trust level
- `complete-agent-example.json` -- all available fields documented

## Related Documentation

- [Quick Start Guide](../quickstart.md) -- getting the registry running
- [Service Management](../service-management.md) -- managing servers, agents, users, and groups
- [API Reference](../api-reference.md) -- REST API endpoints
- [AI Coding Assistants Setup](../ai-coding-assistants-setup.md) -- connecting AI tools to registered servers
